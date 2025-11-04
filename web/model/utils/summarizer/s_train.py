# utils/summarizer/s_train.py
import os, math, time, contextlib, torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# ---- AMP 호환 유틸 (신/구버전, CUDA/MPS/CPU) ---------------------------------
def _resolve_device_and_type(device=None):
    """device가 None이면 자동 선택 + device.type 문자열 반환."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    device_type = device.type  # "cuda" | "mps" | "cpu"
    return device, device_type

def _amp_autocast_ctx(device_type: str, enabled: bool):
    """
    torch.amp.autocast(device_type=...) 우선 시도,
    실패 시 구버전/대체 컨텍스트로 폴백.
    """
    if not enabled:
        return contextlib.nullcontext()

    # MPS는 AMP 미지원(2025 현재), 안전하게 끔
    if device_type == "mps":
        return contextlib.nullcontext()

    # 신 API
    try:
        from torch.amp import autocast as new_autocast  # type: ignore
        return new_autocast(device_type=device_type, enabled=True)
    except Exception:
        pass

    # CUDA 구버전
    if device_type == "cuda":
        try:
            from torch.cuda.amp import autocast as cuda_autocast
            return cuda_autocast(enabled=True)
        except Exception:
            return contextlib.nullcontext()

    # CPU AMP (있으면)
    if device_type == "cpu":
        try:
            from torch.cpu.amp import autocast as cpu_autocast  # may not exist
            return cpu_autocast(enabled=True)
        except Exception:
            return contextlib.nullcontext()

    return contextlib.nullcontext()

def _amp_scaler(device_type: str, enabled: bool):
    """
    GradScaler를 버전에 맞춰 생성. 미지원(또는 MPS 등) 시 no-op 스케일러 반환.
    """
    if not enabled or device_type in ("mps", "cpu"):
        # no-op scaler
        class _NoScaler:
            def scale(self, x): return x
            def unscale_(self, *args, **kw): pass
            def step(self, opt): opt.step()
            def update(self): pass
        return _NoScaler()

    # 시도1: torch.amp.GradScaler (신버전)
    try:
        from torch.amp import GradScaler as NewGradScaler  # type: ignore
        try:
            # 일부 버전은 device_type 위치 인자를 안 받음
            return NewGradScaler(enabled=True)
        except TypeError:
            return NewGradScaler()
    except Exception:
        pass

    # 시도2: torch.cuda.amp.GradScaler (구버전)
    try:
        from torch.cuda.amp import GradScaler as CudaGradScaler
        return CudaGradScaler(enabled=True)
    except Exception:
        # 최후 폴백: no-op
        class _NoScaler:
            def scale(self, x): return x
            def unscale_(self, *args, **kw): pass
            def step(self, opt): opt.step()
            def update(self): pass
        return _NoScaler()

# ---- 모델 호출/로스 -----------------------------------------------------------
def _forward_model(model, X, lengths, mask):
    """모델 시그니처가 (X, lengths) | (X, mask=) | (X) 무엇이든 처리."""
    try:
        return model(X, lengths)
    except TypeError:
        try:
            return model(X, mask=mask)
        except TypeError:
            return model(X)

def _masked_mse(pred, target, mask):
    """마스크가 있으면 유효 위치만 MSE."""
    if mask is None:
        return nn.functional.mse_loss(pred, target)
    m = mask.unsqueeze(-1).expand_as(target)
    diff = (pred - target)[m]
    if diff.numel() == 0:
        return torch.tensor(0.0, device=target.device)
    return torch.mean(diff * diff)

@torch.no_grad()
def validate_one_epoch(model, loader, device):
    model.eval()
    running, n = 0.0, 0
    for batch in loader:
        X = torch.nan_to_num(batch["X"].to(device), nan=0.0, posinf=1e6, neginf=-1e6)
        mask = batch.get("mask", None)
        if mask is not None:
            mask = mask.to(device)
        lengths = batch.get("lengths", None)
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.to(device)

        out = _forward_model(model, X, lengths, mask)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
        loss = _masked_mse(out, X, mask)
        running += float(loss.item())
        n += 1
    return running / max(n, 1)

# ---- 학습 루프 ----------------------------------------------------------------
def train(
    model,
    train_loader,
    val_loader=None,
    *,
    epochs=20,
    lr=1e-3,
    weight_decay=0.0,
    device=None,
    grad_clip=1.0,
    amp=True,
    scheduler=None,
    ckpt_dir="./checkpoints",
    ckpt_name="best.pt",
    log_every=50,
    tb_logdir="./runs/ae",
    tb_hist_every=None,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_logdir)

    device, device_type = _resolve_device_and_type(device)
    # MPS에서는 AMP 자동 비활성화
    if device_type == "mps":
        amp = False

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = _amp_scaler(device_type, enabled=amp)

    best_val = math.inf
    global_step = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running, n_batches = 0.0, 0

        for i, batch in enumerate(train_loader, start=1):
            X = torch.nan_to_num(batch["X"].to(device), nan=0.0, posinf=1e6, neginf=-1e6)
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(device)
            lengths = batch.get("lengths", None)
            if isinstance(lengths, torch.Tensor):
                lengths = lengths.to(device)

            optimizer.zero_grad(set_to_none=True)

            with _amp_autocast_ctx(device_type, enabled=amp):
                out = _forward_model(model, X, lengths, mask)
                out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
                loss = _masked_mse(out, X, mask)

            # 비정상 loss 방지
            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss detected: {loss.item():.4g} -> skip step")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # logs
            global_step += 1
            running += float(loss.item()); n_batches += 1
            writer.add_scalar("loss/train_step", float(loss.item()), global_step)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)

            if (i % log_every) == 0:
                avg = running / max(n_batches, 1)
                print(f"[Epoch {epoch}/{epochs}] step {i:04d} | train_loss={avg:.6f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        train_loss = running / max(n_batches, 1)
        writer.add_scalar("loss/train_epoch", train_loss, epoch)

        # Validation
        if val_loader is not None:
            val_loss = validate_one_epoch(model, val_loader, device)
            writer.add_scalar("loss/val_epoch", val_loss, epoch)
            print(f"==> Epoch {epoch}: train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | elapsed={time.time()-t0:.1f}s")

            if scheduler is not None:
                if hasattr(scheduler, "step") and "ReduceLROnPlateau" in scheduler.__class__.__name__:
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # best ckpt
            if val_loss < best_val and torch.isfinite(torch.tensor(val_loss)):
                best_val = val_loss
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                    os.path.join(ckpt_dir, ckpt_name),
                )
                print(f"    ✓ New best saved (val_loss={val_loss:.6f})")
        else:
            print(f"==> Epoch {epoch}: train_loss={train_loss:.6f} | elapsed={time.time()-t0:.1f}s")
            if scheduler is not None:
                scheduler.step()

        # (옵션) 히스토그램 기록
        if tb_hist_every is not None and (epoch % tb_hist_every == 0):
            for name, p in model.named_parameters():
                writer.add_histogram(f"params/{name}", p.detach().cpu().numpy(), epoch)
                if p.grad is not None:
                    writer.add_histogram(f"grads/{name}", p.grad.detach().cpu().numpy(), epoch)

    writer.close()
    print("Training finished. Logs at:", tb_logdir)