# # training.py
# # -*- coding: utf-8 -*-
# """
# High-performance trainer for UTI hadm-centric pipeline.

# - Data: HadmTableDatasetV2 (no admissions.csv)
# - Model: TableTransformerPredictor (masked table-attention + presence + missing tokens + FiLM)
# - Loss:
#     total = w_los * LOS_loss + w_cls * CLS_loss
#   * LOS_loss options: huber (default), mae, mse, quantile (pinball; tau controllable)
#   * CLS_loss options: bce (default, with pos_weight auto), focal (alpha, gamma)
# - Metrics:
#   * readmission: epoch AUROC (torchmetrics if available), accuracy@0.5
#   * LOS custom: accuracy within ±tol days (default tol=1.0 day)
# - Training niceties: AMP, grad clip, schedulers (none/onecycle/cosine/cosine_warmup),
#   early stopping, resume, TensorBoard logging.

# This file is intentionally arg-agnostic; call train(...) from your main.py.
# """

# import os, math, time, json, random, contextlib
# from typing import Optional, List

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter


# # =======================
# # Utilities
# # =======================
# def set_seed(seed: int = 1337):
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# def resolve_device(device=None):
#     if device is not None:
#         return torch.device(device)
#     if torch.cuda.is_available():
#         return torch.device("cuda")
#     if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
#         return torch.device("mps")
#     return torch.device("cpu")

# def autocast_ctx(device: torch.device, enabled: bool):
#     if not enabled:
#         return contextlib.nullcontext()
#     try:
#         from torch.amp import autocast as amp_autocast
#         return amp_autocast(device_type=device.type, enabled=True)
#     except Exception:
#         if device.type == "cuda":
#             try:
#                 from torch.cuda.amp import autocast as cuda_autocast
#                 return cuda_autocast(enabled=True)
#             except Exception:
#                 return contextlib.nullcontext()
#         return contextlib.nullcontext()

# def make_scaler(device: torch.device, enabled: bool):
#     if not enabled:
#         class _Null:
#             def scale(self, x): return x
#             def step(self, opt): opt.step()
#             def update(self): pass
#             def unscale_(self, opt): pass
#         return _Null()
#     try:
#         from torch.amp import GradScaler
#         return GradScaler(enabled=True)
#     except Exception:
#         try:
#             from torch.cuda.amp import GradScaler as CudaGradScaler
#             return CudaGradScaler(enabled=(device.type=="cuda"))
#         except Exception:
#             return make_scaler(device, enabled=False)


# # =======================
# # Losses & Metrics
# # =======================
# def _valid_mask(y: Optional[torch.Tensor]):
#     if y is None:
#         return None
#     return torch.isfinite(y)

# def _huber(x, y, delta=1.0, mask=None):
#     if mask is not None:
#         x = x[mask]; y = y[mask]
#     if x.numel() == 0:
#         dev = x.device if torch.is_tensor(x) else (y.device if torch.is_tensor(y) else torch.device("cpu"))
#         return torch.tensor(0.0, device=dev)
#     return F.smooth_l1_loss(x, y, beta=delta)

# def _mae(x, y, mask=None):
#     if mask is not None:
#         x = x[mask]; y = y[mask]
#     if x.numel() == 0:
#         return torch.tensor(0.0, device=y.device)
#     return F.l1_loss(x, y)

# def _mse(x, y, mask=None):
#     if mask is not None:
#         x = x[mask]; y = y[mask]
#     if x.numel() == 0:
#         return torch.tensor(0.0, device=y.device)
#     return F.mse_loss(x, y)

# def _quantile(pred, target, tau=0.5, mask=None):
#     if mask is not None:
#         pred = pred[mask]; target = target[mask]
#     if pred.numel() == 0:
#         return torch.tensor(0.0, device=target.device)
#     e = target - pred
#     return torch.maximum(tau * e, (tau - 1.0) * e).mean()

# def _bce_with_logits_masked(logits, target, pos_weight=None, label_smoothing=0.0, mask=None):
#     if mask is not None:
#         logits = logits[mask]; target = target[mask]
#     if logits.numel() == 0:
#         return torch.tensor(0.0, device=target.device)
#     if label_smoothing and label_smoothing > 0:
#         target = target * (1.0 - label_smoothing) + 0.5 * label_smoothing
#     return F.binary_cross_entropy_with_logits(
#         logits, target.float(), pos_weight=pos_weight
#     )

# def _focal_bce_with_logits(logits, target, alpha=0.25, gamma=2.0, label_smoothing=0.0, mask=None):
#     if mask is not None:
#         logits = logits[mask]; target = target[mask]
#     if logits.numel() == 0:
#         return torch.tensor(0.0, device=target.device)
#     if label_smoothing and label_smoothing > 0:
#         target = target * (1.0 - label_smoothing) + 0.5 * label_smoothing

#     bce = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
#     p = torch.sigmoid(logits)
#     pt = torch.where(target > 0.5, p, 1.0 - p)
#     alpha_t = torch.where(target > 0.5, torch.full_like(target, alpha), torch.full_like(target, 1 - alpha))
#     loss = alpha_t * ((1 - pt) ** gamma) * bce
#     return loss.mean()

# @torch.no_grad()
# def _acc_from_logits(logits, target):
#     if (logits is None) or (target is None):
#         dev = logits.device if logits is not None else (target.device if target is not None else torch.device("cpu"))
#         return torch.tensor(float("nan"), device=dev)
#     m = _valid_mask(target)
#     if (m is None) or (m.sum() == 0):
#         return torch.tensor(float("nan"), device=logits.device)
#     prob = torch.sigmoid(logits[m])
#     pred = (prob >= 0.5).long()
#     return (pred == target[m].long()).float().mean().detach()

# @torch.no_grad()
# def _los_within_tol_acc(pred, target, tol: float = 1.0, mask=None):
#     """
#     Accuracy for LOS regression:
#     counts a prediction 'correct' if |pred - target| <= tol (days).
#     """
#     if mask is not None:
#         pred = pred[mask]; target = target[mask]
#     if pred.numel() == 0:
#         return torch.tensor(float("nan"))
#     diff = torch.abs(pred - target)
#     return (diff <= tol).float().mean()

# @torch.no_grad()
# def _gather_probs_and_labels(logits, target, store_probs: List[torch.Tensor], store_labels: List[torch.Tensor]):
#     if (logits is None) or (target is None):
#         return
#     m = _valid_mask(target)
#     if (m is None) or (m.sum() == 0):
#         return
#     prob = torch.sigmoid(logits[m]).detach()
#     lab  = target[m].to(prob.dtype).detach()
#     store_probs.append(prob)
#     store_labels.append(lab)

# @torch.no_grad()
# def _epoch_auroc(probs_list: List[torch.Tensor], labels_list: List[torch.Tensor]):
#     if len(probs_list) == 0:
#         return torch.tensor(float("nan"))
#     prob = torch.cat([p.cpu() for p in probs_list], dim=0)
#     lab  = torch.cat([l.cpu() for l in labels_list], dim=0)
#     if lab.unique().numel() < 2:
#         return torch.tensor(float("nan"))
#     try:
#         from torchmetrics.functional.classification import binary_auroc
#         return binary_auroc(prob, lab)
#     except Exception:
#         return torch.tensor(float("nan"))


# @torch.no_grad()
# def infer_pos_weight_from_loader(loader, device, max_batches=200):
#     pos = neg = 0
#     seen = 0
#     for batch in loader:
#         y = batch.get("y_readmit")
#         if y is None:
#             continue
#         y = y.to(device)
#         m = _valid_mask(y)
#         if (m is None) or (m.sum() == 0):
#             continue
#         yy = y[m].view(-1).long()
#         pos += int((yy == 1).sum().item())
#         neg += int((yy == 0).sum().item())
#         seen += int(yy.numel())
#         if seen >= max_batches * getattr(loader, "batch_size", 1):
#             break
#     if pos == 0:
#         return None
#     return torch.tensor([neg / max(1, pos)], device=device, dtype=torch.float32)


# # =======================
# # Validation
# # =======================
# @torch.no_grad()
# def validate(model, loader, device, amp=False,
#              w_los=1.0, w_cls=0.5,
#              los_loss_type="huber", huber_delta=1.0, quantile_tau=0.5,
#              cls_loss_type="bce", focal_alpha=0.25, focal_gamma=2.0, label_smoothing=0.0,
#              pos_weight=None,
#              los_acc_tolerance: float = 1.0):
#     model.eval()
#     tot = los_sum = cls_sum = acc_sum = pmean_sum = 0.0
#     n = 0
#     epoch_probs, epoch_labels = [], []
#     los_acc_sum = 0.0

#     for batch in loader:
#         for key in ["base", "exam_z", "exam_mask", "y_los", "y_readmit"]:
#             if key not in batch:
#                 raise KeyError(f"Batch is missing required key: '{key}'. "
#                                f"Check collate function & dataset output dict.")

#         base = batch["base"].to(device)
#         exam_z = batch["exam_z"].to(device)
#         exam_mask = batch["exam_mask"].to(device)
#         y_los = batch["y_los"].to(device).view(-1)
#         y_readmit = batch["y_readmit"].to(device).view(-1)

#         with autocast_ctx(device, amp and device.type=="cuda"):
#             los_pred, readmit_logit = model(base=base, exam_z=exam_z, exam_mask=exam_mask)

#             # LOS loss
#             m_los = _valid_mask(y_los)
#             if los_loss_type == "mae":
#                 loss_los = _mae(los_pred.view(-1), y_los, mask=m_los)
#             elif los_loss_type == "mse":
#                 loss_los = _mse(los_pred.view(-1), y_los, mask=m_los)
#             elif los_loss_type == "quantile":
#                 loss_los = _quantile(los_pred.view(-1), y_los, tau=quantile_tau, mask=m_los)
#             else:
#                 loss_los = _huber(los_pred.view(-1), y_los, delta=huber_delta, mask=m_los)

#             # CLS loss
#             m_cls = _valid_mask(y_readmit)
#             if cls_loss_type == "focal":
#                 loss_cls = _focal_bce_with_logits(readmit_logit.view(-1), y_readmit,
#                                                   alpha=focal_alpha, gamma=focal_gamma,
#                                                   label_smoothing=label_smoothing, mask=m_cls)
#             else:
#                 loss_cls = _bce_with_logits_masked(readmit_logit.view(-1), y_readmit,
#                                                    pos_weight=pos_weight,
#                                                    label_smoothing=label_smoothing,
#                                                    mask=m_cls)

#             loss = w_los * loss_los + w_cls * loss_cls

#         # metrics
#         acc = _acc_from_logits(readmit_logit.view(-1), y_readmit)
#         _gather_probs_and_labels(readmit_logit.view(-1), y_readmit, epoch_probs, epoch_labels)

#         # LOS accuracy (±tol)
#         los_acc = _los_within_tol_acc(los_pred.view(-1), y_los, tol=los_acc_tolerance, mask=_valid_mask(y_los))

#         bsz = base.size(0)
#         tot     += float(loss.item())     * bsz
#         los_sum += float(loss_los.item()) * bsz
#         cls_sum += float(loss_cls.item()) * bsz
#         if torch.isfinite(acc):     acc_sum += float(acc.item()) * bsz
#         if torch.isfinite(los_acc): los_acc_sum += float(los_acc.item()) * bsz

#         if torch.isfinite(readmit_logit).all():
#             m = _valid_mask(y_readmit)
#             if (m is not None) and (m.sum() > 0):
#                 pmean_sum += float(torch.sigmoid(readmit_logit[m]).mean().item()) * bsz

#         n += bsz

#     auroc = _epoch_auroc(epoch_probs, epoch_labels)
#     return {
#         "loss":     tot / max(n,1),
#         "loss_los": los_sum / max(n,1),
#         "loss_cls": cls_sum / max(n,1),
#         "acc":      acc_sum / max(n,1) if n>0 else float("nan"),
#         "auroc":    float(auroc.item()) if torch.isfinite(auroc) else float("nan"),
#         "pmean":    pmean_sum / max(n,1) if n>0 else float("nan"),
#         "los_acc":  los_acc_sum / max(n,1) if n>0 else float("nan"),
#     }


# # =======================
# # Training
# # =======================
# def train(
#     model,
#     train_loader,
#     val_loader=None,
#     *,
#     epochs=30,
#     lr=1e-3,
#     weight_decay=0.0,
#     device=None,
#     amp=True,
#     grad_clip=1.0,
#     w_los=1.0, w_cls=0.5,
#     los_loss_type="huber", huber_delta=1.0, quantile_tau=0.5,
#     cls_loss_type="bce", focal_alpha=0.25, focal_gamma=2.0, label_smoothing=0.0,
#     use_auto_pos_weight=True, explicit_pos_weight=None,
#     log_every=100,
#     tb_logdir="./runs/train",
#     ckpt_dir="./checkpoints",
#     ckpt_name="model_best.pt",
#     early_stop_patience=5,
#     grad_accum_steps=1,
#     scheduler_type="onecycle",   # 'none' | 'onecycle' | 'cosine' | 'cosine_warmup'
#     resume_ckpt=None,
#     seed=1337,
#     los_acc_tolerance: float = 1.0,  # <-- NEW: ±tol days for LOS accuracy
# ):
#     set_seed(seed)
#     os.makedirs(tb_logdir, exist_ok=True)
#     os.makedirs(ckpt_dir, exist_ok=True)
#     writer = SummaryWriter(log_dir=tb_logdir)

#     device = resolve_device(device)
#     model  = model.to(device)

#     # pos_weight
#     pos_weight = None
#     if explicit_pos_weight is not None:
#         pos_weight = torch.tensor([float(explicit_pos_weight)], device=device)
#         print(f"[Imbalance] explicit pos_weight = {pos_weight.item():.4f}")
#     elif use_auto_pos_weight:
#         inferred = infer_pos_weight_from_loader(train_loader, device)
#         if inferred is not None:
#             pos_weight = inferred
#             print(f"[Imbalance] auto pos_weight = {pos_weight.item():.4f} (≈ neg/pos)")
#         else:
#             print("[Imbalance] auto pos_weight unavailable; using None.")

#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scaler    = make_scaler(device, enabled=(amp and device.type=="cuda"))

#     # Scheduler
#     if scheduler_type == "onecycle":
#         steps_per_epoch = max(1, len(train_loader))
#         scheduler = torch.optim.lr_scheduler.OneCycleLR(
#             optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch
#         )
#         step_on_batch = True
#     elif scheduler_type == "cosine_warmup":
#         warmup_epochs = max(1, int(0.1 * epochs))
#         cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
#         scheduler = torch.optim.lr_scheduler.SequentialLR(
#             optimizer,
#             schedulers=[
#                 torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs),
#                 cosine
#             ],
#             milestones=[warmup_epochs]
#         )
#         step_on_batch = False
#     elif scheduler_type == "cosine":
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
#         step_on_batch = False
#     else:
#         scheduler = None
#         step_on_batch = False

#     start_epoch = 1
#     best_val   = math.inf
#     no_improve = 0
#     gstep      = 0

#     # Resume
#     if resume_ckpt and os.path.isfile(resume_ckpt):
#         ck = torch.load(resume_ckpt, map_location="cpu")
#         try:
#             model.load_state_dict(ck["model"])
#             if "optimizer" in ck:
#                 optimizer.load_state_dict(ck["optimizer"])
#             if "scheduler" in ck and scheduler is not None and ck["scheduler"] is not None:
#                 try:
#                     scheduler.load_state_dict(ck["scheduler"])
#                 except Exception:
#                     pass
#             if "epoch" in ck:
#                 start_epoch = int(ck["epoch"]) + 1
#             if "best_val" in ck and math.isfinite(ck["best_val"]):
#                 best_val = float(ck["best_val"])
#             print(f"[Resume] from {resume_ckpt} at epoch {start_epoch}, best_val={best_val:.6f}")
#         except Exception as e:
#             print(f"[Resume] failed to load: {e}")

#     # Log static hparams
#     hparams = {
#         "epochs": epochs, "lr": lr, "weight_decay": weight_decay,
#         "w_los": w_los, "w_cls": w_cls,
#         "los_loss_type": los_loss_type, "huber_delta": huber_delta, "quantile_tau": quantile_tau,
#         "cls_loss_type": cls_loss_type, "focal_alpha": focal_alpha, "focal_gamma": focal_gamma,
#         "label_smoothing": label_smoothing,
#         "scheduler": scheduler_type, "grad_clip": grad_clip, "grad_accum_steps": grad_accum_steps,
#         "seed": seed,
#         "los_acc_tolerance": los_acc_tolerance,
#     }
#     writer.add_text("hparams", json.dumps(hparams, ensure_ascii=False, indent=2))

#     t0 = time.time()
#     for epoch in range(start_epoch, epochs+1):
#         model.train()
#         run_loss = run_los = run_cls = 0.0
#         n_seen = 0
#         # accumulate LOS acc (train)
#         los_acc_sum_train = 0.0
#         optimizer.zero_grad(set_to_none=True)

#         for i, batch in enumerate(train_loader, start=1):
#             for key in ["base", "exam_z", "exam_mask", "y_los", "y_readmit"]:
#                 if key not in batch:
#                     raise KeyError(f"Batch is missing required key: '{key}'. "
#                                    f"Check collate function & dataset output dict.")

#             base = batch["base"].to(device)
#             exam_z = batch["exam_z"].to(device)
#             exam_mask = batch["exam_mask"].to(device)
#             y_los = batch["y_los"].to(device).view(-1)
#             y_readmit = batch["y_readmit"].to(device).view(-1)

#             with autocast_ctx(device, amp and device.type=="cuda"):
#                 los_pred, readmit_logit = model(base=base, exam_z=exam_z, exam_mask=exam_mask)

#                 # LOS
#                 m_los = _valid_mask(y_los)
#                 if los_loss_type == "mae":
#                     loss_los = _mae(los_pred.view(-1), y_los, mask=m_los)
#                 elif los_loss_type == "mse":
#                     loss_los = _mse(los_pred.view(-1), y_los, mask=m_los)
#                 elif los_loss_type == "quantile":
#                     loss_los = _quantile(los_pred.view(-1), y_los, tau=quantile_tau, mask=m_los)
#                 else:
#                     loss_los = _huber(los_pred.view(-1), y_los, delta=huber_delta, mask=m_los)

#                 # CLS
#                 m_cls = _valid_mask(y_readmit)
#                 if cls_loss_type == "focal":
#                     loss_cls = _focal_bce_with_logits(readmit_logit.view(-1), y_readmit,
#                                                       alpha=focal_alpha, gamma=focal_gamma,
#                                                       label_smoothing=label_smoothing, mask=m_cls)
#                 else:
#                     loss_cls = _bce_with_logits_masked(readmit_logit.view(-1), y_readmit,
#                                                        pos_weight=pos_weight,
#                                                        label_smoothing=label_smoothing, mask=m_cls)

#                 loss = w_los * loss_los + w_cls * loss_cls
#                 loss_scaled = loss / max(1, grad_accum_steps)

#             if not torch.isfinite(loss):
#                 print(f"[warn] non-finite loss: {loss.item():.4g} -> skip")
#                 optimizer.zero_grad(set_to_none=True)
#                 continue

#             # --- LOS accuracy (train, running) ---
#             los_acc_b = _los_within_tol_acc(los_pred.view(-1), y_los, tol=los_acc_tolerance, mask=_valid_mask(y_los))
#             if torch.isfinite(los_acc_b):
#                 los_acc_sum_train += float(los_acc_b.item()) * base.size(0)

#             # backward/step
#             scaler.scale(loss_scaled).backward()

#             should_step = (i % grad_accum_steps) == 0
#             is_last_iter = (i == len(train_loader))
#             if should_step or is_last_iter:
#                 if grad_clip and grad_clip > 0:
#                     scaler.unscale_(optimizer)
#                     nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad(set_to_none=True)
#                 if scheduler is not None and step_on_batch:
#                     scheduler.step()

#             # stats
#             bsz = base.size(0)
#             run_loss += float(loss.item())     * bsz
#             run_los  += float(loss_los.item()) * bsz
#             run_cls  += float(loss_cls.item()) * bsz
#             n_seen   += bsz

#             gstep += 1
#             writer.add_scalar("train/loss_step", float(loss.item()), gstep)
#             writer.add_scalar("train/loss_los_step", float(loss_los.item()), gstep)
#             writer.add_scalar("train/loss_cls_step", float(loss_cls.item()), gstep)
#             writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], gstep)

#             if "readmit_logit" in locals():
#                 writer.add_scalar("train/logit_mean", readmit_logit.mean().item(), gstep)
#                 writer.add_scalar("train/logit_std",  readmit_logit.std(unbiased=False).item(), gstep)

#             if (i % log_every) == 0:
#                 avg = run_loss / max(n_seen,1)
#                 lr_now = optimizer.param_groups[0]['lr']
#                 print(f"[Epoch {epoch}/{epochs}] step {i:04d} | train_loss={avg:.6f} | lr={lr_now:.2e}")

#         # epoch train logs
#         train_logs = {
#             "loss":     run_loss / max(n_seen,1),
#             "loss_los": run_los  / max(n_seen,1),
#             "loss_cls": run_cls  / max(n_seen,1),
#             "los_acc":  los_acc_sum_train / max(n_seen,1) if n_seen>0 else float("nan"),
#         }
#         writer.add_scalar("train/loss_epoch", train_logs["loss"], epoch)
#         writer.add_scalar("train/loss_los_epoch", train_logs["loss_los"], epoch)
#         writer.add_scalar("train/loss_cls_epoch", train_logs["loss_cls"], epoch)
#         if not math.isnan(train_logs["los_acc"]):
#             writer.add_scalar("train/los_acc_epoch", train_logs["los_acc"], epoch)

#         # validation
#         if val_loader is not None:
#             if scheduler is not None and not step_on_batch:
#                 scheduler.step()

#             val_logs = validate(
#                 model, val_loader, device,
#                 amp=(amp and device.type=="cuda"),
#                 w_los=w_los, w_cls=w_cls,
#                 los_loss_type=los_loss_type, huber_delta=huber_delta, quantile_tau=quantile_tau,
#                 cls_loss_type=cls_loss_type, focal_alpha=focal_alpha, focal_gamma=focal_gamma, label_smoothing=label_smoothing,
#                 pos_weight=pos_weight,
#                 los_acc_tolerance=los_acc_tolerance
#             )
#             writer.add_scalar("val/loss",      val_logs["loss"], epoch)
#             writer.add_scalar("val/loss_los",  val_logs["loss_los"], epoch)
#             writer.add_scalar("val/loss_cls",  val_logs["loss_cls"], epoch)
#             if not math.isnan(val_logs["acc"]):     writer.add_scalar("val/acc",     val_logs["acc"], epoch)
#             if not math.isnan(val_logs["auroc"]):   writer.add_scalar("val/auroc",   val_logs["auroc"], epoch)
#             if not math.isnan(val_logs["pmean"]):   writer.add_scalar("val/prob_mean", val_logs["pmean"], epoch)
#             if not math.isnan(val_logs["los_acc"]): writer.add_scalar("val/los_acc", val_logs["los_acc"], epoch)

#             print(f"==> Epoch {epoch}: train_loss={train_logs['loss']:.6f} | "
#                   f"train_los_acc@±{los_acc_tolerance:.1f}d={train_logs['los_acc']:.4f} | "
#                   f"val_loss={val_logs['loss']:.6f} | val_los_acc@±{los_acc_tolerance:.1f}d={val_logs['los_acc']:.4f} | "
#                   f"val_acc={val_logs['acc']:.4f} | val_auroc={val_logs['auroc']:.4f} | "
#                   f"elapsed={time.time()-t0:.1f}s")

#             # best by val_loss
#             cur_is_best = (val_logs["loss"] < best_val) and math.isfinite(val_logs["loss"])
#             if cur_is_best:
#                 best_val = val_logs["loss"]
#                 save_path = os.path.join(ckpt_dir, ckpt_name)
#                 torch.save(
#                     {
#                         "model": model.state_dict(),
#                         "optimizer": optimizer.state_dict(),
#                         "scheduler": (scheduler.state_dict() if scheduler is not None else None),
#                         "epoch": epoch,
#                         "val":   val_logs,
#                         "best_val": best_val,
#                         "hparams": hparams,
#                     },
#                     save_path,
#                 )
#                 no_improve = 0
#                 print(f"    ✓ New best saved to {save_path}")
#             else:
#                 no_improve += 1
#                 if (early_stop_patience is not None) and (no_improve >= early_stop_patience):
#                     print(f"Early stopping at epoch {epoch} (no improve {no_improve}/{early_stop_patience})")
#                     break
#         else:
#             print(f"==> Epoch {epoch}: train_loss={train_logs['loss']:.6f} | "
#                   f"train_los_acc@±{los_acc_tolerance:.1f}d={train_logs['los_acc']:.4f}")

#     writer.close()
#     print("Training finished. Logs at:", tb_logdir)

# training.py
# -*- coding: utf-8 -*-
"""
High-performance trainer for UTI hadm-centric pipeline.

- Data: HadmTableDatasetV2 (no admissions.csv)
- Model: TableTransformerPredictor (masked table-attention + presence + missing tokens + FiLM)
- Loss:
    total = w_los * LOS_loss + w_cls * CLS_loss
  * LOS_loss options: huber (default), mae, mse, quantile (pinball; tau controllable)
  * CLS_loss options: bce (default, with pos_weight auto), focal (alpha, gamma)
- Metrics:
  * readmission: epoch AUROC (torchmetrics if available), accuracy@0.5
  * LOS custom: accuracy within ±tol days (default tol=1.0 day)
- Training niceties: AMP, grad clip, schedulers (none/onecycle/cosine/cosine_warmup),
  early stopping, resume, TensorBoard logging.

This file is intentionally arg-agnostic; call train(...) from your main.py.
"""

import os, math, time, json, random, contextlib
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


# =======================
# Utilities
# =======================
def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def resolve_device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def autocast_ctx(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    try:
        from torch.amp import autocast as amp_autocast
        return amp_autocast(device_type=device.type, enabled=True)
    except Exception:
        if device.type == "cuda":
            try:
                from torch.cuda.amp import autocast as cuda_autocast
                return cuda_autocast(enabled=True)
            except Exception:
                return contextlib.nullcontext()
        return contextlib.nullcontext()

def make_scaler(device: torch.device, enabled: bool):
    if not enabled:
        class _Null:
            def scale(self, x): return x
            def step(self, opt): opt.step()
            def update(self): pass
            def unscale_(self, opt): pass
        return _Null()
    try:
        from torch.amp import GradScaler
        return GradScaler(enabled=True)
    except Exception:
        try:
            from torch.cuda.amp import GradScaler as CudaGradScaler
            return CudaGradScaler(enabled=(device.type=="cuda"))
        except Exception:
            return make_scaler(device, enabled=False)


# =======================
# Losses & Metrics
# =======================
def _valid_mask(y: Optional[torch.Tensor]):
    if y is None:
        return None
    return torch.isfinite(y)

def _huber(x, y, delta=1.0, mask=None):
    if mask is not None:
        x = x[mask]; y = y[mask]
    if x.numel() == 0:
        dev = x.device if torch.is_tensor(x) else (y.device if torch.is_tensor(y) else torch.device("cpu"))
        return torch.tensor(0.0, device=dev)
    return F.smooth_l1_loss(x, y, beta=delta)

def _mae(x, y, mask=None):
    if mask is not None:
        x = x[mask]; y = y[mask]
    if x.numel() == 0:
        return torch.tensor(0.0, device=y.device)
    return F.l1_loss(x, y)

def _mse(x, y, mask=None):
    if mask is not None:
        x = x[mask]; y = y[mask]
    if x.numel() == 0:
        return torch.tensor(0.0, device=y.device)
    return F.mse_loss(x, y)

def _quantile(pred, target, tau=0.5, mask=None):
    if mask is not None:
        pred = pred[mask]; target = target[mask]
    if pred.numel() == 0:
        return torch.tensor(0.0, device=target.device)
    e = target - pred
    return torch.maximum(tau * e, (tau - 1.0) * e).mean()

def _bce_with_logits_masked(logits, target, pos_weight=None, label_smoothing=0.0, mask=None):
    if mask is not None:
        logits = logits[mask]; target = target[mask]
    if logits.numel() == 0:
        return torch.tensor(0.0, device=target.device)
    if label_smoothing and label_smoothing > 0:
        target = target * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy_with_logits(
        logits, target.float(), pos_weight=pos_weight
    )

def _focal_bce_with_logits(logits, target, alpha=0.25, gamma=2.0, label_smoothing=0.0, mask=None):
    if mask is not None:
        logits = logits[mask]; target = target[mask]
    if logits.numel() == 0:
        return torch.tensor(0.0, device=target.device)
    if label_smoothing and label_smoothing > 0:
        target = target * (1.0 - label_smoothing) + 0.5 * label_smoothing

    bce = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, p, 1.0 - p)
    alpha_t = torch.where(target > 0.5, torch.full_like(target, alpha), torch.full_like(target, 1 - alpha))
    loss = alpha_t * ((1 - pt) ** gamma) * bce
    return loss.mean()

@torch.no_grad()
def _acc_from_logits(logits, target):
    if (logits is None) or (target is None):
        dev = logits.device if logits is not None else (target.device if target is not None else torch.device("cpu"))
        return torch.tensor(float("nan"), device=dev)
    m = _valid_mask(target)
    if (m is None) or (m.sum() == 0):
        return torch.tensor(float("nan"), device=logits.device)
    prob = torch.sigmoid(logits[m])
    pred = (prob >= 0.5).long()
    return (pred == target[m].long()).float().mean().detach()

@torch.no_grad()
def _los_within_tol_acc(pred, target, tol: float = 1.0, mask=None):
    """
    Accuracy for LOS regression:
    counts a prediction 'correct' if |pred - target| <= tol (days).
    """
    if mask is not None:
        pred = pred[mask]; target = target[mask]
    if pred.numel() == 0:
        return torch.tensor(float("nan"))
    diff = torch.abs(pred - target)
    return (diff <= tol).float().mean()

@torch.no_grad()
def _gather_probs_and_labels(logits, target, store_probs: List[torch.Tensor], store_labels: List[torch.Tensor]):
    if (logits is None) or (target is None):
        return
    m = _valid_mask(target)
    if (m is None) or (m.sum() == 0):
        return
    prob = torch.sigmoid(logits[m]).detach()
    lab  = target[m].to(prob.dtype).detach()
    store_probs.append(prob)
    store_labels.append(lab)

@torch.no_grad()
def _epoch_auroc(probs_list: List[torch.Tensor], labels_list: List[torch.Tensor]):
    if len(probs_list) == 0:
        return torch.tensor(float("nan"))
    prob = torch.cat([p.cpu() for p in probs_list], dim=0)
    lab  = torch.cat([l.cpu() for l in labels_list], dim=0)
    if lab.unique().numel() < 2:
        return torch.tensor(float("nan"))
    try:
        from torchmetrics.functional.classification import binary_auroc
        return binary_auroc(prob, lab)
    except Exception:
        return torch.tensor(float("nan"))


@torch.no_grad()
def infer_pos_weight_from_loader(loader, device, max_batches=200):
    pos = neg = 0
    seen = 0
    for batch in loader:
        y = batch.get("y_readmit")
        if y is None:
            continue
        y = y.to(device)
        m = _valid_mask(y)
        if (m is None) or (m.sum() == 0):
            continue
        yy = y[m].view(-1).long()
        pos += int((yy == 1).sum().item())
        neg += int((yy == 0).sum().item())
        seen += int(yy.numel())
        if seen >= max_batches * getattr(loader, "batch_size", 1):
            break
    if pos == 0:
        return None
    return torch.tensor([neg / max(1, pos)], device=device, dtype=torch.float32)


# =======================
# Validation
# =======================
@torch.no_grad()
def validate(model, loader, device, amp=False,
             w_los=1.0, w_cls=0.5,
             los_loss_type="huber", huber_delta=1.0, quantile_tau=0.5,
             cls_loss_type="bce", focal_alpha=0.25, focal_gamma=2.0, label_smoothing=0.0,
             pos_weight=None,
             los_acc_tolerance: float = 1.0):
    model.eval()
    tot = los_sum = cls_sum = acc_sum = pmean_sum = 0.0
    n = 0
    epoch_probs, epoch_labels = [], []
    los_acc_sum = 0.0

    for batch in loader:
        for key in ["base", "exam_z", "exam_mask", "y_los", "y_readmit"]:
            if key not in batch:
                raise KeyError(f"Batch is missing required key: '{key}'. "
                               f"Check collate function & dataset output dict.")

        base = batch["base"].to(device)
        exam_z = batch["exam_z"].to(device)
        exam_mask = batch["exam_mask"].to(device)
        y_los = batch["y_los"].to(device).view(-1)
        y_readmit = batch["y_readmit"].to(device).view(-1)

        with autocast_ctx(device, amp and device.type=="cuda"):
            los_pred, readmit_logit = model(base=base, exam_z=exam_z, exam_mask=exam_mask)

            # LOS loss
            m_los = _valid_mask(y_los)
            if los_loss_type == "mae":
                loss_los = _mae(los_pred.view(-1), y_los, mask=m_los)
            elif los_loss_type == "mse":
                loss_los = _mse(los_pred.view(-1), y_los, mask=m_los)
            elif los_loss_type == "quantile":
                loss_los = _quantile(los_pred.view(-1), y_los, tau=quantile_tau, mask=m_los)
            else:
                loss_los = _huber(los_pred.view(-1), y_los, delta=huber_delta, mask=m_los)

            # CLS loss
            m_cls = _valid_mask(y_readmit)
            if cls_loss_type == "focal":
                loss_cls = _focal_bce_with_logits(readmit_logit.view(-1), y_readmit,
                                                  alpha=focal_alpha, gamma=focal_gamma,
                                                  label_smoothing=label_smoothing, mask=m_cls)
            else:
                loss_cls = _bce_with_logits_masked(readmit_logit.view(-1), y_readmit,
                                                   pos_weight=pos_weight,
                                                   label_smoothing=label_smoothing,
                                                   mask=m_cls)

            loss = w_los * loss_los + w_cls * loss_cls

        # metrics
        acc = _acc_from_logits(readmit_logit.view(-1), y_readmit)
        _gather_probs_and_labels(readmit_logit.view(-1), y_readmit, epoch_probs, epoch_labels)

        # LOS accuracy (±tol)
        los_acc = _los_within_tol_acc(los_pred.view(-1), y_los, tol=los_acc_tolerance, mask=_valid_mask(y_los))

        bsz = base.size(0)
        tot     += float(loss.item())     * bsz
        los_sum += float(loss_los.item()) * bsz
        cls_sum += float(loss_cls.item()) * bsz
        if torch.isfinite(acc):     acc_sum += float(acc.item()) * bsz
        if torch.isfinite(los_acc): los_acc_sum += float(los_acc.item()) * bsz

        if torch.isfinite(readmit_logit).all():
            m = _valid_mask(y_readmit)
            if (m is not None) and (m.sum() > 0):
                pmean_sum += float(torch.sigmoid(readmit_logit[m]).mean().item()) * bsz

        n += bsz

    auroc = _epoch_auroc(epoch_probs, epoch_labels)
    return {
        "loss":     tot / max(n,1),
        "loss_los": los_sum / max(n,1),
        "loss_cls": cls_sum / max(n,1),
        "acc":      acc_sum / max(n,1) if n>0 else float("nan"),
        "auroc":    float(auroc.item()) if torch.isfinite(auroc) else float("nan"),
        "pmean":    pmean_sum / max(n,1) if n>0 else float("nan"),
        "los_acc":  los_acc_sum / max(n,1) if n>0 else float("nan"),
    }


# =======================
# Training
# =======================
def train(
    model,
    train_loader,
    val_loader=None,
    *,
    epochs=30,
    lr=3e-4,
    weight_decay=0.0,
    device=None,
    amp=True,
    grad_clip=1.0,
    w_los=1.0, w_cls=0.5,
    los_loss_type="huber", huber_delta=1.0, quantile_tau=0.5,
    cls_loss_type="bce", focal_alpha=0.25, focal_gamma=2.0, label_smoothing=0.0,
    use_auto_pos_weight=True, explicit_pos_weight=None,
    log_every=100,
    tb_logdir="./runs/train",
    ckpt_dir="./checkpoints",
    ckpt_name="model_best.pt",
    early_stop_patience=5,
    grad_accum_steps=1,
    scheduler_type="cosine_warmup",   # 'none' | 'onecycle' | 'cosine' | 'cosine_warmup'
    resume_ckpt=None,
    seed=1337,
    los_acc_tolerance: float = 1.0,  # <-- NEW: ±tol days for LOS accuracy
):
    set_seed(seed)
    os.makedirs(tb_logdir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_logdir)

    device = resolve_device(device)
    model  = model.to(device)

    # pos_weight
    pos_weight = None
    if explicit_pos_weight is not None:
        pos_weight = torch.tensor([float(explicit_pos_weight)], device=device)
        print(f"[Imbalance] explicit pos_weight = {pos_weight.item():.4f}")
    elif use_auto_pos_weight:
        inferred = infer_pos_weight_from_loader(train_loader, device)
        if inferred is not None:
            pos_weight = inferred
            print(f"[Imbalance] auto pos_weight = {pos_weight.item():.4f} (≈ neg/pos)")
        else:
            print("[Imbalance] auto pos_weight unavailable; using None.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler    = make_scaler(device, enabled=(amp and device.type=="cuda"))

    # Scheduler
    if scheduler_type == "onecycle":
        steps_per_epoch = max(1, len(train_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch
        )
        step_on_batch = True
    elif scheduler_type == "cosine_warmup":
        warmup_epochs = max(1, int(0.1 * epochs))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs),
                cosine
            ],
            milestones=[warmup_epochs]
        )
        step_on_batch = False
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        step_on_batch = False
    else:
        scheduler = None
        step_on_batch = False

    start_epoch = 1
    best_val   = math.inf
    no_improve = 0
    gstep      = 0

    # Resume
    if resume_ckpt and os.path.isfile(resume_ckpt):
        ck = torch.load(resume_ckpt, map_location="cpu")
        try:
            model.load_state_dict(ck["model"])
            if "optimizer" in ck:
                optimizer.load_state_dict(ck["optimizer"])
            if "scheduler" in ck and scheduler is not None and ck["scheduler"] is not None:
                try:
                    scheduler.load_state_dict(ck["scheduler"])
                except Exception:
                    pass
            if "epoch" in ck:
                start_epoch = int(ck["epoch"]) + 1
            if "best_val" in ck and math.isfinite(ck["best_val"]):
                best_val = float(ck["best_val"])
            print(f"[Resume] from {resume_ckpt} at epoch {start_epoch}, best_val={best_val:.6f}")
        except Exception as e:
            print(f"[Resume] failed to load: {e}")

    # Log static hparams
    hparams = {
        "epochs": epochs, "lr": lr, "weight_decay": weight_decay,
        "w_los": w_los, "w_cls": w_cls,
        "los_loss_type": los_loss_type, "huber_delta": huber_delta, "quantile_tau": quantile_tau,
        "cls_loss_type": cls_loss_type, "focal_alpha": focal_alpha, "focal_gamma": focal_gamma,
        "label_smoothing": label_smoothing,
        "scheduler": scheduler_type, "grad_clip": grad_clip, "grad_accum_steps": grad_accum_steps,
        "seed": seed,
        "los_acc_tolerance": los_acc_tolerance,
    }
    writer.add_text("hparams", json.dumps(hparams, ensure_ascii=False, indent=2))

    t0 = time.time()
    for epoch in range(start_epoch, epochs+1):
        model.train()
        run_loss = run_los = run_cls = 0.0
        n_seen = 0
        # accumulate LOS acc (train)
        los_acc_sum_train = 0.0
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader, start=1):
            for key in ["base", "exam_z", "exam_mask", "y_los", "y_readmit"]:
                if key not in batch:
                    raise KeyError(f"Batch is missing required key: '{key}'. "
                                   f"Check collate function & dataset output dict.")

            base = batch["base"].to(device)
            exam_z = batch["exam_z"].to(device)
            exam_mask = batch["exam_mask"].to(device)
            y_los = batch["y_los"].to(device).view(-1)
            y_readmit = batch["y_readmit"].to(device).view(-1)

            with autocast_ctx(device, amp and device.type=="cuda"):
                los_pred, readmit_logit = model(base=base, exam_z=exam_z, exam_mask=exam_mask)

                # LOS
                m_los = _valid_mask(y_los)
                if los_loss_type == "mae":
                    loss_los = _mae(los_pred.view(-1), y_los, mask=m_los)
                elif los_loss_type == "mse":
                    loss_los = _mse(los_pred.view(-1), y_los, mask=m_los)
                elif los_loss_type == "quantile":
                    loss_los = _quantile(los_pred.view(-1), y_los, tau=quantile_tau, mask=m_los)
                else:
                    loss_los = _huber(los_pred.view(-1), y_los, delta=huber_delta, mask=m_los)

                # CLS
                m_cls = _valid_mask(y_readmit)
                if cls_loss_type == "focal":
                    loss_cls = _focal_bce_with_logits(readmit_logit.view(-1), y_readmit,
                                                      alpha=focal_alpha, gamma=focal_gamma,
                                                      label_smoothing=label_smoothing, mask=m_cls)
                else:
                    loss_cls = _bce_with_logits_masked(readmit_logit.view(-1), y_readmit,
                                                       pos_weight=pos_weight,
                                                       label_smoothing=label_smoothing, mask=m_cls)

                loss = w_los * loss_los + w_cls * loss_cls
                loss_scaled = loss / max(1, grad_accum_steps)

            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss: {loss.item():.4g} -> skip")
                optimizer.zero_grad(set_to_none=True)
                continue

            # --- LOS accuracy (train, running) ---
            los_acc_b = _los_within_tol_acc(los_pred.view(-1), y_los, tol=los_acc_tolerance, mask=_valid_mask(y_los))
            if torch.isfinite(los_acc_b):
                los_acc_sum_train += float(los_acc_b.item()) * base.size(0)

            # backward/step
            scaler.scale(loss_scaled).backward()

            should_step = (i % grad_accum_steps) == 0
            is_last_iter = (i == len(train_loader))
            if should_step or is_last_iter:
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and step_on_batch:
                    scheduler.step()

            # stats
            bsz = base.size(0)
            run_loss += float(loss.item())     * bsz
            run_los  += float(loss_los.item()) * bsz
            run_cls  += float(loss_cls.item()) * bsz
            n_seen   += bsz

            gstep += 1
            writer.add_scalar("train/loss_step", float(loss.item()), gstep)
            writer.add_scalar("train/loss_los_step", float(loss_los.item()), gstep)
            writer.add_scalar("train/loss_cls_step", float(loss_cls.item()), gstep)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], gstep)

            if "readmit_logit" in locals():
                writer.add_scalar("train/logit_mean", readmit_logit.mean().item(), gstep)
                writer.add_scalar("train/logit_std",  readmit_logit.std(unbiased=False).item(), gstep)

            if (i % log_every) == 0:
                avg = run_loss / max(n_seen,1)
                lr_now = optimizer.param_groups[0]['lr']
                print(f"[Epoch {epoch}/{epochs}] step {i:04d} | train_loss={avg:.6f} | lr={lr_now:.2e}")

        # epoch train logs
        train_logs = {
            "loss":     run_loss / max(n_seen,1),
            "loss_los": run_los  / max(n_seen,1),
            "loss_cls": run_cls  / max(n_seen,1),
            "los_acc":  los_acc_sum_train / max(n_seen,1) if n_seen>0 else float("nan"),
        }
        writer.add_scalar("train/loss_epoch", train_logs["loss"], epoch)
        writer.add_scalar("train/loss_los_epoch", train_logs["loss_los"], epoch)
        writer.add_scalar("train/loss_cls_epoch", train_logs["loss_cls"], epoch)
        if not math.isnan(train_logs["los_acc"]):
            writer.add_scalar("train/los_acc_epoch", train_logs["los_acc"], epoch)

        # validation
        if val_loader is not None:
            if scheduler is not None and not step_on_batch:
                scheduler.step()

            val_logs = validate(
                model, val_loader, device,
                amp=(amp and device.type=="cuda"),
                w_los=w_los, w_cls=w_cls,
                los_loss_type=los_loss_type, huber_delta=huber_delta, quantile_tau=quantile_tau,
                cls_loss_type=cls_loss_type, focal_alpha=focal_alpha, focal_gamma=focal_gamma, label_smoothing=label_smoothing,
                pos_weight=pos_weight,
                los_acc_tolerance=los_acc_tolerance
            )
            writer.add_scalar("val/loss",      val_logs["loss"], epoch)
            writer.add_scalar("val/loss_los",  val_logs["loss_los"], epoch)
            writer.add_scalar("val/loss_cls",  val_logs["loss_cls"], epoch)
            if not math.isnan(val_logs["acc"]):     writer.add_scalar("val/acc",     val_logs["acc"], epoch)
            if not math.isnan(val_logs["auroc"]):   writer.add_scalar("val/auroc",   val_logs["auroc"], epoch)
            if not math.isnan(val_logs["pmean"]):   writer.add_scalar("val/prob_mean", val_logs["pmean"], epoch)
            if not math.isnan(val_logs["los_acc"]): writer.add_scalar("val/los_acc", val_logs["los_acc"], epoch)

            print(f"==> Epoch {epoch}: train_loss={train_logs['loss']:.6f} | "
                  f"train_los_acc@±{los_acc_tolerance:.1f}d={train_logs['los_acc']:.4f} | "
                  f"val_loss={val_logs['loss']:.6f} | val_los_acc@±{los_acc_tolerance:.1f}d={val_logs['los_acc']:.4f} | "
                  f"val_acc={val_logs['acc']:.4f} | val_auroc={val_logs['auroc']:.4f} | "
                  f"elapsed={time.time()-t0:.1f}s")

            # best by val_loss
            cur_is_best = (val_logs["loss"] < best_val) and math.isfinite(val_logs["loss"])
            if cur_is_best:
                best_val = val_logs["loss"]
                save_path = os.path.join(ckpt_dir, ckpt_name)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": (scheduler.state_dict() if scheduler is not None else None),
                        "epoch": epoch,
                        "val":   val_logs,
                        "best_val": best_val,
                        "hparams": hparams,
                    },
                    save_path,
                )
                no_improve = 0
                print(f"    ✓ New best saved to {save_path}")
            else:
                no_improve += 1
                if (early_stop_patience is not None) and (no_improve >= early_stop_patience):
                    print(f"Early stopping at epoch {epoch} (no improve {no_improve}/{early_stop_patience})")
                    break
        else:
            print(f"==> Epoch {epoch}: train_loss={train_logs['loss']:.6f} | "
                  f"train_los_acc@±{los_acc_tolerance:.1f}d={train_logs['los_acc']:.4f}")

    writer.close()
    print("Training finished. Logs at:", tb_logdir)