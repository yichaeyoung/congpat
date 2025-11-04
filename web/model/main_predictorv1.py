# main_hadm_train.py
# -*- coding: utf-8 -*-
"""
End-to-end main (no argparse) for UTI hadm-centric training.
"""

import os
import json
import torch
from torch.utils.data import DataLoader, random_split

from utils.predictor.p_dataloader_5_3 import (
    HadmTableDatasetV3, collate_hadm_batch_v3, example_sources_config_v3
)
from architectures.predictor.predict_modelv2 import TableTransformerPredictor
from utils.predictor.p_train import train, resolve_device, set_seed
# ⬇️ 새 라벨러 임포트
from utils.predictor.p_30readmit_label_maker import label_unified_with_admissions

# --------------------------
# 0) 경로 & 설정 (여기만 수정)
# --------------------------
# (A) 데이터 경로
BASE_DIR               = os.path.dirname(os.path.abspath(__file__))
UNIFIED_CSV            = os.path.join(BASE_DIR, "../../dataset/summarized.csv")
ADMISSIONS_CSV         = os.path.join(BASE_DIR, "../../dataset/admissions.csv")
UNIFIED_WITH_LABELS    = os.path.join(BASE_DIR, "../../dataset/summarized_with_readmit30_test.csv")

SOURCES_JSON           = None
CRRT_LABEL_CSV         = os.path.join(BASE_DIR, "../../dataset/crrt_labels.csv")
MIMIC_DERIVED_LODS_CSV = os.path.join(BASE_DIR, "../../dataset/mimic-iv_derived_dataset/mimiciv_derived_lods.csv")
MIMIC_DERIVED_CRRT_CSV = os.path.join(BASE_DIR, "../../dataset/mimic-iv_derived_dataset/mimiciv_derived_crrt.csv")
CRRT_WITHIN_WINDOW     = True

# (B) 로더 옵션
BATCH_SIZE   = 128
NUM_WORKERS  = 4
VAL_RATIO    = 0.4
SEED         = 42

# (C) 모델 옵션
D_MODEL      = 256
NHEAD        = 8
DEPTH        = 3
DIM_FF       = 768
DROPOUT      = 0.15
HEAD_HIDDEN  = 256
USE_FILM     = True
USE_MASKED_MEAN = True

# (D) 학습 옵션
EXP_NAME = 'exp_final'
EPOCHS          = 35
LR              = 3e-4
WEIGHT_DECAY    = 1e-4
USE_AMP         = True
GRAD_CLIP       = 1.0
W_LOS, W_CLS    = 1.0, 0.6
LOS_LOSS_TYPE   = "huber"
HUBER_DELTA     = 1.0
QUANTILE_TAU    = 0.5
CLS_LOSS_TYPE   = "bce"
FOCAL_ALPHA     = 0.25
FOCAL_GAMMA     = 2.0
LABEL_SMOOTHING = 0.0
USE_AUTO_POSW   = True
EXPL_POSW       = None
LOG_EVERY       = 100
TB_LOGDIR       = f"./runs/highperf_{EXP_NAME}"
CKPT_DIR        = f"./checkpoints_{EXP_NAME}"
CKPT_NAME       = f"highperf_best_{EXP_NAME}.pt"
EARLY_STOP_PATIENCE = 6
GRAD_ACCUM_STEPS    = 1
SCHEDULER_TYPE      = "cosine_warmup"
RESUME_CKPT         = None
DEVICE              = None

# (E) 캐시
LATENT_CACHE_DIR    = f"./latent_cache_v2_{EXP_NAME}"


def _load_sources_config():
    if SOURCES_JSON and os.path.isfile(SOURCES_JSON):
        with open(SOURCES_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return example_sources_config_v3()


def main():
    # --- Seed/Device ---
    set_seed(SEED)
    device = resolve_device(DEVICE)
    print(f"[Device] {device}")

    # --- Sources config ---
    sources = _load_sources_config()

    # --- 30일 재입원 라벨 생성 (원본 보존, 통계 출력 끔) ---
    labeled_csv, _stats = label_unified_with_admissions(
        unified_csv=UNIFIED_CSV,
        admissions_csv=ADMISSIONS_CSV,
        out_csv=UNIFIED_WITH_LABELS,
        window_days=30,
        include_equal=False,
        verbose=True  
    )

    # --- Dataset ---
    ds = HadmTableDatasetV3(
        unified_csv=labeled_csv,                 # ← 라벨이 주입된 CSV 사용
        drop_index_death=False,
        drop_30d_postdischarge_death=False,
        crrt_label_csv=CRRT_LABEL_CSV,
        mimic_derived_lods_csv=MIMIC_DERIVED_LODS_CSV,
        mimic_derived_crrt_csv=MIMIC_DERIVED_CRRT_CSV,
        crrt_restrict_within_admission_window=CRRT_WITHIN_WINDOW,
        sources=sources,
        encode_device="cpu",
        cache_dir=LATENT_CACHE_DIR,
    )
    print(f"[Dataset] #hadm samples = {len(ds)}")

    # --- Split & DataLoader ---
    full_len = len(ds)
    val_len  = max(1, int(full_len * VAL_RATIO))
    train_len = full_len - val_len
    g = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=g)

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_hadm_batch_v3,
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),  # 품질개선 포인트 반영
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # --- Peek to infer sizes ---
    peek = next(iter(train_loader))
    B, N, E = peek["exam_z"].shape
    S = peek["base"].shape[1]
    print(f"[Peek] base_dim={S}, num_tables={N}, latent_dim={E}")

    # --- Model ---
    model = TableTransformerPredictor(
        num_tables=N, latent_dim=E, base_dim=S,
        d_model=D_MODEL, nhead=NHEAD, depth=DEPTH, dim_ff=DIM_FF,
        dropout=DROPOUT, head_hidden=HEAD_HIDDEN,
        use_film=USE_FILM, use_masked_mean=USE_MASKED_MEAN,
    ).to(device)

    # --- Train ---
    os.makedirs(TB_LOGDIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        device=device,
        amp=USE_AMP,
        grad_clip=GRAD_CLIP,
        w_los=W_LOS, w_cls=W_CLS,
        los_loss_type=LOS_LOSS_TYPE, huber_delta=HUBER_DELTA, quantile_tau=QUANTILE_TAU,
        cls_loss_type=CLS_LOSS_TYPE, focal_alpha=FOCAL_ALPHA, focal_gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING,
        use_auto_pos_weight=USE_AUTO_POSW, explicit_pos_weight=EXPL_POSW,
        log_every=LOG_EVERY,
        tb_logdir=TB_LOGDIR,
        ckpt_dir=CKPT_DIR,
        ckpt_name=CKPT_NAME,
        early_stop_patience=EARLY_STOP_PATIENCE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        scheduler_type=SCHEDULER_TYPE,
        resume_ckpt=RESUME_CKPT,
        seed=SEED,
    )


if __name__ == "__main__":
    main()