# app_gradio_subject_predict.py
# -*- coding: utf-8 -*-
"""
subject_id로 검색:
  1) patients.csv에서 환자 정보 조회 (subject_id, create, gender, anchor_age, anchor_year)
  2) 저장된 모델(.pt) 로드 → 동일 파이프라인으로 재추론하여 퇴원일 예측
     (표에는 pred_dischtime, los_pred_days만 표시; los_pred_days는 소수점 3자리)
"""

import os
import json
import torch
import gradio as gr
import pandas as pd
from torch.utils.data import DataLoader, Subset

# --- 프로젝트 내부 모듈 ---
from utils.predictor.p_dataloader_5_3 import (
    HadmTableDatasetV3, collate_hadm_batch_v3, example_sources_config_v3
)
from architectures.predictor.predict_modelv2 import TableTransformerPredictor


# =========================
# 기본 경로/옵션 (환경에 맞게 수정 가능)
# =========================
PATIENTS_CSV = "./filtered/patients2.csv"

UNIFIED_CSV = "../dataset/summarized_with_readmit30_test.csv"
CKPT_PATH   = "./checkpoints_exp_final/highperf_best_exp_final.pt"
USE_EXAMPLE_SOURCES = True
SOURCES_JSON_PATH   = None
CACHE_DIR  = "./latent_cache_v2_exp_final"
BATCH_SIZE = 64

if torch.cuda.is_available():
    ENCODE_DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    ENCODE_DEVICE = "mps"
else:
    ENCODE_DEVICE = "cpu"


# =========================
# Utilities
# =========================
def resolve_device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pick_col(df: pd.DataFrame, candidates):
    """대소문자 무시하고 후보 중 존재하는 첫 컬럼명을 반환"""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


# =========================
# 1) 환자 조회
# =========================
def lookup_patient(subject_id_str: str, patients_csv_path: str):
    if not subject_id_str or not subject_id_str.strip().isdigit():
        return "❌ subject_id는 정수여야 합니다.", None
    sid = int(subject_id_str.strip())

    if not os.path.isfile(patients_csv_path):
        return f"❌ patients.csv 경로가 존재하지 않습니다: {patients_csv_path}", None

    df = pd.read_csv(patients_csv_path, low_memory=False)

    c_subj   = _pick_col(df, ["subject_id"])
    c_name   = _pick_col(df, ["create", "name", "patient_name"])  # 'create'가 기본, 없으면 fallback
    c_gender = _pick_col(df, ["gender"])
    c_age    = _pick_col(df, ["anchor_age", "age"])
    c_year   = _pick_col(df, ["anchor_year", "birth_year"])

    for req, cname in {
        "subject_id": c_subj, "gender": c_gender, "anchor_age": c_age, "anchor_year": c_year
    }.items():
        if cname is None:
            return f"❌ patients.csv에 필요한 컬럼이 없습니다: {req}", None
    if c_name is None:
        # 이름 컬럼이 아예 없으면 빈 문자열로 대체
        df["__name__"] = "김지헌"
        c_name = "__name__"

    # 숫자 변환 후 필터
    df[c_subj] = pd.to_numeric(df[c_subj], errors="coerce").astype("Int64")
    sel = df.loc[df[c_subj] == sid, [c_subj, c_name, c_gender, c_age, c_year]].copy()

    if sel.empty:
        return f"⚠️ subject_id={sid} 에 해당하는 환자 정보가 없습니다.", None

    sel = sel.drop_duplicates().reset_index(drop=True)
    sel.columns = ["환자 번호", "환자 이름", "성별", "나이", "출생년도"]

    info = f"✅ 환자 조회 완료: subject_id={sid} (행 {len(sel)}개)"
    return info, sel


# =========================
# 2) 퇴원일 예측
# =========================
@torch.no_grad()
def run_inference(
    subject_id_str: str,
    unified_csv: str,
    ckpt_path: str,
    use_example_sources: bool,
    sources_json_path: str,
    cache_dir: str,
    encode_device_str: str,  # "cpu" | "cuda" | "mps"
    batch_size: int,
):
    # 입력 검증
    if not subject_id_str or not subject_id_str.strip().isdigit():
        return "❌ subject_id는 정수여야 합니다.", None
    subject_id = int(subject_id_str.strip())

    if not os.path.isfile(unified_csv):
        return f"❌ unified_csv 경로가 존재하지 않습니다: {unified_csv}", None
    if not os.path.isfile(ckpt_path):
        return f"❌ 체크포인트(.pt) 경로가 존재하지 않습니다: {ckpt_path}", None

    # sources 구성
    if use_example_sources:
        sources = example_sources_config_v3()
    else:
        if not sources_json_path or not os.path.isfile(sources_json_path):
            return "❌ sources_json_path가 유효하지 않습니다. 파일 경로를 확인하세요.", None
        with open(sources_json_path, "r", encoding="utf-8") as f:
            sources = json.load(f)

    # (선택) BRITS CKPT hidden_dim 미스매치 보정(필요시)
    if "complete_blood_count" in sources:
        sources["complete_blood_count"]["hidden_dim"] = sources["complete_blood_count"].get("hidden_dim", 64)

    # 단일 DS 생성(전체) → Subset으로 subject만 추출
    ds_all = HadmTableDatasetV3(
        unified_csv=unified_csv,
        drop_index_death=False,
        drop_30d_postdischarge_death=False,
        crrt_label_csv=None,
        mimic_derived_lods_csv=None,
        mimic_derived_crrt_csv=None,
        crrt_restrict_within_admission_window=True,
        sources=sources,
        encode_device=encode_device_str if encode_device_str in ("cpu","cuda","mps") else "cpu",
        cache_dir=cache_dir if (cache_dir and len(cache_dir.strip())>0) else None,
    )

    idxs = ds_all.df.index[ds_all.df["subject_id"] == subject_id].tolist()
    if len(idxs) == 0:
        return f"⚠️ subject_id={subject_id} 에 해당하는 HADM이 없습니다.", None

    subset = Subset(ds_all, idxs)

    # hadm → (admit, disch)
    tmp = ds_all.df.loc[idxs, ["hadm_id", "admittime", "dischtime"]].copy()
    tmp["admittime"] = pd.to_datetime(tmp["admittime"], errors="coerce")
    tmp["dischtime"] = pd.to_datetime(tmp["dischtime"], errors="coerce")
    times_map = {int(r.hadm_id): (r.admittime, r.dischtime) for r in tmp.itertuples(index=False)}

    # Loader
    loader = DataLoader(
        subset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_hadm_batch_v3,
        drop_last=False,
    )

    # peek → (S,N,E)
    peek = next(iter(loader))
    S = peek["base"].shape[1]
    N = peek["exam_z"].shape[1]
    E = peek["exam_z"].shape[2] if N > 0 else 0

    # 모델 로드
    device = resolve_device(None)
    model = TableTransformerPredictor(
        num_tables=N, latent_dim=E, base_dim=S,
        d_model=256, nhead=8, depth=3, dim_ff=768,
        dropout=0.15, head_hidden=256,
        use_film=True, use_masked_mean=True
    ).to(device).eval()

    ck = torch.load(ckpt_path, map_location="cpu")
    state_dict = ck.get("model", ck)  # {"model": ...} 또는 state_dict
    model.load_state_dict(state_dict, strict=False)

    # 추론
    rows, seen = [], set()
    for batch in loader:
        base = batch["base"].to(device)
        exam_z = batch["exam_z"].to(device)
        exam_mask = batch["exam_mask"].to(device)

        los_pred, readmit_logit = model(base=base, exam_z=exam_z, exam_mask=exam_mask)
        # readmit_prob = torch.sigmoid(readmit_logit)  # 필요시 사용

        hadm_list = batch["hadm_id"]
        subj_list = batch["subject_id"]
        for i in range(len(hadm_list)):
            hadm = int(hadm_list[i])
            subj = int(subj_list[i])

            lp = float(los_pred[i].item())
            lp_3 = round(lp, 3)  # 소수점 3자리

            # 실제 입/퇴원일
            admit_dt, disch_true_dt = times_map.get(hadm, (None, None))

            # 예측 퇴원일(LOS 3자리 사용)
            pred_disch_dt = None
            if admit_dt is not None and pd.notna(admit_dt):
                try:
                    pred_disch_dt = admit_dt + pd.to_timedelta(lp_3, unit="D")
                except Exception:
                    pred_disch_dt = None
            
            # 오차(일) → 소수점 3자리
            err_days_3 = None
            if (pred_disch_dt is not None) and (disch_true_dt is not None) and pd.notna(disch_true_dt):
                try:
                    err_days = (pred_disch_dt - disch_true_dt).total_seconds() / (24 * 3600.0)
                    err_days_3 = round(err_days, 3)
                except Exception:
                    err_days_3 = None

            key = (subj, hadm)
            if key in seen:
                continue
            seen.add(key)

            rows.append({
                "subject_id": subj,
                "hadm_id": hadm,
                "admittime": admit_dt,
                "dischtime_true": disch_true_dt,
                "pred_dischtime": pred_disch_dt,   # ← lp_3로 계산된 datetime
                "los_pred_days": lp_3,             # ← 소수점 3자리
                "error_days": (float(err_days_3) if err_days_3 is not None else None),  # ← 소수점 3자리
            })

    df = pd.DataFrame(
        rows,
        columns=[
            "subject_id", "hadm_id", "admittime", "dischtime_true",
            "pred_dischtime", "los_pred_days", "error_days"
        ],
    ).drop_duplicates().reset_index(drop=True)

    # 날짜 보기 좋게
    for c in ["admittime", "dischtime_true", "pred_dischtime"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # 숫자 3자리 반올림 (표시는 float로 유지)
    for c in ["los_pred_days", "error_days"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(3)

    info = (
        f"✅ 예측 완료: subject_id={subject_id}, HADM {len(df)}건\n"
        f" - 모델 체크포인트: {os.path.basename(ckpt_path)}\n"
        f" - 테이블 수 N={N}, 잠재 차원 E={E}, base 차원 S={S}"
    )
    return info, df


# subject_id만 받아 예측 실행 (고정 인자 래핑)
def run_inference_defaults(subject_id_str: str):
    return run_inference(
        subject_id_str=subject_id_str,
        unified_csv=UNIFIED_CSV,
        ckpt_path=CKPT_PATH,
        use_example_sources=USE_EXAMPLE_SOURCES,
        sources_json_path=SOURCES_JSON_PATH,
        cache_dir=CACHE_DIR,
        encode_device_str=ENCODE_DEVICE,
        batch_size=BATCH_SIZE,
    )


# =========================
# Gradio UI
# =========================
with gr.Blocks(title="UTI 환자 조회 & 퇴원일 예측") as demo:
    gr.Markdown("## 🧑‍⚕️ 환자 조회 → 🏥 퇴원일 예측 (subject_id 기반)\n"
                f"- patients.csv: `{PATIENTS_CSV}`\n"
                f"- 모델: `{CKPT_PATH}`\n"
                f"- 데이터: `{UNIFIED_CSV}`\n"
                f"- sources: {'example_sources_config_v3()' if USE_EXAMPLE_SOURCES else SOURCES_JSON_PATH}\n"
                f"- device: `{ENCODE_DEVICE}`, batch_size: {BATCH_SIZE}")

    with gr.Row():
        subject_id_in = gr.Textbox(label="subject_id (정수)", placeholder="예: 10000232")
    with gr.Row():
        patients_csv_in = gr.Textbox(label="patients.csv 경로", value=PATIENTS_CSV)

    with gr.Row():
        btn_lookup = gr.Button("🔎 환자 조회")
        btn_predict = gr.Button("🧮 퇴원일 예측")

    with gr.Row():
        # 조회 결과
        patient_info_out = gr.Textbox(label="환자 조회 상태")
        patient_table_out = gr.Dataframe(label="환자 정보", interactive=False)

    with gr.Row():
        # 예측 결과
        pred_info_out = gr.Textbox(label="예측 상태")
        pred_table_out = gr.Dataframe(
            label="예측 결과",
            interactive=False
        )
    # 동작 바인딩
    btn_lookup.click(
        fn=lookup_patient,
        inputs=[subject_id_in, patients_csv_in],
        outputs=[patient_info_out, patient_table_out]
    )

    btn_predict.click(
        fn=run_inference_defaults,
        inputs=[subject_id_in],
        outputs=[pred_info_out, pred_table_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
