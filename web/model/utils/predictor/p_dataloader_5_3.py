# hadm_dataloader_v3.py
# -*- coding: utf-8 -*-
"""
Hadm-centric DataLoader (no admissions.csv)
- Unified table(단일 CSV)만 신뢰 소스로 사용
- UTI→UTI 30일 재입원 라벨을 unified 내부에서 생성
- 테이블별 Summarizer를 사전학습 가중치로 로드하여 latent Z 생성
- 모델 타입: LSTM-AE 또는 BRITS를 테이블별로 선택 (model_type: "lstm" | "brits")
- kdigo_stages는 제외
- CRRT 0/1 라벨: 파일 있으면 재사용, 없으면 derived 소스에서 생성(옵션 저장)
- 청크 로딩/결측 견고/캐시 지원

Batch:
{
  "base": (B,S), "exam_z": (B,N,E), "exam_mask": (B,N),
  "y_los": (B,), "y_readmit": (B,),
  "hadm_id": List[int], "subject_id": List[int]
}
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Iterable, Union
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# =========================================================
# 0) Shared utils
# =========================================================
def lengths_to_padding_mask(lengths, max_len=None):
    if lengths is None: return None
    T = int(max_len) if max_len is not None else int(lengths.max().item())
    rng = torch.arange(T, device=lengths.device)[None, :]
    return rng >= lengths[:, None]  # True == pad

def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def _to_dt(x) -> pd.Series:
    return pd.to_datetime(x, errors="coerce", utc=False)

def _safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    return df

def _read_cols(csv_path: str, nrows: int = 5) -> List[str]:
    return pd.read_csv(csv_path, nrows=nrows).columns.tolist()

def _choose_join_key(csv_cols: List[str], prefer: Optional[str] = None) -> str:
    cand = ([prefer] if prefer else []) + ["hadm_id", "stay_id", "subject_id"]
    for k in cand:
        if k and (k in csv_cols):
            return k
    return csv_cols[0]

def _chunk_iter(csv_path: str, usecols: Optional[List[str]] = None, chunksize: int = 500_000):
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        yield chunk

def _ensure_dir(p: Union[str, Path]):
    Path(p).mkdir(parents=True, exist_ok=True)

def _onehot(val: Optional[str], vocab: List[str]) -> np.ndarray:
    v = np.zeros(len(vocab), dtype=np.float32)
    if val is None or not isinstance(val, str):
        return v
    try:
        idx = vocab.index(val)
        v[idx] = 1.0
    except ValueError:
        pass
    return v


# =========================================================
# 1) Summarizers
#    - LSTM AutoEncoder (encode-only)
#    - BRITS-style AutoEncoder (encode-only)
# =========================================================
class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 64,
                 num_layers: int = 1, bidirectional: bool = False, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_dirs = 2 if bidirectional else 1
        assert hidden_dim % self.num_dirs == 0

        self.enc_in = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // self.num_dirs,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def _pack_if_needed(self, x, lengths):
        if lengths is None: return x, None
        return nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False), lengths

    def _unpack_if_needed(self, x_packed, lengths):
        if lengths is None: return x_packed
        x, _ = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True); return x

    def encode(self, X, lengths=None):
        h = torch.relu(self.enc_in(X))
        if lengths is not None:
            packed, _ = self._pack_if_needed(h, lengths)
            _, (hn, _) = self.encoder(packed)
        else:
            _, (hn, _) = self.encoder(h)
        hn = hn.view(self.num_layers, self.num_dirs, X.size(0), self.hidden_dim // self.num_dirs)
        last = hn[-1].transpose(0, 1).contiguous()
        h_final = last.view(X.size(0), self.hidden_dim)
        z = self.to_latent(h_final)
        return z

    def decode(self, z, T: int):
        B = z.size(0)
        seed = torch.relu(self.from_latent(z))
        dec_in = seed.unsqueeze(1).repeat(1, T, 1)
        dec_out, _ = self.decoder(dec_in)
        return self.out_proj(dec_out)

    def forward(self, X, lengths=None, mask=None, return_latent=False):
        if lengths is None and mask is not None:
            lengths = (~mask).sum(dim=1)
        z = self.encode(X, lengths=lengths)
        T = X.size(1)
        X_hat = self.decode(z, T)
        return (X_hat, z) if return_latent else X_hat


# ----- BRITS helpers -----
def _masked_mean_time(x, pad_mask):  # x: (B,T,D), pad_mask: (B,T) True=pad
    if pad_mask is None: return x.mean(dim=1)
    valid = (~pad_mask).float()
    denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * valid.unsqueeze(-1)).sum(dim=1) / denom

def _nan_to_num(x, val=0.0):
    if torch.isnan(x).any(): return torch.nan_to_num(x, nan=val)
    return x

class BRITSAutoEncoder(nn.Module):
    """
    (B,T,F) -> bidirectional imputation encoder -> z (B,E) -> decoder (unused here)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 64,
                 num_layers: int = 1, bidirectional: bool = False, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.f_gru = nn.GRUCell(input_size=3 * input_dim, hidden_size=hidden_dim)
        self.b_gru = nn.GRUCell(input_size=3 * input_dim, hidden_size=hidden_dim)

        self.W_dh_f = nn.Linear(input_dim, hidden_dim)
        self.W_dh_b = nn.Linear(input_dim, hidden_dim)

        self.beta_x = nn.Parameter(torch.ones(input_dim))
        self.bias_x = nn.Parameter(torch.zeros(input_dim))

        self.out_x_f = nn.Linear(hidden_dim, input_dim)
        self.out_x_b = nn.Linear(hidden_dim, input_dim)

        self.fuse_gate = nn.Sequential(nn.Linear(2 * hidden_dim, input_dim), nn.Sigmoid())

        self.register_buffer("feat_mean", torch.zeros(input_dim))

        self.to_latent = nn.Linear(2 * hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers,
            bidirectional=False, dropout=dropout if num_layers > 1 else 0.0, batch_first=True,
        )
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    @staticmethod
    def _compute_deltas(obs_mask_feat, pad_mask):
        B, T, D = obs_mask_feat.shape
        delta = torch.zeros(B, T, D, device=obs_mask_feat.device, dtype=obs_mask_feat.dtype)
        if T <= 1: return delta
        for t in range(1, T):
            prev = delta[:, t - 1, :] + 1.0
            delta[:, t, :] = prev * (1.0 - obs_mask_feat[:, t, :])
            if pad_mask is not None:
                pad_t = pad_mask[:, t].unsqueeze(-1).float()
                delta[:, t, :] = delta[:, t, :] * (1.0 - pad_t)
        return delta

    def _run_direction(self, X, M_feat, Delta, pad_mask, forward=True):
        B, T, D = X.shape
        device = X.device
        if not forward:
            X = torch.flip(X, dims=[1]); M_feat = torch.flip(M_feat, dims=[1]); Delta = torch.flip(Delta, dims=[1])
            pad_mask = torch.flip(pad_mask, dims=[1]) if pad_mask is not None else None

        h = torch.zeros(B, self.hidden_dim, device=device)
        x_last = torch.zeros(B, D, device=device)
        H_list, Xhat_list = [], []
        Delta_norm = torch.log1p(Delta)

        for t in range(T):
            m_t = M_feat[:, t, :]
            x_t = X[:, t, :]
            d_t = Delta[:, t, :]
            dn_t = Delta_norm[:, t, :]

            Wdh = self.W_dh_f if forward else self.W_dh_b
            gamma_h = torch.exp(-F.relu(Wdh(d_t)))
            h = gamma_h * h

            gamma_x = torch.exp(-F.relu(self.beta_x * d_t + self.bias_x))
            x_decay = gamma_x * x_last + (1.0 - gamma_x) * self.feat_mean
            x_c = m_t * x_t + (1.0 - m_t) * x_decay

            gru_in = torch.cat([x_c, m_t, dn_t], dim=-1)
            if forward:
                h = self.f_gru(gru_in, h)
                x_hat = self.out_x_f(h)
            else:
                h = self.b_gru(gru_in, h)
                x_hat = self.out_x_b(h)

            x_last = torch.where(m_t > 0, x_t, x_last)

            if pad_mask is not None:
                step_mask = (~pad_mask[:, t]).float().unsqueeze(-1)
                h = step_mask * h + (1 - step_mask) * h.detach()
                x_hat = step_mask * x_hat + (1 - step_mask) * 0.0

            H_list.append(h.unsqueeze(1)); Xhat_list.append(x_hat.unsqueeze(1))

        H = torch.cat(H_list, dim=1); X_hat = torch.cat(Xhat_list, dim=1)
        if not forward: H = torch.flip(H, dims=[1]); X_hat = torch.flip(X_hat, dims=[1])
        return H, X_hat

    def encode(self, X, lengths=None, mask=None):
        B, T, D = X.shape
        pad_mask = mask if mask is not None else (lengths_to_padding_mask(lengths, T) if lengths is not None else None)
        if pad_mask is None:
            pad_mask = torch.zeros(B, T, dtype=torch.bool, device=X.device)
        X_in = _nan_to_num(X, 0.0)
        M_feat = (~torch.isnan(X)).float() if torch.isnan(X).any() else torch.ones_like(X_in)
        M_feat = M_feat * (~pad_mask).unsqueeze(-1).float()
        Delta = self._compute_deltas(M_feat, pad_mask)

        H_f, Xhat_f = self._run_direction(X_in, M_feat, Delta, pad_mask, forward=True)
        H_b, Xhat_b = self._run_direction(X_in, M_feat, Delta, pad_mask, forward=False)

        gate_in = torch.cat([H_f, H_b], dim=-1)
        alpha = self.fuse_gate(gate_in)
        _ = alpha * Xhat_f + (1.0 - alpha) * Xhat_b  # imputed (디코더는 inference에선 불필요)

        pooled = _masked_mean_time(gate_in, pad_mask)   # (B, 2H)
        z = self.to_latent(self.dropout(pooled))        # (B, latent)
        return z

    def decode(self, z, T, imputed=None, pad_mask=None):
        B = z.size(0)
        seed = torch.relu(self.from_latent(z))
        dec_in = seed.unsqueeze(1).repeat(1, T, 1)
        dec_out, _ = self.decoder(dec_in)
        X_dec = self.out_proj(self.dropout(dec_out))
        if pad_mask is not None:
            X_dec = X_dec * (~pad_mask).unsqueeze(-1).float()
        return X_dec

    def forward(self, X, lengths=None, mask=None, return_latent=False):
        z = self.encode(X, lengths=lengths, mask=mask)
        X_hat = self.decode(z, T=X.size(1))
        return (X_hat, z) if return_latent else X_hat


# =========================================================
# 2) CRRT presence: load-if-exists else build-and-save
# =========================================================
def load_or_build_crrt_labels(
    *, hadm_ids: Iterable[int], crrt_label_csv: Optional[str],
    mimic_derived_lods_csv: Optional[str], mimic_derived_crrt_csv: Optional[str],
    admit_cut_map: Optional[Dict[int, Tuple[pd.Timestamp,pd.Timestamp]]] = None,
    save_if_built: bool = True,
) -> Dict[int, int]:
    hadm_ids = [int(h) for h in hadm_ids]
    hadm_set = set(hadm_ids)
    pres = {h: 0 for h in hadm_ids}

    if crrt_label_csv and os.path.isfile(crrt_label_csv):
        df = pd.read_csv(crrt_label_csv, low_memory=False)
        lower = {c.lower(): c for c in df.columns}
        if ("hadm_id" in lower) and (("crrt_used" in lower) or ("crrt_flag" in lower) or ("label" in lower)):
            col_hadm = lower["hadm_id"]
            col_lab = lower.get("crrt_used", lower.get("crrt_flag", lower.get("label")))
            sub = df[[col_hadm, col_lab]].dropna()
            sub[col_hadm] = pd.to_numeric(sub[col_hadm], errors="coerce").astype("Int64")
            sub[col_lab] = pd.to_numeric(sub[col_lab], errors="coerce").fillna(0).astype(int)
            for r in sub.itertuples(index=False):
                h = int(r[0])
                if h in hadm_set:
                    pres[h] = int(r[1])
            return pres
        else:
            warnings.warn(f"[CRRT] {crrt_label_csv} malformed; will rebuild if sources available.")

    if (mimic_derived_lods_csv is not None) and (mimic_derived_crrt_csv is not None) and \
       os.path.isfile(mimic_derived_lods_csv) and os.path.isfile(mimic_derived_crrt_csv):

        lods = pd.read_csv(mimic_derived_lods_csv, usecols=["hadm_id","stay_id"], low_memory=False)
        lods["hadm_id"] = pd.to_numeric(lods["hadm_id"], errors="coerce").astype("Int64")
        lods["stay_id"] = pd.to_numeric(lods["stay_id"], errors="coerce").astype("Int64")
        lods = lods.dropna().astype(int)
        hadm2stays_all: Dict[int, Set[int]] = lods.groupby("hadm_id")["stay_id"].apply(lambda s: set(s.tolist())).to_dict()
        hadm2stays = {h: hadm2stays_all.get(h, set()) for h in hadm_ids}

        c_cols = _read_cols(mimic_derived_crrt_csv)
        usecols = ["stay_id"]; tcol = None
        for cand in ["charttime", "intime", "starttime"]:
            if cand in c_cols:
                tcol = cand; usecols.append(cand); break

        for chunk in _chunk_iter(mimic_derived_crrt_csv, usecols=usecols, chunksize=500_000):
            chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")
            if tcol: chunk[tcol] = _to_dt(chunk[tcol])
            by_stay = chunk.groupby("stay_id")
            for h, stays in hadm2stays.items():
                if pres[h] == 1: continue
                for sid in stays:
                    if sid in by_stay.groups:
                        if (admit_cut_map is not None) and (tcol is not None) and (h in admit_cut_map):
                            a, d = admit_cut_map[h]
                            sub = by_stay.get_group(sid)
                            sub = sub.loc[(sub[tcol].notna()) & (sub[tcol] >= a) & (sub[tcol] <= d)]
                            if len(sub) > 0:
                                pres[h] = 1; break
                        else:
                            pres[h] = 1; break

        if crrt_label_csv and save_if_built:
            out = pd.DataFrame({"hadm_id": list(pres.keys()), "crrt_used": list(pres.values())})
            Path(crrt_label_csv).parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(crrt_label_csv, index=False)
        return pres

    warnings.warn("[CRRT] No label file and no source files provided; using zeros.")
    return pres


# =========================================================
# 3) Exam default schema (kdigo_stages excluded)
# =========================================================
DEFAULT_REFERENCE_KEYS = {
    'apsiii':               ['subject_id','hadm_id','stay_id','apsiii'],
    'bg':                   ['subject_id','hadm_id','po2','pco2','ph','baseexcess','bicarbonate','totalco2'],
    'chemistry':            ['subject_id','hadm_id','albumin','aniongap','bun','calcium','chloride','glucose','sodium','potassium'],
    'complete_blood_count': ['subject_id','hadm_id','hematocrit','hemoglobin','platelet','rbc','wbc'],
    'creatinine_baseline':  ['hadm_id','mdrd_est','scr_baseline'],
    'enzyme':               ['subject_id','hadm_id','alt','alp','ast','bilirubin_total','bilirubin_direct','bilirubin_indirect','ggt'],
    'inflammation':         ['subject_id','hadm_id','crp'],
    'kdigo_creatinine':     ['hadm_id','stay_id','creat'],
}
DEFAULT_TIME_COL = {
    'apsiii': None, 'bg': 'charttime', 'chemistry': 'charttime',
    'complete_blood_count': 'charttime', 'creatinine_baseline': None,
    'enzyme': 'charttime', 'inflammation': 'charttime', 'kdigo_creatinine': 'charttime',
}

def _load_seq_by_keys_and_window(
    csv_path: str, key_col: str, key_vals: Set[Any], feats: List[str], time_col: Optional[str],
    hadm_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None, chunksize: int = 500_000
) -> np.ndarray:
    usecols = [key_col] + feats + ([time_col] if time_col else [])
    usecols = list(dict.fromkeys([c for c in usecols if c is not None]))
    parts = []; got_any = False
    for chunk in _chunk_iter(csv_path, usecols=usecols, chunksize=chunksize):
        m = chunk[key_col].isin(list(key_vals))
        if not m.any(): continue
        sub = chunk.loc[m].copy()
        if time_col and (time_col in sub.columns):
            sub[time_col] = _to_dt(sub[time_col])
            if hadm_window is not None:
                a, d = hadm_window
                sub = sub.loc[(sub[time_col].notna()) & (sub[time_col] >= a) & (sub[time_col] <= d)]
        if len(sub) > 0:
            parts.append(sub); got_any = True
    if not got_any:
        return np.zeros((0, len(feats)), dtype=np.float32)
    df = pd.concat(parts, ignore_index=True)
    if time_col and (time_col in df.columns):
        df = df.sort_values(time_col, ascending=True, kind="mergesort")
    df = _safe_numeric(df, feats)
    return df[feats].to_numpy(np.float32)


# =========================================================
# 4) UTI→UTI readmission labels within unified table
# =========================================================
def build_readmit_labels_from_unified(
    unified_df: pd.DataFrame,
    *, subject_col: str = "subject_id", hadm_col: str = "hadm_id",
    admit_col: str = "admittime", disch_col: str = "dischtime",
    drop_index_death: bool = False, drop_30d_postdischarge_death: bool = False,
) -> pd.DataFrame:
    df = unified_df[[subject_col, hadm_col, admit_col, disch_col] + (["deathtime"] if "deathtime" in unified_df.columns else [])].copy()
    df = df.rename(columns={subject_col:"subject_id", hadm_col:"hadm_id", admit_col:"admittime", disch_col:"dischtime"})
    df["admittime"] = _to_dt(df["admittime"]); df["dischtime"] = _to_dt(df["dischtime"])
    if "deathtime" in df.columns: df["deathtime"] = _to_dt(df["deathtime"])

    df = df.dropna(subset=["subject_id","hadm_id","admittime","dischtime"]).copy()
    df = df.sort_values(["subject_id","admittime","hadm_id"]).reset_index(drop=True)

    if "deathtime" in df.columns:
        if drop_index_death:
            m = (df["deathtime"].notna()) & (df["deathtime"] >= df["admittime"]) & (df["deathtime"] <= df["dischtime"])
            df = df.loc[~m].copy()
        if drop_30d_postdischarge_death:
            m2 = (df["deathtime"].notna()) & (df["deathtime"] > df["dischtime"]) & \
                 (df["deathtime"] <= (df["dischtime"] + pd.to_timedelta(30, unit="D")))
            df = df.loc[~m2].copy()

    df["next_hadm_id"] = df.groupby("subject_id")["hadm_id"].shift(-1)
    df["next_admittime"] = df.groupby("subject_id")["admittime"].shift(-1)
    valid_next = df["next_admittime"].notna() & (df["next_admittime"] >= df["dischtime"])
    df.loc[~valid_next, ["next_hadm_id","next_admittime"]] = np.nan

    df["los_days"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / (24*3600.0)
    df["days_to_readmit"] = (pd.to_datetime(df["next_admittime"]) - df["dischtime"]).dt.total_seconds() / (24*3600.0)
    df["readmit_30d"] = ((df["next_admittime"].notna()) & (df["days_to_readmit"] >= 0.0) & (df["days_to_readmit"] <= 30.0)).astype(int)

    return df[["hadm_id","subject_id","admittime","dischtime","los_days","next_hadm_id","next_admittime","days_to_readmit","readmit_30d"]].copy()


# =========================================================
# 5) Hadm-centric dataset (NO admissions.csv)
#     - model_type per table: "lstm" or "brits"
# =========================================================
class HadmTableDatasetV3(Dataset):
    """
    Inputs:
      - unified_csv: 단일 테이블(UTI admissions only)
      - sources: { table_name: {
            "csv_path": str,
            "ckpt_path": str,
            "feature_cols": List[str],
            "join_key": "hadm_id" | "subject_id" | "stay_id",
            "time_col": Optional[str],
            "latent_dim": int,
            "hidden_dim": int (opt),
            "num_layers": int (opt),
            "bidirectional": bool (opt; LSTM용),
            "dropout": float (opt),
            "model_type": "lstm" | "brits"   # <- NEW (default "lstm")
        }}
      - CRRT 라벨 재사용/생성 옵션

    Output:
      {
        "base": (S,), "exam_z": (N,E), "exam_mask": (N,),
        "y_los": float, "y_readmit": int, "hadm_id": int, "subject_id": int
      }
    """
    def __init__(
        self,
        *,
        unified_csv: str,
        drop_index_death: bool = False,
        drop_30d_postdischarge_death: bool = False,
        crrt_label_csv: Optional[str] = None,
        mimic_derived_lods_csv: Optional[str] = None,
        mimic_derived_crrt_csv: Optional[str] = None,
        crrt_restrict_within_admission_window: bool = True,
        sources: Dict[str, Dict[str, Any]],
        encode_device: str = "cpu",
        cache_dir: Optional[str] = None,
        limit_hadm_ids: Optional[Iterable[int]] = None,
    ):
        super().__init__()
        self.device = get_device()
        self.encode_device = torch.device(encode_device)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None: _ensure_dir(self.cache_dir)

        # ----- unified 읽기 -----
        base = pd.read_csv(unified_csv, low_memory=False)
        required = ["subject_id","hadm_id","admittime","dischtime","gender","anchor_age","race","diabetes"]
        for col in required:
            if col not in base.columns:
                raise KeyError(f"[unified_csv] missing required column: {col}")

        for tcol in ["admittime","dischtime","deathtime","intime","outtime","charttime"]:
            if tcol in base.columns: base[tcol] = _to_dt(base[tcol])

        first_keys = ["hadm_id","charttime"] if "charttime" in base.columns else ["hadm_id"]
        static_first = base.sort_values(first_keys, na_position="first").groupby("hadm_id").first().reset_index()

        static_cols = ["subject_id","hadm_id","admittime","dischtime","deathtime","gender","anchor_age","race","diabetes"]
        static_cols = [c for c in static_cols if c in static_first.columns]
        static = static_first[static_cols].copy()

        if "days_of_visit" in base.columns:
            dov = base[["hadm_id","days_of_visit"]].dropna().drop_duplicates("hadm_id")
            static = static.merge(dov, on="hadm_id", how="left")
        static["days_of_visit"] = static["days_of_visit"].where(
            static["days_of_visit"].notna(),
            (static["dischtime"] - static["admittime"]).dt.total_seconds() / (24*3600.0)
        )

        static = self._attach_icu_engineered_stats(static, base)

        labels = build_readmit_labels_from_unified(
            base, drop_index_death=drop_index_death, drop_30d_postdischarge_death=drop_30d_postdischarge_death
        )
        self.df = static.merge(
            labels[["hadm_id","subject_id","readmit_30d","los_days","admittime","dischtime"]],
            on=["hadm_id","subject_id"], how="inner", suffixes=("","_lbl")
        )

        self.df["y_los"] = pd.to_numeric(self.df["days_of_visit"], errors="coerce")
        self.df["y_readmit"] = self.df["readmit_30d"].fillna(0).astype(int)

        if limit_hadm_ids is not None:
            sel = set(int(h) for h in limit_hadm_ids)
            self.df = self.df.loc[self.df["hadm_id"].isin(sel)].reset_index(drop=True)

        admit_cut_map = None
        if crrt_restrict_within_admission_window:
            admit_cut_map = {int(r.hadm_id):(r.admittime, r.dischtime)
                             for r in self.df[["hadm_id","admittime","dischtime"]].itertuples(index=False)}

        crrt_map = load_or_build_crrt_labels(
            hadm_ids=self.df["hadm_id"].tolist(),
            crrt_label_csv=crrt_label_csv,
            mimic_derived_lods_csv=mimic_derived_lods_csv,
            mimic_derived_crrt_csv=mimic_derived_crrt_csv,
            admit_cut_map=admit_cut_map,
            save_if_built=True,
        )
        self.df["crrt_flag"] = self.df["hadm_id"].map(lambda h: int(crrt_map.get(int(h), 0)))

        self.genders = sorted([g for g in self.df["gender"].dropna().unique().tolist() if isinstance(g, str)])
        self.races = sorted([r for r in self.df["race"].dropna().unique().tolist() if isinstance(r, str)])

        # ----- sources 준비 (모델 타입별 로드) -----
        self.sources_cfg: Dict[str, Dict[str, Any]] = {}
        for name, cfg in sources.items():
            if not cfg.get("enabled", True): continue
            if name == "kdigo_stages": continue  # excluded
            if not os.path.isfile(cfg["csv_path"]):
                warnings.warn(f"[{name}] csv_path not found → skipped"); continue

            csv_cols = _read_cols(cfg["csv_path"])
            feats = cfg.get("feature_cols") or DEFAULT_REFERENCE_KEYS.get(name, [])
            feats = [c for c in feats if c in csv_cols]
            if len(feats) == 0:
                warnings.warn(f"[{name}] no valid features → skipped"); continue

            time_col = cfg.get("time_col", DEFAULT_TIME_COL.get(name))
            if time_col and (time_col not in csv_cols): time_col = None

            join_key = cfg.get("join_key") or _choose_join_key(csv_cols, prefer="hadm_id")

            latent_dim = int(cfg.get("latent_dim", 64))
            hidden_dim = int(cfg.get("hidden_dim", 128))
            num_layers = int(cfg.get("num_layers", 1))
            dropout = float(cfg.get("dropout", 0.0))
            bidirectional = bool(cfg.get("bidirectional", False))
            model_type = str(cfg.get("model_type", "lstm")).lower()

            ckpt_path = cfg.get("ckpt_path")
            if (ckpt_path is None) or (not os.path.isfile(ckpt_path)):
                warnings.warn(f"[{name}] ckpt missing → skipped"); continue

            # --- 모델 생성 ---
            if model_type == "brits":
                model = BRITSAutoEncoder(
                    input_dim=len(feats), hidden_dim=hidden_dim, latent_dim=latent_dim,
                    num_layers=num_layers, dropout=dropout
                )
            elif model_type == "lstm":
                model = LSTMAutoEncoder(
                    input_dim=len(feats), hidden_dim=hidden_dim, latent_dim=latent_dim,
                    num_layers=num_layers, bidirectional=bidirectional, dropout=dropout
                )
            else:
                warnings.warn(f"[{name}] unknown model_type={model_type} → skipped"); continue

            model = model.to(self.encode_device).eval()

            # --- 가중치 로드 ---
            try:
                state = torch.load(ckpt_path, map_location="cpu")
                sd = state.get("model", state)
                # input_dim 일치 여부(두 모델 모두 .input_dim 보유)
                if hasattr(model, "input_dim") and model.input_dim != len(feats):
                    warnings.warn(f"[{name}] ckpt/model feature mismatch: model expects {model.input_dim} vs file feats {len(feats)}")
                model.load_state_dict(sd, strict=False)
            except Exception as e:
                warnings.warn(f"[{name}] ckpt load failed: {e} → skipped"); continue

            self.sources_cfg[name] = {
                "csv_path": cfg["csv_path"],
                "feature_cols": feats,
                "join_key": join_key,
                "time_col": time_col,
                "latent_dim": latent_dim,
                "model": model,
                "model_type": model_type,
            }

        self.exam_names: List[str] = list(self.sources_cfg.keys())
        if len(self.exam_names) == 0:
            warnings.warn("No active summarizer sources; exam_z will be empty.")

        if self.cache_dir is not None:
            for name in self.exam_names: _ensure_dir(self.cache_dir / name)

        self.hadms: List[int] = [int(h) for h in self.df["hadm_id"].tolist()]
        self.hadm2subject: Dict[int,int] = {int(r.hadm_id): int(r.subject_id)
                                            for r in self.df[["hadm_id","subject_id"]].itertuples(index=False)}

    @staticmethod
    def _attach_icu_engineered_stats(static: pd.DataFrame, unified: pd.DataFrame) -> pd.DataFrame:
        if not (("intime" in unified.columns) and ("outtime" in unified.columns)):
            if "icu_days" in unified.columns:
                agg = unified.groupby("hadm_id")["icu_days"].sum().rename("icu_days_total").reset_index()
            else:
                agg = pd.DataFrame({"hadm_id": unified["hadm_id"].unique(), "icu_days_total": 0.0})
            out = static.merge(agg, on="hadm_id", how="left")
            out["icu_days_total"] = pd.to_numeric(out["icu_days_total"], errors="coerce").fillna(0.0)
            out["icu_visit_count"] = 0.0; out["icu_duration_max"] = 0.0
            out["icu_time_fraction"] = out["icu_days_total"] / (out["days_of_visit"].replace(0, np.nan))
            out["icu_time_fraction"] = out["icu_time_fraction"].replace([np.inf,-np.inf], np.nan).fillna(0.0)
            return out

        ev = unified[["hadm_id","intime","outtime","admittime","dischtime"]].copy()
        ev = ev.dropna(subset=["hadm_id"]).copy()
        ev["intime"] = _to_dt(ev["intime"]); ev["outtime"] = _to_dt(ev["outtime"])
        ev["admittime"] = _to_dt(ev["admittime"]); ev["dischtime"] = _to_dt(ev["dischtime"])

        ev = ev.loc[ev["intime"].notna()].copy()
        ev.loc[ev["outtime"].isna(), "outtime"] = ev.loc[ev["outtime"].isna(), "intime"]

        m_a = ev["admittime"].notna() & (ev["intime"] < ev["admittime"]); ev.loc[m_a, "intime"] = ev.loc[m_a, "admittime"]
        m_d = ev["dischtime"].notna() & (ev["outtime"] > ev["dischtime"]); ev.loc[m_d, "outtime"] = ev.loc[m_d, "dischtime"]

        dur = (ev["outtime"] - ev["intime"]).dt.total_seconds() / (24*3600.0); dur = dur.clip(lower=0.0)
        ev["dur_days"] = dur

        agg = ev.groupby("hadm_id").agg(
            icu_days_total=("dur_days","sum"),
            icu_visit_count=("dur_days","count"),
            icu_duration_max=("dur_days","max"),
        ).reset_index()

        out = static.merge(agg, on="hadm_id", how="left")
        for c in ["icu_days_total","icu_visit_count","icu_duration_max"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        out["icu_time_fraction"] = out["icu_days_total"] / (out["days_of_visit"].replace(0, np.nan))
        out["icu_time_fraction"] = out["icu_time_fraction"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    @torch.no_grad()
    def _summarize_table_for_hadm(self, hadm_id: int, name: str) -> Tuple[np.ndarray, int]:
        cfg = self.sources_cfg[name]
        csv_path, feats, time_col, join_key = cfg["csv_path"], cfg["feature_cols"], cfg["time_col"], cfg["join_key"]
        latent_dim, model, model_type = cfg["latent_dim"], cfg["model"], cfg["model_type"]

        row = self.df.loc[self.df["hadm_id"] == hadm_id].iloc[0]
        hadm_window = (row["admittime"], row["dischtime"])
        if join_key == "hadm_id":
            key_vals = {int(hadm_id)}
        elif join_key == "subject_id":
            key_vals = {int(row["subject_id"])}
        elif join_key == "stay_id":
            return np.zeros((latent_dim,), dtype=np.float32), 0
        else:
            key_vals = {int(hadm_id)}

        X_np = _load_seq_by_keys_and_window(
            csv_path=csv_path, key_col=join_key, key_vals=key_vals, feats=feats,
            time_col=time_col, hadm_window=hadm_window if time_col else None,
        )
        if X_np.shape[0] == 0:
            return np.zeros((latent_dim,), dtype=np.float32), 0

        # cache
        if self.cache_dir is not None:
            cpath = self.cache_dir / name / f"{hadm_id}__{model_type}.npz"
            if cpath.exists():
                try:
                    dat = np.load(cpath)
                    z = dat["z"].astype(np.float32, copy=False)
                    L = int(dat.get("length", X_np.shape[0]))
                    if z.shape[0] == latent_dim:
                        return z, L
                except Exception:
                    pass

        X = torch.from_numpy(X_np).unsqueeze(0).to(self.encode_device)  # (1,T,F)
        lengths = torch.tensor([X_np.shape[0]], device=self.encode_device)
        # 두 모델 모두 encode(lengths=...) 지원
        z = model.encode(X, lengths=lengths)     # (1,E)
        z_np = z.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

        if self.cache_dir is not None:
            np.savez_compressed(self.cache_dir / name / f"{hadm_id}__{model_type}.npz", z=z_np, length=int(X_np.shape[0]))
        return z_np, int(X_np.shape[0])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]; hadm = int(row["hadm_id"])
        age = float(row["anchor_age"]) if pd.notna(row["anchor_age"]) else 0.0
        diabetes = float(row["diabetes"]) if pd.notna(row["diabetes"]) else 0.0
        los = float(row.get("days_of_visit", np.nan))
        if not np.isfinite(los):
            if pd.notna(row["admittime"]) and pd.notna(row["dischtime"]):
                los = float((row["dischtime"] - row["admittime"]).total_seconds() / (24*3600.0))
            else:
                los = 0.0

        icu_days_total = float(row.get("icu_days_total", 0.0) or 0.0)
        icu_visit_count = float(row.get("icu_visit_count", 0.0) or 0.0)
        icu_duration_max = float(row.get("icu_duration_max", 0.0) or 0.0)
        icu_time_fraction = float(row.get("icu_time_fraction", 0.0) or 0.0)
        crrt_flag = float(row.get("crrt_flag", 0.0) or 0.0)

        g_one = _onehot((row["gender"] if isinstance(row["gender"], str) else None), self.genders)
        r_one = _onehot((row["race"] if isinstance(row["race"], str) else None), self.races)

        base_vec = np.concatenate([
            np.array([age, diabetes, los,
                      icu_days_total, icu_visit_count, icu_duration_max, icu_time_fraction,
                      crrt_flag], dtype=np.float32),
            g_one.astype(np.float32), r_one.astype(np.float32),
        ], axis=0)

        zs, masks = [], []
        for name in self.exam_names:
            z, L = self._summarize_table_for_hadm(hadm, name)
            zs.append(z); masks.append(1.0 if L > 0 else 0.0)

        exam_z = np.stack(zs, axis=0) if len(zs) > 0 else np.zeros((0, 0), dtype=np.float32)
        exam_mask = np.array(masks, dtype=np.float32) if len(masks) > 0 else np.zeros((0,), dtype=np.float32)

        return {
            "subject_id": int(row["subject_id"]),
            "hadm_id": hadm,
            "base": torch.from_numpy(base_vec),
            "exam_z": torch.from_numpy(exam_z),
            "exam_mask": torch.from_numpy(exam_mask),
            "y_los": torch.tensor(float(row["y_los"]), dtype=torch.float32),
            "y_readmit": torch.tensor(int(row["y_readmit"]), dtype=torch.int64),
        }


# =========================================================
# 6) Collate
# =========================================================
def collate_hadm_batch_v3(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    base = torch.stack([b["base"] for b in batch], dim=0)              # (B,S)
    exam_z = [b["exam_z"] for b in batch]
    if len(exam_z) > 0 and exam_z[0].ndim == 2:
        exam_z = torch.stack(exam_z, dim=0)                            # (B,N,E)
    else:
        exam_z = torch.zeros((len(batch), 0, 0), dtype=torch.float32)
    exam_mask = torch.stack([b["exam_mask"] for b in batch], dim=0)    # (B,N)

    return {
        "base": base,
        "exam_z": exam_z,
        "exam_mask": exam_mask,
        "y_los": torch.stack([b["y_los"] for b in batch], dim=0).view(-1),
        "y_readmit": torch.stack([b["y_readmit"] for b in batch], dim=0).view(-1),
        "hadm_id": [b["hadm_id"] for b in batch],
        "subject_id": [b["subject_id"] for b in batch],
    }


# =========================================================
# 7) Example sources config (모델 타입 포함)
# =========================================================
REFERENCE_KEYS = {
    'apsiii':               ['apsiii'],
    'bg':                   ['po2','pco2','ph','baseexcess','bicarbonate','totalco2'],
    'chemistry':            ['albumin','aniongap','bun','calcium','chloride','glucose','sodium','potassium'],
    'complete_blood_count': ['hematocrit','hemoglobin','platelet','rbc','wbc'],
    'creatinine_baseline':  ['mdrd_est','scr_baseline'],
    'crrt':                 [],
    'enzyme':               ['alt','alp','ast','bilirubin_total','bilirubin_direct','bilirubin_indirect','ggt'],
    'inflammation':         ['crp'],
    'kdigo_creatinine':     ['creat'],
    'kdigo_stages':         ['uo_rt_24hr'],
}

def example_sources_config_v3() -> Dict[str, Dict[str, Any]]:
    return {
        "apsiii": {
            "csv_path": "../dataset_building/outputs/apsiii_filtered.csv",
            "ckpt_path": "./trained_models/summarizer/summarizer_apsiii_500epoch.pth",
            "feature_cols": REFERENCE_KEYS["apsiii"],
            "join_key": "hadm_id",
            "latent_dim": 64,
            "model_type": "brits",   # or "brits"
        },
        "bg": {
            "csv_path": "../dataset_building/outputs/bg_filtered.csv",
            "ckpt_path": "./trained_models/summarizer/summarizer_bg_500epoch.pth",
            "feature_cols": REFERENCE_KEYS["bg"],
            "join_key": "hadm_id",
            "time_col": "charttime",
            "latent_dim": 64,
            "model_type": "brits",  # 예시: BRITS 사용
        },
        "chemistry": {
            "csv_path": "../dataset_building/outputs/chemistry_filtered.csv",
            "ckpt_path": "./trained_models/summarizer/summarizer_chemistry_500epoch.pth",
            "feature_cols": REFERENCE_KEYS["chemistry"],
            "join_key": "hadm_id",
            "time_col": "charttime",
            "latent_dim": 64,
            "model_type": "lstm",
        },
        "complete_blood_count": {
            "csv_path": "../dataset_building/outputs/complete_blood_count_filtered.csv",
            "ckpt_path": "./trained_models/summarizer/summarizer_complete_blood_count_500epoch.pth",
            "feature_cols": REFERENCE_KEYS["complete_blood_count"],
            "join_key": "hadm_id",
            "time_col": "charttime",
            "latent_dim": 64,
            "model_type": "brits",
        },
        "creatinine_baseline": {
            "csv_path": "../dataset_building/outputs/creatinine_baseline_filtered.csv",
            "ckpt_path": "./trained_models/summarizer/summarizer_creatinine_baseline_500epoch.pth",
            "feature_cols": REFERENCE_KEYS["creatinine_baseline"],
            "join_key": "hadm_id",
            "latent_dim": 64,
            "model_type": "lstm",
        },
        "enzyme": {
            "csv_path": "../dataset_building/outputs/enzyme_filtered.csv",
            "ckpt_path": "./trained_models/summarizer/summarizer_enzyme_500epoch.pth",
            "feature_cols": REFERENCE_KEYS["enzyme"],
            "join_key": "hadm_id",
            "time_col": "charttime",
            "latent_dim": 64,
            "model_type": "lstm",
        },
        "inflammation": {
            "csv_path": "../dataset_building/outputs/inflammation_filtered.csv",
            "ckpt_path": "./trained_models/summarizer/summarizer_inflammation_500epoch.pth",
            "feature_cols": REFERENCE_KEYS["inflammation"],
            "join_key": "hadm_id",
            "time_col": "charttime",
            "latent_dim": 64,
            "model_type": "lstm",
        },
        "kdigo_creatinine": {
            "csv_path": "../dataset_building/outputs/kdigo_creatinine_filtered.csv",
            "ckpt_path": "./trained_models/summarizer/summarizer_kdigo_creatinine_500epoch.pth",
            "feature_cols": REFERENCE_KEYS["kdigo_creatinine"],
            "join_key": "hadm_id",
            "time_col": "charttime",
            "latent_dim": 64,
            "model_type": "lstm",
        },
        # 'kdigo_stages': excluded
    }

# ------------------------------------------------------------------------------------
# 사용 예)
# from torch.utils.data import DataLoader
# from hadm_dataloader_v3 import HadmTableDatasetV3, collate_hadm_batch_v3, example_sources_config_v3
#
# unified_csv = "/path/to/your_unified_table.csv"
# sources = example_sources_config_v3()
#
# ds = HadmTableDatasetV3(
#     unified_csv=unified_csv,
#     crrt_label_csv="/path/to/crrt_labels.csv",
#     mimic_derived_lods_csv="/path/to/Mimic_derived_lods.csv",
#     mimic_derived_crrt_csv="/path/to/Mimic_derived_crrt.csv",
#     sources=sources,
#     encode_device="cuda",
#     cache_dir="./latent_cache_v3",
# )
# loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4,
#                     pin_memory=torch.cuda.is_available(), collate_fn=collate_hadm_batch_v3)
# batch = next(iter(loader))
# print("base:", tuple(batch["base"].shape))
# print("exam_z:", tuple(batch["exam_z"].shape))
# print("exam_mask:", tuple(batch["exam_mask"].shape))
# print("y_los[:5]:", batch["y_los"][:5])
# print("y_readmit[:5]:", batch["y_readmit"][:5])
# print("hadm sample:", batch["hadm_id"][:5])