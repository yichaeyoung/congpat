import os
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Optional
from torch.utils.data import Dataset

# ========= 공용 유틸 =========
def normalize_by_stats(X: np.ndarray, stats: Dict, feature_cols: List[str]) -> np.ndarray:
    means = np.array([stats["mean"].get(c, 0.0) for c in feature_cols], dtype=np.float32)
    stds  = np.array([stats["std"].get(c, 1.0) for c in feature_cols], dtype=np.float32)
    stds[stds == 0] = 1.0
    return (X - means) / stds

def pad_batch(samples, pad_value: float = 0.0, max_len=None, to_torch: bool = True):
    # samples: [{"static": {...}, "sequence": {"X": (T,F), "length": T, ...}}]
    seqs = [s["sequence"] for s in samples]
    lengths = [int(max(0, seq.get("length", 0))) for seq in seqs]
    if any(l <= 0 for l in lengths):
        for seq in seqs:
            if int(seq.get("length", 0)) <= 0:
                X = seq.get("X", None)
                F = X.shape[1] if isinstance(X, np.ndarray) and X.ndim == 2 else 0
                seq["X"] = np.zeros((1, F), dtype=np.float32)
                seq["length"] = 1
        lengths = [seq["length"] for seq in seqs]

    B = len(seqs)
    F = seqs[0]["X"].shape[1] if (B > 0 and isinstance(seqs[0]["X"], np.ndarray) and seqs[0]["X"].ndim == 2) else 0
    T = max(lengths) if max_len is None else min(max(lengths), max_len)

    X_pad = np.full((B, T, F), pad_value, np.float32)
    mask  = np.zeros((B, T), dtype=bool)
    subject_ids = []
    for i, s in enumerate(samples):
        seq = s["sequence"]
        t = min(int(seq["length"]), T)
        if F > 0 and t > 0:
            X_pad[i, :t] = seq["X"][:t]
        mask[i, :t] = True
        sid = s.get("static", {}).get("subject_id", None)
        subject_ids.append(sid)

    if to_torch:
        return {
            "X": torch.from_numpy(X_pad),
            "mask": torch.from_numpy(mask),
            "lengths": torch.tensor(lengths),
            "subject_id": subject_ids,
        }
    else:
        return {"X": X_pad, "mask": mask, "lengths": np.array(lengths), "subject_id": subject_ids}

# ========= 단일 CSV 기반 DataLoader =========
class summarizer_dataloader(Dataset):
    """
    - master CSV(환자 목록)에서 한 행을 고르고
    - derived CSV(한 파일)에서 표에 정의된 조인 키(subject_id/hadm_id/stay_id 우선순위)로 시퀀스를 수집
    - 피처는 표에서 ID 컬럼을 뺀 것만 사용(= static은 모델 입력에서 제외)
    - (T,F) numpy(+옵션 정규화), 빈 시퀀스 환자는 __init__에서 drop-empty
    """
    # 네가 정리한 표 그대로
    FILE_NAMES = [
        'apsiii','bg','chemistry','complete_blood_count','creatinine_baseline',
        'crrt','enzyme','inflammation','kdigo_creatinine','kdigo_stages',
    ]
    REFERENCE_KEYS = {
        'apsiii':               ['subject_id','hadm_id','stay_id','apsiii'],
        'bg':                   ['subject_id','hadm_id','po2','pco2','ph','baseexcess','bicarbonate','totalco2'],
        'chemistry':            ['subject_id','hadm_id','albumin','aniongap','bun','calcium','chloride','glucose','sodium','potassium'],
        'complete_blood_count': ['subject_id','hadm_id','hematocrit','hemoglobin','platelet','rbc','wbc'],
        'creatinine_baseline':  ['hadm_id','mdrd_est','scr_baseline'],
        'crrt':                 [],  # 데이터 컬럼 정의가 비어있음(필요 시 직접 feature_cols 지정)
        'enzyme':               ['subject_id','hadm_id','alt','alp','ast','bilirubin_total','bilirubin_direct','bilirubin_indirect','ggt'],
        'inflammation':         ['subject_id','hadm_id','crp'],
        'kdigo_creatinine':     ['hadm_id','stay_id','creat'],
        'kdigo_stages':         ['stay_id','uo_rt_24hr'],
    }
    # 시간열 추론 후보
    TIME_CANDIDATES = ["charttime","eventtime","starttime","time"]
    # ID 계열
    ID_COLS = {"subject_id","hadm_id","stay_id"}

    def __init__(
        self,
        column_key_name: str,                 # master 기준키(예: 'subject_id')
        master_csv_path: str,                 # 환자 마스터 CSV
        derived_csv_path: str,                # 단일 파생 CSV
        derived_table_name: Optional[str] = None,  # 없으면 파일명에서 추론
        feature_cols: Optional[List[str]] = None,  # 지정 안하면 표 기반에서 ID 제외
        norm_stats: Optional[Dict] = None,
        time_col: Optional[str] = None,
        order_asc: bool = True,
        load_into_memory: bool = True,
    ):
        self.master_csv_path = master_csv_path
        self.derived_csv_path = derived_csv_path
        self.column_key_name = column_key_name
        self.norm_stats = norm_stats
        self.time_col = time_col
        self.order_asc = order_asc
        self.load_into_memory = load_into_memory

        # 0) derived_table_name 추론
        if derived_table_name is None:
            fname = os.path.basename(derived_csv_path)
            stem = os.path.splitext(fname)[0].lower()
            # 파일명이 "apsiii_filtered"처럼 접두어가 들어가도 앞에서 매칭
            for n in self.FILE_NAMES:
                if stem.startswith(n):
                    derived_table_name = n
                    break
        if derived_table_name not in self.FILE_NAMES:
            raise ValueError(f"derived_table_name must be one of {self.FILE_NAMES} (got: {derived_table_name})")
        self.derived_table_name = derived_table_name

        # 1) master 로드 & 키 검사
        self.master = pd.read_csv(self.master_csv_path)
        if self.column_key_name not in self.master.columns:
            raise ValueError(f"column_key_name '{self.column_key_name}' not in master CSV: {self.master.columns.tolist()}")

        # 2) 파생 CSV 열 확인
        try:
            _peek = pd.read_csv(self.derived_csv_path, nrows=5)
            self.derived_cols = _peek.columns.tolist()
        except FileNotFoundError:
            raise FileNotFoundError(f"Derived CSV not found: {self.derived_csv_path}")

        # 3) 조인 키: 표에서 ID 우선순위 subject_id > hadm_id > stay_id 중 가능한 첫 키
        join_priority = [k for k in ['subject_id','hadm_id','stay_id'] if k in self.REFERENCE_KEYS[self.derived_table_name]]
        possible = [k for k in join_priority if k in self.derived_cols and k in self.master.columns]
        if self.column_key_name in possible:
            self.active_join_key = self.column_key_name
        elif possible:
            self.active_join_key = possible[0]
        else:
            raise ValueError(f"No common join key between master/derived for {self.derived_table_name}. "
                             f"Checked {join_priority}. master={self.master.columns.tolist()}, derived={self.derived_cols}")

        # 4) time_col 자동 추론
        if self.time_col is None:
            for cand in self.TIME_CANDIDATES:
                if cand in self.derived_cols:
                    self.time_col = cand
                    break

        # 5) feature_cols 기본: 표에서 ID 제외
        if feature_cols is None:
            base = [c for c in self.REFERENCE_KEYS[self.derived_table_name] if c not in self.ID_COLS]
            # 실제 파일에 존재하는 것만
            self.feature_cols = [c for c in base if c in self.derived_cols]
        else:
            # 사용자가 준 리스트에서 ID는 제거(혹시 들어왔을 경우)
            self.feature_cols = [c for c in feature_cols if (c in self.derived_cols and c not in self.ID_COLS)]

        if len(self.feature_cols) == 0 and self.derived_table_name != 'crrt':
            raise ValueError(f"No valid feature columns for {self.derived_table_name}. "
                             f"Got {feature_cols}, derived has {self.derived_cols}")

        # 6) 최종 usecols
        usecols = [self.active_join_key] + self.feature_cols
        if self.time_col and self.time_col in self.derived_cols:
            usecols.append(self.time_col)
        self.usecols = list(dict.fromkeys([c for c in usecols if c in self.derived_cols]))

        # 7) 파생 CSV 적재(옵션)
        self.derived_df = None
        if self.load_into_memory:
            self.derived_df = pd.read_csv(self.derived_csv_path, usecols=self.usecols)

        # 8) drop-empty: 파생에 키가 없는 master 행 제거
        if self.load_into_memory:
            derived_keys = set(self.derived_df[self.active_join_key].dropna().unique().tolist())
        else:
            only_key = pd.read_csv(self.derived_csv_path, usecols=[self.active_join_key])
            derived_keys = set(only_key[self.active_join_key].dropna().unique().tolist())

        before = len(self.master)
        keep_mask = self.master[self.column_key_name].isin(derived_keys)
        self.master = self.master.loc[keep_mask].reset_index(drop=True)
        after = len(self.master)

        print("== DataLoader (single CSV) ==")
        print(f"- master:   {self.master_csv_path}")
        print(f"- derived:  {self.derived_csv_path}")
        print(f"- table:    {self.derived_table_name}")
        print(f"- join key: {self.active_join_key}")
        print(f"- time_col: {self.time_col}")
        print(f"- features: {self.feature_cols}")
        print(f"- load_into_memory: {self.load_into_memory}")
        print(f"- [Filter drop-empty] master {before} -> {after} rows")

    def __len__(self):
        return len(self.master)

    def _row_to_sequence(self, row: pd.Series) -> Dict:
        key_col = self.active_join_key
        key_val = row[key_col] if key_col in row.index else row[self.column_key_name]

        df = self.derived_df if self.derived_df is not None else pd.read_csv(self.derived_csv_path, usecols=self.usecols)
        df = df.loc[df[key_col] == key_val]

        if df.empty:
            return {"subject_id": row.get("subject_id", None), "X": np.zeros((0, len(self.feature_cols)), np.float32), "length": 0}

        if self.time_col and self.time_col in df.columns:
            df = df.sort_values(self.time_col, ascending=self.order_asc, kind="mergesort")

        feats = [c for c in self.feature_cols if c in df.columns]
        X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)

        if self.norm_stats is not None and len(feats) > 0:
            X = normalize_by_stats(X, self.norm_stats, feats)

        out = {
            "subject_id": row.get("subject_id", None),
            "X": X,
            "length": X.shape[0],
            "features": feats,
        }
        if self.time_col and self.time_col in df.columns:
            out["time"] = df[self.time_col].astype(str).tolist()
        return out

    def __getitem__(self, idx: int) -> Dict:
        row = self.master.iloc[idx]
        seq = self._row_to_sequence(row)

        # static(식별/메타)만 별도로 전달: 모델 입력 피처에는 포함 안 됨
        static_keys = [k for k in ["subject_id","hadm_id","stay_id","gender","anchor_age","diabetes","hospitalization_day"] if k in self.master.columns]
        static_info = {k: row[k] for k in static_keys}

        if seq["length"] == 0:  # 혹시 남아있다면 더미로 가드
            seq["X"] = np.zeros((1, len(seq["features"])), dtype=np.float32)
            seq["length"] = 1
            if "time" in seq:
                seq["time"] = [""]

        return {"static": static_info, "sequence": seq}