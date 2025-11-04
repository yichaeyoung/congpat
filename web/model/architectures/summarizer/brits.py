import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- helpers -----
def lengths_to_padding_mask(lengths, max_len=None):
    if lengths is None:
        return None
    T = int(max_len) if max_len is not None else int(lengths.max().item())
    rng = torch.arange(T, device=lengths.device)[None, :]
    return rng >= lengths[:, None]  # True == pad

def _masked_mean_time(x, pad_mask):  # x: (B,T,D), pad_mask: (B,T) True=pad
    if pad_mask is None:
        return x.mean(dim=1)
    valid = (~pad_mask).float()
    denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * valid.unsqueeze(-1)).sum(dim=1) / denom

def _nan_to_num(x, val=0.0):
    if torch.isnan(x).any():
        return torch.nan_to_num(x, nan=val)
    return x

# ----- BRITS-style AutoEncoder -----
class BRITSAutoEncoder(nn.Module):
    """
    (B,T,F) -> BRITS-style bidirectional imputation (encoder) -> z -> decoder -> (B,T,F)

    - 입력 X: (B,T,F), mask: (B,T) with True=pad
    - 결측 처리: X에서 NaN을 '미관측'으로 간주 (권장). pad는 mask로 구분.
    - 전/후방 GRUCell로 순차 임퓨트(+hidden decay), 두 방향을 게이팅으로 융합.
    - 인코더 잠복 z는 [H_fwd; H_bwd]의 가려진-평균 풀링 후 선형 투영.
    - 디코더는 z로부터 (B,T,F) 재구성 (임퓨트 결과에 residual로 얹어 안정화).

    Args (드롭인 호환):
        input_dim: 특성 F
        hidden_dim: 은닉차원 (각 방향의 GRUCell hidden)
        latent_dim: 잠복차원
        num_layers: 디코더 LSTM 층 수 (BRITS 인코더는 GRUCell 1층 고정)
        bidirectional: (호환용) 무시됨. BRITS는 내부적으로 양방향.
        dropout: 디코더/투영 일부에 사용
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 1,
        bidirectional: bool = False,   # 호환 파라미터 (내부에서 양방향 고정)
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ====== BRITS 핵심 컴포넌트 ======
        # (1) 전/후방 GRUCell (입력: [x_c, m, delta_norm] concat)
        self.f_gru = nn.GRUCell(input_size=3 * input_dim, hidden_size=hidden_dim)
        self.b_gru = nn.GRUCell(input_size=3 * input_dim, hidden_size=hidden_dim)

        # (2) hidden decay gamma_h = exp(-relu(W_h * delta + b_h)) per step
        self.W_dh_f = nn.Linear(input_dim, hidden_dim)
        self.W_dh_b = nn.Linear(input_dim, hidden_dim)

        # (3) value decay for last observation vs feature mean:
        #     x_c = m*x + (1-m)*(gamma_x * x_last + (1-gamma_x)*feat_mean)
        #     gamma_x = exp(-relu(beta_x * delta + bias_x)) per-feature
        self.beta_x = nn.Parameter(torch.ones(input_dim))   # (F,)
        self.bias_x = nn.Parameter(torch.zeros(input_dim))  # (F,)

        # (4) forward/backward에서 예측한 값 (imputation) 헤드
        self.out_x_f = nn.Linear(hidden_dim, input_dim)
        self.out_x_b = nn.Linear(hidden_dim, input_dim)

        # (5) 두 방향 융합 게이트: alpha_t in (0,1) per feature
        self.fuse_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, input_dim),
            nn.Sigmoid()
        )

        # (6) 특성별 전역 평균(임퓨트 백업값). 기본 0, 필요시 외부에서 덮어쓰기 권장
        self.register_buffer("feat_mean", torch.zeros(input_dim))

        # ====== 잠복 z 및 디코더 ======
        self.to_latent = nn.Linear(2 * hidden_dim, latent_dim)
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

    # ---------- delta 계산: 관측 간격 카운트 (NaN/패딩 제외) ----------
    @staticmethod
    def _compute_deltas(obs_mask_feat, pad_mask):
        """
        obs_mask_feat: (B,T,F) 1=관측, 0=미관측
        pad_mask: (B,T) True=pad
        return delta: (B,T,F) 각 특성별 '마지막 관측 이후 경과 step'
        """
        B, T, D = obs_mask_feat.shape
        delta = torch.zeros(B, T, D, device=obs_mask_feat.device, dtype=obs_mask_feat.dtype)
        if T <= 1:
            return delta
        # 첫 스텝: 0
        for t in range(1, T):
            prev = delta[:, t - 1, :] + 1.0
            # 관측되면 0, 아니면 누적
            delta[:, t, :] = prev * (1.0 - obs_mask_feat[:, t, :])
            # pad 구간은 0으로 유지
            if pad_mask is not None:
                pad_t = pad_mask[:, t].unsqueeze(-1).float()
                delta[:, t, :] = delta[:, t, :] * (1.0 - pad_t)
        return delta

    # ---------- 한 방향 RNN 진행(순차 임퓨트) ----------
    def _run_direction(self, X, M_feat, Delta, pad_mask, forward=True):
        """
        X: (B,T,F) 입력값 (NaN->0 치환됨)
        M_feat: (B,T,F) 1=관측, 0=미관측 (pad는 0)
        Delta: (B,T,F) 경과 step
        pad_mask: (B,T) True=pad
        forward: True면 정방향, False면 역방향
        Returns:
            H: (B,T,H), X_hat: (B,T,F)  (각 방향의 예측)
        """
        B, T, D = X.shape
        device = X.device

        if not forward:
            X = torch.flip(X, dims=[1])
            M_feat = torch.flip(M_feat, dims=[1])
            Delta = torch.flip(Delta, dims=[1])
            pad_mask = torch.flip(pad_mask, dims=[1]) if pad_mask is not None else None

        # 초기값
        h = torch.zeros(B, self.hidden_dim, device=device)
        x_last = torch.zeros(B, D, device=device)  # 마지막 관측값(특성별)
        H_list, Xhat_list = [], []

        # delta 정규화 (log1p)
        Delta_norm = torch.log1p(Delta)

        for t in range(T):
            m_t = M_feat[:, t, :]                   # (B,F)
            x_t = X[:, t, :]                        # (B,F)
            d_t = Delta[:, t, :]                    # (B,F)
            dn_t = Delta_norm[:, t, :]              # (B,F)

            # hidden decay
            Wdh = self.W_dh_f if forward else self.W_dh_b
            gamma_h = torch.exp(-F.relu(Wdh(d_t)))  # (B,H)
            h = gamma_h * h

            # value decay for imputation mixture
            gamma_x = torch.exp(-F.relu(self.beta_x * d_t + self.bias_x))  # (B,F)
            x_decay = gamma_x * x_last + (1.0 - gamma_x) * self.feat_mean  # (B,F)
            x_c = m_t * x_t + (1.0 - m_t) * x_decay                        # candidate input

            # GRUCell 입력: [x_c, m_t, dn_t]
            gru_in = torch.cat([x_c, m_t, dn_t], dim=-1)                   # (B,3F)
            if forward:
                h = self.f_gru(gru_in, h)                                  # (B,H)
                x_hat = self.out_x_f(h)                                    # (B,F)
            else:
                h = self.b_gru(gru_in, h)
                x_hat = self.out_x_b(h)

            # 관측이 있으면 last 갱신
            x_last = torch.where(m_t > 0, x_t, x_last)

            # pad면 상태/출력 유지(업데이트 무시)
            if pad_mask is not None:
                step_mask = (~pad_mask[:, t]).float().unsqueeze(-1)  # (B,1)
                h = step_mask * h + (1 - step_mask) * h.detach()     # keep same value
                x_hat = step_mask * x_hat + (1 - step_mask) * 0.0

            H_list.append(h.unsqueeze(1))
            Xhat_list.append(x_hat.unsqueeze(1))

        H = torch.cat(H_list, dim=1)       # (B,T,H)
        X_hat = torch.cat(Xhat_list, dim=1)# (B,T,F)

        if not forward:
            H = torch.flip(H, dims=[1])
            X_hat = torch.flip(X_hat, dims=[1])
        return H, X_hat

    # ---------- encode ----------
    def encode(self, X, lengths=None, mask=None):
        """
        X: (B,T,F), mask: (B,T) True=pad
        return: z (B,latent), fused_memory (B,T,2H), fused_impute (B,T,F)
        """
        B, T, D = X.shape
        pad_mask = mask if mask is not None else (lengths_to_padding_mask(lengths, T) if lengths is not None else None)

        # 관측 마스크: pad 제외 + NaN=미관측으로 처리 권장
        if pad_mask is None:
            pad_mask = torch.zeros(B, T, dtype=torch.bool, device=X.device)
        X_in = _nan_to_num(X, 0.0)
        M_feat = (~torch.isnan(X)).float() if torch.isnan(X).any() else torch.ones_like(X_in)
        M_feat = M_feat * (~pad_mask).unsqueeze(-1).float()

        # 델타 계산
        Delta = self._compute_deltas(M_feat, pad_mask)  # (B,T,F)

        # 전/후방 진행
        H_f, Xhat_f = self._run_direction(X_in, M_feat, Delta, pad_mask, forward=True)
        H_b, Xhat_b = self._run_direction(X_in, M_feat, Delta, pad_mask, forward=False)

        # 두 방향 융합(게이팅)
        gate_in = torch.cat([H_f, H_b], dim=-1)             # (B,T,2H)
        alpha = self.fuse_gate(gate_in)                     # (B,T,F) in (0,1)
        X_hat = alpha * Xhat_f + (1.0 - alpha) * Xhat_b     # (B,T,F)

        # 잠복 표현
        H_cat = gate_in                                    # (B,T,2H)
        pooled = _masked_mean_time(H_cat, pad_mask)        # (B,2H)
        z = self.to_latent(self.dropout(pooled))           # (B,latent)
        return z, H_cat, X_hat, pad_mask

    # ---------- decode ----------
    def decode(self, z, T, imputed=None, pad_mask=None):
        """
        z: (B,latent), T: 길이, imputed: (B,T,F) (전/후방 융합 임퓨트)
        최종 출력은 디코더 출력 + 임퓨트 residual로 안정화.
        """
        B = z.size(0)
        seed = torch.relu(self.from_latent(z))        # (B,H)
        dec_in = seed.unsqueeze(1).repeat(1, T, 1)    # (B,T,H)
        dec_out, _ = self.decoder(dec_in)             # (B,T,H)
        X_dec = self.out_proj(self.dropout(dec_out))  # (B,T,F)
        if imputed is not None:
            X_hat = X_dec + imputed                   # residual guidance
        else:
            X_hat = X_dec
        # pad는 0으로 무해화
        if pad_mask is not None:
            X_hat = X_hat * (~pad_mask).unsqueeze(-1).float()
        return X_hat

    # ---------- forward ----------
    def forward(self, X, lengths=None, mask=None, return_latent=False):
        """
        X: (B,T,F)
        lengths: (B,)  or  mask: (B,T) True=pad
        """
        z, memory, imputed, pad_mask = self.encode(X, lengths=lengths, mask=mask)
        X_hat = self.decode(z, T=X.size(1), imputed=imputed, pad_mask=pad_mask)
        return (X_hat, z) if return_latent else X_hat
