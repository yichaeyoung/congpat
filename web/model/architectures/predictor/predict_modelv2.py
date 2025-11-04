# architectures/predictor/table_transformer_predictor.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x): return self.net(x)

class TableTransformerPredictor(nn.Module):
    """
    inputs:
      - base: (B, S)
      - exam_z: (B, N, E)
      - exam_mask: (B, N)  # 1=존재, 0=없음
    outputs:
      - los_pred: (B,)
      - readmit_logit: (B,)
    """
    def __init__(
        self,
        *,
        num_tables: int,     # N
        latent_dim: int,     # E
        base_dim: int,       # S
        d_model: int = 256,
        nhead: int = 8,
        depth: int = 3,
        dim_ff: int = 512,
        dropout: float = 0.1,
        head_hidden: int = 256,
        use_film: bool = True,
        use_masked_mean: bool = True,
    ):
        super().__init__()
        self.N = num_tables
        self.E = latent_dim
        self.S = base_dim
        self.D = d_model
        self.use_film = use_film
        self.use_masked_mean = use_masked_mean

        # --- Tokenizers ---
        self.z_proj = nn.Linear(self.E, self.D) if self.E > 0 else None
        self.table_embed = nn.Embedding(max(1, self.N), self.D)     # table type
        self.presence_embed = nn.Embedding(2, self.D)               # 0/1
        self.missing_token = nn.Parameter(torch.zeros(max(1, self.N), self.D))
        nn.init.xavier_uniform_(self.missing_token)

        # CLS from base
        self.base_proj = MLP(self.S, max(self.S, self.D), self.D, dropout=dropout)
        self.cls_token_bias = nn.Parameter(torch.zeros(self.D))

        # FiLM conditioning from base
        if self.use_film:
            self.film = nn.Sequential(
                nn.Linear(self.S, self.D * 2),
                nn.GELU(),
                nn.Linear(self.D * 2, self.D * 2),
            )
        else:
            self.film = None

        # --- Transformer encoder over tokens (CLS + tables) ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.D, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.enc_norm = nn.LayerNorm(self.D)

        # --- Heads ---
        # pool: [CLS, masked_mean(table_tokens), valid_ratio]
        head_in = self.D + self.D + 1  # CLS + pooled + valid_ratio
        self.shared = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.GELU(),
            nn.LayerNorm(head_hidden),
            nn.Dropout(dropout),
        )
        self.head_los = nn.Linear(head_hidden, 1)
        self.head_readmit = nn.Linear(head_hidden, 1)

    def forward(self, *, base: torch.Tensor, exam_z: torch.Tensor, exam_mask: torch.Tensor):
        """
        base: (B,S)         exam_z: (B,N,E) or (B,0,0)    exam_mask: (B,N) or (B,0)
        """
        B = base.size(0)
        device = base.device

        # ---- valid ratio ----
        if exam_mask.numel() == 0:
            valid_ratio = torch.zeros(B, 1, device=device)
        else:
            vr = torch.clamp(exam_mask, 0, 1).to(base.dtype)
            valid_ratio = vr.mean(dim=-1, keepdim=True)

        # ---- CLS from base ----
        cls = self.base_proj(torch.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0))  # (B,D)
        cls = cls + self.cls_token_bias

        tokens = [cls.unsqueeze(1)]  # (B,1,D)

        # ---- Table tokens ----
        if self.N > 0 and self.E > 0 and exam_z.numel() > 0:
            present = (exam_mask > 0.5).long()             # (B,N) 0/1
            z = torch.nan_to_num(exam_z, nan=0.0, posinf=0.0, neginf=0.0)  # (B,N,E)
            z = self.z_proj(z)                              # (B,N,D)

            # table type + presence emb
            table_ids = torch.arange(self.N, device=device).view(1, self.N).expand(B, self.N)  # (B,N)
            type_emb = self.table_embed(table_ids)                 # (B,N,D)
            pres_emb = self.presence_embed(present)                # (B,N,D)

            # learned missing token per table
            miss_tok = self.missing_token.unsqueeze(0).expand(B, self.N, self.D)  # (B,N,D)

            token_exam = torch.where(present.unsqueeze(-1).bool(), z, miss_tok)   # (B,N,D)
            token_exam = token_exam + type_emb + pres_emb

            tokens.append(token_exam)  # -> list: [(B,1,D), (B,N,D)]

        # ---- concat & FiLM ----
        X = torch.cat(tokens, dim=1)  # (B, 1+N, D)
        if self.use_film:
            gamma_beta = self.film(base)                   # (B, 2D)
            gamma, beta = gamma_beta.chunk(2, dim=-1)      # each (B,D)
            X = X * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        # ---- encoder ----
        H = self.encoder(X)                                # (B,1+N,D)
        H = self.enc_norm(H)

        cls_out = H[:, 0, :]                               # (B,D)
        if H.size(1) > 1 and self.use_masked_mean:
            # masked mean over table tokens
            if exam_mask.numel() == 0:
                pooled = torch.zeros_like(cls_out)
            else:
                present_f = torch.clamp(exam_mask, 0, 1).to(H.dtype).unsqueeze(-1)   # (B,N,1)
                denom = present_f.sum(dim=1).clamp(min=1.0)                          # (B,1)
                pooled = (H[:, 1:, :] * present_f).sum(dim=1) / denom                # (B,D)
        else:
            pooled = torch.zeros_like(cls_out)

        head_in = torch.cat([cls_out, pooled, valid_ratio], dim=-1)  # (B, D+D+1)
        Z = self.shared(head_in)

        los = self.head_los(Z).squeeze(-1)                 # (B,)
        readmit_logit = self.head_readmit(Z).squeeze(-1)   # (B,)
        return los, readmit_logit