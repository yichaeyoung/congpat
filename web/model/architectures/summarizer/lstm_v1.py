import torch
import torch.nn as nn

def lengths_to_padding_mask(lengths, max_len=None):
    """
    lengths: (B,)
    return: (B, T)  # True == pad 위치
    """
    if lengths is None:
        return None
    B = lengths.shape[0]
    T = int(max_len) if max_len is not None else int(lengths.max().item())
    rng = torch.arange(T, device=lengths.device)[None, :]
    return rng >= lengths[:, None]

class LSTMAutoEncoder(nn.Module):
    """
    (B,T,F) → Encoder → z → Decoder → (B,T,F)
    - lengths 또는 mask 중 하나만 넘겨도 동작.
    - bidirectional 인코더 지원(디코더는 단방향).
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_dirs = 2 if bidirectional else 1
        assert hidden_dim % self.num_dirs == 0, "hidden_dim must be divisible by num_dirs"

        # Encoder
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

        # Decoder
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,           # 단방향 디코더
            num_layers=num_layers,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    # ----------------- internals -----------------
    def _pack_if_needed(self, x, lengths):
        if lengths is None:
            return x, None
        return nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        ), lengths

    def _unpack_if_needed(self, x_packed, lengths):
        if lengths is None:
            return x_packed
        x, _ = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
        return x

    # ----------------- encoder/decoder -----------------
    def encode(self, X, lengths=None):
        """
        X: (B,T,F)
        return: z (B, latent_dim)
        """
        h = torch.relu(self.enc_in(X))  # (B,T,H)
        if lengths is not None:
            packed, _ = self._pack_if_needed(h, lengths)
            _, (hn, _) = self.encoder(packed)
        else:
            _, (hn, _) = self.encoder(h)

        # hn: (num_layers*num_dirs, B, H//num_dirs)
        hn = hn.view(self.num_layers, self.num_dirs, X.size(0), self.hidden_dim // self.num_dirs)
        last = hn[-1]                               # (num_dirs, B, H//num_dirs)
        last = last.transpose(0, 1).contiguous()    # (B, num_dirs, H//num_dirs)
        h_final = last.view(X.size(0), self.hidden_dim)  # (B, H)
        z = self.to_latent(h_final)                 # (B, latent)
        return z

    def decode(self, z, T: int):
        """
        z: (B, latent_dim)
        T: decode length
        return: X_hat (B,T,F)
        """
        B = z.size(0)
        seed = torch.relu(self.from_latent(z))      # (B,H)
        dec_in = seed.unsqueeze(1).repeat(1, T, 1)  # (B,T,H)  간단한 반복 입력
        dec_out, _ = self.decoder(dec_in)           # (B,T,H)
        X_hat = self.out_proj(dec_out)              # (B,T,F)
        return X_hat

    # ----------------- forward -----------------
    def forward(self, X, lengths=None, mask=None, return_latent=False):
        """
        X: (B,T,F)
        lengths: (B,) 실제 길이. 없으면 mask로부터 추정.
        mask: (B,T) True=pad
        """
        if lengths is None and mask is not None:
            lengths = (~mask).sum(dim=1)

        z = self.encode(X, lengths=lengths)
        T = X.size(1)
        X_hat = self.decode(z, T)
        return (X_hat, z) if return_latent else X_hat