import torch
import torch.nn as nn
from torch import Tensor


class GRUFFNBlock(nn.Module):
    """
    GRU-based position-aware feed-forward block.
    Implements: GRU -> ReLU -> Linear -> Dropout
    Input/Output: (batch, seq_len, hidden_dim)
    """
    def __init__(self, in_dim: int, gru_hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=gru_hidden_size,
                          batch_first=True, bidirectional=False)
        self.act = nn.ReLU()
        self.proj = nn.Linear(gru_hidden_size, in_dim)
        self.dropout = nn.Dropout(dropout)

        # Weight initialization
        nn.init.kaiming_uniform_(self.proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.proj.bias)
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)        # (batch, seq_len, gru_hidden_size)
        activated = self.act(gru_out)   # ReLU
        out = self.proj(activated)      # project back to in_dim
        return self.dropout(out)


class TransformerEncoder(nn.Module):
    """
    Single Transformer Encoder:
      1. Multi-head self-attention + residual + LayerNorm
      2. GRU-FFN (GRU -> ReLU -> Linear) + residual + LayerNorm

    Input/Output: (batch, seq_len, hidden_dim)
    """
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        gru_dim: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        norm_layer: type = nn.LayerNorm,
    ):
        super().__init__()
        # Multi-head attention
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = norm_layer(hidden_dim, eps=1e-6)

        # GRU-based feed-forward
        self.gru_ffn = GRUFFNBlock(hidden_dim, gru_dim, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = norm_layer(hidden_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        torch._assert(x.dim() == 3, f"Expected (batch, seq_len, hidden_dim) got {x.shape}")

        # ---- Multi-head attention block ----
        attn_out, _ = self.self_attention(x, x, x, need_weights=False)
        x = x + self.dropout1(attn_out)   # residual
        x = self.ln1(x)                   # norm

        # ---- GRU-FFN block ----
        ffn_out = self.gru_ffn(x)
        x = x + self.dropout2(ffn_out)    # residual
        x = self.ln2(x)                   # norm

        return x
