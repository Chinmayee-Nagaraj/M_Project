import torch
import torch.nn as nn
from transformerEncoder import TransformerEncoder

# ------------------------------------------------------
# Single Bottleneck Block = Time → Frequency → Global
# ------------------------------------------------------
class BottleneckBlock(nn.Module):
    """
    Bottleneck block with three transformer stages:
      1. Time Transformer  : operates across temporal axis (per frequency bin).
      2. Frequency Transformer : operates across frequency axis (per time frame).
      3. Global Transformer: operates jointly across time–frequency tokens.

    Each stage uses residual connections + GroupNorm for stability.
    Input / output format:
        Input : (B, C, T, F)
        Output: (B, T*F, C) sequence for downstream modules
    """
    def __init__(self, hidden_dim, num_heads, gn_groups=8):
        super().__init__()

        # TransformerEncoders (one per axis)
        self.time_transformer = TransformerEncoder(
            num_heads=num_heads, hidden_dim=hidden_dim,
            gru_dim=hidden_dim * 2, dropout=0.1, attention_dropout=0.1
        )
        self.freq_transformer = TransformerEncoder(
            num_heads=num_heads, hidden_dim=hidden_dim,
            gru_dim=hidden_dim * 2, dropout=0.1, attention_dropout=0.1
        )
        self.global_transformer = TransformerEncoder(
            num_heads=num_heads, hidden_dim=hidden_dim,
            gru_dim=hidden_dim * 2, dropout=0.1, attention_dropout=0.1
        )

        # Normalization layers (applied in channels-first format)
        self.norm_t = nn.GroupNorm(gn_groups, hidden_dim)
        self.norm_f = nn.GroupNorm(gn_groups, hidden_dim)
        self.norm_g = nn.GroupNorm(gn_groups, hidden_dim)

    def forward(self, x, T, F):
        """
        Args:
            x: (B, C, T, F) input tensor
            T, F: expected time and frequency dims

        Returns:
            (B, T*F, C) sequence after Time–Freq–Global transformers
        """
        B, C, t, f = x.shape
        assert t == T and f == F, f"Expected input ({T},{F}), got ({t},{f})"

        # Put channels last for easier reshaping
        feat = x.permute(0, 2, 3, 1)  # (B, T, F, C)

        # --- Time Transformer (over T, for each freq bin) ---
        t_in = feat.permute(0, 2, 1, 3).reshape(B * F, T, C)   # (B*F, T, C)
        t_out = self.time_transformer(t_in)                    # (B*F, T, C)
        t_out = t_out.reshape(B, F, T, C).permute(0, 2, 1, 3)  # (B, T, F, C)
        feat = self.norm_t((feat + t_out).permute(0, 3, 1, 2)) # residual + norm
        feat = feat.permute(0, 2, 3, 1)                        # back to (B, T, F, C)

        # --- Frequency Transformer (over F, for each time step) ---
        f_in = feat.reshape(B * T, F, C)                       # (B*T, F, C)
        f_out = self.freq_transformer(f_in)                    # (B*T, F, C)
        f_out = f_out.reshape(B, T, F, C)
        feat = self.norm_f((feat + f_out).permute(0, 3, 1, 2))
        feat = feat.permute(0, 2, 3, 1)

        # --- Global Transformer (over flattened T*F tokens) ---
        g_in = feat.reshape(B, T * F, C)                       # (B, T*F, C)
        g_out = self.global_transformer(g_in)                  # (B, T*F, C)
        g_out = g_out.reshape(B, T, F, C)
        feat = self.norm_g((feat + g_out).permute(0, 3, 1, 2))
        feat = feat.permute(0, 2, 3, 1)

        return feat.reshape(B, T * F, C)  # flatten tokens


# ------------------------------------------------------
# Multi-block Bottleneck Wrapper (TFT)
# ------------------------------------------------------
class TFT(nn.Module):
    """
    Time-Frequency Transformer (TFT).

    Pipeline:
      1. 1x1 Conv → project input channels → hidden_dim
      2. N BottleneckBlocks stacked sequentially
      3. 1x1 Conv → refine output
      4. Optional grouped 1x1 Conv → project to out_channels

    Args:
        in_channels:  input channel dim
        hidden_dim:   embedding dim inside transformer
        num_heads:    attention heads
        num_blocks:   number of stacked bottlenecks
        gn_groups:    groups for GroupNorm + grouped conv
        out_channels: final output channels (defaults to hidden_dim)
    """
    def __init__(self, in_channels, hidden_dim, num_heads, num_blocks=2, gn_groups=8, out_channels=None):
        super().__init__()
        out_channels = out_channels or hidden_dim

        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.bottlenecks = nn.ModuleList([
            BottleneckBlock(hidden_dim, num_heads, gn_groups)
            for _ in range(num_blocks)
        ])
        self.output_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.gconv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, groups=gn_groups)

    def forward(self, x, T, F):
        """
        Args:
            x: (B, C, T, F)
            T, F: time and frequency dims

        Returns:
            (B, out_channels, T, F)
        """
        B, C, t, f = x.shape
        assert t == T and f == F, f"Expected input ({T},{F}), got ({t},{f})"

        # Step 1: Input projection
        x = self.input_proj(x)  # (B, hidden_dim, T, F)

        # Step 2: Sequential bottlenecks
        for block in self.bottlenecks:
            x_seq = block(x, T, F)                              # (B, T*F, hidden_dim)
            x = x_seq.permute(0, 2, 1).reshape(B, x.shape[1], T, F)

        # Step 3: Output projection
        x = self.output_proj(x)

        # Step 4: Grouped 1x1 conv (final projection)
        x = self.gconv(x)

        return x  # (B, out_channels, T, F)
