import torch
import torch.nn as nn
from _transformer import Encoder   # your existing Transformer encoder

# ------------------------------------------------------
# Single Bottleneck Block = Time → Freq → Global
# ------------------------------------------------------
class BottleneckBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers=1, gn_groups=8):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Time Transformer (sequence = time, per frequency bin)
        self.time_transformer = Encoder(
            seq_length=None,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            gru_dim=hidden_dim * 2,
            dropout=0.1,
            attention_dropout=0.1
        )

        # Frequency Transformer (sequence = freq, per time step)
        self.freq_transformer = Encoder(
            seq_length=None,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            gru_dim=hidden_dim * 2,
            dropout=0.1,
            attention_dropout=0.1
        )

        # Global Transformer (sequence = T*F, whole grid)
        self.global_transformer = Encoder(
            seq_length=None,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            gru_dim=hidden_dim * 2,
            dropout=0.1,
            attention_dropout=0.1
        )

        # Normalization for stability
        self.norm_t = nn.GroupNorm(gn_groups, hidden_dim)
        self.norm_f = nn.GroupNorm(gn_groups, hidden_dim)
        self.norm_g = nn.GroupNorm(gn_groups, hidden_dim)

    def forward(self, x, T, F):
        """
        Args:
            x: (B, T*F, C) tokens
            T, F: time and frequency dimensions
        Returns:
            (B, T*F, C) tokens
        """
        B, L, C = x.shape
        assert L == T * F, f"Expected T*F={T*F}, got {L}"

        # reshape back to (B, T, F, C)
        feat = x.reshape(B, T, F, C)

        # --- Time Transformer ---
        t_in = feat.permute(0, 2, 1, 3).contiguous().reshape(B * F, T, C)  # (B*F, T, C)
        t_out = self.time_transformer(t_in)
        t_out = t_out.reshape(B, F, T, C).permute(0, 2, 1, 3).contiguous()
        # Apply GroupNorm: channels must be 2nd dim
        feat = self.norm_t((feat + t_out).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # --- Frequency Transformer ---
        f_in = feat.reshape(B * T, F, C)
        f_out = self.freq_transformer(f_in)
        f_out = f_out.reshape(B, T, F, C)
        feat = self.norm_f((feat + f_out).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # --- Global Transformer ---
        g_in = feat.reshape(B, T * F, C)
        g_out = self.global_transformer(g_in)
        g_out = g_out.reshape(B, T, F, C)
        feat = self.norm_g((feat + g_out).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return feat.reshape(B, T * F, C)


# ------------------------------------------------------
# Multi-block Bottleneck Wrapper
# ------------------------------------------------------
class TimeFreqGlobalBottleneck(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_blocks=2, num_layers=1, gn_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            BottleneckBlock(hidden_dim, num_heads, num_layers, gn_groups)
            for _ in range(num_blocks)
        ])

    def forward(self, x, T, F):
        for block in self.blocks:
            x = block(x, T, F)
        return x


# ------------------------------------------------------
# Test Scenario
# ------------------------------------------------------
if __name__ == "__main__":
    B, T, F, C = 2, 16, 8, 64   # batch, time, freq, channels
    dummy = torch.randn(B, T * F, C)

    model = TimeFreqGlobalBottleneck(hidden_dim=C, num_heads=4, num_blocks=2)
    out = model(dummy, T, F)

    print("Input :", dummy.shape)    # (B, T*F, C)
    print("Output:", out.shape)      # (B, T*F, C)
