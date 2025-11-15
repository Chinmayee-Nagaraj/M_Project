import torch
import torch.nn as nn
from torch import Tensor
from transformer_encoder import TransformerEncoder


class TFTblock(nn.Module):
    """
    Temporal-Frequency-Transformer (TFT) block.

    Consists of three sequential transformer stages:
        1. Time Transformer  : operates across temporal axis (per frequency bin).
        2. Frequency Transformer : operates across frequency axis (per time frame).
        3. Global Transformer: operates jointly across flattened time–frequency tokens.

    Each stage applies residual connections and GroupNorm for stability.

    Input shape  : (B, C, T, F)
    Output shape : (B, C, T, F)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        attention_dropout: float,
        gn_groups: int
    ) -> None:

        super().__init__()

        # TransformerEncoders for each stage
        self.time_transformer = TransformerEncoder(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            gru_dim=hidden_dim * 4,
            dropout=dropout,
            attention_dropout=attention_dropout
        )
        self.freq_transformer = TransformerEncoder(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            gru_dim=hidden_dim * 4,
            dropout=dropout,
            attention_dropout=attention_dropout
        )
        self.global_transformer = TransformerEncoder(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            gru_dim=hidden_dim * 4,
            dropout=dropout,
            attention_dropout=attention_dropout
        )

        # Normalization layers (channel-wise)
        self.norm_t = nn.GroupNorm(gn_groups, hidden_dim)
        self.norm_f = nn.GroupNorm(gn_groups, hidden_dim)
        self.norm_g = nn.GroupNorm(gn_groups, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the TFT block.

        Args:
            x (Tensor): Input tensor of shape (B, C, T, F)

        Returns:
            Tensor: Output tensor of shape (B, C, T, F)
        """
        B, C, T, F = x.shape  

        residual1 = x   # (B, C, T, F)
        
        # --- Time Transformer (across T for each F) ---
        t_in = residual1.permute(0, 3, 2, 1).reshape(B * F, T, C)   # (B*F, T, C)
        t_out = self.time_transformer(t_in)                         # (B*F, T, C)
        t_out = t_out.reshape(B, F, T, C).permute(0, 3, 2, 1)       # (B, C, T, F)
        feat = self.norm_t(t_out) + residual1                       # residual + norm (B, C, T, F)
        residual2 = feat.permute(0, 2, 3, 1)                        # (B, T, F, C)

        # --- Frequency Transformer (across F for each T) ---
        f_in = residual2.reshape(B * T, F, C)                       # (B*T, F, C)
        f_out = self.freq_transformer(f_in)                         # (B*T, F, C)
        f_out = f_out.reshape(B, T, F, C).permute(0, 3, 1, 2)       # (B, C, T, F)
        feat = self.norm_f(f_out) + residual2.permute(0, 3, 1, 2)   # residual + norm (B, C, T, F)
        residual3 = feat.permute(0, 2, 3, 1)                        # back to (B, T, F, C)

        # --- Global Transformer (over all T*F tokens) ---
        g_in = residual3.reshape(B, T * F, C)                       # (B, T*F, C)
        g_out = self.global_transformer(g_in)                       # (B, T*F, C)
        g_out = g_out.reshape(B, T, F, C).permute(0, 3, 1, 2)       # (B, C, T, F)
        feat = self.norm_g(g_out) + residual3.permute(0, 3, 1, 2)   # residual + norm (B, C, T, F)

        return feat


class BottleNeckBlock(nn.Module):
    """
    Bottleneck block that stacks multiple TFT blocks sequentially.

    Pipeline:
        1. 1x1 Conv → project input channels to hidden_dim
        2. N sequential TFT blocks
        3. 1x1 Conv → refine intermediate output
        4. Grouped 1x1 Conv → project to out_channels

    Args:
        in_channels (int): Input channels
        hidden_dim (int): Internal hidden dimension
        num_heads (int): Attention heads in each transformer
        num_blocks (int): Number of stacked TFT blocks
        gn_groups (int): Groups for GroupNorm and grouped 1x1 conv
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 4,
        gn_groups: int = 8
    ) -> None:

        super().__init__()
        out_channels = in_channels

        # Project input to hidden dimension
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        # Stack TFT blocks
        self.bottlenecks = nn.ModuleList([
            TFTblock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=0.1,               # default dropout
                attention_dropout=0.1,     # default attention dropout
                gn_groups=gn_groups
            )
            for _ in range(num_blocks)
        ])

        # Output projection
        self.output_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        # Grouped 1x1 convolution
        self.gconv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, groups=gn_groups)

    def forward(self, x: Tensor, T: int, F: int) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor (B, C, T, F)
            T (int): Expected time dimension
            F (int): Expected frequency dimension

        Returns:
            Tensor: Output tensor (B, out_channels, T, F)
        """
        B, C, t, f = x.shape
        assert (t, f) == (T, F), f"Expected input ({T},{F}), got ({t},{f})"

        # Input projection
        x = self.input_proj(x)  # (B, hidden_dim, T, F)

        # Sequential TFT blocks
        for block in self.bottlenecks:
            x = block(x)        # (B, hidden_dim, T, F)

        # Output projection
        x = self.output_proj(x)

        # Final grouped conv
        x = self.gconv(x)

        return x                # (B, out_channels, T, F)
