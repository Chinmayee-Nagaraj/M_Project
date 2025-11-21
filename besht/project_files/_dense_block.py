import torch
import torch.nn as nn

class ConvNetBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.ln = nn.LayerNorm(channels)
        self.pw_conv1 = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.dw_conv1 = nn.Conv1d(channels * 2, channels * 2,
                                  kernel_size=kernel_size,
                                  padding=kernel_size // 2,
                                  groups=channels * 2)
        self.glu = nn.GLU(dim=1)
        self.pw_conv2 = nn.Conv1d(channels, channels, kernel_size=1)

        # second conv path
        self.dw_conv2 = nn.Conv1d(channels, channels,
                                  kernel_size=kernel_size,
                                  padding=kernel_size // 2,
                                  groups=channels)
        self.pw_conv3 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        # Input shape: (B, F, T)
        residual_1 = x

        # (B, F, T) -> (B, T, F) for LayerNorm
        x_norm = x.transpose(1, 2)
        x_norm = self.ln(x_norm)
        x_norm = x_norm.transpose(1, 2)  # back to (B, F, T)

        # First conv path
        x = self.pw_conv1(x_norm)
        x = self.dw_conv1(x)
        x = self.glu(x)
        x = self.pw_conv2(x)

        # First residual
        x = x + residual_1

        # Second conv path
        residual_2 = x
        x = self.dw_conv2(x)
        x = self.pw_conv3(x)

        # Second residual
        x = x + residual_2

        return x  # shape (B, F, T)


# -----------------------
# Improved DenseBlock (Replica of Fig. 2)
# -----------------------
class ImprovedDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, dilations=[1, 3, 5]):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_channels = in_channels
        for d in dilations:
            block = nn.Sequential(
                # 2×3 Conv with dilation
                nn.Conv1d(current_channels, growth_rate,
                          kernel_size=3, dilation=d, padding=d),
                nn.ReLU(),
                nn.Conv1d(growth_rate, growth_rate,
                          kernel_size=3, dilation=d, padding=d),
                nn.ReLU(),
                # ConvNet after 2×3 conv
                ConvNetBlock(growth_rate)
            )
            self.blocks.append(block)
            current_channels += growth_rate

    def forward(self, x):
        # input shape: (B, T, F)
        x = x.transpose(1, 2)  # -> (B, F, T)

        features = [x]
        for block in self.blocks:
            concatenated_features = torch.cat(features, dim=1)
            out = block(concatenated_features)
            features.append(out)

        return torch.cat(features, dim=1).transpose(1, 2)  # -> (B, T, F)


# -----------------------
# 🔎 Test
# -----------------------
if __name__ == "__main__":
    B, T, F = 2, 100, 80
    dummy = torch.randn(B, T, F)

    denseblock = ImprovedDenseBlock(in_channels=F, growth_rate=16)
    out = denseblock(dummy)

    expected_channels = F + len(denseblock.blocks) * 16

    print("Input shape :", dummy.shape)
    print("Output shape:", out.shape)
    print("Expected output channels (F):", expected_channels)

    assert out.shape == (B, T, expected_channels)
    print("\n✅ All Checks Passed.")
