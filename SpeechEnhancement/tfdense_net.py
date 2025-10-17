import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as Func

from dense_block import DenseBlock
from bottleneck import BottleNeckBlock


class EncoderBlock(nn.Module):
    """Encoder block: DenseBlock → Conv3x3 → PReLU
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.dense = DenseBlock(in_channels)  # feature extractor
        self.conv = nn.Conv2d(
            self.dense.out_channels, out_channels,
            kernel_size=3, stride=2, padding=1
        )
        self.prelu = nn.PReLU()

    def forward(self, x: Tensor) -> Tensor:
        t_down, f_down = x.shape[2], x.shape[3]
        x = self.dense(x)
        x = self.prelu(self.conv(x))
        return x


class DecoderBlock(nn.Module):
    """Decoder block: concatenate skip → DenseBlock → ConvTranspose3x3 → PReLU
    Args:
        in_channels (int): channels from previous decoder
        en_channels (int): channels from encoder skip connection
        out_channels (int): output channels of decoder
    """
    def __init__(self, in_channels: int, en_channels: int, out_channels: int) -> None:
        super().__init__()
        total_in = in_channels + en_channels
        self.dense = DenseBlock(total_in)
        self.conv = nn.ConvTranspose2d(
            self.dense.out_channels, out_channels,
            kernel_size=3, stride=2, padding=1, output_padding=(1, 0)
        )
        self.prelu = nn.PReLU()

    def forward(self, x: Tensor, en: Tensor) -> Tensor:
        x = torch.cat([x, en], dim=1)  # concatenate along channel dim
        x = self.dense(x)
        x = self.prelu(self.conv(x))
        return x

class TFDense_Net(nn.Module):
    """TFDense-Net backbone: encoder → bottleneck → decoder
    Args:
        in_channels (int): input channels (e.g., 1 for spectrogram)
        base_channels (int): initial feature channels
        num_layers (int): number of encoder/decoder blocks
        hidden_dim (int): hidden dim for bottleneck transformer
        num_heads (int): num attention heads in bottleneck
        num_blocks (int): num transformer blocks in bottleneck
        gn_groups (int): groups for group normalization in bottleneck
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 64, num_layers: int = 3,
                 hidden_dim: int = 64, num_heads: int = 4, num_blocks: int = 4, gn_groups: int = 8) -> None:
        super().__init__()

        # Initial 1x1 conv to project input channels
        self.conv1x1 = nn.Conv2d(in_channels, base_channels, kernel_size=1)

        # Encoder blocks
        self.encoders = nn.ModuleList()
        channels = base_channels
        for _ in range(num_layers):
            self.encoders.append(EncoderBlock(channels, channels))

        # Bottleneck block
        self.bottleneck = BottleNeckBlock(
            in_channels=channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            gn_groups=gn_groups
        )

        # Decoder blocks
        self.decoders = nn.ModuleList()
        for _ in range(num_layers):
            self.decoders.append(DecoderBlock(channels, channels, channels))

        # Final output projection
        self.final = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        # Initial projection
        x = self.conv1x1(x)

        # Encoder with skip connections
        encoder_skips = []

        # Encoder output dimensions
        encoder_dims = []

        for enc in self.encoders:
            t_down, f_down = x.shape[2], x.shape[3]
            x = enc(x)

            encoder_skips.append(x)
            encoder_dims.append((t_down, f_down))

        # Bottleneck
        T_down, F_down = x.shape[2], x.shape[3]
        x = self.bottleneck(x, T_down, F_down)
        
        # Decoder with skip connections
        for dec, en , (t_req, f_req) in zip(self.decoders, reversed(encoder_skips), reversed(encoder_dims)):
            x = dec(x, en)
            x = Func.interpolate(x, size=(t_req,f_req), mode='bilinear')

        # Final output
        return self.final(x)
