import torch
import torch.nn as nn
from denseBlock import DenseBlock
from bottleneck import TFT

class EncoderBlock(nn.Module):
    """Encoder: DenseBlock → Conv3x3 → PReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dense = DenseBlock(in_channels, growth_rate=out_channels)                
        self.conv = nn.Conv2d(self.dense.out_channels, out_channels, 
                              kernel_size=3, stride=2, padding=1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.dense(x)
        skip = x                     # save BEFORE downsampling
        x = self.prelu(self.conv(x)) # downsampled path
        return x, skip


class DecoderBlock(nn.Module):
    """Decoder: DenseBlock → ConvTranspose3x3 → PReLU"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        total_in = in_channels + skip_channels
        self.dense = DenseBlock(total_in, growth_rate=out_channels)

        self.conv = nn.ConvTranspose2d(self.dense.out_channels, out_channels,kernel_size=3, stride=2, padding=1, output_padding=1)
        self.prelu = nn.PReLU()
    
    def forward(self, x, skip):
        # align spatial sizes if off by 1 pixel
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.dense(x)
        x = self.conv(x)
        x = self.prelu(x)
        return x

# -----------------------
# Full U-Net with TFT bottleneck
# -----------------------
class UNetWithTFT(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, num_layers=3,
                 hidden_dim=128, num_heads=4, num_blocks=2, gn_groups=8):
        super().__init__()
        # Encoder
        self.encoders = nn.ModuleList()
        channels = in_channels

        
        for i in range(num_layers):
            enc = EncoderBlock(channels, base_channels * (2 ** i))
            self.encoders.append(enc)
            channels = base_channels * (2 ** i) 

        # Bottleneck = TFT
        self.bottleneck = TFT(
            in_channels=channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            gn_groups=gn_groups,
            out_channels=channels  # keep same for smooth decoder input
        )        

        # Decoder
        self.decoders = nn.ModuleList()
        for i in reversed(range(num_layers)):
            skip_channels = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** i)
            dec = DecoderBlock(channels, skip_channels, out_ch)
            self.decoders.append(dec)
            channels = out_ch   # instead of formula
                     
        # Final output
        self.final = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x, T=None, F=None):
        B, C, T, F = x.shape
        
        # Encoder
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        # Use the actual downsampled dims
        T_down, F_down = x.shape[2], x.shape[3]
        
        # Bottleneck
        x = self.bottleneck(x, T_down, F_down)
        
        # Decoder
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.final(x)
