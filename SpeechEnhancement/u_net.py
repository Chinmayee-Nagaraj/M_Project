import torch
import torch.nn as nn

from dense_block import DenseBlock
from bottleneck import BottleNeckBlock


class EncoderBlock(nn.Module):
    """Encoder: DenseBlock → Conv3x3 → PReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dense = DenseBlock(in_channels)                
        self.conv = nn.Conv2d(self.dense.out_channels, out_channels, 
                              kernel_size=3, stride=2, padding=1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.dense(x)
        x = self.prelu(self.conv(x)) 
        return x


class DecoderBlock(nn.Module):
    """Decoder: DenseBlock → ConvTranspose3x3 → PReLU"""
    def __init__(self, in_channels, en_channels, out_channels):
        super().__init__()
        total_in = in_channels + en_channels
        self.dense = DenseBlock(total_in)
        self.conv = nn.ConvTranspose2d(self.dense.out_channels, out_channels,
                                       kernel_size=3, stride=2, padding=1, output_padding=(1,0))
        self.prelu = nn.PReLU()
    
    def forward(self, x, en):
        x = torch.cat([x, en], dim=1)
        x = self.dense(x)
        x = self.prelu(self.conv(x)) 
        return x


class TFDense_Net(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, num_layers=3,
                 hidden_dim=64, num_heads=4, num_blocks=4, gn_groups=8):
        super().__init__()

        conv1x1 = nn.Conv2d(in_channels=1, out_channels=base_channels, kernel_size=1)
                     
        # Encoder
        self.encoders = nn.ModuleList()
        channels = base_channels

        for i in range(num_layers):
            enc = EncoderBlock(channels, channels)
            self.encoders.append(enc)

        # Bottleneck
        self.bottleneck = BottleNeckBlock(
            in_channels=channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            gn_groups=gn_groups
        )        

        # Decoder
        self.decoders = nn.ModuleList()
        for i in reversed(range(num_layers)):
            dec = DecoderBlock(channels, channels, channels)
            self.decoders.append(dec)
                     
        # Final output
        self.final = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x_proj = conv1x1(x)
        
        B, C, T, F = x.shape
        
        # Encoder
        encoder_ops = []
        for enc in self.encoders:
            x = enc(x)
            encoder_ops.append(skip)

        # Use the actual downsampled dims
        T_down, F_down = x.shape[2], x.shape[3]
        
        # Bottleneck
        x = self.bottleneck(x, T_down, F_down)
        
        # Decoder
        for dec, en in zip(self.decoders, reversed(encoder_ops)):
            x = dec(x, en)

        return self.final(x)
