import warnings

import torch
import torch.nn as nn
from typing import Tuple


# Suppress just the UserWarning from F.conv2d about padding
warnings.filterwarnings(
    "ignore",
    message="Using padding='same' with even kernel lengths and odd dilation"
)


class ConvNetBlock(nn.Module):
    """
    A convolutional network block that uses:
    - Layer normalization across channels
    - Pointwise (1x1) and depthwise convolutions
    - Gated Linear Unit (GLU) activation
    - Two residual connections for stability and feature reuse
    
    Input and Output shape : (B, C_in, T, F)
    """
    def __init__(self, in_channels: int, kernel_size: Tuple[int, int]=(3,3)):
        """
        Args:
            in_channels (int): Number of input channels (C_in)
            kernel_size (tuple): Kernel size for depthwise convolutions
        """
        super().__init__()

        c = in_channels
        
        # Layer norm over channel dimension
        self.layer_norm = nn.LayerNorm(c)

        # Pointwise convolution   
        self.pw_conv1 = nn.Conv2d(c, 2*c, kernel_size=1)

        # Depthwise convolution 
        self.dw_conv1 = nn.Conv2d(2*c, 2*c, kernel_size=kernel_size, 
                                  padding='same', groups=2*c)
        
        # GLU activation splits channel dim into 2 halves -> output = half channels
        self.glu = nn.GLU(dim=1)

        self.pw_conv2 = nn.Conv2d(c, c, kernel_size=1)
        
        self.dw_conv2 = nn.Conv2d(c, c, kernel_size=kernel_size, 
                                  padding='same', groups=c)
        self.pw_conv3 = nn.Conv2d(c, c, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the block.
        Args:
            x: Tensor of shape (B, C_in, T, F)
        Returns:
            Tensor of shape (B, C_in, T, F)
        """
        residual1 = x
        
        # Layer norm over channels requires permuting to B,T,F,C
        x = x.permute(0, 2, 3, 1)  # B,T,F,C
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)  # B,C,T,F
        
        x = self.pw_conv1(x)
        x = self.dw_conv1(x)
        x = self.glu(x)
        x = self.pw_conv2(x)
        
        # Residual connection 1 (align channels)
        residual2 = x + residual1
        x = self.dw_conv2(residual2)
        x = self.pw_conv3(x)

        # Residual connection 2 (align channels)
        x = x + residual2
        return x


class DenseBlock(nn.Module):
    """
    Improved DenseBlock for spectrogram-like inputs.
    
    Structure:
    - Alternating 2x3 convolution (dilated) and ConvNetBlock
    - Dense concatenation of features (channel-wise)
    - Progressive channel growth in concatenated inputs
    - Output after final dilated conv has c channels
    
    Input shape : (B, C, T, F)
    Output shape: (B, C, T, F)
    """

    def __init__(self, in_channels: int, growth_rate: int):
        super().__init__()
        in_c = in_channels
        g = growth_rate

        # First 2x3 convolution
        self.conv1 = nn.Conv2d(in_c, g, kernel_size=(2, 3), padding='same')

        # ConvNetBlock after first concat 
        self.convnet1 = ConvNetBlock(in_channels=in_c + g)

        # Dilated conv (dilation=1)
        self.dilconv1 = nn.Conv2d(in_c + g, g, kernel_size=(2, 3), padding='same', dilation=1)

        # ConvNetBlock after second concat
        self.convnet2 = ConvNetBlock(in_channels=in_c + 2*g)

        # Dilated conv (dilation=3)
        self.dilconv2 = nn.Conv2d(in_c + 2*g, g, kernel_size=(2, 3), padding='same', dilation=3)

        # ConvNetBlock after third concat 
        self.convnet3 = ConvNetBlock(in_channels=in_c + 3*g)

        # Final dilated conv (dilation=5)
        self.dilconv3 = nn.Conv2d(in_c + 3*g, g, kernel_size=(2, 3), padding='same', dilation=5)

        self.out_channels = g

    def forward(self, x):
        """
        Forward pass of DenseBlock.
        Args:
            x: Tensor of shape (B, C, T, F)
        Returns:
            Tensor of shape (B, G, T, F)
        """

        # First 2x3 convolution
        out1 = self.conv1(x)                        # (B, G, T, F)
        cat1 = torch.cat([x, out1], dim=1)          # (B, C+G, T, F)
        inter2 = self.convnet1(cat1)                # (B, C+G, T, F)
        
        # Dilated conv1 (dilation=1) and concat
        out2 = self.dilconv1(inter2)                # (B, G, T, F)
        cat2 = torch.cat([x, out1, out2], dim=1)    # (B, C+2G, T, F)
        inter3 = self.convnet2(cat2)                # (B, C+2G, T, F)

        # Dilated conv2 (dilation=3) and concat
        out3 = self.dilconv2(inter3)                        # (B, G, T, F)
        cat3 = torch.cat([x, out1, out2, out3], dim=1)      # (B, C+3G, T, F)
        inter4 = self.convnet3(cat3)                        # (B, C+3G, T, F)

        # Final dilated conv (dilation=5)
        out_final = self.dilconv3(inter4)                   # (B, G, T, F)
        return out_final

