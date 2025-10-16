import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, in_channels, kernel_size=(3,3)):
        """
        Args:
            in_channels (int): Number of input channels (C_in)
            kernel_size (tuple): Kernel size for depthwise convolutions
        """
        super().__init__()
        
        # Layer norm over channel dimension
        self.layer_norm = nn.LayerNorm(in_channels)

        # Pointwise convolution   
        self.pw_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Depthwise convolution 
        self.dw_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                  padding='same', groups=in_channels)
        
        # GLU activation splits channel dim into 2 halves -> output = half channels
        self.glu = nn.GLU(dim=1)

        self.pw_conv2 = nn.Conv2d(in_channels//2, in_channels, kernel_size=1)
        
        self.dw_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                  padding='same', groups=in_channels)
        self.pw_conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

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

    def __init__(self, in_channels):
        super().__init__()
        c = in_channels

        # First 2x3 convolution
        self.conv1 = nn.Conv2d(in_channels, c, kernel_size=(2, 3), padding='same')

        # ConvNetBlock after first concat 
        self.convnet1 = ConvNetBlock(in_channels=2*c)

        # Dilated conv (dilation=1)
        self.dilconv1 = nn.Conv2d(2*c, c, kernel_size=(2, 3), padding='same', dilation=1)

        # ConvNetBlock after second concat
        self.convnet2 = ConvNetBlock(in_channels=3*c)

        # Dilated conv (dilation=3)
        self.dilconv2 = nn.Conv2d(3*c, c, kernel_size=(2, 3), padding='same', dilation=3)

        # ConvNetBlock after third concat 
        self.convnet3 = ConvNetBlock(in_channels=4*c)

        # Final dilated conv (dilation=5)
        self.dilconv3 = nn.Conv2d(4*c, c, kernel_size=(2, 3), padding='same', dilation=5)

        self.out_channels = c

    def forward(self, x):
        """
        Forward pass of DenseBlock.
        Args:
            x: Tensor of shape (B, C, T, F)
        Returns:
            Tensor of shape (B, C, T, F)
        """

        # First 2x3 convolution
        out1 = self.conv1(x)                        # (B, C, T, F)
        cat1 = torch.cat([x, out1], dim=1)          # (B, 2C, T, F)
        out2 = self.convnet1(cat1)                  # (B, 2C, T, F)
        
        # Dilated conv1 (dilation=1) and concat
        out3 = self.dilconv1(out2)                  # (B, C, T, F)
        cat2 = torch.cat([out2, out3], dim=1)       # (B, 3C, T, F)
        out3 = self.convnet2(cat2)                  # (B, 3C, T, F)

        # Dilated conv2 (dilation=3) and concat
        out4 = self.dilconv2(out3)                  # (B, C, T, F)
        cat3 = torch.cat([out3, out4], dim=1)       # (B, 4C, T, F)
        out4 = self.convnet3(cat3)                  # (B, 4C, T, F)

        # Final dilated conv (dilation=5)
        out_final = self.dilconv3(out4)             # (B, C, T, F)
        return out_final

