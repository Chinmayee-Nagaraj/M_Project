import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, in_channels, growth_rate=64):
        super().__init__()
        c = growth_rate

        self.conv1 = DilatedConv(in_channels, c, (2,3), dilation=1)
        self.proj1 = nn.Conv2d(in_channels + c, 2*c, kernel_size=1)

        self.conv2 = DilatedConv(2*c, c, (2,3), dilation=1)
        self.proj2 = nn.Conv2d(2*c + c, 3*c, kernel_size=1)
        self.cn2   = ConvNet(2*c)

        self.conv3 = DilatedConv(3*c, c, (2,3), dilation=3)
        self.proj3 = nn.Conv2d(3*c + c, 4*c, kernel_size=1)
        self.cn3   = ConvNet(3*c)

        self.conv4 = DilatedConv(4*c, c, (2,3), dilation=5)
        self.cn4   = ConvNet(4*c)

        # Final output is compressed to c
        self.out_channels = c    

    def forward(self, x):
        # x: (B,C,T,F)
        out1 = self.conv1(x)                      # (B,c,T,F)
        cat1 = torch.cat([x, out1], dim=1)        # (B,1+c,T,F)
        cat1 = self.proj1(cat1)                   # → (B,2c,T,F)
        out2 = self.cn2(cat1)                     # (B,2c,T,F)
        
        out3_in = self.conv2(out2)                # (B,c,T,F)
        cat2 = torch.cat([out2, out3_in], dim=1)  # (B,2c+c,T,F)
        cat2 = self.proj2(cat2)                   # → (B,3c,T,F)
        out3 = self.cn3(cat2)                     # (B,3c,T,F)

        out4_in = self.conv3(out3)                # (B,c,T,F)
        cat3 = torch.cat([out3, out4_in], dim=1)  # (B,3c+c,T,F)
        cat3 = self.proj3(cat3)                   # → (B,4c,T,F)
        out4 = self.cn4(cat3)                     # (B,4c,T,F)

        out_final = self.conv4(out4)              # (B,c,T,F)
        return out_final
