import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, channels, kernel_size=(3,3), dilation=1):
        super().__init__()
        padding = (
            (kernel_size[0] - 1) // 2 * dilation,
            (kernel_size[1] - 1) // 2 * dilation
        )

        self.ln = nn.LayerNorm(channels)

        self.pw1 = nn.Conv2d(channels, channels, kernel_size=1)

        self.dw1 = nn.Conv2d(
            channels, channels, kernel_size, padding=padding,
            dilation=dilation, groups=channels
        )

        self.glu_fc = nn.Conv2d(channels, channels * 2, kernel_size=1)

        self.pw2 = nn.Conv2d(channels, channels, kernel_size=1)

        self.dw2 = nn.Conv2d(
            channels, channels, kernel_size, padding=padding,
            dilation=dilation, groups=channels
        )
        self.pw3 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        #B, C, T, F = x.shape
        # LayerNorm -> (B,T,F,C)
        y = x.permute(0, 2, 3, 1)
        y = self.ln(y)
        y = y.permute(0, 3, 1, 2)

        y = self.pw1(y)
        y = self.dw1(y)

        y = self.glu_fc(y)
        y = F.glu(y, dim =1)

        y = self.pw2(y)
        y = y + x  # residual 1

        z = self.dw2(y)
        z = self.pw3(z)
        out = y + z  # residual 2
        return out


class DilatedConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(2,3), dilation=1):
        super().__init__()
        kH, kW = kernel
        d = dilation

        # compute asymmetric padding
        pad_h = d * (kH - 1)
        pad_w = d * (kW - 1)

        # split into top/bottom, left/right
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        self.pad = nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom))
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=0, dilation=d)

    def forward(self, x):
        return self.conv(self.pad(x))


class DenseBlock(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.c = c

        # convs (use same padding to keep T,F unchanged)
        self.conv1 = DilatedConv(1, c, (2,3), dilation=1)
        self.conv2 = DilatedConv(2*c, c, (2,3), dilation=1)
        self.conv3 = DilatedConv(3*c, c, (2,3), dilation=3)
        self.conv4 = DilatedConv(4*c, c, (2,3), dilation=5)
                
        # projections so concat always matches ConvNet input channels
        self.proj1 = nn.Conv2d(1+c, 2*c, kernel_size=1)
        self.proj2 = nn.Conv2d(2*c+c, 3*c, kernel_size=1)
        self.proj3 = nn.Conv2d(3*c+c, 4*c, kernel_size=1)

        # convnets
        self.cn2 = ConvNet(2*c)
        self.cn3 = ConvNet(3*c)
        self.cn4 = ConvNet(4*c)

    def forward(self, x):
        # x: (B,F,T)
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2) 
        # x: (B,1,T,F)
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


