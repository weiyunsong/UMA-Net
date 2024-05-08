from einops.einops import rearrange

#把输入tensor转成三维
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c' )

#把输入tensor转成四维
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

    #若是通道之间的位置变换使用一下代码
    x = x.permute(0, 2, 3, 1)    #[b, c, h, w] -->[b, h, w, c]
    x = x.permute(0, 3, 1, 2)   #[b, h, w, c] -->[b, c, h, w]

#常见tensor的输入形式： B C H W  ,  B N C  ,  B H W C

import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward

class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True),
                                    )
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)

        return x


