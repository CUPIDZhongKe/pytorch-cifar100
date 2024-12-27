import torch
import torch.nn as nn

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y



class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y
    

class Residual_Attention_Block(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(Residual_Attention_Block, self).__init__()
        kernel_size = 3
        stride = 1
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        self.res = nn.Sequential(*layers)
        self.ca = CALayer(in_channels)
        self.pa = PALayer(in_channels)

    def forward(self, x):
        x1 = x + self.res(x)

        x2 = x1 + self.res(x1)

        x3 = x2 + self.res(x2)

        x3_1 = x1 + x3

        x4 = self.ca(x3_1)  
        x5 = self.pa(x4)

        return x5 + x


class CPRFNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=64):
        super(CPRFNet, self).__init__()
        self.dim = dim
        kernel_size = 3

        def conv(in_channels, out_channels, kernel_size, bias=True):
            return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
        
        pre_process = [conv(3, self.dim, kernel_size, True)]

        self.g1 = Residual_Attention_Block(self.dim, self.dim)
        self.g2 = Residual_Attention_Block(self.dim, self.dim)
        self.g3 = Residual_Attention_Block(self.dim, self.dim)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * 3 * 2, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * 3 * 2, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.palayer = PALayer(self.dim)

        post_process = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)
        ]

        self.post = nn.Sequential(*post_process)
        self.pre = nn.Sequential(*pre_process)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        y1 = self.pre(x1)
        res1 = self.g1(y1)
        res2 = self.g2(res1)
        res3 = self.g3(res2)

        y2= self.pre(x2)
        res4 = self.g1(y2)
        res5 = self.g2(res4)
        res6 = self.g3(res5)

        w = self.ca(torch.cat([res1, res2, res3, res4, res5, res6], dim=1))
        w = w.view(-1, 3 * 2, self.dim)[:, :, :, None, None]
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3 + w[:, 3, ::] * res4 + w[:, 4, ::] * res5 + w[:, 5, ::] * res6
        out = self.palayer(out)
        x = self.post(out)
        return x


if __name__ == "__main__":

    x = torch.randn(1, 3, 32, 32) # B C H W

    y = torch.randn(1, 3, 32, 32) # B C H W

    model = CPRFNet()

    output = model([x, y])

    print(x.size())
    print(output.size())