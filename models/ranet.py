import torch
import torch.nn as nn
import torch.nn.functional as F


class Up(nn.Module):

    def __init__(self, nc, bias):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=nc, out_channels=nc, kernel_size=2, stride=2, bias=bias)

    def forward(self, x1, x):
        x2 = self.up(x1)

        diffY = x.size()[2] - x2.size()[2]
        diffX = x.size()[3] - x2.size()[3]
        x3 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x3


## Spatial Attention
class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding=0, bias=False):
        super(Basic, self).__init__()
        self.out_channels = out_planes
        groups = 1
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def __init__(self):
        super(ChannelPool, self).__init__()

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SAB(nn.Module):
    def __init__(self):
        super(SAB, self).__init__()
        kernel_size = 5
        self.compress = ChannelPool()
        self.spatial = Basic(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


## Channel Attention Layer
class CAB(nn.Module):
    def __init__(self, nc, reduction=8, bias=False):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(nc, nc // reduction, kernel_size=1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(nc // reduction, nc, kernel_size=1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RAB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(RAB, self).__init__()
        kernel_size = 3
        stride = 1
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        self.res = nn.Sequential(*layers)
        self.sab = SAB()

    def forward(self, x):
        x1 = x + self.res(x)
        x2 = x1 + self.res(x1)
        x3 = x2 + self.res(x2)

        x3_1 = x1 + x3
        x4 = x3_1 + self.res(x3_1)
        x4_1 = x + x4

        x5 = self.sab(x4_1)
        x5_1 = x + x5

        return x5_1


class RANet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=128, num_classes=100, bias=True):
        super(RANet, self).__init__()
        kernel_size = 3

        self.conv_head = nn.Conv2d(in_nc, nc, kernel_size=kernel_size, padding=1, bias=bias)

        self.rab = RAB(nc, nc, bias)

        self.conv_tail = nn.Conv2d(nc, out_nc, kernel_size=kernel_size, padding=1, bias=bias)

        self.down = nn.Conv2d(nc, nc, kernel_size=2, stride=2, bias=bias)

        self.up = Up(nc, bias)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_nc, num_classes)

    def forward(self, x):
        x1 = self.conv_head(x)
        x2 = self.rab(x1)
        x2_1 = self.down(x2)
        x3 = self.rab(x2_1)
        x3_1 = self.down(x3)
        x4 = self.rab(x3_1)
        x4_1 = self.up(x4, x3)
        x5 = self.rab(x4_1 + x3)
        x5_1 = self.up(x5, x2)
        x6 = self.rab(x5_1 + x2)
        x7 = self.conv_tail(x6 + x1)
        X = x - x7
        x8 = self.avg_pool(X)
        x9 = x8.view(x8.size(0), -1)
        x10 = self.fc(x9)

        return x10
    
if __name__ == "__main__":

    x = torch.randn(1, 3, 32, 32) # B C H W

    model = RANet()

    output = model(x)

    print(x.size())
    print(output.size())