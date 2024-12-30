import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

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
    

class Residual_Attention_Block(nn.Module):

    expansion = 1   #the expansion of Residual Block

    def __init__(self, in_channels=64, out_channels=64, stride=1):
        super().__init__()

        #residual function
        self.res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * Residual_Attention_Block.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * Residual_Attention_Block.expansion),
            CALayer(out_channels),
            PALayer(out_channels)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != Residual_Attention_Block.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Residual_Attention_Block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Residual_Attention_Block.expansion)
            )
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.res(x) + self.shortcut(x))
    
# FPN构建
# fpn_list中包含以下特征维度,对应章节1.3中的图
# C2 [2, 64, 64, 64]
# C3 [2, 128, 32, 32]
# C4 [2, 256, 16, 16]
# C5 [2, 512, 8, 8]
class FPN(nn.Module):
    def __init__(self,in_channel_list,out_channel):
        super().__init__()
        self.inner_layer=[]  # 1x1卷积，统一通道数
        self.out_layer=[]  # 3x3卷积，对add后的特征图进一步融合
        for in_channel in in_channel_list:  
            self.inner_layer.append(nn.Conv2d(in_channel,out_channel,1))
            self.out_layer.append(nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1))
    def forward(self,x):
        head_output=[]  # 存放最终输出特征图
        corent_inner=self.inner_layer[-1](x[-1])  # 过1x1卷积，对C5统一通道数操作
        head_output.append(self.out_layer[-1](corent_inner)) # 过3x3卷积，对统一通道后过的特征进一步融合，加入head_output列表
        # print(self.out_layer[-1](corent_inner).shape)
        
        for i in range(len(x)-2,-1,-1):  # 通过for循环，对C4，C3，C2进行
            pre_inner=corent_inner
            corent_inner=self.inner_layer[i](x[i])  # 1x1卷积，统一通道数操作
            size=corent_inner.shape[2:]  # 获取上采样的大小（size）
            pre_top_down=F.interpolate(pre_inner,size=size)  # 上采样操作（这里大家去看一下interpolate这个上采样api）
            add_pre2corent=pre_top_down+corent_inner  # add操作
            head_output.append(self.out_layer[i](add_pre2corent))  # 3x3卷积，特征进一步融合操作，并加入head_output列表
            # print(self.out_layer[i](add_pre2corent).shape)
        
        return head_output

    
class RANet(nn.Module):
    def __init__(self, block, num_block, num_classes=100):
        super(RANet, self).__init__()
        
        self.in_channels = 64 

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.ca = CALayer(256)
        self.pa = PALayer(256)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        self.up = Up(256, True)

        self.fpn = FPN([64,128,256,512], 256)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        FPN_list = []
        print(x.size())
        x1 = self.conv1(x)
        print("layer1 ", x1.shape)
        x2 = self.conv2_x(x1)
        print("layer2 ", x2.shape)
        x3 = self.conv3_x(x2)
        print("layer3 ", x3.shape)
        x4 = self.conv4_x(x3)
        x5 = self.conv5_x(x4)

        FPN_list.append(x2)
        FPN_list.append(x3)
        FPN_list.append(x4)
        FPN_list.append(x5)

        fpn_output = self.fpn(FPN_list)

        # 上采样FPN输出
        # x2 = fpn_output[0]
        # x3_up = self.up(fpn_output[1], fpn_output[0])
        # x4_up = self.up(fpn_output[2], fpn_output[0])
        # x5_up = self.up(fpn_output[3], fpn_output[0])

        # output = self.ca(torch.cat([x2, x3_up, x4_up, x5_up], dim=1))
        # output = self.pa(output)

        # 使用FPN最底层特征图作为多尺度融合输出
        output = self.ca(fpn_output[3])
        output = self.pa(output)

        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output
    
def ranet18():
    """ return a RANet 18 object
    """
    return RANet(Residual_Attention_Block, [2, 2, 2, 2])

    
if __name__ == "__main__":

    x = torch.randn(1, 3, 64, 64) # B C H W

    model = ranet18()

    # summary(model, input_size=(1, 3, 32, 32))

    output = model(x)

    # print(x.size())
    # print(output.size())