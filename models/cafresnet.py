import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from timm.models.layers import DropPath
import antialiased_cnns
import torchinfo

class ConvBlock_down(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, kernel_size=3, stride=1, padding=1, if_down=True):
        super(ConvBlock_down, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hid_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(hid_channels, out_channels, kernel_size, stride, padding)

        self.bn1 = nn.BatchNorm2d(hid_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.GELU()

        self.if_down = if_down
        self.down = nn.MaxPool2d(kernel_size=2, stride=1)
        self.down_anti = antialiased_cnns.BlurPool(in_channels, stride=2)

    def forward(self, x):
        if self.if_down:
            x = self.down(x)
            x = self.down_anti(x)
            x = self.act(x)

        x1 = self.conv1(x)
        x2 = self.bn1(x1)

        x3 = self.conv2(x2)
        x3 = self.bn2(x3)
        out = self.act(x3)

        return out

class encoder_convblock(nn.Module):
    def __init__(self):
        super(encoder_convblock, self).__init__()
        self.inlayer = nn.Conv2d(3, 64, 1)
        self.block1 = ConvBlock_down(64, 32, 112, if_down=False)
        self.block2 = ConvBlock_down(112, 56, 160)
        self.block3 = ConvBlock_down(160, 80, 208)
        self.block4 = ConvBlock_down(208, 104, 256)

    def forward(self, img):
        img = self.inlayer(img)
        x1 = self.block1(img)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        return x1, x2, x3, x4

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, patch_size=4, in_c=1, embed_dim=256, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class DePatch(nn.Module):
    def __init__(self, channel, embed_dim=256, patch_size=4):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, patch_size ** 2 * channel),
        )

    def forward(self, x, ori):
        b, c, h, w = ori  # [1, 32, 56, 56]
        h_ = h // self.patch_size
        w_ = w // self.patch_size
        x = self.projection(x)  # [1, 196, 16]
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h_, w=w_, p1=self.patch_size, p2=self.patch_size)
        return x

class Block(nn.Module):
    # Transformer Encoder Block
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # ==Dropout
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.norm1(x)
        attn, q, k, v = self.attn(x)

        # x = x + self.drop_path(attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, q, k, v
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # q,k,v分别对应3个通道
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # batch_size, num_patch+1, total_embed_dim
        B, N, C = x.shape

        # [batch_size, num_patch+1, 3 * total_embed_dim] -> [3, batch_size, num_head, num_patch+1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        # [batch_size, num_head, num_patch+1, num_patch+1]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # @矩阵乘法
        attn = attn.softmax(dim=-1)  # 对每一行进行softmax操作
        attn = self.attn_drop(attn)

        # [batch_size, num_patch+1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, q, k, v

class TransCAF(nn.Module):
    def __init__(self, patch_size, dim, num_heads, channel, proj_drop, depth, qk_scale, attn_drop):
        super(TransCAF, self).__init__()

        self.patchembed1 = PatchEmbed(patch_size=patch_size, in_c=channel, embed_dim=dim)
        self.patchembed2 = PatchEmbed(patch_size=patch_size, in_c=channel, embed_dim=dim)

        self.TransformerEncoderBlocks1 = nn.Sequential(*[
            TransformerEncoderBlock(dim=dim, num_heads=num_heads)
            for i in range(depth)
        ])
        self.TransformerEncoderBlocks2 = nn.Sequential(*[
            TransformerEncoderBlock(dim=dim, num_heads=num_heads)
            for i in range(depth)
        ])

        self.QKV_Block1 = Block(dim=dim, num_heads=num_heads)
        self.QKV_Block2 = Block(dim=dim, num_heads=num_heads)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj_drop1 = nn.Dropout(proj_drop)
        self.proj_drop2 = nn.Dropout(proj_drop)

        self.depatch = DePatch(channel=channel, embed_dim=dim, patch_size=patch_size)

    def forward(self, in_1, in_2):
        # Patch Embeding1
        in_emb1 = self.patchembed1(in_1)
        B, N, C = in_emb1.shape

        # Transformer Encoder1
        in_emb1 = self.TransformerEncoderBlocks1(in_emb1)

        # cross self-attention Feature Extraction
        _, q1, k1, v1 = self.QKV_Block1(in_emb1)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        # Patch Embeding2
        in_emb2 = self.patchembed2(in_2)

        # Transformer Encoder2
        in_emb2 = self.TransformerEncoderBlocks2(in_emb2)

        _, q2, k2, v2 = self.QKV_Block2(in_emb2)

        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)

        # cross attention
        x_attn1 = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)
        x_attn1 = self.proj1(x_attn1)
        x_attn1 = self.proj_drop1(x_attn1)

        x_attn2 = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)
        x_attn2 = self.proj2(x_attn2)
        x_attn2 = self.proj_drop2(x_attn2)

        x_attn = (x_attn1 + x_attn2) / 2

        print("x_attn: ", x_attn.shape)

        # Patch Rearrange
        ori = in_2.shape  # b,c,h,w
        out1 = self.depatch(x_attn, ori)

        # out = in_1 + in_2 + out1

        return out1
    
class CAF(nn.Module):
    def __init__(self, patch_size, dim, num_heads, channel, proj_drop, depth, qk_scale, attn_drop):
        super(CAF, self).__init__()

        self.patchembed1 = PatchEmbed(patch_size=patch_size, in_c=channel, embed_dim=dim)
        self.patchembed2 = PatchEmbed(patch_size=patch_size, in_c=channel, embed_dim=dim)

        self.TransformerEncoderBlocks1 = nn.Sequential(*[
            TransformerEncoderBlock(dim=dim, num_heads=num_heads)
            for i in range(depth)
        ])
        self.TransformerEncoderBlocks2 = nn.Sequential(*[
            TransformerEncoderBlock(dim=dim, num_heads=num_heads)
            for i in range(depth)
        ])

        self.QKV_Block1 = Block(dim=dim, num_heads=num_heads)
        self.QKV_Block2 = Block(dim=dim, num_heads=num_heads)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj_drop1 = nn.Dropout(proj_drop)
        self.proj_drop2 = nn.Dropout(proj_drop)

        self.depatch = DePatch(channel=channel, embed_dim=dim, patch_size=patch_size)

    def forward(self, in_1, in_2):
        # pure cross attention fusion
        # 将特征图展平并调整维度顺序
        batch_size, channels, height, width = in_1.shape
        num_patches = height * width
        embedding_dim = channels

        # 重新排列为 [batch_size, num_patches, embedding_dim]
        reshaped_1 = rearrange(in_1, 'b c h w -> b (h w) c')
        reshaped_2 = rearrange(in_2, 'b c h w -> b (h w) c')

        _, q1, k1, v1 = self.QKV_Block1(reshaped_1)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        _, q2, k2, v2 = self.QKV_Block2(reshaped_2)

        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)

        x_attn1 = (attn1 @ v2).transpose(1, 2).reshape(batch_size, num_patches, embedding_dim)
        x_attn1 = self.proj1(x_attn1)
        x_attn1 = self.proj_drop1(x_attn1)

        x_attn2 = (attn2 @ v1).transpose(1, 2).reshape(batch_size, num_patches, embedding_dim)
        x_attn2 = self.proj2(x_attn2)
        x_attn2 = self.proj_drop2(x_attn2)

        x_attn = (x_attn1 + x_attn2) / 2

        # 恢复原始形状 [batch_size, channels, height, width]
        out1 = rearrange(x_attn, 'b (h w) c -> b c h w', h=height, w=width)

        # out = in_1 + in_2 + out1

        return out1


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # ==Dropout
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x1 = self.norm1(x)
        attn_list = self.attn(x1)  # x,q,k,v
        attn = attn_list[0]
        x1 = self.drop_path(attn)
        x = x + x1

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class sff(nn.Module):
    def __init__(self, patch_size=16, dim=256, num_heads=8, channels=256, fusionblock_depth=3,
                 qk_scale=None, attn_drop=0., proj_drop=0., img_size=256):
        super(sff, self).__init__()

        # Fusion Block
        self.FusionBlock1 = TransCAF(patch_size=patch_size, dim=dim, num_heads=num_heads, channel=channels,
                                proj_drop=proj_drop, depth=fusionblock_depth, qk_scale=qk_scale,
                                attn_drop=attn_drop)
        self.FusionBlock2 = TransCAF(patch_size=patch_size, dim=dim, num_heads=num_heads, channel=channels,
                                proj_drop=proj_drop, depth=fusionblock_depth, qk_scale=qk_scale,
                                attn_drop=attn_drop)
        self.FusionBlock_final = TransCAF(patch_size=patch_size, dim=dim, num_heads=num_heads,
                                     channel=channels,
                                     proj_drop=proj_drop, depth=fusionblock_depth, qk_scale=qk_scale,
                                     attn_drop=attn_drop)

        self.conv1x1 = nn.Conv2d(1, 1, 1)

    def forward(self, img1, img2):
        x = img1
        y = img2

        y_f = torch.fft.fft2(y)  # Fourier Transform
        y_f = torch.fft.fftshift(y_f)
        y_f = torch.log(1 + torch.abs(y_f))

        x_f = torch.fft.fft2(x)
        x_f = torch.fft.fftshift(x_f)
        x_f = torch.log(1 + torch.abs(x_f))

        feature_y = self.FusionBlock1(x_f, y_f)
        feature_x = self.FusionBlock2(x, y)

        feature_y = torch.fft.ifftshift(feature_y)
        feature_y = torch.fft.ifft2(feature_y)
        feature_y = torch.abs(feature_y)

        z = self.FusionBlock_final(feature_x, feature_y)

        return z

class adafuse(nn.Module):
    def __init__(self, patch_size=16, dim=256, num_heads=8, channels=[112, 160, 208, 256],
                 fusionblock_depth=[4, 4, 4, 4], qk_scale=None, attn_drop=0., proj_drop=0.):
        super(adafuse, self).__init__()

        self.encoder = encoder_convblock()

        # self.conv_up4 = ConvBlock_up(256, 104, 208)
        # self.conv_up3 = ConvBlock_up(208 * 2, 80, 160)
        # self.conv_up2 = ConvBlock_up(160 * 2, 56, 112)
        # self.conv_up1 = ConvBlock_up(112 * 2, 8, 16, if_up=False)

        # Fusion Block
        self.fusionnet1 = sff(patch_size=patch_size, dim=dim, num_heads=num_heads,
                              channels=channels[0],
                              fusionblock_depth=fusionblock_depth[0],
                              qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=proj_drop)
        self.fusionnet2 = sff(patch_size=patch_size, dim=dim, num_heads=num_heads,
                              channels=channels[1],
                              fusionblock_depth=fusionblock_depth[1],
                              qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=proj_drop)
        self.fusionnet3 = sff(patch_size=patch_size, dim=dim, num_heads=num_heads,
                              channels=channels[2],
                              fusionblock_depth=fusionblock_depth[2],
                              qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=proj_drop)
        self.fusionnet4 = sff(patch_size=patch_size, dim=dim, num_heads=num_heads,
                              channels=channels[3],
                              fusionblock_depth=fusionblock_depth[3],
                              qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=proj_drop)

        # Conv 1x1
        self.outlayer = nn.Conv2d(16, 1, 1)

    def forward(self, img1, img2):
        x1, x2, x3, x4 = self.encoder(img1)
        y1, y2, y3, y4 = self.encoder(img2)
        print(x1.shape, y1.shape)
        print(x2.shape, y2.shape)
        print(x3.shape, y3.shape)
        print(x4.shape, y4.shape)

        z1 = self.fusionnet1(x1, y1)
        z2 = self.fusionnet2(x2, y2)
        z3 = self.fusionnet3(x3, y3)
        z4 = self.fusionnet4(x4, y4)

        # out4 = self.conv_up4(z4)
        # out3 = self.conv_up3(torch.cat((out4, z3), dim=1))
        # out2 = self.conv_up2(torch.cat((out3, z2), dim=1))
        # out1 = self.conv_up1(torch.cat((out2, z1), dim=1))

        # img_fusion = self.outlayer(out1)

        return z1, z2, z3, z4


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

class ResNet_SFF(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.sff = sff(patch_size=16, dim=256, num_heads=8, channels=512, fusionblock_depth=3,
                 qk_scale=None, attn_drop=0., proj_drop=0., img_size=256)

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

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2_x(x1)
        x3 = self.conv3_x(x2)
        x4 = self.conv4_x(x3)
        x5 = self.conv5_x(x4)

        y1 = self.conv1(y)
        y2 = self.conv2_x(y1)
        y3 = self.conv3_x(y2)
        y4 = self.conv4_x(y3)
        y5 = self.conv5_x(y4)

        output = self.sff(x5, y5)

        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


class ResNet_CAF(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.caf = CAF(patch_size=16, dim=512, num_heads=8, channel=512, depth=3,
                 qk_scale=None, attn_drop=0., proj_drop=0.)

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

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2_x(x1)
        x3 = self.conv3_x(x2)
        x4 = self.conv4_x(x3)
        x5 = self.conv5_x(x4)

        y1 = self.conv1(y)
        y2 = self.conv2_x(y1)
        y3 = self.conv3_x(y2)
        y4 = self.conv4_x(y3)
        y5 = self.conv5_x(y4)

        output = self.caf(x5, y5)

        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def resnet18_sff():
    """ return a ResNet 18 object
    """
    return ResNet_SFF(BasicBlock, [2, 2, 2, 2])

def resnet18_caf():
    """ return a ResNet 18 object
    """
    return ResNet_CAF(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])


if __name__ == "__main__":
    img1 = torch.randn(32, 3, 256, 256)
    img2 = torch.randn(32, 3, 256, 256)

    net = resnet18_sff()

    torchinfo.summary(net, input_data=(img1, img2))

    # z = net(img1, img2)

    # print(z.shape)



