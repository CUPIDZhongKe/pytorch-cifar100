import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from timm.models.layers import DropPath
import antialiased_cnns
import torchinfo
import matplotlib.pyplot as plt
import math

''' FCMNet: Frequency-aware cross-modality attention networks for RGB-D salient object detection
    https://github.com/XiaoJinNK/FCMNet.git
'''

def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i+0.5)/L) / math.sqrt(L)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)
def get_dct_weights(width,height,channel,fidx_u,fidx_v):
    dct_weights = torch.zeros(1, channel, width, height)
    c_part = channel // len(fidx_u)
    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for t_x in range(width):
            for t_y in range(height):
                dct_weights[:, i*c_part: (i+1)*c_part, t_x, t_y] = get_1d_dct(t_x, u_x, width) * get_1d_dct(t_y, v_y, height)
    return dct_weights

class FCABlock(nn.Module):
    """
        FcaNet: Frequency Channel Attention Networks
        https://arxiv.org/pdf/2012.11879.pdf
    """
    def __init__(self, channel,width,height,fidx_u, fidx_v, reduction=16):
        super(FCABlock, self).__init__()
        mid_channel = channel // reduction
        self.register_buffer('pre_computed_dct_weights', get_dct_weights(width,height,channel,fidx_u,fidx_v))
        self.excitation = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.sum(x * self.pre_computed_dct_weights, dim=[2,3])
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)
    
class SFCA(nn.Module):
    def __init__(self, in_channel,width,height,fidx_u,fidx_v):
        super(SFCA, self).__init__()

        fidx_u = [temp_u * (width // 8) for temp_u in fidx_u]
        fidx_v = [temp_v * (width // 8) for temp_v in fidx_v]
        self.FCA = FCABlock(in_channel, width, height, fidx_u, fidx_v)
        self.conv1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()
    def forward(self, x):
        # FCA
        F_fca = self.FCA(x)
        #context attention
        con = self.conv1(x) # c,h,w -> 1,h,w
        con = self.norm(con)
        F_con = x * con
        return F_fca + F_con
    
class FACMA(nn.Module):
    def __init__(self, in_channel, width, height, fidx_u, fidx_v):
        super(FACMA, self).__init__()
        self.sfca_depth = SFCA(in_channel, width, height, fidx_u, fidx_v)
        self.sfca_rgb   = SFCA(in_channel, width, height, fidx_u, fidx_v)
    def forward(self, rgb, depth):
        out_d = self.sfca_depth(depth)
        # out_d = rgb * out_d

        out_rgb = self.sfca_rgb(rgb)
        # out_rgb = depth * out_rgb
        return out_rgb, out_d


''' PlainUSR: Chasing Faster ConvNet for Efficient Super-Resolution
    github：https://github.com/icandle/PlainUSR
'''
class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None,padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size,stride,padding, count_include_pad=False)
    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool
    
class LocalAttention(nn.Module):
    ''' attention based on local importance'''
    def __init__(self, channels, f=16):
        super().__init__()
        self.body = nn.Sequential(
            # sample importance
            nn.Conv2d(channels, f, 1),
            SoftPooling2D(3, stride=1),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(f, channels, 3, padding=1),
            # to heatmap
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )            
    def forward(self, x):
        ''' forward '''
        # interpolate the heat map
        g = self.gate(x[:,:1].clone())
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return x * w * g #(w + g) #self.gate(x, w) 

''' PIDNet: A Real-time Semantic Segmentation Network Inspired from PID Controller
    https://github.com/XuJiacong/PIDNet.git
'''
class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=True, with_channel=True, BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        self.f_y = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        if with_channel:
            self.up = nn.Sequential(
                                    nn.Conv2d(mid_channels, in_channels, 
                                              kernel_size=1, bias=False),
                                    BatchNorm(in_channels)
                                   )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

        self.maxpool = LocalAttention(channels=mid_channels)

        self.edge_enhance_in = EdgeEnhancer(in_channels, nn.BatchNorm2d, None)
        self.edge_enhance_mid = EdgeEnhancer(mid_channels, nn.BatchNorm2d, None)

    def forward(self, x, y): 
        input_size = x.size()
        y = self.edge_enhance_in(y)
        x = self.edge_enhance_in(x)

        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        
        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x_k = self.f_x(x)

        y_q = self.maxpool(y_q)
        x_k = self.maxpool(x_k)

        if self.with_channel:
            sim_map = self.up(x_k * y_q)
            sim_map = self.edge_enhance_in(sim_map)
            sim_map = torch.sigmoid(sim_map)
        else:
            sim_map = torch.sum(x_k * y_q, dim=1).unsqueeze(1)
            sim_map = torch.sigmoid(sim_map)
        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        
        x_sim = x * (1 - sim_map)
        y_sim = y * sim_map

        z = x_sim + y_sim
        z = self.edge_enhance_in(z)

        # # 可视化
        # # 将 output 从计算图中分离出来，并转换为 NumPy 数组
        # output_x = sim_map.detach().cpu().numpy().squeeze()
        # output_y = (1 - sim_map).detach().cpu().numpy().squeeze()
        # x_sim = x_sim.detach().cpu().numpy().squeeze()
        # y_sim = y_sim.detach().cpu().numpy().squeeze()

        # # 显示原图和滤波后的图像
        # plt.figure(figsize=(10, 5)) 
        # plt.subplot(2, 4, 1)
        # plt.title("dis sim map")
        # plt.imshow(output_y, cmap='gray')

        # plt.subplot(2, 4, 2)
        # plt.title("sim map")
        # plt.imshow(output_x, cmap='gray')

        # plt.subplot(2, 4, 3)
        # plt.title("x Image sim")
        # plt.imshow(x_sim, cmap='gray')

        # plt.subplot(2, 4, 4)
        # plt.title("y Image sim")
        # plt.imshow(y_sim, cmap='gray')

        # # 将 output 从计算图中分离出来，并转换为 NumPy 数组
        # output_np = z.detach().cpu().numpy().squeeze()
        # x = x.detach().cpu().numpy().squeeze()
        # y = y.detach().cpu().numpy().squeeze()


        # # 显示原图和滤波后的图像
        # plt.subplot(2, 4, 5)
        # plt.title("Original vis Image")
        # plt.imshow(x, cmap='gray')

        # plt.subplot(2, 4, 6)
        # plt.title("Original trans Image")
        # plt.imshow(y, cmap='gray')

        # plt.subplot(2, 4, 7)
        # plt.title("Fusion Image")
        # plt.imshow(output_np, cmap='gray')

        # plt.show()

        return z

''' Multi-Scale and Detail-Enhanced Segment Anything Model for Salient Object Detection
    https://github.com/BellyBeauty/MDSAM.git
'''
class MEEM(nn.Module):
    def __init__(self, in_dim, hidden_dim, width, norm, act):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias = False),
            norm(hidden_dim),
            nn.Sigmoid()
        )

        self.pool = nn.AvgPool2d(3, stride= 1,padding = 1)

        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()
        for i in range(width - 1):
            self.mid_conv.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias = False),
                norm(hidden_dim),
                nn.Sigmoid()
            ))
            self.edge_enhance.append(EdgeEnhancer(hidden_dim, norm, act))

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * width, in_dim, 1, bias = False),
            norm(in_dim),
            act()
        )
    
    def forward(self, x):
        mid = self.in_conv(x)

        out = mid
        #print(out.shape)
        
        for i in range(self.width - 1):
            mid = self.pool(mid)
            mid = self.mid_conv[i](mid)

            out = torch.cat([out, self.edge_enhance[i](mid)], dim = 1)
        
        out = self.out_conv(out)

        return out

class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias = False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride= 1, padding = 1)
    
    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge
    
class DetailEnhancement(nn.Module):
    def __init__(self, img_dim, feature_dim, norm, act):
        super().__init__()
        self.img_in_conv = nn.Sequential(
            nn.Conv2d(3, img_dim, 3, padding = 1, bias = False),
            norm(img_dim),
            act()
        )
        self.img_er = MEEM(img_dim, img_dim  // 2, 4, norm, act)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim + img_dim, 32, 3, padding = 1, bias = False),
            norm(32),
            act(),
            nn.Conv2d(32, 16, 3, padding = 1, bias = False),
            norm(16),
            act(),
        )

        self.out_conv = nn.Conv2d(16, 1, 1)
        
        self.feature_upsample = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding = 1, bias = False),
            norm(feature_dim),
            act(),
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            nn.Conv2d(feature_dim, feature_dim, 3, padding = 1, bias = False),
            norm(feature_dim),
            act(),
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            nn.Conv2d(feature_dim, feature_dim, 3, padding = 1, bias = False),
            norm(feature_dim),
            act(),
        )
    
    def forward(self, img, feature, b_feature):

        feature = torch.cat([feature, b_feature], dim = 1)
        feature = self.feature_upsample(feature)

        img_feature = self.img_in_conv(img)
        img_feature = self.img_er(img_feature) + img_feature

        out_feature = torch.cat([feature, img_feature], dim = 1)
        out_feature = self.fusion_conv(out_feature)
        out = self.out_conv(out_feature)

        return out


''' Adaptive Medical Image Fusion Based on Spatial-Frequential Cross Attention (AdaFuse)
    https://github.com/xianming-gu/AdaFuse.git
'''
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

        # Patch Rearrange
        ori = in_2.shape  # b,c,h,w
        out1 = self.depatch(x_attn, ori)

        out = in_1 + in_2 + out1

        return out
    
class CAF(nn.Module):
    def __init__(self, patch_size, dim, num_heads, channel, proj_drop, depth, qk_scale, attn_drop):
        super(CAF, self).__init__()

        self.patchembed1 = PatchEmbed(patch_size=patch_size, in_c=channel, embed_dim=dim)

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
        # # 将特征图展平并调整维度顺序
        # batch_size, channels, height, width = in_1.shape
        # num_patches = height * width
        # embedding_dim = channels

        # # 重新排列为 [batch_size, num_patches, embedding_dim]
        # reshaped_1 = rearrange(in_1, 'b c h w -> b (h w) c')
        # reshaped_2 = rearrange(in_2, 'b c h w -> b (h w) c')


        # Patch Embeding1
        in_emb1 = self.patchembed1(in_1)
        in_emb2 = self.patchembed1(in_2)
        B, N, C = in_emb1.shape
        input1 = in_emb1
        _, q1, k1, v1 = self.QKV_Block1(in_emb1)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        _, q2, k2, v2 = self.QKV_Block2(in_emb2)

        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)

        x_attn1 = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)
        x_attn1 = self.proj1(x_attn1)
        x_attn1 = self.proj_drop1(x_attn1)

        x_attn2 = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)
        x_attn2 = self.proj2(x_attn2)
        x_attn2 = self.proj_drop2(x_attn2)

        x_attn = (x_attn1 + x_attn2) / 2

        # # 恢复原始形状 [batch_size, channels, height, width]
        # out1 = rearrange(x_attn, 'b (h w) c -> b c h w', h=height, w=width)

        # Patch Rearrange
        ori = in_2.shape  # b,c,h,w
        out1 = self.depatch(x_attn, ori)

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

class TransSFF(nn.Module):
    def __init__(self, patch_size=16, dim=256, num_heads=8, channels=512, fusionblock_depth=3,
                 qk_scale=None, attn_drop=0., proj_drop=0., img_size=256):
        super(TransSFF, self).__init__()

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
    
class SFF(nn.Module):
    def __init__(self, patch_size=16, dim=256, num_heads=8, channels=256, fusionblock_depth=3,
                 qk_scale=None, attn_drop=0., proj_drop=0., img_size=256):
        super(SFF, self).__init__()

        # Fusion Block
        self.FusionBlock1 = CAF(patch_size=patch_size, dim=dim, num_heads=num_heads, channel=channels,
                                proj_drop=proj_drop, depth=fusionblock_depth, qk_scale=qk_scale,
                                attn_drop=attn_drop)
        self.FusionBlock2 = CAF(patch_size=patch_size, dim=dim, num_heads=num_heads, channel=channels,
                                proj_drop=proj_drop, depth=fusionblock_depth, qk_scale=qk_scale,
                                attn_drop=attn_drop)
        self.FusionBlock_final = CAF(patch_size=patch_size, dim=dim, num_heads=num_heads,
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
        self.fusionnet1 = TransSFF(patch_size=patch_size, dim=dim, num_heads=num_heads,
                              channels=channels[0],
                              fusionblock_depth=fusionblock_depth[0],
                              qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=proj_drop)
        self.fusionnet2 = TransSFF(patch_size=patch_size, dim=dim, num_heads=num_heads,
                              channels=channels[1],
                              fusionblock_depth=fusionblock_depth[1],
                              qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=proj_drop)
        self.fusionnet3 = TransSFF(patch_size=patch_size, dim=dim, num_heads=num_heads,
                              channels=channels[2],
                              fusionblock_depth=fusionblock_depth[2],
                              qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=proj_drop)
        self.fusionnet4 = TransSFF(patch_size=patch_size, dim=dim, num_heads=num_heads,
                              channels=channels[3],
                              fusionblock_depth=fusionblock_depth[3],
                              qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=proj_drop)

        # Conv 1x1
        self.outlayer = nn.Conv2d(16, 1, 1)

    def forward(self, img1, img2):
        x1, x2, x3, x4 = self.encoder(img1)
        y1, y2, y3, y4 = self.encoder(img2)

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

        self.sff = SFF(patch_size=16, dim=256, num_heads=8, channels=512, fusionblock_depth=3,
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

class ResNet_TransSFF(nn.Module):

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

        self.sff = TransSFF(patch_size=16, dim=256, num_heads=8, channels=512, fusionblock_depth=4,
                 qk_scale=None, attn_drop=0., proj_drop=0., img_size=256)

        # self.caf = CAF(patch_size=16, dim=512, num_heads=8, channel=512, depth=3,
        #          qk_scale=None, attn_drop=0., proj_drop=0.)

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

        self.sff = SFF(patch_size=16, dim=256, num_heads=8, channels=512, fusionblock_depth=3,
                 qk_scale=None, attn_drop=0., proj_drop=0., img_size=256)

        # self.caf = CAF(patch_size=16, dim=512, num_heads=8, channel=512, depth=3,
        #          qk_scale=None, attn_drop=0., proj_drop=0.)

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

        self.caf = CAF(patch_size=16, dim=256, num_heads=8, channel=512, depth=4,
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

        # y_f = torch.fft.fft2(y5)  # Fourier Transform
        # y_f = torch.fft.fftshift(y_f)
        # y_f = torch.log(1 + torch.abs(y_f))

        # x_f = torch.fft.fft2(x5)
        # x_f = torch.fft.fftshift(x_f)
        # x_f = torch.log(1 + torch.abs(x_f))

        output = self.caf(x5, y5)

        # output = torch.fft.ifftshift(output)
        # output = torch.fft.ifft2(output)
        # output = torch.abs(output)


        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output
    
class ResNet_PagFM(nn.Module):

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

        self.pagfm = PagFM(in_channels=512, mid_channels=512)

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
        '''
            x: vis image
            y: trans image
        '''
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

        output = self.pagfm(x5, y5)

        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def resnet18_caf():
    """ return a ResNet 18 object
    """
    return ResNet_CAF(BasicBlock, [2, 2, 2, 2])

def resnet18_sff():
    """ return a ResNet 18 object
    """
    return ResNet_SFF(BasicBlock, [2, 2, 2, 2])

def resnet18_trans_sff():
    """ return a ResNet 18 object
    """
    return ResNet_TransSFF(BasicBlock, [2, 2, 2, 2])

def resnet18_pag():
    """ return a ResNet 18 object
    """
    return ResNet_PagFM(BasicBlock, [2, 2, 2, 2])
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
    img1 = torch.randn(1, 3, 256, 256)
    img2 = torch.randn(1, 3, 256, 256)
    fidx_u = [0, 1]
    fidx_v = [0, 1]

    # net = FACMA(in_channel=3, width=256, height=256, fidx_u=fidx_u, fidx_v=fidx_v)
    # out_rgb, out_d = net(img1, img2)

    # net = MEEM(in_dim=3, hidden_dim=6)
    # net = PatchEmbed(in_c=512)
    # net = PagFM(in_channels=3, mid_channels=64, with_channel=False)
    net = resnet18_pag()
    

    torchinfo.summary(net, input_data=(img1, img2))

    # z = net(img1, img2)

    # print(z.shape)



