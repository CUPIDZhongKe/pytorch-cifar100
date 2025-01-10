import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from models import cafresnet
import torch
import torch.nn as nn
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from utils import get_network, get_test_dataloader, calculate_mean_std, get_paired_dataloaders


net = cafresnet.PagFM(in_channels=1, mid_channels=512, with_channel=True, fusion=False)
# net_with_ee =  cafresnet.PagFM(in_channels=1, mid_channels=512, with_channel=True, with_ee=True, with_lia=False)
# net_with_lia = cafresnet.PagFM(in_channels=1, mid_channels=512, with_channel=True, with_ee=False, with_lia=True)
# net_with_meem = cafresnet.PagFM(in_channels=1, mid_channels=512, with_channel=True, with_ee=True, with_lia=True)
# net = cafresnet.FACMA(in_channel=1, width=256, height=256, fidx_u=[0, 1], fidx_v=[0, 1])
# net = cafresnet.MEEM(in_dim=1, hidden_dim=64, width=3, norm=nn.BatchNorm2d, act=nn.ReLU)
# net = cafresnet.EdgeEnhancer(in_dim=1, norm=nn.BatchNorm2d, act=nn.ReLU)
# net = cafresnet.LocalAttention(channels=1)

# 读取包含中文名的图片
image_path_vis = r"F:\datasets\visset\6\1_10.jpg"
image_path_trans = r"F:\datasets\transset\6\1_10.jpg"

# 使用 cv2.imread 读取图片
image_vis = cv2.imread(image_path_vis, cv2.IMREAD_GRAYSCALE)
image_trans = cv2.imread(image_path_trans, cv2.IMREAD_GRAYSCALE)

# 将图像转换为 numpy 数组
image_vis = image_vis.astype(np.float32)
image_trans = image_trans.astype(np.float32)

# 将 numpy 数组转换为 PyTorch 张量，并调整维度顺序
tensor_vis = torch.from_numpy(image_vis).unsqueeze(0).unsqueeze(0)  # 转换为 [1, C, H, W]
tensor_trans = torch.from_numpy(image_trans).unsqueeze(0).unsqueeze(0)  # 转换为 [1, C, H, W]# 打印张量的形状

print(f"tensor_vis shape: {tensor_vis.shape}")
print(f"tensor_trans shape: {tensor_trans.shape}")

out_vis, out_trans = net(tensor_vis, tensor_trans)
# out_with_ee = net_with_ee(tensor_vis, tensor_trans)
# out_with_lia = net_with_lia(tensor_vis, tensor_trans)
# out_with_meem = net_with_meem(tensor_vis, tensor_trans)

out_vis = out_vis.detach().cpu().numpy().squeeze()
out_trans = out_trans.detach().cpu().numpy().squeeze()
# out_with_ee = out_with_ee.detach().cpu().numpy().squeeze()
# out_with_lia = out_with_lia.detach().cpu().numpy().squeeze()
# out_with_meem = out_with_meem.detach().cpu().numpy().squeeze()

plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.title("vis Image")
plt.imshow(image_vis, cmap='gray')

plt.subplot(2, 2, 2)
plt.title("trans Image")
plt.imshow(image_trans, cmap='gray')

plt.subplot(2, 2, 3)
plt.title("out vis Image")
plt.imshow(out_vis, cmap='gray')

plt.subplot(2, 2, 4)
plt.title("out trans Image")
plt.imshow(out_trans, cmap='gray')

# plt.subplot(2, 4, 7)
# plt.title("out with lia Image")
# plt.imshow(out_with_lia, cmap='gray')

# plt.subplot(2, 4, 8)
# plt.title("out with ee and lia Image")
# plt.imshow(out_with_meem, cmap='gray')

plt.show()
