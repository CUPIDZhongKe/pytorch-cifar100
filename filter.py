import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from models import cafresnet
import torch

net = cafresnet.PagFM(in_channels=512, mid_channels=512, with_channel=True)
# net = cafresnet.FACMA(in_channel=1, width=256, height=256, fidx_u=[0, 1], fidx_v=[0, 1])

# 读取包含中文名的图片
image_path_vis = r"F:\datasets\visset\1\8_10.jpg"
image_path_trans = r"F:\datasets\transset\1\8_10.jpg"

# 使用 cv2.imread 读取图片
image_vis = cv2.imread(image_path_vis, cv2.IMREAD_GRAYSCALE)
image_trans = cv2.imread(image_path_trans, cv2.IMREAD_GRAYSCALE)

# 将图像转换为 numpy 数组
image_vis = image_vis.astype(np.float32)
image_trans = image_trans.astype(np.float32)

plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.title("vis Image")
plt.imshow(image_vis, cmap='gray')

plt.subplot(2, 2, 2)
plt.title("trans Image")
plt.imshow(image_trans, cmap='gray')

# 将 numpy 数组转换为 PyTorch 张量，并调整维度顺序
tensor_vis = torch.from_numpy(image_vis).unsqueeze(0).unsqueeze(0)  # 转换为 [1, C, H, W]
tensor_trans = torch.from_numpy(image_trans).unsqueeze(0).unsqueeze(0)  # 转换为 [1, C, H, W]# 打印张量的形状

print(f"tensor_vis shape: {tensor_vis.shape}")
print(f"tensor_trans shape: {tensor_trans.shape}")

img1 = torch.randn(1, 512, 32, 32)
img2 = torch.randn(1, 512, 32, 32)
# out_vis, out_trans = net(tensor_vis, tensor_trans)
output = net(img1, img2)

print(output.shape)

# out_vis = out_vis.detach().cpu().numpy().squeeze()
# out_trans = out_trans.detach().cpu().numpy().squeeze()

# plt.subplot(2, 2, 3)
# plt.title("vis Fusion Image")
# plt.imshow(out_vis, cmap='gray')

# plt.subplot(2, 2, 4)
# plt.title("trans Fusion Image")
# plt.imshow(out_trans, cmap='gray')

# plt.show()
