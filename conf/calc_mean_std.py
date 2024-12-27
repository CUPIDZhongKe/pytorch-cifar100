import torch
from torchvision import datasets, transforms

# 定义数据集的路径
data_path = 'path_to_your_dataset'

# 定义数据集的转换
transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载数据集
dataset = datasets.ImageFolder(root=data_path, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

# 初始化变量
mean = 0.0
std = 0.0
nb_samples = 0

# 计算均值和标准差
for data, _ in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(f'Mean: {mean}')
print(f'Std: {std}')