""" helper function

author baiyu
"""
import os
import sys
import re
import datetime
# import cv2
from PIL import Image
import numpy
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from conf import settings
from dataset import PairedDataset
from sklearn.model_selection import train_test_split

def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()
    elif args.net == 'single_branch_cprfnet':
        from models.single_branch_cprfnet import CPRFNet
        net = CPRFNet()
    elif args.net == 'cprfnet':
        from models.cafresnet import CPRFNet
        net = CPRFNet()
    elif args.net == 'ranet18':
        from models.ranet import ranet18
        net = ranet18()
    elif args.net == 'ranet18-fpn':
        from models.resnet_fpn import ranet18_fpn
        net = ranet18_fpn()
    elif args.net == 'resnet18-fpn':
        from models.resnet_fpn import resnet18_fpn
        net = resnet18_fpn()
    elif args.net == 'resnet18-sff':
        from models.cafresnet import resnet18_sff
        net = resnet18_sff()    
    elif args.net == 'resnet18-caf':
        from models.cafresnet import resnet18_caf
        net = resnet18_caf()   
    elif args.net == 'resnet18-trans-sff':
        from models.cafresnet import resnet18_trans_sff
        net = resnet18_trans_sff()
    elif args.net == 'resnet18-pag':
        from models.cafresnet import resnet18_pag
        net = resnet18_pag()
    elif args.net == 'resnet18-pag-pretrain':
        from models.cafresnet import resnet18_pag_pretrain
        net = resnet18_pag_pretrain
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        # 检查是否有多个 GPU
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # 使用 DataParallel 包装模型
            net = nn.DataParallel(net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = net.cuda()

    return net


# def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
#     """ return training dataloader
#     Args:
#         mean: mean of cifar100 training dataset
#         std: std of cifar100 training dataset
#         path: path to cifar100 training python dataset
#         batch_size: dataloader batchsize
#         num_workers: dataloader num_works
#         shuffle: whether to shuffle
#     Returns: train_data_loader:torch dataloader object
#     """

#     transform_train = transforms.Compose([
#         #transforms.ToPILImage(),
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(15),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)
#     ])
#     #cifar100_training = CIFAR100Train(path, transform=transform_train)
#     cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
#     cifar100_training_loader = DataLoader(
#         cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

#     return cifar100_training_loader


# def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
#     """ return training dataloader
#     Args:
#         mean: mean of cifar100 test dataset
#         std: std of cifar100 test dataset
#         path: path to cifar100 test python dataset
#         batch_size: dataloader batchsize
#         num_workers: dataloader num_works
#         shuffle: whether to shuffle
#     Returns: cifar100_test_loader:torch dataloader object
#     """

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)
#     ])
#     #cifar100_test = CIFAR100Test(path, transform=transform_test)
#     cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
#     cifar100_test_loader = DataLoader(
#         cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

#     return cifar100_test_loader

def get_training_dataloader(data_dir, mean, std, batch_size=16, num_workers=2, shuffle=True, pin_memory=True):
    """ return training dataloader
    Args:
        data_dir: path to training dataset
        mean: mean of training dataset
        std: std of training dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_loader: torch dataloader object
    """
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小为 224x224
        transforms.RandomCrop(256, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
    train_loader = DataLoader(train_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
    return train_loader

def get_test_dataloader(data_dir, mean, std, batch_size=16, num_workers=2, shuffle=True, pin_memory=True):
    """ return test dataloader
    Args:
        data_dir: path to test dataset
        mean: mean of test dataset
        std: std of test dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: test_loader: torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小为 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_dataset = ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform_test)
    test_loader = DataLoader(test_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
    return test_loader

def calculate_mean_std(data_dir):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小为 224x224
        transforms.ToTensor()
    ])
    
    dataset = ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    mean = 0.0
    std = 0.0
    nb_samples = 0

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
    return mean, std

# def calculate_mean_std(vis_dir, trans_dir):
#     means = []
#     stds = []
#     transform = transforms.ToTensor()

#     # 遍历可见光图像文件夹
#     for class_id in range(1, 8):
#         vis_class_dir = os.path.join(vis_dir, str(class_id))
#         for vis_img in tqdm(os.listdir(vis_class_dir), desc=f"Processing {vis_class_dir}"):
#             vis_path = os.path.join(vis_class_dir, vis_img)
#             image = Image.open(vis_path).convert('RGB')
#             tensor_image = transform(image)
#             means.append(tensor_image.mean(dim=(1, 2)))
#             stds.append(tensor_image.std(dim=(1, 2)))

#     # 遍历透射可见图像文件夹
#     for class_id in range(1, 8):
#         trans_class_dir = os.path.join(trans_dir, str(class_id))
#         for trans_img in tqdm(os.listdir(trans_class_dir), desc=f"Processing {trans_class_dir}"):
#             trans_path = os.path.join(trans_class_dir, trans_img)
#             image = Image.open(trans_path).convert('RGB')
#             tensor_image = transform(image)
#             means.append(tensor_image.mean(dim=(1, 2)))
#             stds.append(tensor_image.std(dim=(1, 2)))

#     # 计算所有图像的均值和标准差
#     mean = torch.stack(means).mean(dim=0)
#     std = torch.stack(stds).mean(dim=0)
#     return mean, std

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    print("folders: ", folders)

    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[-2])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]

def make_pairs_and_labels(vis_dir, trans_dir):
    pairs = []
    labels = []
    for class_id in range(1, 8):
        vis_class_dir = os.path.join(vis_dir, str(class_id))
        trans_class_dir = os.path.join(trans_dir, str(class_id))
        vis_images = sorted(os.listdir(vis_class_dir))
        trans_images = sorted(os.listdir(trans_class_dir))
        for vis_img, trans_img in zip(vis_images, trans_images):
            vis_path = os.path.join(vis_class_dir, vis_img)
            trans_path = os.path.join(trans_class_dir, trans_img)
            pairs.append((vis_path, trans_path))
            labels.append(class_id)  # 类标签从0开始
    return pairs, labels

def get_paired_dataloaders(vis_dir, trans_dir, mean, std, batch_size=16, num_workers=2, shuffle=True, pin_memory=True, test_size=0.2):
    pairs, labels = make_pairs_and_labels(vis_dir, trans_dir)
    train_pairs, test_pairs, train_labels, test_labels = train_test_split(pairs, labels, test_size=test_size, stratify=labels)

    transform_vis = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(256, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[0], std[0])
    ])

    transform_trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(256, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[1], std[1])
    ])

    train_dataset = PairedDataset(train_pairs, train_labels, transform_vis=transform_vis, transform_trans=transform_trans)
    test_dataset = PairedDataset(test_pairs, test_labels, transform_vis=transform_vis, transform_trans=transform_trans)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, test_loader

def show_images(vis_image, trans_image):
    vis_image = vis_image.permute(1, 2, 0).cpu().numpy()
    trans_image = trans_image.permute(1, 2, 0).cpu().numpy()

    vis_image = (vis_image * 255).astype(numpy.uint8)
    trans_image = (trans_image * 255).astype(numpy.uint8)

    combined_image = numpy.hstack((vis_image, trans_image))
    cv2.imshow('Vis and Trans Images', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    data_dir = r'F:\datasets'
    mean = [0.0, 0.0]
    std = [0.0, 0.0]
    mean[0], std[0] = calculate_mean_std(
        os.path.join(data_dir, 'visset'),
        # os.path.join(data_dir, 'transset')
    )
    mean[1], std[1] = calculate_mean_std(
        # os.path.join(data_dir, 'visset'),
        os.path.join(data_dir, 'transset')
    )
    #data preprocessing:
    training_loader, test_loader = get_paired_dataloaders(
        os.path.join(data_dir, 'visset'),
        os.path.join(data_dir, 'transset'),
        mean, 
        std, 
        num_workers=1,  # 增加 num_workers
        batch_size=32,
        shuffle=True,
        pin_memory=True,  # 使用 pin_memory
        test_size=0.3  # 设置测试集比例
    )          

    # 示例：遍历训练数据
    for batch_index, (vis_images, trans_images, labels) in enumerate(training_loader):
        print(batch_index, vis_images.shape, trans_images.shape, labels.shape)
        for i in range(vis_images.size(0)):
            x = vis_images[i]
            y = trans_images[i]

            y_f = torch.fft.fft2(y)  # Fourier Transform
            y_f = torch.fft.fftshift(y_f)
            y_f = torch.log(1 + torch.abs(y_f))

            x_f = torch.fft.fft2(x)
            x_f = torch.fft.fftshift(x_f)
            x_f = torch.log(1 + torch.abs(x_f))

            show_images(x_f, y_f)

            # feature_y = torch.fft.ifftshift(feature_y)
            # feature_y = torch.fft.ifft2(feature_y)
            # feature_y = torch.abs(feature_y)
            
    #     # break