#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse
import os

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader, calculate_mean_std, get_paired_dataloaders

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-datadir', type=str, default='F:\\datasets', help='the path of data')
    parser.add_argument('-batchsize', type=int, default=32, help='batch size for dataloader')
    args = parser.parse_args()

    net = get_network(args)

   #data preprocessing:
    mean = [0.0, 0.0]
    std = [0.0, 0.0]
    mean[0], std[0] = calculate_mean_std(
        os.path.join(args.datadir, 'visset')
    )
    mean[1], std[1] = calculate_mean_std(
        os.path.join(args.datadir, 'transset')
    )
    #data preprocessing:
    _, test_loader = get_paired_dataloaders(
        os.path.join(args.datadir, 'visset'),
        os.path.join(args.datadir, 'transset'),
        mean, 
        std, 
        num_workers=8,  # 增加 num_workers
        batch_size=32,
        shuffle=True,
        pin_memory=True,  # 使用 pin_memory
        test_size=0.5  # 设置测试集比例
    )   


    # mean, std = calculate_mean_std(os.path.join(args.datadir, 'test'))
    # test_loader = get_test_dataloader(
    #     args.datadir,
    #     mean,
    #     std,
    #     batch_size=args.batchsize,
    #     num_workers=8
    # )

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (vis_image, trans_image, label) in enumerate(test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

            if args.gpu:
                vis_image = vis_image.cuda()
                trans_image = trans_image.cuda()
                label = label.cuda()
                # print('GPU INFO.....')
                # print(torch.cuda.memory_summary(), end='')


            output = net(vis_image, trans_image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print()
    print("Top 1 err: ", 1 - correct_1 / len(test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
