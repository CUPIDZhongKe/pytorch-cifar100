import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import re
import random

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
            filenames.append(filename)
        else:
            print(f"Warning: Unable to read image {img_path}")
    return images, filenames

def calculate_similarity(img1, img2):
    img1 = cv2.resize(img1, (32, 32)).flatten()
    img2 = cv2.resize(img2, (32, 32)).flatten()
    return cosine_similarity([img1], [img2])[0][0]

def rename_images(folder):
    images, filenames = load_images_from_folder(folder)
    num_images = len(images)
    similarity_matrix = np.zeros((num_images, num_images))

    # 计算相似度矩阵
    for i in range(num_images):
        for j in range(i + 1, num_images):
            similarity_matrix[i, j] = calculate_similarity(images[i], images[j])

    used = set()
    pair_index = 1

    for i in range(num_images):
        if i in used:
            continue
        most_similar_index = np.argmax(similarity_matrix[i])
        if similarity_matrix[i, most_similar_index] == 0:
            continue
        used.add(i)
        used.add(most_similar_index)

        # 获取原始文件名
        filename1 = filenames[i]
        filename2 = filenames[most_similar_index]

        # 构造新文件名
        new_filename1 = f"{pair_index}_可见光_{filename1.split('_')[-1]}"
        new_filename2 = f"{pair_index}_透视可见_{filename2.split('_')[-1]}"

        # 重命名文件
        shutil.move(os.path.join(folder, filename1), os.path.join(folder, new_filename1))
        shutil.move(os.path.join(folder, filename2), os.path.join(folder, new_filename2))

        pair_index += 1

def rename_images_2(folder):
    for filename in os.listdir(folder):
        # 匹配文件名模式 "i_可见光_xxxxxxxxxx.jpg" 或 "i_透视可见_xxxxxxxxxx.jpg"
        match = re.match(r"(\d+)_([可见光|透视可见]+)_(\d+).jpg", filename)
        if match:
            i, type, x = match.groups()
            new_filename = f"Image_{type}_{x}.jpg"
            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_filename)
            shutil.move(old_path, new_path)
            print(f"Renamed {old_path} to {new_path}")

def rename_files_3(folder):
    for filename in os.listdir(folder):
        # 匹配文件名模式 "i_可见光_xxxxxxxxxx.jpg" 或 "i_透视可见_xxxxxxxxxx.jpg"
        match = re.match(r"(\d+)_([可见光|透视可见]+)_(\d+).jpg", filename)
        if match:
            i, type, x = match.groups()
            new_filename = f"{i}_{type}.jpg"
            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_filename)
            shutil.move(old_path, new_path)
            print(f"Renamed {old_path} to {new_path}")

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_index = 1

    filenames = os.listdir(input_folder)
    visible_files = [f for f in filenames if re.match(r"(\d+)_可见光.jpg", f)]
    perspective_files = [f for f in filenames if re.match(r"(\d+)_透视可见.jpg", f)]

    visible_dict = {re.match(r"(\d+)_可见光.jpg", f).group(1): f for f in visible_files}
    perspective_dict = {re.match(r"(\d+)_透视可见.jpg", f).group(1): f for f in perspective_files}

    for i in visible_dict:
        if i not in perspective_dict:
            print(f"Warning: Matching file for {visible_dict[i]} not found")
            continue

        visible_file = visible_dict[i]
        perspective_file = perspective_dict[i]

        visible_img_path = os.path.join(input_folder, visible_file)
        perspective_img_path = os.path.join(input_folder, perspective_file)

        visible_img = cv2.imdecode(np.fromfile(visible_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        perspective_img = cv2.imdecode(np.fromfile(perspective_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        if visible_img is None or perspective_img is None:
            print(f"Warning: Unable to read image {visible_img_path} or {perspective_img_path}")
            continue

        height, width = visible_img.shape[:2]
        blocks = []

        for y in range(0, height, 256):
            for x in range(0, width, 256):
                visible_block = visible_img[y:y+256, x:x+256]
                perspective_block = perspective_img[y:y+256, x:x+256]
                if visible_block.shape[0] != 256 or visible_block.shape[1] != 256:
                    continue

                gray_block = cv2.cvtColor(visible_block, cv2.COLOR_BGR2GRAY)
                _, binary_block = cv2.threshold(gray_block, 127, 255, cv2.THRESH_BINARY)

                zero_pixel_count = np.sum(binary_block == 0)
                total_pixel_count = binary_block.size
                zero_pixel_ratio = zero_pixel_count / total_pixel_count

                if 0.1 <= zero_pixel_ratio <= 0.9:
                    blocks.append((visible_block, perspective_block))

import os
import cv2
import numpy as np
import re
import random

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_index = 1

    filenames = os.listdir(input_folder)
    visible_files = [f for f in filenames if re.match(r"(\d+)_可见光.jpg", f)]
    perspective_files = [f for f in filenames if re.match(r"(\d+)_透视可见.jpg", f)]

    visible_dict = {re.match(r"(\d+)_可见光.jpg", f).group(1): f for f in visible_files}
    perspective_dict = {re.match(r"(\d+)_透视可见.jpg", f).group(1): f for f in perspective_files}

    for i in visible_dict:
        if i not in perspective_dict:
            print(f"Warning: Matching file for {visible_dict[i]} not found")
            continue

        visible_file = visible_dict[i]
        perspective_file = perspective_dict[i]

        visible_img_path = os.path.join(input_folder, visible_file)
        perspective_img_path = os.path.join(input_folder, perspective_file)

        visible_img = cv2.imdecode(np.fromfile(visible_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        perspective_img = cv2.imdecode(np.fromfile(perspective_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        if visible_img is None or perspective_img is None:
            print(f"Warning: Unable to read image {visible_img_path} or {perspective_img_path}")
            continue

        def get_blocks(img, block_size):
            height, width = img.shape[:2]
            blocks = []
            positions = []
            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    if y + block_size > height or x + block_size > width:
                        continue
                    block = img[y:y+block_size, x:x+block_size]
                    gray_block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
                    _, binary_block = cv2.threshold(gray_block, 127, 255, cv2.THRESH_BINARY)
                    zero_pixel_count = np.sum(binary_block == 0)
                    total_pixel_count = binary_block.size
                    zero_pixel_ratio = zero_pixel_count / total_pixel_count
                    if 0.1 <= zero_pixel_ratio <= 0.9:
                        blocks.append(block)
                        positions.append((y, x))
            return blocks, positions

        blocks = []
        positions = []
        block_size = 256
        while len(blocks) < 30 and block_size >= 64:
            blocks, positions = get_blocks(visible_img, block_size)
            if len(blocks) < 30:
                block_size = int(block_size * 0.75)

        # 随机保留 30 个分割块，确保位置一致
        if len(blocks) > 30:
            selected_indices = random.sample(range(len(blocks)), 30)
            blocks = [blocks[i] for i in selected_indices]
            positions = [positions[i] for i in selected_indices]

        perspective_blocks = []
        for (y, x) in positions:
            if y + block_size > perspective_img.shape[0] or x + block_size > perspective_img.shape[1]:
                continue
            perspective_block = perspective_img[y:y+block_size, x:x+block_size]
            perspective_blocks.append(perspective_block)

        for visible_block, perspective_block in zip(blocks, perspective_blocks):
            # 上采样至 256x256
            visible_block_resized = cv2.resize(visible_block, (256, 256), interpolation=cv2.INTER_LINEAR)
            perspective_block_resized = cv2.resize(perspective_block, (256, 256), interpolation=cv2.INTER_LINEAR)

            new_visible_filename = f"{image_index}_{visible_file}"
            new_perspective_filename = f"{image_index}_{perspective_file}"
            new_visible_path = os.path.join(output_folder, new_visible_filename)
            new_perspective_path = os.path.join(output_folder, new_perspective_filename)
            cv2.imencode('.jpg', visible_block_resized)[1].tofile(new_visible_path)
            cv2.imencode('.jpg', perspective_block_resized)[1].tofile(new_perspective_path)
            print(f"Saved {new_visible_path} and {new_perspective_path}")
            image_index += 1

def move_images(input_folder, output_folder_vis, output_folder_trans):
    if not os.path.exists(output_folder_vis):
        os.makedirs(output_folder_vis)
    if not os.path.exists(output_folder_trans):
        os.makedirs(output_folder_trans)

    filenames = os.listdir(input_folder)
    for filename in filenames:
        if "可见光" in filename:
            src_path = os.path.join(input_folder, filename)
            dst_path = os.path.join(output_folder_vis, filename)
            shutil.move(src_path, dst_path)
            print(f"Moved {filename} to {output_folder_vis}")
        elif "透视可见" in filename:
            src_path = os.path.join(input_folder, filename)
            dst_path = os.path.join(output_folder_trans, filename)
            shutil.move(src_path, dst_path)
            print(f"Moved {filename} to {output_folder_trans}")

def copy_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filenames = os.listdir(input_folder)
    for filename in filenames:
        match = re.match(r'(\d+)_\d+_透视可见\.jpg', filename)
        if match:
            x = int(match.group(1))
            if 1 <= x <= 1680:
                src_path = os.path.join(input_folder, filename)
                dst_path = os.path.join(output_folder, filename)
                shutil.copy(src_path, dst_path)
                print(f"Copied {filename} to {output_folder}")

import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_train_dataloader(data_dir, mean, std, batch_size=16, num_workers=2, shuffle=True):
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
        transforms.Resize((224, 224)),  # 调整图像大小为 224x224
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
    train_loader = DataLoader(train_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return train_loader

def get_test_dataloader(data_dir, mean, std, batch_size=16, num_workers=2, shuffle=True):
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
        transforms.Resize((224, 224)),  # 调整图像大小为 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_dataset = ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform_test)
    test_loader = DataLoader(test_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return test_loader

def calculate_mean_std(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为 224x224
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


if __name__ == "__main__":
    data_dir = r'F:\datasets\docDataset_oneside_vis'
    mean, std = calculate_mean_std(os.path.join(data_dir, 'train'))
    train_loader = get_train_dataloader(data_dir, mean, std)
    test_loader = get_test_dataloader(data_dir, mean, std)

    # 示例：遍历训练数据
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        break

