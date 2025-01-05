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
        match = re.match(r"(\d+)_([可见光|透射可见]+)_(\d+).jpg", filename)
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
    perspective_files = [f for f in filenames if re.match(r"(\d+)_透射可见.jpg", f)]

    visible_dict = {re.match(r"(\d+)_可见光.jpg", f).group(1): f for f in visible_files}
    perspective_dict = {re.match(r"(\d+)_透射可见.jpg", f).group(1): f for f in perspective_files}

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
        elif "透射可见" in filename:
            src_path = os.path.join(input_folder, filename)
            dst_path = os.path.join(output_folder_trans, filename)
            shutil.move(src_path, dst_path)
            print(f"Moved {filename} to {output_folder_trans}")

def copy_images(input_folder, train_folder, test_folder):
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    filenames = os.listdir(input_folder)
    for filename in filenames:
        match = re.match(r'(\d+)_\d+_([可见光|透射可见]+)\.jpg', filename)
        if match:
            x = int(match.group(1))
            if 1 <= x <= 630:
                src_path = os.path.join(input_folder, filename)
                train_path = os.path.join(train_folder, filename)
                shutil.copy(src_path, train_path)
                print(f"Copied {filename} to {train_folder}")
            else:
                src_path = os.path.join(input_folder, filename)
                test_path = os.path.join(test_folder, filename)
                shutil.copy(src_path, test_path)
                print(f"Copied {filename} to {test_folder}")

def delete_all_files(folder):
    if not os.path.exists(folder):
        print(f"The folder {folder} does not exist.")
        return

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"Deleted file {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Deleted directory {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def move_and_rename_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filenames = os.listdir(input_folder)
    for filename in filenames:
        match = re.match(r'(\d+)_(\d+)_(可见光|透射可见)\.jpg', filename)
        if match:
            i = int(match.group(1))
            j = match.group(2)
            suffix = match.group(3)
            new_i = i + 30
            new_filename = f"{new_i}_{j}_{suffix}.jpg"
            src_path = os.path.join(input_folder, filename)
            dst_path = os.path.join(output_folder, new_filename)
            shutil.copy(src_path, dst_path)
            print(f"Copy and renamed {filename} to {new_filename}")


if __name__ == "__main__":
    for i in range(1, 8):
        # input_folder = rf"F:\datasets\docDataset\{i}\双面"
        # output_folder = rf"F:\datasets\docDataset\{i}\patches"
        # process_images(input_folder, output_folder)

        # input_folder = rf"F:\datasets\docDataset\{i}\patches"
        # output_folder_vis = rf"F:\datasets\docDataset\{i}\patches_vis_dual"
        # output_folder_trans = rf"F:\datasets\docDataset\{i}\patches_trans_dual"
        # move_images(input_folder, output_folder_vis, output_folder_trans)

        # input_folder_vis = rf"F:\datasets\docDataset\{i}\patches_vis_dual"
        # input_folder_trans = rf"F:\datasets\docDataset\{i}\patches_trans_dual"
        # train_folder_trans = rf"F:\datasets\docDataset_twoside_trans\train\{i}"
        # train_folder_vis = rf"F:\datasets\docDataset_twoside_vis\train\{i}"
        # test_folder_trans = rf"F:\datasets\docDataset_twoside_trans\test\{i}"
        # test_folder_vis = rf"F:\datasets\docDataset_twoside_vis\test\{i}"
        # copy_images(input_folder_vis, train_folder_vis, test_folder_vis)
        # copy_images(input_folder_trans, train_folder_trans, test_folder_trans)

    #     folder_to_clean = rf"F:\datasets\docDataset\{i}\patches"
    #     delete_all_files(folder_to_clean)
        # folder_to_clean = rf"F:\datasets\docDataset\{i}\patches_vis_dual"
        # delete_all_files(folder_to_clean)
        # folder_to_clean = rf"F:\datasets\docDataset\{i}\patches_trans_dual"
        # delete_all_files(folder_to_clean)
        # input_folder_trans = r"F:\datasets\docDataset\docDataset_twoside_vis"
        # input_folder_vis = r"F:\datasets\docDataset\docDataset_twoside_vis"
        # delete_all_files(input_folder_vis)
        # delete_all_files(input_folder_trans)

        input_folder_path = rf"F:\datasets\docDataset_twoside_vis\train\{i}"
        output_folder_path = rf"F:\datasets\docDataset_vis_mix\train\{i}"
        move_and_rename_images(input_folder_path, output_folder_path)
        input_folder_path = rf"F:\datasets\docDataset_twoside_vis\test\{i}"
        output_folder_path = rf"F:\datasets\docDataset_vis_mix\test\{i}"
        move_and_rename_images(input_folder_path, output_folder_path)

        input_folder_path = rf"F:\datasets\docDataset_twoside_trans\train\{i}"
        output_folder_path = rf"F:\datasets\docDataset_trans_mix\train\{i}"
        move_and_rename_images(input_folder_path, output_folder_path)
        input_folder_path = rf"F:\datasets\docDataset_twoside_trans\test\{i}"
        output_folder_path = rf"F:\datasets\docDataset_trans_mix\test\{i}"
        move_and_rename_images(input_folder_path, output_folder_path)