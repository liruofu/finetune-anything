import os
import shutil
import numpy as np
from PIL import Image

# 定义源文件夹和目标文件夹路径、
ann_src_dir = ['/home/yelu/ICPR_CV/cropped/ann_dir/train', '/home/yelu/ICPR_CV/cropped/ann_dir/val']
ann_dst_dir = ['/home/yelu/ICPR_CV/cropped/non_black_ann_dir/train', '/home/yelu/ICPR_CV/cropped/non_black_ann_dir/val']

img_src_dir = ['/home/yelu/ICPR_CV/cropped/img_dir/train', '/home/yelu/ICPR_CV/cropped/img_dir/val']
img_dst_dir = ['/home/yelu/ICPR_CV/cropped/non_black_img_dir/train', '/home/yelu/ICPR_CV/cropped/non_black_img_dir/val']

# 确保目标文件夹存在
for path in ann_dst_dir:
    if not os.path.exists(path):
        os.makedirs(path)

for path in img_dst_dir:
    if not os.path.exists(path):
        os.makedirs(path)


# 判断图片是否全黑的函数
def is_image_not_pure_black(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)

    # 验证像素是否存在问题
    print(np.max(img_array))

    # 计算所有像素RGB值的总和
    total_sum = img_array.sum()

    # 如果总和不为0，则认为图像不是纯黑
    return total_sum != 0


# 遍历源文件夹中的所有.jpg和.png文件
for i in range(2):
    ann_dir = ann_src_dir[i]
    ann_des_dir = ann_dst_dir[i]

    img_dir = img_src_dir[i]
    img_des_dir = img_dst_dir[i]

    for filename in os.listdir(ann_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            out_ann_path = os.path.join(ann_dir, filename)

            if is_image_not_pure_black(out_ann_path):
                # move gray image
                dst_file = os.path.join(ann_des_dir, filename)
                shutil.copy2(out_ann_path, dst_file)

                # move RGB image
                name = filename[:-23] + ".png"
                read_img_path = os.path.join(img_dir, name)
                dst_file = os.path.join(img_des_dir, name)
                shutil.copy2(read_img_path, dst_file)

                # output result
                print(f'Moved {filename} to {dst_file}')