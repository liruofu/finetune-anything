import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

#sam的图片输入大小必须是1024
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
])

'''

'''

class PotsdamSemanticDataset(Dataset):
    # 对于label的像素值0	no-data	无效值（使用时应被忽略）
    def __init__(self, root, image_set, transform=transform, with_id=False, with_mask=False, target_transform=None):
        super(PotsdamSemanticDataset, self).__init__()
        self.root_dir = root
        self.image_dir = os.path.join(self.root_dir, 'img_dir', image_set)
        self.mask_dir = os.path.join(self.root_dir, 'ann_dir', image_set)
        self.image_id_list = [file for file in os.listdir(self.image_dir) if file.endswith('.png')]
        self.transform = transform
        self.with_id = with_id
        self.with_mask = with_mask
        self.dataset_meta = {
            'classes': ['impervious surfaces', 'building', 'low vegetation', 'tree', 'car', 'clutter'],  # 类别名称列表
            'num_classes': 6  # 类别数量
        }
        self.pixel_mean = torch.as_tensor([123.675, 116.28, 103.53])
        self.pixel_std =  torch.as_tensor([58.395, 57.12, 57.375])

    def __len__(self):
        return len(self.image_id_list)

    def get_image(self, image_id):
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert('RGB')
        return image, image_path

    def get_mask(self, image_id):
        mask_path = os.path.join(self.mask_dir, image_id)
        if os.path.isfile(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = None
        return mask


    def __getitem__(self, index):
        image_id = self.image_id_list[index]
        image ,image_path = self.get_image(image_id)
        image=  self.preprocess(transform(torch.as_tensor(np.array(image)).permute(2,0,1)))
        mask = self.get_mask(image_id)
        mask = np.array(mask)

        return image, image_path, mask

    # sam官方处理图片时正则化的值
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values."""
        # Normalize colors
        x = (x - self.pixel_mean[:, None, None]) / self.pixel_std[:, None, None]

        return x


class LovedaSemanticDataset(Dataset):
    # 对于label的像素值0	no-data	无效值（使用时应被忽略）
    def __init__(self, root, image_set, transform=transform, with_id=False, with_mask=False, target_transform=None):
        super(LovedaSemanticDataset, self).__init__()
        self.root_dir = root
        self.image_dir = os.path.join(self.root_dir, 'img_dir', image_set)
        self.mask_dir = os.path.join(self.root_dir, 'ann_dir', image_set)
        self.image_id_list = [file for file in os.listdir(self.image_dir) if
                              file.endswith('.jpg') or file.endswith('.png')]
        self.transform = transform
        self.with_id = with_id
        self.with_mask = with_mask
        self.dataset_meta = {
            'classes': ['background', 'building', 'road', 'water', 'barren', 'forest', 'agriculture'],  # 类别名称列表
            'num_classes': 7  # 类别数量
        }

        self.pixel_mean = torch.as_tensor([123.675, 116.28, 103.53])
        self.pixel_std = torch.as_tensor([58.395, 57.12, 57.375])


    def __len__(self):
        return len(self.image_id_list)


    def get_image(self, image_id):
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert('RGB')
        return image, image_path


    def get_mask(self, image_id):
        mask_path = os.path.join(self.mask_dir, image_id)
        if os.path.isfile(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = None
        return mask


    def __getitem__(self, index):
        image_id = self.image_id_list[index]
        image, image_path = self.get_image(image_id)
        image = self.preprocess(transform(torch.as_tensor(np.array(image)).permute(2, 0, 1)))
        mask = self.get_mask(image_id)
        mask = np.array(mask)

        return image, image_path, mask


    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values."""
        # Normalize colors
        x = (x - self.pixel_mean[:, None, None]) / self.pixel_std[:, None, None]

        return x


class IsaidSemanticDataset(Dataset):
    # 对于label的像素值0	no-data	无效值（使用时应被忽略）
    def __init__(self, root, image_set, transform=transform, with_id=False, with_mask=False, target_transform=None):
        super(IsaidSemanticDataset, self).__init__()
        self.root_dir = root
        self.image_dir = os.path.join(self.root_dir, 'non_black_img_dir', image_set)
        self.mask_dir = os.path.join(self.root_dir, 'non_black_ann_dir', image_set)
        self.image_id_list = [file for file in os.listdir(self.image_dir) if file.endswith('.png')]
        self.transform = transform
        self.with_id = with_id
        self.with_mask = with_mask
        self.class_names = ['ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court', 'ground track field', 'bridge', 'large vehicle', 'small vehicle', 'helicopter', 'swimming pool', 'roundabout', 'soccer ball field', 'plane', 'harbor']
        self.pixel_mean = torch.as_tensor([123.675, 116.28, 103.53])
        self.pixel_std = torch.as_tensor([58.395, 57.12, 57.375])
        self.dataset_meta = {
            'classes': ['ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court', 'ground track field', 'bridge', 'large vehicle', 'small vehicle', 'helicopter', 'swimming pool', 'roundabout', 'soccer ball field', 'plane', 'harbor'],  # 类别名称列表
            'num_classes': 15  # 类别数量
        }

    def __len__(self):
        return len(self.image_id_list)

    def get_image(self, image_id):
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert('RGB')
        return image, image_path

    def get_mask(self, image_id):
        mask_path = os.path.join(self.mask_dir, image_id)

        # 此处使用mmsegmentation进行数据处理时label与image
        mask_path = mask_path[:-4] + "_instance_color_RGB.png"
        if os.path.isfile(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = None
        return mask

    def __getitem__(self, index):
        image_id = self.image_id_list[index]
        image, image_path = self.get_image(image_id)
        image = self.preprocess(transform(torch.as_tensor(np.array(image)).permute(2, 0, 1)))
        mask = self.get_mask(image_id)
        mask = np.array(mask)

        # print("label_max:", np.max(mask).item())

        return image, image_path, mask

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values."""
        # Normalize colors
        x = (x - self.pixel_mean[:, None, None]) / self.pixel_std[:, None, None]

        return x