# copyright ziqi-jin finetune anything
# modified by fjy
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
from .segment_anything_ori import sam_model_registry
from .image_encoder_adapter import BaseImgEncodeAdapter
from .mask_decoder_adapter import BaseMaskDecoderAdapter, SemMaskDecoderAdapter
from .prompt_encoder_adapter import BasePromptEncodeAdapter
import torch.nn.functional as F
from sampaper_new.sampaper.utils import mask_to_binary_images, draw_rectangles_and_points_on_masks, show_masks_isaid, draw_rectangles_and_points_on_masks_isaid
from .DilatedConvNet import FourLayerDilatedConvNetB
import matplotlib.pyplot as plt


class ECANet(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECANet, self).__init__()
        self.in_channels = in_channels
        self.fgp = nn.AdaptiveAvgPool2d((1, 1))
        kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.act1 = nn.Sigmoid()

    def forward(self, x):
        output = self.fgp(x)
        output = output.squeeze(-1).transpose(-1, -2)
        output = self.con1(output).transpose(-1, -2).unsqueeze(-1)
        output = self.act1(output)
        output = torch.multiply(x, output)
        return output


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class BaseExtendSam(nn.Module):

    def __init__(self, class_num,   ckpt_path=None, fix_img_en=False, fix_prompt_en=False, fix_mask_de=False, model_type='vit_b',):
        super(BaseExtendSam, self).__init__()
        assert model_type in ['default', 'vit_b', 'vit_l', 'vit_h'], print(
            "Wrong model_type, SAM only can be built as vit_b, vot_l, vit_h and default ")
        self.ori_sam = sam_model_registry[model_type](ckpt_path)
        self.img_adapter = BaseImgEncodeAdapter(self.ori_sam, fix=fix_img_en)
        self.prompt_adapter = BasePromptEncodeAdapter(self.ori_sam, fix=fix_prompt_en)
        self.mask_adapter = BaseMaskDecoderAdapter(self.ori_sam, fix=fix_mask_de)
        self.num = class_num

        # 空洞卷积以扩大感受野
        # self.dilated_conv = FourLayerDilatedConvNetB(class_num)

    def multi_prompt(self, prompt, x, all_rectangles, all_points, labels, final_masks, i):
        boxes = None
        point_coords = None
        point_labels = None
        mask = None
        if all_rectangles[i] != []:
            boxes = torch.from_numpy(np.array(all_rectangles[i])).to('cuda')

        if all_points[i] != []:
            point_coords = torch.reshape(torch.from_numpy(np.array(all_points[i])).to('cuda'), (boxes.shape[0], -1, 2))
            point_labels = torch.reshape(torch.from_numpy(np.array(labels[i])).to('cuda'), (boxes.shape[0], -1))

            # print(point_coords.shape)
            # print(point_labels.shape)

        if i == 4:  # potsdam tree class
            mask = (F.interpolate(prompt.unsqueeze(0).float(), (256, 256), mode="bilinear", align_corners=False)==i).float()

        # print box.shape
        # print(boxes.shape)

        # 更改prompt类型，三种prompts作为输入
        sparse_embeddings, dense_embeddings = self.prompt_adapter(  # 框、点、mask提示方式
            points=(point_coords, point_labels),
            boxes=boxes,
            # mask提示的大小必须是256，float类型，=None为没有输入
            masks=(F.interpolate(prompt.unsqueeze(0).float(), (256, 256), mode="bilinear", align_corners=False)==i).float()
        )

        multimask_output = False

        low_res_masks, iou_predictions = self.mask_adapter(
            image_embeddings=x,
            prompt_adapter=self.prompt_adapter,
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        # print(f'每一个类别mask大小{low_res_masks.shape}')
        # print(low_res_masks)

        values, _ = torch.max(low_res_masks, dim=0)  #由于每个提示输出一个分割mask，每一类有多个mask，取置信度最大值为该点值，将多mask变为一个

        final_masks[i] = values.unsqueeze(0)

    def forward(self, img, prompt, shape):

        x, mylist = self.img_adapter(img)

        all_rectangles = []
        all_points = []
        labels = []
        final_masks = (torch.zeros(self.num, 1, 256, 256)).to('cuda')   #存放每一类经过sam处理后的图

        # final_masks[0]=torch.ones(256, 256)*-10     #ignore index 0

        # sam的提示点、框的尺度必须是1024
        outputs = F.interpolate(prompt.unsqueeze(0).float(), (1024, 1024), mode="bilinear",
                                align_corners=False).squeeze(0).to('cuda')
        binary_images = mask_to_binary_images(outputs.detach().cpu().numpy(), self.num)

        # 重点，此处使用的是isaid_function
        # draw_rectangles_and_points_on_masks(all_points, labels, all_rectangles, binary_images)
        draw_rectangles_and_points_on_masks(all_points, labels, all_rectangles, binary_images)

        for i in range(1, self.num):
            # print(i)
            # if np.array(all_rectangles[i]).shape[0] > 100 and cont == 0:
            #     show_masks_isaid(label.squeeze(0).detach().cpu().numpy(), plt.gca())
            #     plt.show()
            #     show_masks_isaid(outputs.squeeze(0).detach().cpu().numpy(), plt.gca())
            #     plt.show()
            #     cont += 1

            if all_rectangles[i] == [] or np.array(all_rectangles[i]).shape[0] > 70:
                # final_masks[i] = torch.zeros((256, 256))
                continue
            self.multi_prompt(prompt, x, all_rectangles, all_points, labels, final_masks, i)

        # 将sam输出结果变为与label同大小，方便损失计算
        final_maskss = F.interpolate(final_masks, shape, mode="bilinear", align_corners=False).permute(1, 0, 2, 3)

        # 此处插入扩大感受野所使用的空洞卷积
        # final_masksss = self.dilated_conv(final_maskss)

        return final_maskss




