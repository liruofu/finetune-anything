# copyright ziqi-jin
import torch
import torch.nn as nn
from torch.nn import functional as F
from .segment_anything_ori import sam_model_registry
from .image_encoder_adapter import BaseImgEncodeAdapter
from .mask_decoder_adapter import BaseMaskDecoderAdapter, SemMaskDecoderAdapter
from .prompt_encoder_adapter import BasePromptEncodeAdapter
from .segment_anything_ori.modeling.common import LayerNorm2d
from torchvision.ops import DeformConv2d
from typing import Type

class DConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(DConv, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias, dilation=dilation)
        self.conv2 = DeformConv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                                  padding=padding, bias=bias, dilation=dilation, groups=planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(x, out)
        return out


class DetailCapture(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DetailCapture, self).__init__()
        self.dconv1 = nn.Sequential(DConv(in_channel, out_channel, 3, 1, 1, 1),
                                    LayerNorm2d(out_channel), nn.GELU())
        self.dconv2 = nn.Sequential(DConv(in_channel, out_channel, 3, 1, 9, 9),
                                    LayerNorm2d(out_channel), nn.GELU())
        self.dconv3 = nn.Sequential(DConv(in_channel, out_channel, 3, 1, 12, 12),
                                    LayerNorm2d(out_channel), nn.GELU())
        self.norm1 = LayerNorm2d(out_channel)
        self.norm2 = LayerNorm2d(out_channel)
        self.conv = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.mlp = MLPBlock(out_channel, out_channel*2)

    def forward(self, x):
        x1 = self.dconv1(x)
        x2 = self.dconv2(x)
        x3 = self.dconv3(x)
        output = self.norm1(x1+x2+x3+self.conv(x))
        output = self.norm2(self.mlp(output.permute(0,2,3,1)).permute(0,3,1,2)+output)
        return output

class BaseExtendSam(nn.Module):

    def __init__(self, ckpt_path=None, fix_img_en=False, fix_prompt_en=False, fix_mask_de=False, model_type='vit_b'):
        super(BaseExtendSam, self).__init__()
        # assert model_type in ['default', 'vit_b', 'vit_l', 'vit_h'], print(
        #     "Wrong model_type, SAM only can be built as vit_b, vot_l, vit_h and default ")
        self.ori_sam = sam_model_registry[model_type](ckpt_path)
        self.img_adapter = BaseImgEncodeAdapter(self.ori_sam, fix=fix_img_en)
        self.prompt_adapter = BasePromptEncodeAdapter(self.ori_sam, fix=fix_prompt_en)
        self.mask_adapter = BaseMaskDecoderAdapter(self.ori_sam, fix=fix_mask_de)
        self.detail = DetailCapture(64, 32)


    def forward(self, img):
        x, x0 = self.img_adapter(img)
        details = self.detail(x0)
        dark_channel, _ = torch.min(img, dim=1)
        inverted_matrix = -dark_channel
        max_filtered_matrix = F.max_pool2d(inverted_matrix.unsqueeze(1), kernel_size=3, stride=1, padding=1)
        points = None
        boxes = None
        # masks = None

        sparse_embeddings, dense_embeddings = self.prompt_adapter(
            points=points,
            boxes=boxes,
            masks=-max_filtered_matrix,
        )
        multimask_output = True
        low_res_masks, iou_predictions = self.mask_adapter(
            image_embeddings=x,
            prompt_adapter=self.prompt_adapter,
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            details=details
        )
        # print(low_res_masks.size())
        return low_res_masks, iou_predictions


class SemanticSam(BaseExtendSam):

    def __init__(self, ckpt_path=None, fix_img_en=False, fix_prompt_en=False, fix_mask_de=False, class_num=20,
                 model_type='vit_b'):
        super().__init__(ckpt_path=ckpt_path, fix_img_en=fix_img_en, fix_prompt_en=fix_prompt_en,
                         fix_mask_de=fix_mask_de, model_type=model_type)
        self.mask_adapter = SemMaskDecoderAdapter(self.ori_sam, fix=fix_mask_de, class_num=class_num)


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))