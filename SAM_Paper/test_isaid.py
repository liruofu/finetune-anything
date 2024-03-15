'''
adapting sam for segmentation refinement
'''
import random
import cv2
import future
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from mmseg.apis import inference_model, init_model
from extend_sam import BaseExtendSam
from dataset import PotsdamSemanticDataset, LovedaSemanticDataset, IsaidSemanticDataset
from utils import Accuracy, compute_miou, show_masks_potsdam, show_masks_isaid, show_masks_loveda
from  iou_metric import IoUMetric


# 提示模型
def net(img, model):
    for param in model.parameters():
        param.requires_grad = False
    result = inference_model(model, img)
    return result


# 计算每类iou，有点问题
def showiou(list):
    miou = []
    for i in range(len(list)):
        if list[i] == []:
            miou.append("没有这一类")
        else:
            miou.append(sum(list[i]) / len(list[i]))
    for i in range(len(miou)):
        print(f'第{i + 1}类的miou是{miou[i]}')


if __name__ == '__main__':

    model = init_model("/home/yelu/ICPR_CV/mmsegmentation-main/configs/"
                       "segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py",
                       "/home/yelu/ICPR_CV/result/segformer_isaid/iter_240000.pth", device='cuda')

    test_dir ="/home/yelu/ICPR_CV/cropped"

    test_dataset = IsaidSemanticDataset(test_dir, 'val')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    sam_checkpoint = "/home/yelu/ICPR_CV/sampaper_new/sampaper/sam_vit_b_01ec64.pth"
    device = "cuda"
    model_type = "vit_b"
    SAM = BaseExtendSam(ckpt_path=sam_checkpoint, model_type=model_type,
                        class_num=16, fix_prompt_en=True, fix_img_en=True, fix_mask_de=True).to(device)

    SAM.load_state_dict(torch.load("/home/yelu/ICPR_CV/result/sam/isaid/segformer_10_1_M+P+B_Con/SAM_potsdam.pth"))

    prompt_miou = 0
    sam_miou = 0
    sam_accuracy = 0
    prompt_accuracy = 0

    compute1 = compute_miou(16)
    compute2 = compute_miou(16)

    sam_iou_list = []
    prompt_iou_list = []

    dataset_meta = test_dataset.dataset_meta
    iou_metric = IoUMetric(dataset_meta=dataset_meta)

    with torch.no_grad():
        data_samples = []
        SAM.eval()
        for j, (input, path, label) in enumerate(test_loader, 0):
            print(j)

            input, path, label = input.to(device), path, label.to(device)
            prompt = net(path, model)[0].pred_sem_seg.data   # + torch.tensor(1)

            result = SAM(input, prompt, label.shape[1:])
            result = torch.argmax(result, dim=1).to('cuda')

            # 调用mmsegmentation官方miou计算函数
            # data_samples = [{'pred_sem_seg': result, 'gt_sem_seg': label}]
            data_samples = [{'pred_sem_seg': result, 'gt_sem_seg': label}]
            iou_metric.process(data_batch=1, data_samples=data_samples)

            sam_accuracy += Accuracy(result, label)
            prompt_accuracy += Accuracy(prompt, label)

            label = label.squeeze(1).to(torch.int64)

            # sam_iou, sam_iou_list = compute1(result, label)
            # sam_miou += sam_iou
            #
            # prompt_iou, prompt_iou_list = compute2(prompt, label)
            # prompt_miou += prompt_iou

            metrics = iou_metric.compute_metrics(iou_metric.results)
            # 输出官方评估结果
            print(metrics)
            '''
              
            '''

            show_masks_isaid(np.array(label.squeeze(0).detach().cpu().numpy()), plt.gca())
            plt.show()
            show_masks_isaid(result.squeeze(0).detach().cpu().numpy(), plt.gca())
            plt.show()
            show_masks_isaid(np.array(prompt.squeeze(0).detach().cpu().numpy()), plt.gca())
            plt.show()
            '''     
            
            '''


        # 计算评估指标
        metrics = iou_metric.compute_metrics(iou_metric.results)
        # 输出官方评估结果
        print(metrics)

        # 自己写的稍微有点问题，偏低
        # print(f'SAM的输出miou{sam_miou / len(test_loader)}')
        # print(f'SAM的输出accuracy{sam_accuracy / len(test_loader)}')
        # showiou(sam_iou_list)
        # print(f'原始的输出miou{prompt_miou / len(test_loader)}')
        # print(f'原始的输出accuracy{prompt_accuracy / len(test_loader)}')
        # showiou(prompt_iou_list)



