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
from dsm_dataset import PotsdamSemanticDataset,LovedaSemanticDataset
from utils import Accuracy, compute_miou, show_masks_potsdam, show_masks_isaid, show_masks_loveda
from  iou_metric_potsdam import IoUMetric

#提示模型
def net(img, model):
    for param in model.parameters():
        param.requires_grad = False
    result = inference_model(model, img)
    return result

#计算每类iou，有点问题
def showiou(list):
    miou = []
    for i in range(len(list)):
        if list[i]==[]:
            miou.append("没有这一类")
        else:
            miou.append(sum(list[i]) / len(list[i]))
    for i in range(len(miou)):
        print(f'第{i + 1}类的miou是{miou[i]}')


if __name__ == '__main__':

    model = init_model(
        "/home/yelu/ICPR_CV/result/swin_postdam/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py",
        "/home/yelu/ICPR_CV/result/swin_postdam/iter_160000.pth", device='cuda')
    test_dir ="/home/yelu/ICPR_CV/potsdam/cropped_Potsdam"

    # 初始化数据集
    test_dataset = PotsdamSemanticDataset(test_dir, 'val')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = "cuda"
    model_type = "vit_b"
    SAM = BaseExtendSam(ckpt_path="/home/yelu/ICPR_CV/sampaper_new/sampaper/sam_vit_b_01ec64.pth",
                        class_num=7, fix_prompt_en=False, fix_img_en=True, fix_mask_de=False).to(device)

    SAM.load_state_dict(torch.load('/home/yelu/ICPR_CV/result/dsm_train_log/sam_segb0_fine_p_d_b_m.pth'))


    prompt_miou = 0
    sam_miou = 0
    sam_accuracy = 0
    prompt_accuracy = 0

    compute1 = compute_miou(7)
    compute2 = compute_miou(7)

    sam_iou_list = []
    prompt_iou_list = []

    dataset_meta = test_dataset.dataset_meta
    iou_metric = IoUMetric(dataset_meta=dataset_meta)

    dataset_meta = test_dataset.dataset_meta
    iou_metric = IoUMetric(dataset_meta=dataset_meta)
    iou_metric_ori = IoUMetric(dataset_meta=dataset_meta)

    with torch.no_grad():
        data_samples = []
        SAM.eval()
        for j, (input, path, label,dsm) in enumerate(test_loader, 0):
            print(j)

            input, path, label,dsm = input.to(device), path, label.to(device),(dsm.to(device).permute(0,3,1,2))[:,0,:,:]
            prompt = net(path, model)[0].pred_sem_seg.data + torch.tensor(1)

            # g in the forward function
            pred = (net(path, model)[0].seg_logits.data)[0:6, :, :]

            result = SAM(input, prompt, label.shape[1],dsm)
            result[:,1:,:,:]=torch.sigmoid(result[:,1:,:,:])
            # result = SAM(input, prompt, label.shape[1:])
            result = torch.argmax(result, dim=1).to('cuda')

            # 调用mmsegmentation官方miou计算函数
            data_samples = [{'pred_sem_seg': result, 'gt_sem_seg': label}]
            data_samples_ori = [{'pred_sem_seg': prompt, 'gt_sem_seg': label}]

            iou_metric.process(data_batch=1, data_samples=data_samples)
            iou_metric_ori.process(data_batch=1, data_samples=data_samples_ori)

            sam_accuracy += Accuracy(result, label)
            prompt_accuracy += Accuracy(prompt, label)

            label = label.squeeze(1).to(torch.int64)

            sam_iou, sam_iou_list = compute1(result, label)
            sam_miou += sam_iou

            prompt_iou, prompt_iou_list = compute2(prompt, label)
            prompt_miou += prompt_iou

            # 计算评估指标
            # metrics = iou_metric.compute_metrics(iou_metric.results)
            # # 输出官方评估结果
            # print(metrics)
            '''

            '''

            '''      
                        if j % 16 == 0:
                show_masks_potsdam(np.array(label.squeeze(0).detach().cpu().numpy()), plt.gca())
                plt.show()
                show_masks_potsdam(result.squeeze(0).detach().cpu().numpy(), plt.gca())
                plt.show()
                show_masks_potsdam(np.array(prompt.squeeze(0).detach().cpu().numpy()), plt.gca())
                plt.show()
            '''

        # 计算评估指标
        metrics = iou_metric.compute_metrics(iou_metric.results)
        metrics_ori = iou_metric_ori.compute_metrics(iou_metric_ori.results)
        # 输出官方评估结果
        print("Sam result:")
        print(metrics)
        print("forward model result:")
        print(metrics_ori)




