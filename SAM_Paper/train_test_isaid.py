
import cv2
import os
import numpy as np
from torch import nn
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import time
from dataset import PotsdamSemanticDataset, IsaidSemanticDataset
import torchvision.transforms as T
from extend_sam import BaseExtendSam
from mmseg.apis import inference_model, init_model
import torch.nn.functional as F
from utils import mask_to_binary_images, show_masks_potsdam, fill_holes, show_masks_isaid
import matplotlib.pyplot as plt
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


def Accuracy(output, label):
    # 将输出转换为预测的类别
    pred = torch.argmax(output, dim=1)

    # 计算预测正确的像素数
    correct_pixels = torch.sum(pred == label)
    # 计算总的像素数
    total_pixels = label.numel()
    # 计算精度
    accuracy = correct_pixels.float() / total_pixels
    return accuracy


def train_net(SAM, device, file_path, epochs=1, batch_size=1, lr=1e-4):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # train_val_loader= DataLoader(train_val_dataset, batch_size=1, shuffle=True)
    # 定义损失函数和优化器
    optimizer = torch.optim.SGD(SAM.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(SAM.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)           #一些可调整的参数与函数
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    loss_threshold = 0.8
    loss_counter = 0
    times_threshold = 20

    # criterion=nn.BCEWithLogitsLoss()
    # criterion=dice_ce_loss()

    # 训练模型
    for epoch in range(epochs):
        total_train_loss = 0
        total_train_val_loss = 0
        accuracy = 0
        losss = 0
        miou = 0


        for i, (inputs, image_path, labels) in enumerate(train_loader, 0):

            #if torch.max(labels).item() == 255:
            #    continue

            # 获取提示预测图，通道数为1
            prompt = net(image_path, model)[0].pred_sem_seg.data

            # 比较生成图片与label类别是否相同
            # show_masks_isaid(np.array(labels.squeeze(0).detach().cpu().numpy()), plt.gca())
            # plt.show()
            # show_masks_isaid(prompt.squeeze(0).detach().cpu().numpy(), plt.gca())
            # plt.show()

            SAM.train()
            inputs, labels = inputs.to(device), labels.to(device)

            # print("label_max:", torch.max(labels).item())

            optimizer.zero_grad()

            # sam输入为图片、提示与label大小，输出结果与label大小保持一致
            # 此处调用的是extend_sam中的forward函数
            outputss = SAM(inputs, prompt, labels.shape[1:])

            # print(torch.max(torch.argmax(outputss, dim=1), dim=1), torch.max(labels.long(), dim=1))

            loss = criterion(outputss, labels.long())

            losss += loss.item()
            #total_train_loss += loss.item()

            # print(torch.max(outputss).item())
            # print(outputss)

            if torch.max(outputss).item() == 0:
                continue
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, batch {i+1} loss: {loss.item()} ')
            # 保存模型
            if (i+1) % 100 == 0:
                averageLoss=losss/100
                print(f'Epoch {epoch + 1}, batch {i+1}, average loss: {averageLoss} ')
                with open(file_path+'SAM'+'_train_'+'log.txt','a') as file:
                    file.write(f'Epoch {epoch+1}, batch{i+1}, train loss:{averageLoss}')
                if (averageLoss>loss_threshold):
                    loss_counter=0
                else :
                    loss_counter+=1
                if(loss_counter>times_threshold):
                    print("Training is Done")
                    break

                losss = 0
            #if (i+1) % 100 == 0:

                #with open(file_path + 'SAM' + '_train_' + 'log.txt', 'a') as file:
                #    file.write(f'Epoch {epoch + 1}, batch {i+1}, train loss: {total_train_loss / 1000} \n')
                #total_train_loss = 0

        scheduler.step()
        torch.save(SAM.state_dict(), file_path + "SAM_potsdam.pth")


def test_net(file_path):
    model = init_model("/home/yelu/ICPR_CV/mmsegmentation-main/configs/"
                       "segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py",
                       "/home/yelu/ICPR_CV/result/segformer_isaid/iter_240000.pth", device='cuda')

    test_dir = "/home/yelu/ICPR_CV/cropped"

    test_dataset = IsaidSemanticDataset(test_dir, 'val')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    sam_checkpoint = "/home/yelu/ICPR_CV/sampaper_new/sampaper/sam_vit_b_01ec64.pth"
    device = "cuda"
    model_type = "vit_b"
    SAM = BaseExtendSam(ckpt_path=sam_checkpoint, model_type=model_type,
                        class_num=16, fix_prompt_en=True, fix_img_en=True, fix_mask_de=True).to(device)

    SAM.load_state_dict(torch.load(file_path + "SAM_potsdam.pth"))

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
            prompt = net(path, model)[0].pred_sem_seg.data  # + torch.tensor(1)

            result = SAM(input, prompt, label.shape[1:])
            result = torch.argmax(result, dim=1).to('cuda')

            # 调用mmsegmentation官方miou计算函数
            data_samples = [{'pred_sem_seg': result, 'gt_sem_seg': label}]
            iou_metric.process(data_batch=1, data_samples=data_samples)

            sam_accuracy += Accuracy(result, label)
            prompt_accuracy += Accuracy(prompt, label)

            label = label.squeeze(1).to(torch.int64)

            sam_iou, sam_iou_list = compute1(result, label)
            sam_miou += sam_iou

            prompt_iou, prompt_iou_list = compute2(prompt, label)
            prompt_miou += prompt_iou

        # 计算评估指标
        metrics = iou_metric.compute_metrics(iou_metric.results)
        # 输出官方评估结果
        print(metrics)


if __name__ == '__main__':

    # 数据集路径
    data_root = "/home/yelu/ICPR_CV/cropped"

    # 初始化数据集
    train_dataset = IsaidSemanticDataset(data_root, 'train')
    # train_val_dataset = PotsdamSemanticDataset(data_root, 'val')
    # 指定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建模型并移动到设备

    # 利用mmsegmentation初始化提示分割模型
    model = init_model("/home/yelu/ICPR_CV/mmsegmentation-main/configs/"
                       "segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py",
                       "/home/yelu/ICPR_CV/result/segformer_isaid/iter_240000.pth", device='cuda')

    # 初始化sam
    SAM = BaseExtendSam(ckpt_path="/home/yelu/ICPR_CV/sampaper_new/sampaper/sam_vit_b_01ec64.pth",
                        class_num=16, fix_prompt_en=True, fix_img_en=True, fix_mask_de=False).to(device)
    # SAM.load_state_dict(torch.load("SAM_potsdam.pth"))

    # 文件存储路径
    file_path = "/home/yelu/ICPR_CV/result/sam/isaid/dif_prompt/segformer_10_1_B_Con/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    start = time.time()

    train_net(SAM, device, file_path)

    test_net(file_path)

    end = time.time()
    print(f'训练时间为 {end-start}s')

    # 训练模型
