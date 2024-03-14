
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
from sampaper_new.sampaper.extend_sam.extend_sam_post import BaseExtendSam
from mmseg.apis import inference_model, init_model
import torch.nn.functional as F
from utils import mask_to_binary_images, show_masks_potsdam, fill_holes, show_masks_isaid
import matplotlib.pyplot as plt


#提示模型
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


def train_net(SAM, device, file_path, epochs=3, batch_size=1, lr=1e-4):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # train_val_loader= DataLoader(train_val_dataset, batch_size=1, shuffle=True)
    # 定义损失函数和优化器
    optimizer = torch.optim.SGD(SAM.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(SAM.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.3)           # 一些可调整的参数与函数
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # criterion=nn.BCEWithLogitsLoss()
    # criterion=dice_ce_loss()

    # 训练模型
    for epoch in range(epochs):
        total_train_loss = 0
        total_train_val_loss = 0
        accuracy = 0
        losss = 0

        for i, (inputs, image_path, labels) in enumerate(train_loader, 0):

            if torch.max(labels).item() == 255:
                continue

            # 获取提示预测图，通道数为1
            prompt = net(image_path, model)[0].pred_sem_seg.data + torch.tensor(1)
            SAM.train()
            inputs, labels = inputs.to(device), labels.to(device)

            # print("label_max:", torch.max(labels).item())

            optimizer.zero_grad()

            # sam输入为图片、提示与label大小，输出结果与label大小保持一致
            # g in the forward function
            pred = (net(image_path, model)[0].seg_logits.data)[0:6, :, :]

            result = SAM(inputs, prompt, labels.shape[1], pred)
            # outputss = SAM(inputs, prompt, labels.shape[1:])

            if i % 50 == 0:
                show_masks_potsdam(np.array(labels.squeeze(0).detach().cpu().numpy()), plt.gca())
                plt.show()
                show_masks_potsdam(prompt.squeeze(0).detach().cpu().numpy(), plt.gca())
                plt.show()
                show_masks_potsdam(torch.argmax(result, dim=1).squeeze(0).detach().cpu().numpy(), plt.gca())
                plt.show()
                torch.save(SAM.state_dict(), file_path + str(epoch) + str(i) + 'SAM_potsdam.pth')

            # print(torch.max(torch.argmax(outputss, dim=1), dim=1), torch.max(labels.long(), dim=1))

            loss = criterion(result, labels.long())

            losss += loss.item()
            total_train_loss += loss.item()

            # print(torch.max(outputss).item())
            # print(outputss)

            if torch.max(result).item() == 0:
                continue
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, batch {i+1} loss: {loss.item()} ')
            # 保存模型
            if (i+1) % 100 == 0:
                print(f'Epoch {epoch + 1}, batch {i+1}, average loss: {losss/100} ')
                losss = 0
            if (i+1) % 100 == 0:
                with open(file_path + 'SAM' + '_train_' + 'log.txt', 'a') as file:
                    file.write(f'Epoch {epoch + 1}, batch {i+1}, train loss: {total_train_loss / 1000} \n')
                total_train_loss = 0

        scheduler.step()
        torch.save(SAM.state_dict(), file_path + '/SAM_potsdam_post.pth')


if __name__ == '__main__':

    # 数据集路径
    data_root = "/home/yelu/ICPR_CV/potsdam/cropped_Potsdam"

    # 初始化数据集
    train_dataset = PotsdamSemanticDataset(data_root, 'train')
    # train_val_dataset = PotsdamSemanticDataset(data_root, 'val')
    # 指定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建模型并移动到设备

    # 利用mmsegmentation初始化提示分割模型
    model = init_model(
        "/home/yelu/ICPR_CV/result/swin_postdam/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py",
        "/home/yelu/ICPR_CV/result/swin_postdam/iter_160000.pth", device='cuda')

    # 初始化sam
    SAM = BaseExtendSam(ckpt_path="/home/yelu/ICPR_CV/sampaper_new/sampaper/sam_vit_b_01ec64.pth",
                        class_num=7, fix_prompt_en=False, fix_img_en=True, fix_mask_de=False).to(device)
    # SAM.load_state_dict(torch.load("SAM_potsdam.pth"))

    file_path = "/home/yelu/ICPR_CV/fast/se_dual_postprocess_pots_point15_padding5/"
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)


    start = time.time()

    train_net(SAM, device, file_path)

    end = time.time()
    print(f'训练时间为 {end-start}s')

    # 训练模型












