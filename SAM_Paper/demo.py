'''
adapting sam for segmentation refinement
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from mmseg.apis import inference_model, init_model
from torchvision import transforms
from extend_sam import BaseExtendSam
from utils import Accuracy, compute_miou,show_masks_potsdam, show_masks_loveda, show_masks_isaid
from  iou_metric import IoUMetric


def showiou(list):
    miou = []
    for i in range(len(list)):
        if list[i]==[]:
            miou.append("没有这一类")
        else:
            miou.append(sum(list[i]) / len(list[i]))
    for i in range(len(miou)):
        print(f'第{i + 1}类的miou是{miou[i]}')

def net(img,model):
    for param in model.parameters():
        param.requires_grad = False
    result = inference_model(model, img)
    return result


if __name__ == '__main__':

    model = init_model("C:/Users/86159/PycharmProjects/mmsegmentation/mmsegmentation-main/configs/"
                       "segformer/segformer_mit-b5_8xb2-160k_ade20k-512x512.py",
                       "C:/Users/86159/PycharmProjects/mmsegmentation/iter_30000.pth", device='cuda')

    image_path = "C:/Users/86159/Desktop/loveda/img_dir/val/2553.png"
    label_path = "C:/Users/86159/Desktop/loveda/ann_dir/val/2553.png"

    sam_checkpoint = "C:/Users/86159/PycharmProjects/sam/sam_vit_b_01ec64.pth"
    device = "cuda"
    model_type = "vit_b"
    SAM = BaseExtendSam(ckpt_path=sam_checkpoint, model_type=model_type,
                        class_num=8, fix_prompt_en=True, fix_img_en=True, fix_mask_de=True).to(device)

    SAM.load_state_dict(torch.load("SAM_loveda.pth"))


    compute1 = compute_miou(7)  #数据集类别数
    compute2 = compute_miou(7)



    label =  torch.from_numpy(np.array(Image.open(label_path))).to('cuda')

    image = Image.open(image_path)
    image = transforms.Compose([transforms.ToTensor(),transforms.Resize(1024)])(image).to(device).unsqueeze(0)

    prompt = net(image_path, model).pred_sem_seg.data+torch.tensor(1)#
    #outputs = net(image_path, model).pred_sem_seg.data


    result = SAM(image, prompt, label.unsqueeze(0).shape[1:])
    result = torch.argmax(result, dim=1)

    label = label.squeeze(1).to(torch.int64)
    sam_accuracy = Accuracy(result,label)
    prompt_accuracy = Accuracy(prompt, label)

    sam_miou, sam_iou_list = compute1(result, label)
    prompt_miou, prompt_iou_list= compute2(prompt.squeeze(0), label)

    #可视化输出
    show_masks_loveda(np.array(label.squeeze(0).detach().cpu().numpy()), plt.gca())
    plt.show()
    show_masks_loveda(result.squeeze(0).detach().cpu().numpy(), plt.gca())
    plt.show()
    show_masks_loveda(np.array(prompt.squeeze(0).detach().cpu().numpy()), plt.gca())
    plt.show()

    # 测试数据集类别
    dataset_meta =  {
    'classes': ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7'],  # 类别名称列表
    'num_classes': 7 # 类别数量
}
    iou_metric = IoUMetric(dataset_meta=dataset_meta)

    data_samples=[{'pred_sem_seg': result, 'gt_sem_seg': label}]
    iou_metric.process(data_batch=1, data_samples=data_samples)
    # 计算评估指标
    metrics = iou_metric.compute_metrics(iou_metric.results)
    # 输出评估结果
    print(metrics)

    # 单张图片单类iou应该没问题
    print(f'SAM的输出miou {sam_miou}')
    print(f'SAM的输出accuracy {sam_accuracy}')
    showiou(sam_iou_list)
    print(f'原始的输出miou {prompt_miou}')
    print(f'原始的输出accuracy {prompt_accuracy}')
    showiou(prompt_iou_list)



