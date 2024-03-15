'''

'''
import random
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


class compute_miou(nn.Module):
    def __init__(self, num):
        super(compute_miou, self).__init__()
        self.num = num
        self.iou_lists = [[] for _ in range(num)]

    def forward(self, y_pred, y_true):

        iou = []
        for label in range(1, self.num+1):
                intersection = torch.logical_and(y_true == label, y_pred == label).sum().float()
                union = torch.logical_or(y_true == label, y_pred == label).sum().float()
                if  (y_true == label).sum().float() > 0:
                    iou.append((intersection + 1e-6) / (union + 1e-6))
                    self.iou_lists[label-1].append((intersection + 1e-6) / (union + 1e-6))

        # 计算mIoU
        #print(iou_list)
        mIoU = sum(iou) /len(iou)
        return mIoU.item(), self.iou_lists


def Accuracy(output, label):
    # 将输出转换为预测的类别
    #pred = torch.from_numpy(output)
    # 计算预测正确的像素数
    correct_pixels = torch.sum(output == label)
    # 计算总的像素数
    total_pixels = label.numel()
    # 计算精度
    accuracy = correct_pixels.float() / total_pixels
    return accuracy


def show_masks_isaid(mask, ax, color_map=None):
    if color_map is None:
        color_map = {
            1: (0, 0, 63),  # 船舶
            2: (0, 63, 63),  # 储罐
            3: (0, 63, 0),  # 棒球场
            4: (0, 63, 127),  # 网球场
            5: (0, 63, 191),  # 篮球场
            6: (0, 63, 255),  # 田径场
            7: (0, 127, 63),  # 桥梁
            8: (0, 127, 127),  # 大型车辆
            9: (255,255, 200),  # 小型车辆
            10: (0, 0, 191),  # 直升机
            11: (0, 0, 255),  # 游泳池
            12: (0, 191, 127),  # 环岛
            13: (0, 127, 191),  # 足球场
            14: (255, 127, 255),  # 飞机
            15: (0, 100, 155)  # 港口
        }

    h, w = mask.shape[-2:]
    mask_image = np.zeros((h, w, 3), dtype=np.uint8) # 创建一个和 mask 相同大小的 RGB图像

    for value, color in color_map.items():
        mask_image[mask == value] = color

    ax.imshow(mask_image)


def show_masks_loveda(mask, ax, color_map=None):
    if color_map is None:
        color_map = {
            0: [0, 0, 0],  # ignore
            1: [255, 248, 220],  # background
            2: [100, 149, 237],  # building
            3: [102, 205, 170],  # road
            4: [205, 133, 63],  # water
            5: [160, 32, 240],  # barren
            6: [255, 64, 64],  # forest
            7: [139, 69, 19],  # agriculture
        }

    h, w = mask.shape[-2:]
    mask_image = np.zeros((h, w, 3), dtype=np.uint8) # 创建一个和 mask 相同大小的 RGB图像

    for value, color in color_map.items():
        mask_image[mask == value] = color

    ax.imshow(mask_image)


def show_masks_potsdam(mask, ax, color_map=None):
    if color_map is None:
        color_map = {
            0: [0, 0, 0],
            1: [255, 255, 255],  # Impervious surfaces
            2: [0, 0, 255 ],  # Building
            3: [0, 255, 255],  # Low vegetation
            4: [0, 255, 0],  # Tree
            5: [255, 255, 0],  # Car
            6: [255, 0, 0],  # Clutter/background
        }

    h, w = mask.shape[-2:]
    mask_image = np.zeros((h, w, 3), dtype=np.uint8) # 创建一个和 mask 相同大小的 RGB图像

    for value, color in color_map.items():
        mask_image[mask == value] = color

    ax.imshow(mask_image)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):

    if coords!=[]:

        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)


def show_boxes(boxes, ax):
    if boxes !=[]:
        for box in boxes:
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def mask_to_binary_images(mask,num):
    num_classes = num # 类别数，包括背景类别
    binary_images = np.zeros((num_classes, 1, 1024, 1024), dtype=np.uint8)  # 创建用于存储二值图像的数组

    for i in range(num_classes):
        binary_images[i] = (mask == i).astype(np.uint8)  # 将掩码值为 i 的像素转换为二值图像

    return binary_images


def fill_holes(mask):
    mask_filled = np.zeros_like(mask)
    for label in range(0, 7):  # 假设像素值的范围是1到6
        mask_label = (mask == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_label.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_filled, contours, -1, label, thickness=cv2.FILLED)
        # 返回填充后的结果
    mask_filled = mask_filled.astype(np.uint8)

    return mask_filled

def to_result(final_masks,outputs):
    result = np.zeros((1024, 1024), dtype=int)
    # 找到只有一个1的像素点的索引
    single_indices = np.sum(final_masks, axis=0) == 1
    result[single_indices] = np.argmax(final_masks, axis=0)[single_indices]
    # 找到有多个1的像素点的索引
    multiple_indices = (np.sum(final_masks, axis=0) > 1) | (np.sum(final_masks, axis=0) == 0)
    result[multiple_indices] = np.array(outputs.squeeze(0).detach().cpu().numpy())[multiple_indices]
    result = fill_holes(result)
    return result


# TODO: modify params when changing the dataset！
def draw_rectangles_and_points_on_masks(all_points, labels, all_rectangles, masks, min_width=20, min_height=20, point_number=15):
    num_masks, _, height, width = masks.shape  # remained to be modified

    for i in range(num_masks):
        rectangles = []
        point = []
        la = []
        mask = masks[i, 0]  # 获取当前掩码
        point_num_each_mask = 10 #选取多少个正例样本
        remove_size = 20 #小于多少就扔掉
        padding_size = 5 #在外扩一定像素
        image_size = 1024

        #print("i:",i)

        mask_uint8 = (mask * 255).astype(np.uint8)  # 将二值掩码转换为 uint8 类型
        # mask生成连通域
        retval, labels_matrix, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        #print("number:",retval)

        ## 为1说明这一类没有分割结果，跳过
        if(retval==1):
            #print("zero!")
            all_points.append([])
            labels.append([])
            all_rectangles.append([])
            continue

        data_array = stats[1:, :-1]

        # i为6背景这类特殊处理
        if i==6:
            remove_size=5
            point_num_each_mask = 5
            padding_size=0
        #有这些连通域要删除
        rows_to_delete = np.where((data_array[:, 2] < remove_size) & (data_array[:, 3] < remove_size))[0]


        #保留的
        remain_labels = np.setdiff1d(np.arange(retval - 1), rows_to_delete) + 1  # 剩下mask对应的label
        #print(type(remain_labels))

        if remain_labels.shape[0]==0:
            all_points.append([])
            labels.append([])
            all_rectangles.append([])
            continue


        all_selected_points = []
        result_coordinates = {value: [] for value in remain_labels}
        nonzero_coords = np.argwhere(np.isin(labels_matrix, remain_labels))

        # 统计每个连通域中的点
        for coord in nonzero_coords:
            value = labels_matrix[coord[0], coord[1]]
            result_coordinates[value].append((coord[1], coord[0]))

        # 对每个连通域选取指定个数的点
        for value in remain_labels:
            if value in result_coordinates:
                points = random.sample(result_coordinates[value],
                                       min(point_num_each_mask, len(result_coordinates[value])))


                #print(f"值为{value}的点坐标：{points}")
                all_selected_points.extend(points)

        all_selected_points_arr = np.array(all_selected_points)
        #删掉不符合大小的连通域对应的框
        modified_array = np.delete(data_array, rows_to_delete, axis=0)
        #根据个数生成label
        modified_labels = np.ones(modified_array.shape[0] * point_num_each_mask)#label 0/1

        #print(modified_array.shape)
        # 计算出框
        modified_array[:, 2] = modified_array[:, 0] + modified_array[:, 2]
        modified_array[:, 3] = modified_array[:, 1] + modified_array[:, 3]
        # 外阔padding
        modified_array[:, 0:2] = np.maximum(modified_array[:, 0:2] - padding_size, 0)
        modified_array[:, 2:4] = np.minimum(modified_array[:, 2:4] + padding_size, image_size)
        reshape_labels = modified_labels.reshape(-1, point_num_each_mask)
        #print("point:",all_selected_points_arr.shape)
        #print("la:",reshape_labels.shape)
        #ccprint("box",modified_array.shape)

        all_points.append(all_selected_points_arr)
        labels.append(reshape_labels)
        all_rectangles.append(modified_array)
        # cv2.imshow(f'Mask {i}', mask_uint8)  # 显示带有矩形框的掩码
        cv2.waitKey(0)
        cv2.destroyAllWindows()




        #contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找到掩码中的轮廓

        # for contour in contours:
        #     x, y, w, h = cv2.boundingRect(contour)  # 获取轮廓的边界框
        #     if w >= min_width and h >= min_height:  # 添加过滤条件
        #         cv2.rectangle(mask_uint8, (x, y), (x + w, y + h), 255, 2)  # 在掩码上绘制矩形框
        #         x1, y1, x2, y2 = x, y, x + w, y + h
        #         if x1-5 > 0 and y1-5 > 0 and x2+5 < 1024 and y2+5 < 1024:
        #             rectangles.append([x1-5, y1-5, x2+5, y2+5])
        #         else:
        #            rectangles.append([x1, y1, x2, y2])
        #         #rectangles.append([x1, y1, x2, y2])

        # for rec in rectangles:
        #     points = []
        #     nonzero_points = np.transpose(np.nonzero(mask_uint8[rec[1]:rec[3], rec[0]:rec[2]]))
        #     points.extend([[px + rec[0], py + rec[1]] for py, px in nonzero_points])
        #
        #     random_indices = random.sample(range(len(points)), point_number)
        #     point.extend(points[i] for i in random_indices)
        #     la.extend([1] * point_number)
        #
        # all_points.append(point)
        # labels.append(la)
        # all_rectangles.append(rectangles)
        # # cv2.imshow(f'Mask {i}', mask_uint8)  # 显示带有矩形框的掩码
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

# def draw_rectangles_and_points_on_masks(all_points, labels, all_rectangles, masks, min_width=20, min_height=20, point_number=15):
#     num_masks, _, height, width = masks.shape  # remained to be modified
#
#     for i in range(num_masks):
#         rectangles = []
#         point = []
#         la = []
#         mask = masks[i, 0]  # 获取当前掩码
#
#         mask_uint8 = (mask * 255).astype(np.uint8)  # 将二值掩码转换为 uint8 类型
#         contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找到掩码中的轮廓
#
#         for contour in contours:
#             x, y, w, h = cv2.boundingRect(contour)  # 获取轮廓的边界框
#             if w >= min_width and h >= min_height:  # 添加过滤条件
#                 cv2.rectangle(mask_uint8, (x, y), (x + w, y + h), 255, 2)  # 在掩码上绘制矩形框
#                 x1, y1, x2, y2 = x, y, x + w, y + h
#                 if x1-5 > 0 and y1-5 > 0 and x2+5 < 1024 and y2+5 < 1024:
#                     rectangles.append([x1-5, y1-5, x2+5, y2+5])
#                 else:
#                    rectangles.append([x1, y1, x2, y2])
#                 #rectangles.append([x1, y1, x2, y2])
#
#         for rec in rectangles:
#             points = []
#             nonzero_points = np.transpose(np.nonzero(mask_uint8[rec[1]:rec[3], rec[0]:rec[2]]))
#             points.extend([[px + rec[0], py + rec[1]] for py, px in nonzero_points])
#
#             random_indices = random.sample(range(len(points)), point_number)
#             point.extend(points[i] for i in random_indices)
#             la.extend([1] * point_number)
#
#         all_points.append(point)
#         labels.append(la)
#         all_rectangles.append(rectangles)
#         # cv2.imshow(f'Mask {i}', mask_uint8)  # 显示带有矩形框的掩码
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

def draw_rectangles_and_points_on_masks_isaid(all_points, labels, all_rectangles, masks, min_width=20, min_height=20):
    num_masks, _, height, width = masks.shape  # remained to be modified

    for i in range(num_masks):
        rectangles = []
        point = []
        la = []
        mask = masks[i, 0]  # 获取当前掩码

        mask_uint8 = (mask * 255).astype(np.uint8)  # 将二值掩码转换为 uint8 类型
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找到掩码中的轮廓

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)  # 获取轮廓的边界框
            if w >= min_width and h >= min_height:  # 添加过滤条件
                cv2.rectangle(mask_uint8, (x, y), (x + w, y + h), 255, 2)  # 在掩码上绘制矩形框
                x1, y1, x2, y2 = x, y, x + w, y + h
                rectangles.append([x1, y1, x2, y2])

        for rec in rectangles:
            points = []
            nonzero_points = np.transpose(np.nonzero(mask_uint8[rec[1]:rec[3], rec[0]:rec[2]]))
            points.extend([[px + rec[0], py + rec[1]] for py, px in nonzero_points])

            random_indices = random.sample(range(len(points)), 10)
            point.extend(points[i] for i in random_indices)
            la.extend([1] * 10)

        all_points.append(point)
        labels.append(la)
        all_rectangles.append(rectangles)
        # cv2.imshow(f'Mask {i}', mask_uint8)  # 显示带有矩形框的掩码
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def change(final_masks, mask, x):
    result = np.zeros((1024, 1024), dtype=int)
    for i in range(1024):
        for j in range(1024):
            # 如果在 "5" 方向上有一个像素点是1，则将当前像素点设置为1
            if np.any(mask[:, i, j] == 1):
                result[i, j] = 1

    final_masks[x] = result


def multi_prompt(predictor, all_rectangles, all_points,labels, image, final_masks, i):
    if all_rectangles[i] != []:
        box = predictor.transform.apply_boxes_torch(torch.from_numpy(np.array(all_rectangles[i])), image.shape[:2]).to(
            'cuda')

    else:
        box = None
    if all_points[i] != []:
        point_coords = torch.reshape(torch.from_numpy(np.array(all_points[i])).to('cuda'), (box.shape[0], -1, 2))
        point_labels = torch.reshape(torch.from_numpy(np.array(labels[i])).to('cuda'), (box.shape[0], -1))

        print(point_coords.shape)
        print(point_labels.shape)
    else:
        point_coords = None
        point_labels = None

    masks, _, _ = predictor.predict_torch(
        #mask_input=torch.where(img == i, torch.tensor(1.).to('cuda'), torch.tensor(0.).to('cuda')).to('cuda'),
        point_coords=point_coords,
        point_labels=point_labels,
        #point_coords=None,
        #point_labels=None,
        boxes=box,
        multimask_output=False,
    )
    print(masks.shape)
    # x H x W

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        mask = mask.squeeze(0).cpu().numpy().astype(np.uint8) * 255
        # 找到轮廓
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 对每个轮廓进行填充
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
        # 将0-255范围内的值转换为True或False
        mask = mask.astype(bool)

        show_mask(mask, plt.gca())

    change(final_masks, masks.squeeze(1).cpu().numpy().astype(np.uint8), i)
    show_points(np.array(all_points[i]), np.array(labels[i]), plt.gca())
    show_boxes(all_rectangles[i], plt.gca())
    plt.axis('off')
    plt.show()


class dice_ce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_ce_loss, self).__init__()
        self.batch = batch
        self.relu=nn.ReLU()
        self.ce_loss = nn.CrossEntropyLoss()

    def soft_dice_coeff(self, y_true, y_pred):
        y_pred = self.relu(y_pred[:,0,:,:])
        smooth = 0.00001  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_pred,y_true):
        a = self.ce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return  a + b





