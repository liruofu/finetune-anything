import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import cv2
import sys
import random
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


sam_checkpoint = "./checkpoint/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cpu"     # "cuda""cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

# 将输入的图像进行编码
image_path = 'test/potsdam_1024_RGB.tif'
image = cv2.imread('test/potsdam_1024_RGB.tif')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)


#由mask转为框 多个ok
def multi_mask_to_box_prompt():

    label_type="5"
    label_predict_image = cv2.imread('test/spilt_label_seunet/image_'+label_type+'.png')
    label_predict_img = cv2.cvtColor(label_predict_image, cv2.COLOR_BGR2GRAY)

    label_real_image = cv2.imread('test/spilt_label/image_'+label_type+'.png')
    label_real_img = cv2.cvtColor(label_real_image, cv2.COLOR_BGR2GRAY)

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(label_predict_img, connectivity=8)


    #删掉太小的
    data_array = stats[1:, :-1]
    rows_to_delete = np.where((data_array[:, 2] < 15) & (data_array[:, 3] < 15))[0]

    modified_array = np.delete(data_array, rows_to_delete, axis=0)
    # 对每一行进行操作，得到框
    modified_array[:, 2] = modified_array[:, 0] + modified_array[:, 2]
    modified_array[:, 3] = modified_array[:, 1] + modified_array[:, 3]


    # modified_array = np.array([[0, 162, 201, 440],
    #                 [294, 320, 1008, 773], ])

    print(modified_array.shape)

    print(modified_array)


    input_boxes = torch.tensor(
        modified_array

    , device=predictor.device)


    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    print(masks.shape)  # x H x W

    plt.figure(figsize=(10, 10))
    plt.imshow(label_real_img)
    # for mask in masks:
    #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
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
    for box in input_boxes:
        show_box(box.cpu().numpy(), plt.gca())
    plt.axis('off')
    plt.show()
#multi_mask_to_box_prompt()


def multi_mask_to_box_point_prompt():
    label_type = "4"
    label_predict_image = cv2.imread('test/spilt_label_seunet/image_' + label_type + '.png')
    label_predict_img = cv2.cvtColor(label_predict_image, cv2.COLOR_BGR2GRAY)

    label_real_image = cv2.imread('test/spilt_label/image_' + label_type + '.png')
    label_real_img = cv2.cvtColor(label_real_image, cv2.COLOR_BGR2GRAY)
    point_num_each_mask= 10
    remove_size =50
    padding_size =20
    image_size =1024


    retval, labels_matrix, stats, centroids = cv2.connectedComponentsWithStats(label_predict_img, connectivity=8)

    print(stats.shape)

    data_array = stats[1:, :-1]
    rows_to_delete = np.where((data_array[:, 2] < remove_size) & (data_array[:, 3] < remove_size))[0]
    remain_labels = np.setdiff1d(np.arange(retval - 1), rows_to_delete) + 1  # 剩下mask对应的label

    all_selected_points = []
    result_coordinates = {value: [] for value in remain_labels}
    nonzero_coords = np.argwhere(np.isin(labels_matrix, remain_labels))

    # 将坐标按照标签分组
    for coord in nonzero_coords:
        value = labels_matrix[coord[0], coord[1]]
        result_coordinates[value].append((coord[1], coord[0]))

    # 输出结果
    for value in remain_labels:
        if value in result_coordinates:
            points = random.sample(result_coordinates[value], min(point_num_each_mask, len(result_coordinates[value])))
            print(f"值为{value}的点坐标：{points}")
            all_selected_points.append(points)

    all_selected_points_arr = np.array(all_selected_points)


    modified_array = np.delete(data_array, rows_to_delete, axis=0)
    modified_labels = np.ones(modified_array.shape[0]*point_num_each_mask)

    print(modified_array.shape)

    modified_array[:, 2] = modified_array[:, 0] + modified_array[:, 2]
    modified_array[:, 3] = modified_array[:, 1] + modified_array[:, 3]

    modified_array[:, 0:2] = np.maximum(modified_array[:, 0:2] - padding_size, 0)
    modified_array[:, 2:4] = np.minimum(modified_array[:, 2:4] + padding_size, image_size)
    reshape_labels = modified_labels.reshape(-1, point_num_each_mask)



    point_coords = torch.from_numpy(all_selected_points_arr).to('cpu')
    point_labels = torch.from_numpy(reshape_labels).to('cpu')


    input_boxes = torch.tensor(
        modified_array

    , device=predictor.device)

    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=point_coords,
        point_labels= point_labels,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    print(masks.shape)  # x H x W
    print(masks)

    plt.figure(figsize=(10, 10))
    plt.imshow(label_real_img)
    # for mask in masks:
    #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for mask in masks:
        mask = mask.squeeze(0).cpu().numpy().astype(np.uint8) * 255

        mask = mask.astype(bool)
        show_mask(mask, plt.gca())
    for box in input_boxes:
        show_box(box.cpu().numpy(), plt.gca())
    show_points(all_selected_points_arr,reshape_labels, plt.gca())

    plt.axis('off')
    plt.show()
multi_mask_to_box_point_prompt()