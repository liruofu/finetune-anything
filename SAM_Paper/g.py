import torch
from mmseg.evaluation import IoUMetric
from mmseg.models import build_segmentor


iou_metric = IoUMetric()

# 处理数据
iou_metric.process({'pred_sem_seg': pred_label}, [{'gt_sem_seg': label}])

# 计算评估指标
metrics = iou_metric.compute_metrics(iou_metric.results)

# 输出评估结果
print(metrics)
