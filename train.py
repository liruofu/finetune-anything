from extend_sam import BaseExtendSam
import argparse
import OmegaConf
from torch.utils.data import DataLoader
from .datasets import get_dataset
from .utils import get_losses

supported_tasks = ['detection', 'semantic_seg', 'instance_seg']
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', default='instance_seg', type=str)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--cfg', default='./config/coco_instance.yaml', type=str)

if __name__ == '__main__':
    config = OmegaConf.load(parser.cfg)
    task_name = parser.task_name
    if task_name not in supported_tasks:
        print("Please input the supported task name.")
    dataset = get_dataset(config.dataset_name, parser.data_dir)
    data_loader = DataLoader(dataset, batch_size=config.bs, shuffle=True, num_workers=config.num_workers,
                             drop_last=config.drop_last)
    losses = get_losses(config.loss)
    model = BaseExtendSam()
