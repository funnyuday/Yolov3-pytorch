# coding:utf-8
import argparse

class Option():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # 路径
        self.parser.add_argument('--data_dir', default='../data/UA-DETRAC.txt')
        self.parser.add_argument('--project', default='../train')
        self.parser.add_argument('--name', default='car_detect_yolov3_small')
        self.parser.add_argument('--cfg', default='../config/yolov3_small.cfg')
        self.parser.add_argument('--pretrained', default='')
        # 模型参数
        self.parser.add_argument('--mode', default='Train')
        self.parser.add_argument('--img_size', default=(416, 416))
        self.parser.add_argument('--batch_size', default=128)
        self.parser.add_argument('--model_type', default='YOLOv3')
        self.parser.add_argument('--optim_type', default='SGD')
        self.parser.add_argument('--lr_scheduler', default='CosineAnnealingLR')
        self.parser.add_argument('--step_size', default=(30))
        # 数据集参数
        self.parser.add_argument('--shuffle', default=True)
        self.parser.add_argument('--load_thread', default=8)
        self.parser.add_argument('--no_letter', default=False)
        self.parser.add_argument('--mosaic', default=False)
        # 训练超参数
        self.parser.add_argument('--dist', default=False)
        self.parser.add_argument('--local_rank', default=-1)
        self.parser.add_argument('--gpu_ids', default=[0])
        self.parser.add_argument('--warm_up', default=True)
        self.parser.add_argument('--initial_lr', type=float, default=0.000001)
        self.parser.add_argument('--warm_up_epoch', default=5)
        self.parser.add_argument('--max_epoch', default=120)
        self.parser.add_argument('--save_epoch', default=50)
        self.parser.add_argument('--lr', type=float, default=0.001)

    def parse(self):
        opt = self.parser.parse_args()
        opt.world_size = len(opt.gpu_ids)
        opt.lr *= len(opt.gpu_ids)
        return opt
