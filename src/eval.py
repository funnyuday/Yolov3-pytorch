# coding:utf-8
import os
import pdb
import yaml
import torch
import argparse
import logging
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from pathlib import Path
from data import YoloLoader
from utils.norm_trainer import Tester
from model.yolo.deploy import eval_yolo
from utils.general import increment_path, set_logging

logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_root', default='/data/zhengdong/Detection/UA-DETRAC/Insight-MVT_Annotation_Test/')
    parser.add_argument('--data_dir', default='../data/UA-DETRAC_test.txt')
    parser.add_argument('--project', default='../test')
    parser.add_argument('--name', default='car_detect_yolov3')
    parser.add_argument('--cfg', default='../config/yolov3.cfg')
    parser.add_argument('--names', default='../data/car.names')
    parser.add_argument('--model_path', default='../config/yolov3.weights')
    parser.add_argument('--mode', default='Test')
    parser.add_argument('--img_size', default=(416, 416))
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--model_type', default='YOLOv3')
    parser.add_argument('--shuffle', default=False)
    parser.add_argument('--load_thread', default=4)
    parser.add_argument('--no_letter', default=False)
    parser.add_argument('--conf_thr', default=0.3)
    parser.add_argument('--nms_thr', default=0.45)
    parser.add_argument('--iou_thr', default=0.40)
    parser.add_argument('--gpu_id', default=0)
    parser.add_argument('--draw_res', default=True)
    args = parser.parse_args()
    return args



def main(opt):
    set_logging()
    # Show GPUs info
    s = f"Yolov3 detection eval factory torch {torch.__version__}"
    n = torch.cuda.device_count()
    if n > 1 and opt.batch_size:
        assert opt.batch_size % n ==0, f"batch-size {opt.batch_size} not multiple of GPU count {n}"
    p = torch.cuda.get_device_properties(opt.gpu_id)
    s += f" CUDA:{opt.gpu_id} ({p.name}, {p.total_memory / 1024 **2}MB)\n"
    logger.info(s)
    # Make Diectories
    wdir = opt.save_dir
    opt.accu_detail_file = opt.save_dir / 'eval.xls'
    wdir.mkdir(parents=True, exist_ok=True)
    # Save run setings
    with open(opt.save_dir /"opt.yaml", "w") as f:
        opt.save_dir = str(opt.save_dir)
        yaml.dump(vars(opt), f, sort_keys=False)
    # Build Tester
    tester = Tester(opt)
    tester.model.cuda(opt.gpu_id)
    # Data Loader
    data_loader = YoloLoader(opt)
    test_loader = data_loader.GetDataset()
    # Test
    cudnn.benchmark = True
    ns = len(test_loader)
    pbar = enumerate(test_loader)
    pbar = tqdm(pbar, total=ns, desc="Testing")
    eval_yolo(opt, pbar, tester.model)

if __name__=="__main__":
    opt = parser()
    opt.save_dir = increment_path(Path(opt.project) / opt.name, sep="_")
    main(opt=opt)