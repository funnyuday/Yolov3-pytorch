# coding:utf-8
import os
import cv2
import pdb
import yaml
import torch
import random
import argparse
import logging
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from pathlib import Path
from utils.norm_trainer import Tester
from model.yolo.deploy import predict_images
from model.yolo.utils import load_classes
from utils.general import increment_path, set_logging, gen_list

logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', default='/data0/zhengdong/project/Detection/data/videos')
    parser.add_argument('--project', default='../video_images')
    parser.add_argument('--name', default='labels')
    parser.add_argument('--cfg', default='../config/yolov3_coco.cfg')
    parser.add_argument('--names', default='../data/coco.names')
    parser.add_argument('--model_path', default='../config/yolov3_coco.weights')
    parser.add_argument('--mode', default='Test')
    parser.add_argument('--img_size', default=(608, 608))
    parser.add_argument('--type', default='video', choices=['videos', 'image'])
    parser.add_argument('--gen_txt', default=True)
    parser.add_argument('--show_names', default=['car', 'bus', 'truck'])
    parser.add_argument('--model_type', default='YOLOv3')
    parser.add_argument('--shuffle', default=False)
    parser.add_argument('--load_thread', default=4)
    parser.add_argument('--no_letter', default=False)
    parser.add_argument('--conf_thr', default=0.55)
    parser.add_argument('--nms_thr', default=0.45)
    parser.add_argument('--gpu_id', default=0)
    args = parser.parse_args()
    return args



def main(opt):
    set_logging()
    # Show GPUs info
    s = f"Yolov3 detection predicting factory torch {torch.__version__}"
    n = torch.cuda.device_count()
    if n > 1 and opt.batch_size:
        assert opt.batch_size % n ==0, f"batch-size {opt.batch_size} not multiple of GPU count {n}"
    p = torch.cuda.get_device_properties(opt.gpu_id)
    s += f" CUDA:{opt.gpu_id} ({p.name}, {p.total_memory / 1024 **2}MB)\n"
    logger.info(s)
    # Make Diectories
    wdir = opt.save_dir
    wdir.mkdir(parents=True, exist_ok=True)
    # Save run setings
    with open(opt.save_dir /"opt.yaml", "w") as f:
        opt.save_dir = str(opt.save_dir)
        yaml.dump(vars(opt), f, sort_keys=False)
    # Build Tester
    tester = Tester(opt)
    tester.model.cuda(opt.gpu_id)
    # Data Loader
    cudnn.benchmark = True
    # Names
    class_names = load_classes(opt.names)
    colors = {name:(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for name in class_names}
    
    if opt.type == 'video':
        file_list = gen_list(opt.data_dir, file_name=['.mp4', '.avi'])
        pbar = tqdm(file_list, total=len(file_list), desc="Videos Predicting")
        
        for file in pbar:
            save_path = Path(opt.save_dir) / 'result' / opt.data_dir.split('/')[-1] / os.path.splitext('/'.join(file.split('/')[-2:]))[0]
            save_path.mkdir(parents=True, exist_ok=True)
            videoCapture = cv2.VideoCapture(file)
            frame_cnt = 0
            success, frame = videoCapture.read()
            while success:
                res_frame, rects = predict_images(opt, tester.model, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), colors, class_names)
                save_frame = str(save_path / f'frame_{frame_cnt}.jpg')
                if not opt.gen_txt:
                    cv2.imwrite(save_frame, res_frame)
                if len(rects) != 0:
                    save_txt = os.path.splitext(save_frame)[0] + '.txt'
                    with open(save_txt, "w") as f_rect:
                        for rect in rects:
                            f_rect.writelines(rect)
                success, frame = videoCapture.read() 
                frame_cnt += 1
            videoCapture.release()

    # elif opt.type == 'image':
    #     file_list = gen_list(opt.data_dir, file_name=['.png', '.jpg', 'jpeg', 'bmp'])
    #     pbar = tqdm(file_list, total=len(file_list), desc="Images Predicting")
    #     for file in pbar:
    #         image = cv2.imread(file)
    #         res_image = predict_images(opt, tester.model, cv2.cvtColor(image, cv2.COLOR_BGR2RGB), colors, class_names)

if __name__=="__main__":
    opt = parser()
    opt.save_dir = increment_path(Path(opt.project) / opt.name, sep="_")
    main(opt=opt)