from cProfile import label
import os
import cv2
import ipdb
import math
import copy
import torch
import random
import warnings
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
from PIL import ImageFile
from torchvision import transforms
from utils import xywh2xyxy, xyxy2xywh, xywhn2xyxy

ImageFile.LOAD_TRUNCATED_IMAGES = True

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps)) 
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr)

def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    height = img.shape[0] + border[0] * 2
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2
    C[1, 2] = -img.shape[0] / 2

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 100)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 100)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    # Combined rotation matrix
    M = T @ S @ R @ P @ C
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy= xy @ M.T
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)
        else:
            xy = xy[:, :2].reshape(n, 8)
        
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T, wh_thr=5, ar_thr=20, area_thr=0.3)
        targets = targets[i]
        targets[:, 1:5] = xy[i]
    
    return img, targets

def random_crop(img, labels):
    scale = random.uniform(0.45, 0.95)
    h, w = img.shape[:2]
    size_w = int(w * scale)
    size_h = int(h * scale)
    x = np.random.randint(0, h - size_w)
    y = np.random.randint(0, h - size_h)
    crop = img[y:y + size_h, x:x + size_w, :]
    new_labels = copy.deepcopy(labels)
    new_h, new_w = crop.shape[:2]
    new_labels[..., 0] = (new_labels[..., 0] * w - x)
    new_labels[..., 1] = (new_labels[..., 1] * h - y)
    new_labels[..., 2] = (new_labels[..., 2] * w - x)
    new_labels[..., 3] = (new_labels[..., 3] * h - y)
    new_labels[new_labels < 0] = 0
    new_labels[..., 0][new_labels[..., 0] > new_w] = new_w
    new_labels[..., 1][new_labels[..., 1] > new_h] = new_h
    new_labels[..., 2][new_labels[..., 2] > new_w] = new_w
    new_labels[..., 3][new_labels[..., 3] > new_h] = new_h
    labels[..., 0] *= w
    labels[..., 1] *= h
    labels[..., 2] *= w
    labels[..., 3] *= h
    idx = box_candidates(box1=labels.T, box2=new_labels.T, wh_thr=4, area_thr=0.35)
    new_labels[..., 0] /= new_w
    new_labels[..., 1] /= new_h
    new_labels[..., 2] /= new_w
    new_labels[..., 3] /= new_h
    return crop, new_labels[idx], idx

def random_distort_image(img, hue, saturation, exposure):
    def rand_scale(s):
        scale = random.uniform(1, s)
        if random.uniform(0, 1) < 0.5:
            return scale
        else:
            return 1./scale
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float64)

    hsv[..., 1] *= dsat
    hsv[..., 2] *= dexp
    hsv[..., 1:][hsv[..., 1:] > 255] = 255
    hsv[..., 1:][hsv[..., 1:] < 0] = 0
    hsv[..., 0] += dhue
    hsv[..., 0][hsv[..., 0] > 180] = hsv[..., 0][hsv[..., 0] > 180] - 180
    hsv[..., 0][hsv[..., 0] < 0] = hsv[..., 0][hsv[..., 0] < 0] + 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

class YoloDataset(data.Dataset):
    def __init__(self, opt, data_dir, mode):
        super(YoloDataset, self).__init__()
        self.opt = opt
        self.jitter = self.opt.jitter if mode is 'Train' else 1
        self.hue = self.opt.hue if mode is 'Train' else 0
        self.saturation = self.opt.saturation if mode is 'Train' else 1
        self.exposure = self.opt.exposure if mode is 'Train' else 1

        self.images = self.load_txt(data_dir)
        self.data_size = len(self.images)
        self.img_size = opt.img_size
        self.mode = mode
        self.batch_count = 0
        self.no_letter = opt.no_letter
        self.mosaic = opt.mosaic

    def load_txt(self, file_path):
        f_data = open(file_path, 'r', encoding='utf-8')
        lines = f_data.readlines()
        data = {}
        for idx, line in tqdm(lines, desc='Data loader'):
            line = line.encode('utf8').strip()
            img_path = line.split(' '.encode('utf8'))[0].decode()
            label_path = os.path.splitext(img_path)[0].replace('images', 'labels') + '.txt'
            data[idx] = {'img_path': img_path, 'labels': label_path}
        
        print(f"Datafile {file_path} has {len(data)}!")
        return data

    def load_data(self, index):
        return cv2.cvtColor(cv2.imread(self.images[index]['img_path'], cv2.COLOR_BGR2RGB))
        
    def load_labels(self, index):
        label_path = self.images[index]['labels']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boxes = np.loadtxt(label_path).reshape(-1, 5)                
        return boxes

    def load_mosaic(self, index): 
        labels4 = []
        s = self.img_size
        mosaic_border = [-self.img_size // 2, -self.img_size // 2]
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in mosaic_border]
        indices = [index] + [range(self.data_size)[random.randint(0, self.data_size - 1)] for _ in range(3)]

        for i, index in enumerate(indices):
            img = self.load_data(index)
            img = random_distort_image(img, self.hue, self.saturation, self.exposure)
            h, w = img.shape[0], img.shape[1]
            flip = random.uniform(0, 1) < 0.5
            img = cv2.flip(img, 1) if flip else img

            if i == 0:
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a -x1a), h - (y2a, y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            labels = self.load_labels(index)
            if labels.size:
                if flip:
                    xyxy = xywh2xyxy(labels[:, 1:])
                    xyxy[..., [0, 2]] = 1 - xyxy[..., [2, 0]]
                    xywh = xyxy2xywh(xyxy)
                    labels[:, 1:] = xywh
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)
            labels4.append(labels)

        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])

        img4, labels4 = random_perspective(img4, labels4, 0, 0.1, 0.3, 0, 0, mosaic_border)
        labels4[:, 1:] = xyxy2xywh(labels4[:, 1:]) / self.img_size
        return img4, labels4

    def letter_box(self, img, bboxes_xyxy):
        w_o, h_o = img.shape[1], img.shape[0]
        w, h = self.img_size, self.img_size
        sized = np.full((w, h, 3), 127.5)
        dw = self.jitter * w_o
        dh = self.jitter * h_o
        new_ar = (w_o + random.uniform(-dw, dw)) / (h_o + random.uniform(-dh, dh)) \
            if self.mode is 'Train' else w_o / h_o
        
        if new_ar < 1:
            nh = h
            nw = nh * new_ar
        else:
            nw = w
            nh = nw / new_ar
            
        dx = random.uniform(-(w - nw) / 2, (w - nw) / 2) if self.mode is 'Train' else 0
        dy = random.uniform(-(h - nh) / 2, (h - nh) / 2) if self.mode is 'Train' else 0

        resized = cv2.resize(img, (int(nw), int(nh)), interpolation=cv2.INTER_CUBIC)
        lx = int((w - nw) // 2 + dx)
        ly = int((h - nh) // 2 + dy)
        rx = int(nw) + lx
        ry = int(nh) + ly

        sized[ly:ry, lx:rx, :] = resized
        bboxes_xyxy[..., 0] = (bboxes_xyxy[..., 0] * nw + (w - nw) / 2 + dx) / w
        bboxes_xyxy[..., 1] = (bboxes_xyxy[..., 1] * nh + (h - nh) / 2 + dy) / h
        bboxes_xyxy[..., 2] = (bboxes_xyxy[..., 2] * nw + (w - nw) / 2 + dx) / w
        bboxes_xyxy[..., 3] = (bboxes_xyxy[..., 3] * nh + (h - nh) / 2 + dy) / h
        return sized, bboxes_xyxy

    def common_data(self, idx):
        # Load daa
        img = self.load_data(idx)
        bboxes = self.load_labels(idx)
        bboxes_xyxy = xywh2xyxy(bboxes[..., 1:])
        # Transform
        flip = random.uniform(0, 1) if self.mode is 'Train' else 1
        crop = random.uniform(0, 1) if self.mode is 'Train' else 1
        # Random crop
        if crop < 0.5:
            img, bboxes_xyxy, idx = random_crop(img, bboxes_xyxy)
            bboxes = bboxes[idx]
        # Letter box
        if not self.no_letter:
            img, bboxes_xyxy = self.letter_box(img , bboxes_xyxy)
        else:
            img = cv2.resize(img, (int(self.img_size), int(self.img_size)), interpolation=cv2.INTER_CUBIC)
        # Colors
        img = random_distort_image(img, self.hue, self.saturation, self.exposure)
        # Flip
        if flip < 0.5:
            img = cv2.flip(img, 1)
            bboxes_xyxy[..., [0, 2]] = 1 - bboxes_xyxy[..., [2, 0]]
        bboxes[..., 1:] = xyxy2xywh(bboxes_xyxy)
        return img, bboxes

    def collate_fn(self, batch):
        self.batch_count += 1
        img_path, imgs, bb_targets = list(zip(*batch))
        imgs = torch.stack(imgs)
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)
        return img_path, imgs, bb_targets

    def __getitem__(self, index):
        if self.mosaic and self.mode is 'Train':
            # Mosaic
            img, bboxes = self.load_mosaic(index)
            # Mix up
            if random.uniform(0, 1) < 0.3 and self.opt.mixup:
                img2, bboxes2 = self.load_mosaic(random.randint(0, self.data_size -1))
                r = np.random.beta(0.0, 8.0)
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                bboxes = np.concatenate((bboxes, bboxes2), 0)
        else:
            img, bboxes = self.common_data(index)
        bb_targets = torch.zeros((len(bboxes), 6))
        bb_targets[:, 1:] = transforms.ToTensor()(bboxes)
        img = transforms.ToTensor()(img)
        return self.images[index]['img_path'], img, bb_targets
    
    def __len__(self):
        return self.data_size

class YoloLoader():
    def __init__(self, opt):
        self.opt = opt
        if opt.mode is 'Train':
            print("Load Train Dataset...")
            self.train_set = YoloDataset(opt, opt.train_dir, 'Train')
        else:
            print("Load Test Dataset...")
            self.train_set = YoloDataset(opt, opt.test_dir, 'Test')
    
    def GetTrainset(self):
        return self._DataLoader(self.train_set)
    
    def GetTestset(self):
        return self._DataLoader(self.test_set)

    def _DataLoader(self, dataset):
        dataloader = data.DataLoader(dataset, 
                                    batch_size=self.opt.batch_size,
                                    shuffle=self.opt.shuffle,
                                    num_workers=int(self.opt.num_worker),
                                    collate_fn=dataset.collate_fn)
        return dataloader