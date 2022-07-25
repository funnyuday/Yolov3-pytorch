import os
import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image as save_tensor
from .utils import fn_timer

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    hyperparams.update({
        'batch': int(hyperparams['batch']),
        'subdivisions': int(hyperparams['subdivisions']),
        'width': int(hyperparams['width']),
        'height': int(hyperparams['height']),
        'channels': int(hyperparams['channels']),
        'optimizer': hyperparams.get('optimizer'),
        'momentum': float(hyperparams['momentum']),
        'decay': float(hyperparams['decay']),
        'learning_rate': float(hyperparams['learning_rate']),
        'burn_in': int(hyperparams['burn_in']),
        'max_batches': int(hyperparams['max_batches']),
        'policy': hyperparams['policy'],
        'lr_steps': list(zip(map(int,   hyperparams["steps"].split(",")),
                             map(float, hyperparams["scales"].split(","))))
    })
    assert hyperparams["height"] == hyperparams["width"], \
        "Height and width should be equal! Non square images are padded with zeros."
    output_filters = [hyperparams["channels"]]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}",
                                   nn.BatchNorm2d(filters, momentum=0.1, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
            if module_def["activation"] == "mish":
                modules.add_module(f"mish_{module_i}", Mish())

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                                   padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers]) // int(module_def.get("groups", 1))
            modules.add_module(f"route_{module_i}", nn.Sequential())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", nn.Sequential())

        elif module_def["type"] == "yolo":
            labels_mask = None
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            strides = int(module_def["strides"])
            ignore_thresh = float(module_def["ignore_thresh"])
            num_classes = int(module_def["classes"])
            if "labels" in module_def:
                labels_mask = [int(x) for x in module_def["labels"].split(",")]
            # Define detection layer
            yolo_layer = YOLOLayer([anchors, anchor_idxs], num_classes, strides, ignore_thresh)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, GIoU=False):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        a_x1, a_y1, a_x2, a_y2 = bboxes_a[:, None, 0], bboxes_a[:, None, 1], bboxes_a[:, None, 2], bboxes_a[:, None, 3]
        b_x1, b_y1, b_x2, b_y2 = bboxes_b[:, 0], bboxes_b[:, 1], bboxes_b[:, 2], bboxes_b[:, 3]
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
        a_x1, a_y1, a_x2, a_y2 = (bboxes_a[:, None, 0] - bboxes_a[:, None, 2] / 2), \
                                 (bboxes_a[:, None, 1] - bboxes_a[:, None, 2] / 3), \
                                 (bboxes_a[:, None, 0] + bboxes_a[:, None, 2] / 2), \
                                 (bboxes_a[:, None, 1] + bboxes_a[:, None, 2] / 3)
        b_x1, b_y1, b_x2, b_y2 = (bboxes_b[:, 0] - bboxes_b[:, 2] / 2), \
                                 (bboxes_b[:, 1] - bboxes_b[:, 3] / 2), \
                                 (bboxes_b[:, 0] + bboxes_b[:, 2] / 2), \
                                 (bboxes_b[:, 1] + bboxes_b[:, 3] / 2)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en
    union = (area_a[:, None] + area_b - area_i)
    iou = area_i / union
    if GIoU:
        cw = torch.max(a_x2, b_x2) - torch.min(a_x1, b_x1)
        ch = torch.max(a_y2, b_y2) - torch.min(a_y1, b_y1)
        c_area = cw * ch + 1e-6
        return iou - (c_area - union) / c_area
    return iou

def wh_iou(wh1, wh2, GIoU=False):
    wh1 = wh1[:, None]
    wh2 = wh2[None]
    inter = torch.min(wh1, wh2).prod(2)
    union = (wh1.prod(2) + wh2.prod(2) - inter)
    iou = inter / union
    if GIoU:
        c_area = torch.max(wh1, wh2).prod(2)
        return iou - (c_area - union) / c_area
    return iou

def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class Mish(nn.Module):
    """ The MISH activation function (https://github.com/digantamisra98/Mish) """

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class BCEFcoalloss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFcoalloss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, pt, target):
        pt = pt + 1e-6
        loss = -self.alpha*(1 - pt)**self.gamma*target*torch.log(pt) - \
               (1 - self.alpha)*pt**self.gamma*(1 - target)*torch.log(1 - pt)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class YOLOLayer(nn.Module):
    """Detection layer"""
    def __init__(self, anchors, num_classes, stride, ignore_thr):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors[0]
        self.anchors_mask = anchors[1]
        self.n_anchors = len(self.anchors_mask)
        self.n_classes = num_classes
        self.ignore_thr = ignore_thr
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_focalloss = BCEFcoalloss(reduction='sum')
        self.stride = stride
        self.all_anchors_grid = [(w / self.stride, h / self.stride) for w, h in self.anchors]
        self.all_mask_anchors = [self.all_anchors_grid[i] for i in self.anchors_mask]
        self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)

    def forward(self, x, labels=None, path=None):
        out = x
        bs = out.shape[0]
        fs_x = out.shape[3]
        fs_y = out.shape[2]
        nc = 5 + self.n_classes
        dtype = torch.cuda.FloatTensor if  x.is_cuda else torch.FloatTensor

        out = out.view(bs, self.n_anchors, nc, fs_y, fs_x)
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        # logistic activation for xy, obj, cls
        out[..., np.r_[:2, 4:nc]] = torch.sigmoid(out[..., np.r_[:2, 4:nc]])
        x_shift = dtype(np.broadcast_to(torch.arange(fs_x), out.shape[:4])).cuda(x.device)
        y_shift = dtype(np.broadcast_to(torch.arange(fs_y).reshape(fs_y, 1), out.shape[:4])).cuda(x.device)
        masked_anchors = np.array(self.all_mask_anchors)
        w_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 0], (1, self.n_anchors, 1, 1)), out.shape[:4])).cuda(x.device)
        h_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 1], (1, self.n_anchors, 1, 1)), out.shape[:4])).cuda(x.device)

        pred = out.clone()
        pred[..., 0] = (pred[..., 0] + x_shift) / fs_x
        pred[..., 1] = (pred[..., 1] + y_shift) / fs_y
        pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors / fs_x
        pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors / fs_y

        pred_box = pred.clone()
        pred_box[..., [0, 2]] *= self.stride*fs_x
        pred_box[..., [1, 3]] *= self.stride*fs_y
        pred_box = pred_box.view(bs, -1, nc).data
        if not self.training:
            return None, None, pred_box

        pred = pred[..., :4].data
        # target assignment
        cls_mask = torch.zeros(bs, self.n_anchors, 
                               fs_y, fs_x, 4 + self.n_classes).type(dtype).cuda(x.device)
        obj_mask = torch.ones(bs, self.n_anchors, 
                              fs_y, fs_x).type(dtype).cuda(x.device)
        scale = torch.zeros(bs, self.n_anchors,
                            fs_y, fs_x, 4).type(dtype).cuda(x.device)
        target = torch.zeros(bs, self.n_anchors,
                             fs_y, fs_x, nc).type(dtype).cuda(x.device)

        gt_x, gt_y, gt_w, gt_h = labels[:, 2], labels[:, 3], labels[:, 4], labels[:, 5]
        gi = (gt_x * fs_x).long()
        gj = (gt_y * fs_y).long()
        gi[gi >= fs_x] = fs_x - 1
        gj[gj >= fs_y] = fs_y - 1

        gt_box = dtype(np.zeros((len(labels), 4))).cuda(x.device)
        gt_box[:, 0], gt_box[:, 1], gt_box[:, 2], gt_box[:, 3] = gt_x, gt_y, gt_w, gt_h

        # calculate iou between truth and reference anchors
        anchor_ious_all = wh_iou(gt_box[..., 2:].cpu(), torch.stack([self.ref_anchors[..., 2] / fs_x, self.ref_anchors[..., 3] / fs_y], dim=1), GIoU=True)
        best_iou, best_n_all = anchor_ious_all.max(dim=1)
        best_n = best_n_all % len(self.anchors_mask)
        best_n_mask = ((best_n_all == self.anchors_mask[0]) | (best_n_all == self.anchors_mask[1]) | (best_n_all == self.anchors_mask[2]) | (best_n_all == self.anchors_mask[3])) # | (best_iou > 0.7)

        ti = best_n_mask == 1
        b, a, j, i = labels[:, 0][ti].long(), best_n[ti], gj[ti], gi[ti]

        # make sure one anchor match one label
        pos = torch.stack([b, a.cuda(x.device), j, i]).T
        p_mask = torch.ones_like(b).bool()
        for p in range(len(pos)):
            p_mask_tmp = (pos == pos[p]).sum(1) == 4
            if p_mask_tmp.sum() <= 1: continue
            p_mask_tmp[p] = False
            pos[p] = dtype([99, 99, 99, 99])
            p_mask = p_mask & ~p_mask_tmp

        b, a, j, i = b[p_mask], a[p_mask], j[p_mask], i[p_mask]
        for ba in range(bs):
            cur = labels[:, 0] == ba
            n = len(labels[cur])
            if n == 0: continue
            pred_ious = bboxes_iou(pred[ba].view(-1, 4), gt_box[cur], xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thr)
            pred_best_iou = pred_best_iou.view(pred[ba].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[ba] = ~pred_best_iou

        obj_mask[b, a, j, i] = 1
        cls_mask[b, a, j, i, :] = 1
        target[b, a, j, i, 0] = gt_x[ti][p_mask] * fs_x - i
        target[b, a, j, i, 1] = gt_y[ti][p_mask] * fs_y - j
        target[b, a, j, i, 2] = torch.log(gt_w[ti][p_mask] * self.stride * fs_x / (dtype(self.all_mask_anchors)[a][..., 0] * self.stride).cuda(x.device))
        target[b, a, j, i, 3] = torch.log(gt_h[ti][p_mask] * self.stride * fs_y / (dtype(self.all_mask_anchors)[a][..., 1] * self.stride).cuda(x.device))
        target[b, a, j, i, 4] = 1
        target[b, a, j, i, 5 + labels[:, 1][ti][p_mask].long()] = 1
        scale[b, a, j, i, :] = scale[b, a, j, i, :].copy_(torch.unsqueeze(2 - gt_w[ti][p_mask] * gt_h[ti][p_mask], 1))

        out[..., 4] *= obj_mask
        out[..., np.r_[0:4, 5:nc]] *= cls_mask
        out[..., :4] *= scale

        target[..., 4] *= obj_mask
        target[..., np.r_[0:4, 5:nc]] *= cls_mask
        target[..., :4] *= scale

        loss_xy = self.mse_loss(out[..., :2], target[..., :2])
        loss_wh = self.mse_loss(out[..., 2:4], target[..., 2:4])
        loss_obj = self.mse_loss(out[..., 4], target[..., 4])
        loss_cls = self.mse_loss(out[..., 5:], target[..., 5:])
        loss = loss_xy + loss_wh + loss_cls + loss_obj
        return loss, torch.stack((loss_xy, loss_wh, loss_obj, loss_cls, loss)).detach(), pred_box

class Darknet(nn.Module):
    """YOLOv3 object detection model"""
    def __init__(self, opt):
        super(Darknet, self).__init__()
        self.opt = opt
        self.module_defs = parse_model_config(opt.cfg)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if isinstance(layer[0], YOLOLayer)]
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def weights_init_normal(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x, labels=None, path=None):
        dtype = (torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor)
        layer_outputs, yolo_outputs, loss, loss_l = [], [], dtype([0]).cuda(x.device), dtype([0,0,0,0,0]).cuda(x.device)
        if labels != None:
            if len(labels) == 0:
                return loss, loss_l
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                combined_outputs = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
                group_size = combined_outputs.shape[1] // int(module_def.get("groups", 1))
                group_id = int(module_def.get("group_id", 0))
                x = combined_outputs[:, group_size * group_id : group_size * (group_id + 1)] # Slice groupings used by yolo v4
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                layer_loss, layer_loss_l, x = module[0](x, labels, path)
                yolo_outputs.append(x)
                if layer_loss is not None:
                    loss += layer_loss
                    loss_l += layer_loss_l
                yolo_outputs.append(x)
            layer_outputs.append(x)
        return loss/x.shape[0], (loss_l / len(yolo_outputs) / x.shape[0]) if self.opt.mode == "Train" else torch.cat(yolo_outputs, 1)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        # If the weights file has a cutoff, we can find out about it by looking at the filename
        # examples: darknet53.conv.74 -> cutoff is 74
        filename = os.path.basename(weights_path)
        if ".conv." in filename:
            try:
                cutoff = int(filename.split(".")[-1])  # use last part of filename
            except ValueError:
                pass

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(
                    weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)
        fp.close()