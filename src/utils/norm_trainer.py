# coding:utf-8
import os
import pdb
import torch
import logging
import numpy as np
import torch.optim as optim

from model import Darknet
from model.yolo.utils import fn_timer

logger = logging.getLogger(__name__)

model_dict     = {"YOLOv3": Darknet}
opt_dict       = {"SGD": optim.SGD}
scheduler_dict = {"StepLR": optim.lr_scheduler.StepLR,
                  "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR}

class Trainer():
    def __init__(self, opt):
        self.opt = opt
        self.model = self._build_net()
        self.load_model()
        self.optim = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.start_epoch = 0

    def _build_net(self):
        if self.opt.model_type not in model_dict:
            raise(f"ERROR: unknow {self.opt.model_type}, build model failed.")
        net = model_dict[self.opt.model_type](self.opt)
        net.apply(net.weights_init_normal)
        return net

    def load_model(self):
        if os.path.isfile(self.opt.pretrained):
            if ".weights" in self.opt.pretrained:
                self.model.load_darknet_weights(self.opt.pretrained)
            else:
                print(f'Load pretrained weight {self.opt.pretrained}.')
                state_model_inf = torch.load(self.opt.pretrained)
                weight_inf = state_model_inf
                state_dict = {}
                for key, val in weight_inf.items():
                    state_dict[key.replace('module.', '')] = val
                self.model.load_state_dict(state_dict, strict=True)
        else:
            logging.info("Warring: skip load pretrained model.")

    def _get_optimizer(self):
        if self.opt.optim_type == 'SGD':
            return opt_dict[self.opt.optim_type](self.model.parameters(),
                                                 lr=1e-6 if self.opt.warm_up else self.opt.lr,
                                                 momentum=0.9,
                                                 weight_decay=5e-4)
        logging.error("ERRPR: Wrong optimizer type --> {self.opt.optim_type}.")

    def _get_scheduler(self):
        if self.opt.lr_scheduler == 'StepLR':
            return scheduler_dict[self.opt.lr_scheduler](self.optim, step_size=self.opt.step_size)
        elif self.opt.lr_scheduler == 'CosineAnnealingLR':
            return scheduler_dict[self.opt.lr_scheduler](self.optim,
                                                         T_max=self.opt.max_epoch + 1 if not self.opt.warm_up else
                                                         self.opt.max_epoch + 1 - self.opt.warm_up_epoch, eta_min=0)
        logging.error("ERRPR: Wrong scheduler type --> {self.opt.lr_scheduler}.")

    def process(self, inputs):
        datas, labels, paths = inputs
        datas = datas.type(torch.FloatTensor).cuda(non_blocking=True, device=self.opt.gpu_ids[0])
        labels = labels.type(torch.FloatTensor).cuda(non_blocking=True, device=self.opt.gpu_ids[0])
        # Loss
        loss, loss_l = self.model(datas, labels, paths)
        if loss.item() == 0:
            return loss_l
        # Backward
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        return loss_l

class Tester():
    def __init__(self, opt):
        self.opt = opt
        self.model = self._build_net()
        self.load_model()
        

    def _build_net(self):
        if self.opt.model_type not in model_dict:
            raise(f"ERROR: unknow {self.opt.model_type}, build model failed.")
        net = model_dict[self.opt.model_type](self.opt)
        return net
    
    def load_model(self):
        assert os.path.isfile(self.opt.model_path), " model path is not exist."
        if ".weights" in self.opt.model_path:
            self.model.load_darknet_weights(self.opt.model_path)
        else:
            if os.path.isfile(self.opt.model_path):
                print(f'Load pretrained weight {self.opt.model_path}.')
                state_model_inf = torch.load(self.opt.model_path, map_location={'cuda:2': 'cuda:0'})
                weight_inf = state_model_inf
                state_dict = {}
                for key, val in weight_inf.items():
                    state_dict[key.replace('module.', '')] = val
                self.model.load_state_dict(state_dict, strict=True)
            else:
                logging.info("Warring: skip load pretrained model.")