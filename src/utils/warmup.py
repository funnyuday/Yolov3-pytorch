# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Functions for ramping hyperparameters up or down

Each function takes the current training step or epoch, and the
ramp length in the same format, and returns a multiplier between
0 and 1.
"""


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def warmup_learning_rate(opt, optimizer, epoch):
    '''
    函数作用：学习率 warm up操作
    入参：
        epoch：网络当前运行的epcoh数目
        step_in_epoch: 网络当前在一个epoch中运行的step数目
        total_steps_in_epoch： 一个epoch最大step数目
    '''
    lr = linear_rampup(epoch, opt.warm_up_epoch) * (opt.lr - opt.initial_lr) + opt.initial_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
