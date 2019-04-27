# -*- coding: utf-8 -*-
"""
  @Time   : 2019/4/21 22:52
  @Author : JinnTaoo
"""

# coding:utf-8

from easydict import EasyDict as edict

config = edict()

config.BATCH_SIZE = 384
config.CLS_OHEM = True
config.CLS_OHEM_RATIO = 0.7
config.BBOX_OHEM = False
config.BBOX_OHEM_RATIO = 0.7

config.EPS = 1e-14
config.LR_EPOCH = [6, 14, 20]
