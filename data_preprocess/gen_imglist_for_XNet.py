# -*- coding: utf-8 -*-
"""
  @Time   : 2019/4/20 15:34
  @Author : JinnTaoo
"""

import numpy as np
import numpy.random as npr
import os
import sys
import argparse


proj_name = 'MTCNN-TF2'
root_path = os.path.join(os.path.abspath(sys.path[0]).split(proj_name)[0], proj_name)
print('root_path: ', root_path)
os.chdir(root_path)
sys.path.insert(0, root_path)


data_dir = os.path.abspath(os.path.join(root_path, '../DATA'))


def gen_imglist_for_xnet(stage):
    base_num = 250000
    size_of_net = {'PNet': 12, 'RNet': 24, 'ONet': 48}
    if stage not in size_of_net:
        raise Exception("The net type error!")
    size = size_of_net[stage]
    with open(os.path.join(data_dir, '%s/pos_%s.txt' % (size, size)), 'r') as f:
        pos = f.readlines()
    with open(os.path.join(data_dir, '%s/neg_%s.txt' % (size, size)), 'r') as f:
        neg = f.readlines()
    with open(os.path.join(data_dir, '%s/part_%s.txt' % (size, size)), 'r') as f:
        part = f.readlines()
    with open(os.path.join(data_dir, '%s/landmark_%s_aug.txt' % (size, size)), 'r') as f:
        landmark = f.readlines()
    dir_path = os.path.abspath(os.path.join(data_dir, 'imglist'))
    x_net_path = os.path.abspath(os.path.join(dir_path, '%s' % stage))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(x_net_path):
        os.makedirs(x_net_path)
    with open(os.path.abspath(os.path.join(x_net_path, 'train_%s_landmarks.txt' % stage)), 'w') as fw:
        ratio = [3, 1, 1]
        base_num = 250000
        print(len(neg), len(pos), len(part), base_num)
        # shuffle the order of the initial data
        # if negative examples are more than 750k then only choose 750k
        if len(neg) > base_num * 3:
            neg_keep = np.random.choice(len(neg), size=base_num*3, replace=True)
        else:
            neg_keep = np.random.choice(len(neg), size=len(neg), replace=True)
        pos_keep = npr.choice(len(pos), size=base_num, replace=True)
        part_keep = npr.choice(len(part), size=base_num, replace=True)
        print(len(neg_keep), len(pos_keep), len(part_keep))
        for i in pos_keep:
            fw.write(pos[i])
        for i in neg_keep:
            fw.write(neg[i])
        for i in part_keep:
            fw.write(part[i])
        for item in landmark:
            fw.write(item)
        fw.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Create hard bbox sample...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stage', dest='stage', help='working stage, can be PNet, RNet, ONet',
                        default='unknow', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    stage = args.stage
    if stage not in ['PNet', 'RNet', 'ONet']:
        raise Exception('Please specify stage by --stage=PNet or RNet or ONet')
    gen_imglist_for_xnet(stage)
