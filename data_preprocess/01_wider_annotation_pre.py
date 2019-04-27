# -*- coding: utf-8 -*-
"""
  @Time   : 2019/4/17 12:11
  @Author : JinnTaoo
"""
#  convert to one line per sample
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
os.chdir(os.path.dirname(sys.path[0]))

train_anno_path = './data_preprocess/wider_face_split/wider_face_train_bbx_gt.txt'
fw_path = './data_preprocess/wider_face_train.txt'
print(os.getcwd())


def convert_to_one_line(train_anno_path, fw_path):
    with open(train_anno_path, 'r', encoding='utf-8') as fr:
        lines = fr.read().strip().split('\n')
    annos = list()
    i = 0
    j = i + 1
    while i < len(lines):
        name = lines[i]
        num = int(lines[j])
        if num == 0:
            print(i, name)
            i = i + 3
            j = i + 1
        else:
            boxes = [lines[k].split(' ')[:4] for k in range(j+1, j+1+num)]
            boxes = np.array(boxes, dtype=np.int)
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            boxes = boxes.astype(np.str)
            boxes = boxes.flatten().tolist()
            boxes = ' '.join(boxes)
            annos.append(name + ' ' + boxes)
            i = j + 1 + num
            j = i + 1

    with open(fw_path, 'a+', encoding='utf-8') as fw:
        fw.write('\n'.join(annos))


if __name__ == "__main__":
    convert_to_one_line(train_anno_path, fw_path)
