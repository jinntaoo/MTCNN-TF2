# -*- coding: utf-8 -*-
"""
  @Time   : 2019/4/17 17:02
  @Author : JinnTaoo
"""
import os
import sys
import cv2
import numpy as np

proj_name = 'MTCNN-TF2'
root_path = os.path.join(os.path.abspath(sys.path[0]).split(proj_name)[0], proj_name)
print('root_path: ', root_path)
os.chdir(root_path)
sys.path.insert(0, root_path)

from util.utils import calc_iou


def gen_data_for_p_net(in_size, anno_file_path, img_path, save_dir):
    """
    generate input image data for p net
    :param in_size: image's size, like 12*12, 24*24 or 48*48
    :param anno_file_path: annotations' path
    :param img_path: wilder's path
    :param save_dir: save's dir
    :param pos_save_path:
    :param part_save_path:
    :param neg_save_path:
    :return:
    """
    pos_save_path = os.path.join(save_dir, 'positive')
    part_save_path = os.path.join(save_dir, 'part')
    neg_save_path = os.path.join(save_dir, 'negative')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(pos_save_path):
        os.makedirs(pos_save_path)
    if not os.path.exists(neg_save_path):
        os.makedirs(neg_save_path)
    if not os.path.exists(part_save_path):
        os.makedirs(part_save_path)

    pos_idx = 0  # positive
    neg_idx = 0  # negative
    part_idx = 0  # part
    idx, box_idx = 0, 0
    f1 = open(os.path.join(save_dir, 'pos_%s.txt' % in_size), 'w')
    f2 = open(os.path.join(save_dir, 'neg_%s.txt' % in_size), 'w')
    f3 = open(os.path.join(save_dir, 'part_%s.txt' % in_size), 'w')
    with open(anno_file_path, 'r') as fr:
        annotations = fr.readlines()
    num_anno = len(annotations)
    print('%d imgs in total' % num_anno)

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        image_path = annotation[0]
        bbox = list(map(float, annotation[1:]))
        boxes = np.array(bbox, dtype=np.float32, ).reshape(-1, 4)
        img = cv2.imread(os.path.join(img_path, image_path))

        idx += 1  # print count every 100

        height, width, channel = img.shape
        num_neg = 0
        while num_neg < 50:
            size = np.random.randint(12, min(width, height) / 2)
            # top left coordinate
            nx, ny = np.random.randint(0, width - size), np.random.randint(0, height - size)
            crop_box = np.array([nx, ny, nx + size, ny + size])
            iou = calc_iou(crop_box, boxes)
            croped_img = img[ny: ny + size, nx:nx + size, :]
            resized_img = cv2.resize(croped_img, (in_size, in_size), interpolation=cv2.INTER_LINEAR)
            if np.max(iou) < 0.3:
                save_file = os.path.join(neg_save_path, "%s.jpg" % neg_idx)
                f2.write(os.path.join(neg_save_path, '%s.jpg' % neg_idx) + ' 0\n')
                cv2.imwrite(save_file, resized_img)
                neg_idx += 1
                num_neg += 1

        for box in boxes:
            x1, y1, x2, y2 = box
            w, h = x2 - x1 + 1, y2 - y1 + 1
            if max(w, h) < 20 or x1 < 0 or y1 < 0:
                continue
            # crop another 5 images near the bbox if IoU less than 0.5, save as negative samples
            for i in range(5):
                size = np.random.randint(12, min(width, height) / 2)
                # delta_x and delta_y are offsets of (x1, y1)
                # max can make sure if the delta is a negative number , x1+delta_x >0
                # parameter high of randint make sure there will be intersection between bbox and cropped_box
                # print('-size ', -size,)
                # print('-x1 ', -x1)
                # print('w ', w)
                # print()
                delta_x = np.random.randint(max(-size, -x1), w)
                delta_y = np.random.randint(max(-size, -y1), h)
                nx1 = int(max(0, x1 + delta_x))
                ny1 = int(max(0, y1 + delta_y))
                # if the right bottom point is out of image then skip
                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                iou = calc_iou(crop_box, boxes)
                croped_img = img[ny1: ny1 + size, nx1: nx1 + size, :]
                resized_img = cv2.resize(croped_img, (in_size, in_size), interpolation=cv2.INTER_LINEAR)

                if np.max(iou) < 0.3:
                    save_file = os.path.join(neg_save_path, "%s.jpg" % neg_idx)
                    f2.write(os.path.join(neg_save_path, '%s.jpg' % neg_idx) + ' 0\n')
                    cv2.imwrite(save_file, resized_img)
                    neg_idx += 1
            # generate positive examples and part faces
            for i in range(20):
                # pos and part face size [minsize*0.8, maxsize*1.25]
                size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                # delta here is the offset of box center
                if w < 5:
                    print('w<5, w: ', w)
                    continue
                delta_x = np.random.randint(-w * 0.2, w * 0.2)
                delta_y = np.random.randint(-h * 0.2, h * 0.2)
                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                nx2, ny2 = nx1 + size, ny1 + size
                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])
                # yu ground_truth offset
                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)
                croped_img = img[ny1:ny2, nx1:nx2, :]
                resized_img = cv2.resize(croped_img, (in_size, in_size), interpolation=cv2.INTER_LINEAR)
                box_ = box.reshape(1, -1)
                iou = calc_iou(crop_box, box_)
                if iou >= 0.65:
                    save_file = os.path.join(pos_save_path, '%s.jpg' % pos_idx)
                    f1.write(os.path.join(pos_save_path, '%s.jpg' % pos_idx) + ' 1 %.2f %.2f %.2f %.2f\n'
                             % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_img)
                    pos_idx += 1
                elif iou >= 0.4:
                    save_file = os.path.join(part_save_path, '%s.jpg' % part_idx)
                    f3.write(os.path.join(part_save_path, '%s.jpg' % part_idx) + ' -1 %.2f %.2f %.2f %.2f\n'
                             % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_img)
                    part_idx += 1
            box_idx += 1
            if idx % 100 == 0:
                print("%s images done, pos: %s part: %s neg: %s" % (idx, pos_idx, part_idx, neg_idx))
    f1.close()
    f2.close()
    f3.close()


if __name__ == "__main__":
    gen_data_for_p_net(in_size=12,
                       anno_file_path='./data_preprocess/wider_face_train.txt',
                       img_path='../DATA/WIDER_train/images',
                       save_dir='../DATA/12',)
    pass
