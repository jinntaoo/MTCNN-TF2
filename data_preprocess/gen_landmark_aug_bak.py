# -*- coding: utf-8 -*-
"""
  @Time   : 2019/4/20 19:34
  @Author : JinnTaoo
"""
import os
import sys
import argparse
import cv2
import numpy as np
import random

proj_name = 'MTCNN-TF2'
root_path = os.path.join(sys.path[0].split(proj_name)[0], proj_name)
os.chdir(root_path)
sys.path.insert(0, root_path)

from util.utils import get_bbox_landmark_from_txt, calc_iou, BBox, rotate, flip


def gen_landmark_data(src_txt_path, net, augmet=False):
    """
    generate landmark samples for training
    :param src_txt_path: each line is: 0=img path, 1-4=bbox, 5-14=landmark 5 points
    :param net: PNet or RNet or ONet
    :param augmet: augmentation if True
    :return:
    """
    print(">>>>>> Start landmark data create...Stage: %s" % net)
    save_folder = os.path.join(root_path, '../DATA/12/')
    save_image_folder = os.path.join(save_folder, 'train_%s_landmark_aug' % net)
    size_of_net = {'PNet': 12, 'RNet': 24, 'ONet': 48}
    if net not in size_of_net:
        raise Exception("The net type error!")
    if not os.path.isdir(save_image_folder):
        os.makedirs(save_image_folder)
        print('create folder: ', save_image_folder)
    save_f = open(os.path.join(save_folder, 'landmark_%s_aug.txt' % size_of_net[net]), 'w')
    image_count = 0
    # image_path bbox landmark(5*2)
    bbox_landmark_info = get_bbox_landmark_from_txt(src_txt_path, data_path='../DATA/landmarks_traindata', with_landmark=True)
    for img_path, bbox, landmark_gt in bbox_landmark_info:
        f_imgs = list()
        f_landmarks = list()
        img = cv2.imread(img_path)
        assert(img is not None)
        img_h, img_w, img_c = img.shape
        gt_box = np.array([bbox.left, bbox.top, bbox.right, bbox.bottom])
        f_face = img[bbox.top: bbox.bottom+1, bbox.left: bbox.right+1]
        f_face = cv2.resize(f_face, (size_of_net[net], size_of_net[net]))
        landmark = np.zeros((5, 2))
        # normalize
        for index, one in enumerate(landmark_gt):
            rv = ((one[0]-gt_box[0])/(gt_box[2]-gt_box[0]), (one[1]-gt_box[1])/(gt_box[3]-gt_box[1]))
            landmark[index] = rv
        f_imgs.append(f_face)
        f_landmarks.append(landmark.reshape(10))
        landmark = np.zeros((5, 2))
        if augmet:
            x1, y1, x2, y2 = gt_box
            gt_width = x2 - x1 + 1
            gt_height = y2 - y1 + 1
            if max(gt_width, gt_height) < 40 or x1 < 0 or y1 < 0:
                continue
            # random shift
            for i in range(10):
                bbox_size = np.random.randint(int(min(gt_width, gt_height) * 0.8), np.ceil(1.25 * max(gt_width, gt_height)))
                # delta_x and delta_y are offsets of (x1, y1)
                # max can make sure if the delta is a negative number , x1+delta_x >0
                # parameter high of randint make sure there will be intersection between bbox and cropped_box
                delta_x = np.random.randint(-gt_width*0.2, gt_width*0.2)
                delta_y = np.random.randint(-gt_height*0.2, gt_height*0.2)
                nx1 = int(max(x1+gt_width/2 - bbox_size/2 + delta_x, 0))
                ny1 = int(max(y1+gt_height/2 - bbox_size/2 + delta_y, 0))
                nx2 = nx1 + bbox_size
                ny2 = ny1 + bbox_size
                if nx2 > img_w or ny2 > img_h:
                    continue
                # print(nx1, ny1, nx2, ny2)
                crop_box = np.array([nx1, ny1, nx2, ny2])
                cropped_img = img[ny1: ny2+1, nx1: nx2+1, :]
                resized_img = cv2.resize(cropped_img, (size_of_net[net], size_of_net[net]))
                iou = calc_iou(crop_box, np.expand_dims(gt_box, 0))
                if iou <= 0.65:
                    continue
                f_imgs.append(resized_img)
                # normalize
                for index, one in enumerate(landmark_gt):
                    rv = ((one[0]-nx1)/bbox_size, (one[1]-ny1/bbox_size))
                    landmark[index] = rv
                f_landmarks.append(landmark.reshape(10))
                landmark = np.zeros((5, 2))
                # get last landmark from list
                landmark_ = f_landmarks[-1].reshape((-1, 2))
                bbox = BBox([nx1, ny1, nx2, ny2])

                # mirror
                if random.choice([0, 1]) > 0:
                    face_flipped, landmark_flipped = flip(resized_img, landmark_)
                    face_flipped = cv2.resize(face_flipped, (size_of_net[net], size_of_net[net]))
                    # c*h*w
                    f_imgs.append(face_flipped)
                    f_landmarks.append(landmark_flipped.reshape(10))
                # rotate
                if random.choice([0, 1]) > 0:
                    face_rotated_by_alpha, landmark_rotated = rotate(img, bbox,
                                                                     bbox.reproject_landmark(landmark_), 5)
                    # landmark offset
                    landmark_rotated = bbox.project_landmark(landmark_rotated)
                    face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size_of_net[net], size_of_net[net]))
                    f_imgs.append(face_rotated_by_alpha)
                    f_landmarks.append(landmark_rotated.reshape(10))

                    # flip
                    face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                    face_flipped = cv2.resize(face_flipped, (size_of_net[net], size_of_net[net]))
                    f_imgs.append(face_flipped)
                    f_landmarks.append(landmark_flipped.reshape(10))
                # anti-clockwise rotation
                if random.choice([0, 1]) > 0:
                    face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, bbox.reproject_landmark(landmark_), -5)
                    landmark_rotated = bbox.project_landmark(landmark_rotated)
                    face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size_of_net[net], size_of_net[net]))
                    f_imgs.append(face_rotated_by_alpha)
                    f_landmarks.append(landmark_rotated.reshape(10))

                    face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                    face_flipped = cv2.resize(face_flipped, (size_of_net[net], size_of_net[net]))
                    f_imgs.append(face_flipped)
                    f_landmarks.append(landmark_flipped.reshape(10))
        f_imgs, f_landmarks = np.asarray(f_imgs), np.asarray(f_landmarks)
        for i in range(len(f_imgs)):
            # if np.sum(np.where(f_landmarks[i] <= 0, 1, 0)) > 0:
            #     print('skip image: %d' % i)
            #     print(f_landmarks[i])
            #     continue
            # if np.sum(np.where(f_landmarks[i] >= 1, 1, 0)) > 0:
            #     print('skip image: %d', i)
            #     print(f_landmarks[i])
            #     continue
            path = os.path.join(save_image_folder, '%d.jpg' % image_count)
            cv2.imwrite(path, f_imgs[i])
            landmarks = map(str, list(f_landmarks[i]))
            save_f.write(path + ' -2 ' + ' '.join(landmarks) + '\n')
            image_count += 1
        print_str = "\rCount: {}".format(image_count)
        sys.stdout.write(print_str)
        sys.stdout.flush()
    save_f.close()
    print('\n Landmark create done!')


def parse_args():
    parser = argparse.ArgumentParser(description='Create hard bbox sample...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stage', dest='stage', help='working stage, can be pnet, rnet, onet',
                        default='unknow', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    stage = args.stage
    if stage not in ['PNet', 'RNet', 'ONet']:
        raise Exception('Please specify stage by --stage=PNet or RNet or ONet')
    # augment: data augmentation
    gen_landmark_data('../DATA/landmarks_traindata/trainImageList.txt', stage, augmet=True)
















