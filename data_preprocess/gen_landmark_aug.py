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
import tqdm

proj_name = 'MTCNN-TF2'
root_path = os.path.join(sys.path[0].split(proj_name)[0], proj_name)
os.chdir(root_path)
sys.path.insert(0, root_path)

from util.utils import get_bbox_landmark_from_txt, calc_iou, BBox, rotate, flip


def generate_data(src_txt_path, net, argument=False):
    """

    :param src_txt_path: name/path of the text file that contains image path,
                bounding box, and landmarks
    :param net: one of the net in the cascaded networks
    :param argument: apply augmentation or not
    :return:  images and related landmarks
    """
    print(">>>>>> Start landmark data create...Stage: %s" % net)

    save_folder = os.path.join(root_path, '../DATA/12/')
    save_image_folder = os.path.join(save_folder, 'train_%s_landmark_aug' % net)
    data_path = '../DATA/landmarks_traindata'
    size_of_net = {'PNet': 12, 'RNet': 24, 'ONet': 48}
    if net not in size_of_net:
        raise Exception("The net type error!")
    if not os.path.isdir(save_image_folder):
        os.makedirs(save_image_folder)
        print('create folder: ', save_image_folder)
    save_f = open(os.path.join(save_folder, 'landmark_%s_aug.txt' % size_of_net[net]), 'w')
    image_count = 0

    image_id = 0
    # f = open(os.path.join(OUTPUT, "landmark_%s_aug.txt" % size), 'w')
    # dstdir = "train_landmark_few"
    # get image path , bounding box, and landmarks from file 'ftxt'
    data = get_bbox_landmark_from_txt(src_txt_path, data_path=data_path)
    idx = 0
    # image_path bbox landmark(5*2)
    for (imgPath, bbox, landmarkGt) in data:
        idx += 1
        if idx % 100 == 0:
            print("\n%d origin images done" % idx)
        # print imgPath
        f_imgs = list()
        f_landmarks = list()
        # print(imgPath)
        img = cv2.imread(imgPath)
        assert (img is not None)
        img_h, img_w, img_c = img.shape
        gt_box = np.array([bbox.left, bbox.top, bbox.right, bbox.bottom])
        # get sub-image from bbox
        f_face = img[bbox.top:bbox.bottom + 1, bbox.left:bbox.right + 1]
        # resize the gt image to specified size
        f_face = cv2.resize(f_face, (size_of_net[net], size_of_net[net]))
        # initialize the landmark
        landmark = np.zeros((5, 2))
        # normalize land mark by dividing the width and height of the ground truth bounding box
        # landmakrGt is a list of tuples
        for index, one in enumerate(landmarkGt):
            # (( x - bbox.left)/ width of bounding box, (y - bbox.top)/ height of bounding box
            rv = ((one[0] - gt_box[0]) / (gt_box[2] - gt_box[0]), (one[1] - gt_box[1]) / (gt_box[3] - gt_box[1]))
            # put the normalized value into the new list landmark
            landmark[index] = rv
        f_imgs.append(f_face)
        f_landmarks.append(landmark.reshape(10))
        landmark = np.zeros((5, 2))
        if argument:
            # idx = idx + 1
            x1, y1, x2, y2 = gt_box
            # gt's width
            gt_w = x2 - x1 + 1
            # gt's height
            gt_h = y2 - y1 + 1
            if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
                continue
            # random shift
            for i in range(10):
                bbox_size = np.random.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                delta_x = np.random.randint(-gt_w * 0.2, gt_w * 0.2)
                delta_y = np.random.randint(-gt_h * 0.2, gt_h * 0.2)
                nx1 = int(max(x1 + gt_w / 2 - bbox_size / 2 + delta_x, 0))
                ny1 = int(max(y1 + gt_h / 2 - bbox_size / 2 + delta_y, 0))

                nx2 = nx1 + bbox_size
                ny2 = ny1 + bbox_size
                if nx2 > img_w or ny2 > img_h:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])

                cropped_im = img[ny1:ny2 + 1, nx1:nx2 + 1, :]
                resized_im = cv2.resize(cropped_im, (size_of_net[net], size_of_net[net]))
                iou = calc_iou(crop_box, np.expand_dims(gt_box, 0))
                if iou > 0.65:
                    f_imgs.append(resized_im)
                    # normalize
                    for index, one in enumerate(landmarkGt):
                        rv = ((one[0] - nx1) / bbox_size, (one[1] - ny1) / bbox_size)
                        landmark[index] = rv
                    f_landmarks.append(landmark.reshape(10))
                    landmark = np.zeros((5, 2))
                    landmark_ = f_landmarks[-1].reshape(-1, 2)
                    bbox = BBox([nx1, ny1, nx2, ny2])

                    # mirror
                    if random.choice([0, 1]) > 0:
                        face_flipped, landmark_flipped = flip(resized_im, landmark_)
                        face_flipped = cv2.resize(face_flipped, (size_of_net[net], size_of_net[net]))
                        # c*h*w
                        f_imgs.append(face_flipped)
                        f_landmarks.append(landmark_flipped.reshape(10))
                    # rotate
                    if random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox,
                                                                         bbox.reproject_landmark(landmark_), 5)  # 逆时针旋转
                        # landmark_offset
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
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox,
                                                                         bbox.reproject_landmark(landmark_), -5)  # 顺时针旋转
                        landmark_rotated = bbox.project_landmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size_of_net[net], size_of_net[net]))
                        f_imgs.append(face_rotated_by_alpha)
                        f_landmarks.append(landmark_rotated.reshape(10))

                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size_of_net[net], size_of_net[net]))
                        f_imgs.append(face_flipped)
                        f_landmarks.append(landmark_flipped.reshape(10))

            f_imgs, f_landmarks = np.asarray(f_imgs), np.asarray(f_landmarks)
            # print F_imgs.shape
            # print f_landmarks.shape
            for i in range(len(f_imgs)):
                if np.sum(np.where(f_landmarks[i] <= 0, 1, 0)) > 0:
                    continue
                if np.sum(np.where(f_landmarks[i] >= 1, 1, 0)) > 0:
                    continue
                tmp_path = os.path.join(save_image_folder, "%d.jpg" % image_id)
                cv2.imwrite(tmp_path, f_imgs[i])
                landmarks = map(str, list(f_landmarks[i]))
                save_f.write(tmp_path + " -2 " + " ".join(landmarks) + "\n")
                image_id = image_id + 1
            print_str = "\rimage generated Count: {}".format(image_id)
            sys.stdout.write(print_str)
            sys.stdout.flush()
    # print F_imgs.shape
    # print F_landmarks.shape
    # F_imgs = processImage(F_imgs)
    # shuffle_in_unison_scary(F_imgs, F_landmarks)
    save_f.close()
    print('\nLandmark create done!')
    return f_imgs, f_landmarks


def parse_args():
    parser = argparse.ArgumentParser(description='Create hard bbox sample...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stage', dest='stage', help='working stage, can be PNet, RNet, ONet',
                        default='unknow', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    stage = args.stage
    if stage not in ['PNet', 'RNet', 'ONet']:
        raise Exception('Please specify stage by --stage=PNet or RNet or ONet')
    # augement: data augmentation

    # train data
    # stage = "PNet"
    # the file contains the names of all the landmark training data
    # train_txt = "trainImageList.txt"
    imgs, landmarks = generate_data('../DATA/landmarks_traindata/trainImageList.txt', stage, argument=True)
