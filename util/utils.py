# -*- coding: utf-8 -*-
"""
  @Time   : 2019/4/17 18:15
  @Author : JinnTaoo
"""
import numpy as np
import os, sys
import cv2
import numpy as np
import time


def calc_iou(box, boxes):
    """
    :param box: detected target, shape(5,): x1, y1, x2, y2, score
    :param boxes: ground truth, shape (n, 4)
    :return: float, shape (n, )
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    # compute the width and height of the bbox
    width = np.maximum(0, xx2 - xx1 + 1)
    height = np.maximum(0, yy2 - yy1 + 1)

    inter = width * height
    over = inter / (area + box_area - inter)
    return over


def convert_to_square(bbox):
    """
    :param bbox: numpy array, shape (n, 5)
    :return: square bbox
    """
    square_bbox = bbox.copy()
    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox


""" for landmark utils"""


def flip(face, landmark):
    face_plipped_by_x = cv2.flip(face, 1)
    # mirror
    landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark])
    landmark_[[0, 1]] = landmark_[[1, 0]]
    landmark_[[3, 4]] = landmark_[[4, 3]]
    return face_plipped_by_x, landmark_


def rotate(img, bbox, landmark, alpha):
    """
    given a face with bbox and landmark, rotate with alpha and return
    rotated face with bbox, landmark (absolute position)
    :param img:
    :param bbox:
    :param landmark:
    :param alpha: angle
    :return:
    """
    center = ((bbox.left + bbox.right) / 2, (bbox.top + bbox.bottom) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    # whole image rotate
    # pay attention: 3rd param(col*row)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
    landmark_ = np.asarray([(rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2],
                             rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2]) for (x, y) in landmark])
    # crop face
    face = img_rotated_by_alpha[bbox.top:bbox.bottom + 1, bbox.left:bbox.right + 1]
    return face, landmark_


def show_landmark(face, landmark):
    """view face with landmark for visualization"""
    face_copied = face.copy().astype(np.uint8)
    for (x, y) in landmark:
        xx = int(face.shape[0] * x)
        yy = int(face.shape[1] * y)
        cv2.circle(face_copied, (xx, yy), 2, (0, 0, 0), -1)
        cv2.imshow("face_rot", face_copied)
        cv2.waitKey(0)


def random_shift(landmark_gt, shift):
    """Random shift one time"""
    diff = np.random.rand(5, 2)
    diff = (2*diff-1) * shift
    landmark_p = landmark_gt + diff
    return landmark_p


def random_shift_with_argument(landmark_gt, shift):
    """
        Random Shift more
    """
    n = 2
    landmark_ps = np.zeros((n, 5, 2))
    for i in range(n):
        landmark_ps[i] = random_shift(landmark_gt, shift)
    return landmark_ps


"""for bbox utils"""


def logger(msg):
    """ log message"""
    now = time.ctime()
    print("[%s] %s" % (now, msg))


def create_dir(p):
    if not os.path.exists(p):
        os.mkdir(p)


def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def draw_landmark(img, bbox, landmark):
    cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom),
                  (0, 0, 255), 2)
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y), 2, (0, 255, 0), -1))
    return img


def get_bbox_landmark_from_txt(src_txt_path, data_path, with_landmark=True):
    """
    Generate data from landmark label txt file
    :param txt:
    :param data_path:
    :param with_landmark:
    :return: [(img_path, bbox, landmark)]
             bbox: [left, right, top, bottom]
             landmark: [(x1, y1), (x2, y2), ...]
    """
    with open(src_txt_path, 'r') as fr:
        lines = fr.readlines()
    ret = list()
    for line in lines:
        line = line.strip().split(' ')
        img_path = os.path.join(data_path, line[0]).replace('\\', '/')
        # bounding box, (x1, y1, x2, y2)
        bbox = (line[1], line[3], line[2], line[4])
        bbox = [float(x) for x in bbox]
        bbox = list(map(int, bbox))
        # landmark
        if not with_landmark:
            ret.append((img_path, BBox(bbox)))
            continue
        landmark = np.zeros((5, 2))
        for index in range(5):
            rv = (float(line[5+2*index]), float(line[5+2*index+1]))
            landmark[index] = rv
        # normalize
        """
        for index, one in enumrate(landmakr):
            rv = ((one[0]-bbox[0])/(bbox[2]-bbox[0], (one[1]-bbox[1])/bbox[3]-bbox[1]))
            landmark[index] = rv
        """
        ret.append((img_path, BBox(bbox), landmark))
    return ret


class BBox:
    """bounding box of face"""
    def __init__(self, bbox):
        self.left = bbox[0]
        self.top = bbox[1]
        self.right = bbox[2]
        self.bottom = bbox[3]
        self.x = bbox[0]
        self.y = bbox[1]
        self.w = bbox[2]-bbox[0]
        self.h = bbox[3]-bbox[1]

    def expand(self, scale=0.05):
        bbox = [self.left, self.right, self.top, self.bottom]
        bbox[0] -= int(self.w * scale)
        bbox[1] += int(self.w * scale)
        bbox[2] -= int(self.h * scale)
        bbox[3] += int(self.h * scale)
        return BBox(bbox)

    def project(self, point):
        """offset"""
        x = (point[0] - self.x) / self.w
        y = (point[1] - self.y) / self.h
        return np.asarray([x, y])

    def reproject(self, point):
        """absolute point(image (left, top))"""
        x = self.x + self.w * point[0]
        y = self.y + self.h * point[1]
        return np.asarray([x, y])

    def reproject_landmark(self, landmark):
        """landmark 5*2"""
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p

    def project_landmark(self, landmark):
        """change to offset according to bbox"""
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p

    def sub_bbox(self, left_r, right_r, top_r, bottom_r):
        """
        f_bbox = bbox.sub_bbox(-0.05, 1.05, -0.05, 1.05)
        self.w bounding-box width
        self.h bounding-box height
        :param left_r:
        :param right_r:
        :param top_r:
        :param bottom_r:
        :return:
        """
        left_delta = self.w * left_r
        right_delta = self.w * right_r
        top_delta = self.h * top_r
        bottom_delta = self.h * bottom_r
        left = self.left + left_delta
        right = self.left + right_delta
        top = self.top + top_delta
        bottom = self.top + bottom_delta
        return BBox([left, right, top, bottom])













