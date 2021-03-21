# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: k_meansx.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,八月 05
# ---
"""
## K-means ##
"""
from os import listdir
from os.path import isfile, join
import argparse
# import cv2
import numpy as np
import sys
import os
import shutil
import random
import math
from xml_extract2 import xml_extract

centroids = []

def IOU(x, centroids):
    '''
    :param x: 某一个ground truth的w,h
    :param centroids:  anchor的w,h的集合[(w,h),(),...]，共k个
    :return: 单个ground truth box与所有k个anchor box的IoU值集合
    '''
    IoUs = []
    w, h = x  # ground truth的w,h
    for centroid in centroids:
        c_w, c_h = centroid  # anchor的w,h
        if c_w >= w and c_h >= h:  # anchor包围ground truth
            iou = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:  # anchor宽矮
            iou = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:  # anchor瘦长
            iou = c_w * h / (w * h + c_w * (c_h - h))
        else:  # ground truth包围anchor     means both w,h are bigger than c_w and c_h respectively
            iou = (c_w * c_h) / (w * h)
        IoUs.append(iou)  # will become (k,) shape
    return np.array(IoUs)


def avg_IOU(X, centroids):
    '''
    :param X: ground truth的w,h的集合[(w,h),(),...]
    :param centroids: anchor的w,h的集合[(w,h),(),...]，共k个
    '''
    n, d = len(X), 2
    sum = 0.
    for i in range(len(X)):
        sum += max(IOU(X[i], centroids))  # 返回一个ground truth与所有anchor的IoU中的最大值
    return sum / n  # 对所有ground truth求平均


def write_anchors_to_file(centroids, X, anchor_file, input_shape, yolo_version):
    '''
    :param centroids: anchor的w,h的集合[(w,h),(),...]，共k个
    :param X: ground truth的w,h的集合[(w,h),(),...]
    :param anchor_file: anchor和平均IoU的输出路径
    '''
    f = open(anchor_file, 'w')

    anchors = centroids.copy()
    print(len(anchors))

    if yolo_version == 'yolov2':
        for i in range(len(anchors)):
            # yolo中对图片的缩放倍数为32倍，所以这里除以32，
            # 如果网络架构有改变，根据实际的缩放倍数来
            # 求出anchor相对于缩放32倍以后的特征图的实际大小（yolov2）
            anchors[i][0] *= input_shape / 32.
            anchors[i][1] *= input_shape / 32.
    elif yolo_version == 'yolov3':
        for i in range(anchors.shape[0]):
            # 求出yolov3相对于原图的实际大小
            anchors[i][0] *= input_shape
            anchors[i][1] *= input_shape
    else:
        print("the yolo version is not right!")
        exit(-1)
    widths = []
    for i in range(len(anchors)):
        widths.append(anchors[i][0])
    #widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    #print('Anchors = ', anchors[sorted_indices])

    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, ' % (anchors[i][0], anchors[i][1]))

    # there should not be comma after last anchor, that's why
    f.write('%0.2f,%0.2f\n' % (anchors[int(sorted_indices[-1:])][0], anchors[int(sorted_indices[-1:])][1]))

    f.write('%f\n' % (avg_IOU(X, centroids)))
    print()


def kmeans(X, centroids, eps, anchor_file, input_shape, yolo_version):
    N = len(X) #X.shape[0]  # ground truth的个数
    iterations = 0
    #print("centroids.shape", centroids)
    k = len(centroids)
    dim = 2 # anchor的个数k以及w,h两维，dim默认等于2
    prev_assignments = np.ones(N) * (-1)  # 对每个ground truth分配初始标签
    iter = 0
    old_D = np.zeros((N, k))  # 初始化每个ground truth对每个anchor的IoU

    while True:
        D = []
        iter += 1
        for i in range(N):
            d = 1 - IOU(X[i], centroids)
            D.append(d)
        D = np.array(D)  # D.shape = (N,k)  得到每个ground truth对每个anchor的IoU

        #print("iter {} th: sum of dists = {}".format(iter, np.sum(np.abs(old_D - D))))  # 计算每次迭代和前一次IoU的变化值

        # assign samples to centroids
        assignments = np.argmin(D, axis=1)  # 将每个ground truth分配给距离d最小的anchor序号

        if (assignments == prev_assignments).all():  # 如果前一次分配的结果和这次的结果相同，就输出anchor以及平均IoU
            # print("Centroids = ", centroids)
            # for i in range(len(centroids)):
            #     print(centroids[i])
            #write_anchors_to_file(centroids, X, anchor_file, input_shape, yolo_version)
            return (len(centroids), centroids)

        # calculate new centroids
        centroid_sums = np.zeros((k, dim), np.float)  # 初始化以便对每个簇的w,h求和
        for i in range(N):
            centroid_sums[assignments[i]] += X[i]  # 将每个簇中的ground truth的w和h分别累加
        for j in range(k):  # 对簇中的w,h求平均
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j) + 1)

        prev_assignments = assignments.copy()
        old_D = D.copy()

def mainx(kx):
    parser = argparse.ArgumentParser()
    parser.add_argument('-filelist', default=r'/home/alanc/Documents/faster-rcnn.pytorch-pytorch-1.0/data/VOCdevkit2007/VOC2007/Annotations',
                        help='path to filelist\n')
    parser.add_argument('-output_dir', default=r'E:\BaiduNetdiskDownload\darknetHG8245', type=str,
                        help='Output anchor directory\n')
    parser.add_argument('-num_selected', default=500, type=int,
                       help='number of selected items\n')
    parser.add_argument('-num_clusters', default=kx, type=int,
                        help='number of clusters\n')
    '''
    需要注意的是yolov2输出的值比较小是相对特征图来说的，
    yolov3输出值较大是相对原图来说的，
    所以yolov2和yolov3的输出是有区别的
    '''
    parser.add_argument('-yolo_version', default='yolov2', type=str,
                        help='yolov2 or yolov3\n')
    parser.add_argument('-yolo_input_shape', default=416, type=int,
                        help='input images shape，multiples of 32. etc. 416*416\n')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    annotation_dims = xml_extract(args.filelist, args.num_selected)
    """
    f = open(args.filelist)

    lines = [line.rstrip('\n') for line in f.readlines()]

    annotation_dims = []

    for line in lines:
        line = line.replace('JPEGImages', 'labels')
        line = line.replace('.jpg', '.txt')
        line = line.replace('.png', '.txt')
        print(line)
        f2 = open(line)
        for line in f2.readlines():
            line = line.rstrip('\n')
            w, h = line.split(' ')[3:]
            # print(w,h)
            # what only need is the w and h
            annotation_dims.append((float(w), float(h)))
    annotation_dims = np.array(annotation_dims)  # 保存所有ground truth框的(w,h)
    """
    eps = 0.005

    if args.num_clusters == 0:
        for num_clusters in range(1, 11):  # we make 1 through 10 clusters
            anchor_file = join(args.output_dir, 'anchors%d.txt' % (num_clusters))

            indices = [random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
            for l in indices:
                centroids.append(annotation_dims[l])
            #centroids = annotation_dims[indices]
            nx, center = kmeans(annotation_dims, centroids, eps, anchor_file, args.yolo_input_shape, args.yolo_version)
            print('centroids len()', len(centroids))
    else:
        anchor_file = join(args.output_dir, 'anchors%d.txt' % (args.num_clusters))
        indices = [random.randrange(len(annotation_dims)) for i in range(args.num_clusters)]
        #print(indices)
        for l in indices:
            centroids.append(annotation_dims[l])
        nx, center = kmeans(annotation_dims, centroids, eps, anchor_file, args.yolo_input_shape, args.yolo_version)
        #print('centroids len()', len(centroids))
    return (nx, center)

if __name__ == "__main__":
    kx= 5
    nx,center = mainx(kx)
    print("nx,center:",nx,center)

# if __name__ == "__main__":
#     main(sys.argv,ks)