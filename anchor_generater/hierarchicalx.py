# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: hierarchical.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,八月 06
# ---
import sys, os
import numpy as np
from xml_extract2 import xml_extract


class Hierarchical:
    def __init__(self, center, left=None, right=None, flag=None, distance=0.0):
        self.center = center
        self.left = left
        self.right = right
        self.flag = flag
        self.distance = distance


def traverse(node):
    if node.left == None and node.right == None:
        return [node.center]
    else:
        return traverse(node.left) + traverse(node.right)


def distance(v1, v2):
    if len(v1) != len(v2):
        print("出现错误了")
    distance = 0
    for i in range(len(v1)):
        distance += (v1[i] - v2[i]) ** 2
    distance = np.sqrt(distance)
    return distance


def hcluster(data, n):
    if len(data) <= 0:
        print('invalid data')
    clusters = [Hierarchical(data[i], flag=i) for i in range(len(data))]
    print(clusters)
    distances = {}
    min_id1 = None
    min_id2 = None
    currentCluster = -1

    while len(clusters) > n:
        minDist = 100000000000000

        for i in range(len(clusters) - 1):
            for j in range(i + 1, len(clusters)):
                # print(distances.get((clusters[i], clusters[j])))
                if distances.get((clusters[i], clusters[j])) == None:
                    distances[(clusters[i].flag, clusters[j].flag)] = distance(clusters[i].center, clusters[j].center)

                if distances[(clusters[i].flag, clusters[j].flag)] <= minDist:
                    min_id1 = i
                    min_id2 = j
                    minDist = distances[(clusters[i].flag, clusters[j].flag)]

        if min_id1 != None and min_id2 != None and minDist != 1000000000:
            newCenter = [(clusters[min_id1].center[i] + clusters[min_id2].center[i]) / 2 for i in
                         range(len(clusters[min_id2].center))]
            newFlag = currentCluster
            currentCluster -= 1
            newCluster = Hierarchical(newCenter, clusters[min_id1], clusters[min_id2], newFlag, minDist)
            del clusters[min_id2]
            del clusters[min_id1]
            clusters.append(newCluster)
        finalCluster = [traverse(clusters[i]) for i in range(len(clusters))]
        return finalCluster


if __name__ == '__main__':
    """
    data = [[123, 321, 434, 4325, 345345], [23124, 141241, 434234, 9837489, 34743], \
            [128937, 127, 12381, 424, 8945], [322, 4348, 5040, 8189, 2348], \
            [51249, 42190, 2713, 2319, 4328], [13957, 1871829, 8712847, 34589, 30945],
            [1234, 45094, 23409, 13495, 348052], [49853, 3847, 4728, 4059, 5389]]
    """
    path = '/home/alanc/Documents/faster-rcnn.pytorch-pytorch-1.0/data/VOCdevkit2007/VOC2007/Annotations'
    m = num_items_selected =1500
    data = xml_extract(path, m)
    print("The items loaded successfully from " + path)
    # print(len(data))
    finalCluster = hcluster(data, 3)
    print(finalCluster)
    print(len(finalCluster))