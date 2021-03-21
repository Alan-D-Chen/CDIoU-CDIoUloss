#!/usr/bin/python
# -*- coding: UTF-8 -*-
# get annotation object bndbox location
import os
import cv2
import random

try:
    import xml.etree.cElementTree as ET  # 解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET

BndBoxLoc = []
"""
##get object annotation bndbox loc start
def GetAnnotBoxLoc(AnotPath):  # AnotPath VOC标注文件路径
    tree = ET.ElementTree(file=AnotPath)  # 打开文件，解析成一棵树型结构
    root = tree.getroot()  # 获取树型结构的根
    ObjectSet = root.findall('object')  # 找到文件中所有含有object关键字的地方，这些地方含有标注目标
    ObjBndBoxSet = {}  # 以目标类别为关键字，目标框为值组成的字典结构
    #list1= []
    for Object in ObjectSet:
        ObjName = Object.find('name').text
        BndBox = Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text)  # -1 #-1是因为程序是按0作为起始位置的
        y1 = int(BndBox.find('ymin').text)  # -1
        x2 = int(BndBox.find('xmax').text)  # -1
        y2 = int(BndBox.find('ymax').text)  # -1
        BndBoxLoc.append((float(x2 - x1),float(y2 - y1)))

    return (BndBoxLoc)
"""
def xml_extract(path, m):
    i = 0
    pathes = []
    #path = "/home/alanc/Documents/faster-rcnn.pytorch-pytorch-1.0/data/VOCdevkit2007/VOC2007/Annotations"
    for x in os.listdir(path):
        #print(x)
        i = i+1
        pathes.append(x)

    selected_path = random.sample(pathes, m)
    for l in selected_path:
        paths = os.path.join(path, l)
        path2 = paths

        tree = ET.ElementTree(file=path2)  # 打开文件，解析成一棵树型结构
        root = tree.getroot()  # 获取树型结构的根
        ObjectSet = root.findall('object')  # 找到文件中所有含有object关键字的地方，这些地方含有标注目标
        ObjBndBoxSet = {}  # 以目标类别为关键字，目标框为值组成的字典结构
        # list1= []
        for Object in ObjectSet:
            ObjName = Object.find('name').text
            BndBox = Object.find('bndbox')
            x1 = int(BndBox.find('xmin').text)  # -1 #-1是因为程序是按0作为起始位置的
            y1 = int(BndBox.find('ymin').text)  # -1
            x2 = int(BndBox.find('xmax').text)  # -1
            y2 = int(BndBox.find('ymax').text)  # -1
            BndBoxLoc.append([float(x2 - x1), float(y2 - y1)])
    print("Dataset :", path)
    print("There are ",i , " items in this dataset, ", len(selected_path),"items selected for clustering.")
    print("There are ",len(BndBoxLoc),"potential anchors in ",len(selected_path),"selected items.")
    return (BndBoxLoc)