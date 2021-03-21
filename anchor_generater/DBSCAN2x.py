# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: DBSCAN2x.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,八月 08
# ---
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle  ##python自带的迭代器模块
from sklearn.preprocessing import StandardScaler
from xml_extract2 import xml_extract

##设置分层聚类函数
def dbscanx(path, m):
    dataMat = xml_extract(path, m)
    for dataMats in dataMat:
        dataMats = list(map(int, dataMats))
    #print(type(dataMat), "dataMate", dataMat)
    #print("dataMat[10]:", dataMat[10])
    db = DBSCAN(eps=20, min_samples=20)
    X = dataMat
    ##训练数据
    db.fit(X)
    ##初始化一个全是False的bool类型的数组
    core_samples_mask = np.zeros_like(db.labels_)
    '''
   这里是关键点(针对这行代码：xy = X[class_member_mask & ~core_samples_mask])：
   db.core_sample_indices_  表示的是某个点在寻找核心点集合的过程中暂时被标为噪声点的点(即周围点
   小于min_samples)，并不是最终的噪声点。在对核心点进行联通的过程中，这部分点会被进行重新归类(即标签
   并不会是表示噪声点的-1)，也可也这样理解，这些点不适合做核心点，但是会被包含在某个核心点的范围之内
   '''
    core_samples_mask[db.core_sample_indices_] = 1

    ##每个数据的分类
    lables = db.labels_
    #print("labels",lables)
    #print(len(lables))

    ##分类个数：lables中包含-1，表示噪声点
    n_clusters_ = len(np.unique(lables)) - (1 if -1 in lables else 0)
    return n_clusters_
