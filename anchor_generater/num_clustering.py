# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site:
# @File: num_clustering.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,八月 07
# ---
import pandas as pd
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering, DBSCAN, spectral_clustering
from sklearn import metrics
from sklearn.metrics import calinski_harabasz_score
import matplotlib.pyplot as plt
from xml_extract2 import xml_extract
from DBSCAN2x import dbscanx
import numpy as np
from mean_shiftx import mean_shift
from k_meansx import mainx
from prettytable import PrettyTable
import math

path = '/home/alanc/Documents/faster-rcnn.pytorch-pytorch-1.0/data/VOCdevkit2007/VOC2007/Annotations'
m = num_items_selected = 500
Zdata = xml_extract(path, m)

## just for AgglomerativeClustering
linkages = ['ward', 'average', 'complete']
## just for spectral_clustering
##变换成矩阵，输入必须是对称矩阵
metrics_metrix = (-1 * metrics.pairwise.pairwise_distances(Zdata)).astype(np.int32)
metrics_metrix += -1 * metrics_metrix.min()

## SSE sum of the squared errors
sse_list = []
sse_list2 = []
sse_list3 = []
K = range(1, 15)
for k in range(1,15):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(Zdata)
    sse_list.append([k, kmeans.inertia_, 0])   #model.inertia_返回模型的误差平方和，保存进入列表

# Calculate the slope difference between the two sides of a point #
for i in range(1,13):
    sse_list[i][2] = (sse_list[i][1]-sse_list[i-1][1])/(sse_list[i][0]-sse_list[i-1][0]) - (sse_list[i+1][1]-sse_list[i][1])/(sse_list[i+1][0]-sse_list[i][0])

for i in range(len(sse_list)-1):
    # 获得第一个元素,将其与剩余的元素进行比较,如果大于即交换位置
    for j in range(i+1,len(sse_list)):
        if sse_list[i][2]>sse_list[j][2]:
            temp=sse_list[j]
            sse_list[j]=sse_list[i]
            sse_list[i]=temp
#print("The best number for K-means clustering by SSE(sum of the squared errors) is ", sse_list[0][0])

## 轮廓系数
## silhouette_score & Calinski-Harabaz Index
clusters = range(2,15)

sc_scores = []
sc_scores2 = []

ac_scores = []
ac_scores2 = []

pc_scores = []
pc_scores2 = []
for k in clusters:
    kmeans_model = KMeans(n_clusters=k).fit(Zdata)
    ac_model = AgglomerativeClustering(linkage=linkages[2], n_clusters=k).fit(Zdata)
    pc_model = spectral_clustering(metrics_metrix, n_clusters=k)

    sc_score = metrics.silhouette_score(Zdata, kmeans_model.labels_,sample_size=10000, metric='euclidean')
    sc_scores.append([k, sc_score])
    sc_score2 = metrics.calinski_harabasz_score(Zdata, kmeans_model.labels_)
    sc_scores2.append([k, sc_score2])
    ## Agglomerative
    ac_score = metrics.silhouette_score(Zdata, ac_model.labels_, sample_size=10000, metric='euclidean')
    ac_scores.append([k, ac_score])
    ac_score2 = metrics.calinski_harabasz_score(Zdata, ac_model.labels_)
    ac_scores2.append([k, ac_score2])
    ## spectral_clustering
    pc_score = metrics.silhouette_score(Zdata, pc_model, sample_size=10000, metric='euclidean')
    pc_scores.append([k, pc_score])
    pc_score2 = metrics.calinski_harabasz_score(Zdata, pc_model)
    pc_scores2.append([k, pc_score2])

for i in range(len(sc_scores)-1):
    # 获得第一个元素,将其与剩余的元素进行比较,如果小于即交换位置
    for j in range(i+1,len(sc_scores)):
        if sc_scores[i][1]<sc_scores[j][1]:
            temp=sc_scores[j]
            sc_scores[j]=sc_scores[i]
            sc_scores[i]=temp
        if sc_scores2[i][1] < sc_scores2[j][1]:
            temp = sc_scores2[j]
            sc_scores2[j] = sc_scores2[i]
            sc_scores2[i] = temp

        if ac_scores[i][1]<ac_scores[j][1]:
            temp=ac_scores[j]
            ac_scores[j]=ac_scores[i]
            ac_scores[i]=temp
        if ac_scores2[i][1] < ac_scores2[j][1]:
            temp = ac_scores2[j]
            ac_scores2[j] = ac_scores2[i]
            ac_scores2[i] = temp

        if pc_scores[i][1]<pc_scores[j][1]:
            temp=pc_scores[j]
            pc_scores[j]=pc_scores[i]
            pc_scores[i]=temp
        if pc_scores2[i][1] < pc_scores2[j][1]:
            temp = pc_scores2[j]
            pc_scores2[j] = pc_scores2[i]
            pc_scores2[i] = temp
        # if sc_scores3[i][1] < sc_scores3[j][1]:
        #     temp = sc_scores3[j]
        #     sc_scores3[j] = sc_scores3[i]
        #     sc_scores3[i] = temp
num_cluster, cluster_ids = mean_shift(Zdata, 70.0)
num_cluster_dbscanx = dbscanx(path, m)
# #print(sc_scores)
# print("The best number for K-means clustering by Silhouette Coefficient is ", sc_scores[0][0])
# #print(sc_scores2)
# print("The best number for K-means clustering by Calinski-Harabaz Index is ", sc_scores2[0][0])
#
# #print(ac_scores)
# print("The best number for Agglomerative clustering by Silhouette Coefficient is ", ac_scores[0][0])
# #print(ac_scores2)
# print("The best number for Agglomerative clustering by Calinski-Harabaz Index is ", ac_scores2[0][0])
#
# #print(pc_scores)
# print("The best number for Spectral clustering by Silhouette Coefficient is ", pc_scores[0][0])
# #print(pc_scores2)
# print("The best number for Spectral clustering by Calinski-Harabaz Index is ", pc_scores2[0][0])
#
# print("The best number for DBSCAN clustering is ", num_cluster_dbscanx)

num_clusterx = (sse_list[0][0] + sc_scores[0][0] + sc_scores2[0][0] + ac_scores[0][0] + ac_scores2[0][0]
                + pc_scores[0][0] + pc_scores2[0][0] + num_cluster_dbscanx)/8
num_clusterx = int(math.ceil(num_clusterx))
#################################
x = PrettyTable(["Method for clustering", "Automatic presentation", "SSE(sum of the squared errors)", "Silhouette Coefficient", "Calinski-Harabaz Index"])
x.align["Method for clustering"] = "l" # Left align city names
x.padding_width = 1 # One space between column edges and contents (default)
x.add_row(["K-means/PAM",0,sse_list[0][0],sc_scores[0][0],sc_scores2[0][0]])
x.add_row(["Hierarchical",0, 0,ac_scores[0][0],ac_scores2[0][0]])
x.add_row(["Spectral",0,0,pc_scores[0][0],pc_scores2[0][0]])
x.add_row(["DBSCANx",num_cluster_dbscanx,0,0,0])
x.add_row(["Mean-shift", num_cluster, 0, 0,0])
print(x)
print("Based on the above information, the following suggestions by the clustering system are : \n ")
nx, centerx = mainx(num_clusterx)
print("Plan 1:\n", "Number of clusters(K-means/PAM):",nx,"\n Cluster center:")
for l in range(len(centerx)):
    print(centerx[l])
print("Plan 2:\n", "Number of clusters(mean shift):",num_cluster,"\n Cluster center:")
for l in range(len(cluster_ids)):
    print(cluster_ids[l])
