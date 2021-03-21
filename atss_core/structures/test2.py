# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: test2.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,十月 29
# ---
import numpy as np
import time
import torch
import torch.nn.functional as F
#from sqrtm import sqrtm

# Get the minimum bounding box (including region proprosal and ground truth) #
n = torch.rand([3, 4])
m = torch.rand([5, 4])

print("n.shape[0],n.shape[1]:",n.shape[0],n.shape[1])
print("m.shape[0],m.shape[1]:",m.shape[0],m.shape[1])
print("##################################################################-------->>")
nss = n.unsqueeze(1)
nss = nss.expand(3,5,4)
print("n & n.shape:\n",n,"\n",n.shape)
mss = m.unsqueeze(0)
mss = mss.expand(3,5,4)
print("m & m.shape:\n",m,"\n",m.shape)
nms = torch.cat((nss,mss),dim=2)
print("nms & nms.shape:\n",nms,"\n",nms.shape)

A = nms[:,:,[0,4]]
B = nms[:,:,[1,5]]
C = nms[:,:,[2,6]]
D = nms[:,:,[3,7]]
print("#########################")
print("A & A.shape:\n",A,"\n",A.shape)
Am = torch.max(A,2)[0]
print("Am & Am.shape:\n",Am,"\n",Am.shape)
print(B,B.shape)
Bm = torch.max(B,2)[0]
print(Bm)
print(C,C.shape)
Cm = torch.max(C,2)[0]
print(Cm)
print(D,D.shape)
Dm = torch.max(D,2)[0]
print(Dm)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
AB = torch.cat((Am,Bm),dim = 1)
CD = torch.cat((Cm,Dm),dim = 1)
print("AB & AB.shape:\n",AB,"\n",AB.shape)
print("CD & CD.shape:\n",CD,"\n",CD.shape)
XY = torch.zeros([Am.shape[0],Am.shape[1],4])
print("XY & XY.shape:\n",XY,"\n",XY.shape)
XY[:,:,0]= Am
XY[:,:,1]= Bm
XY[:,:,2]= Cm
XY[:,:,3]= Dm
print("XY & XY.shape:\n",XY,"\n",XY.shape)
XYx = (XY[:,:,[2,3]] - XY[:,:,[0,1]]) ** 2
print("XYx & XYx.shape:\n",XYx,"\n",XYx.shape)
XxY = XYx[:,:,0]+XYx[:,:,1]
XYs = XxY.sqrt()                       ###########################-> to get square root
print("XYs & XYs.shape:\n",XYs,"\n",XYs.shape)

#######################################################
# The average distance between GT and RP is obtained #
nms = torch.cat((n, m),dim=0)
print("nms & nms.shape:\n",nms,"\n",nms.shape)
#########################################################
print("n.shape & n:\n", n.shape,"\n", n)
n0 = n[:,[0,3]] #.unsqueeze(1)
n1 = n[:,[1,2]] #.unsqueeze(1)
print("n0 & n0.shape:\n",n0,"\n",n0.shape)
print("n1 & n1.shape:\n",n1,"\n",n1.shape)
ns = torch.cat((n,n0,n1),dim = 1)
print("ns.shape & ns:\n", ns.shape,"\n", ns)
######################################################
print("m.shape & m:\n", m.shape,"\n", m)
m0 = m[:,[0,3]] #.unsqueeze(1)
m1 = m[:,[1,2]] #.unsqueeze(1)
print("m0 & m0.shape:\n",m0,"\n",m0.shape)
print("m1 & m1.shape:\n",m1,"\n",m1.shape)
ms = torch.cat((m,m0,m1),dim = 1)
print("ms.shape & ms:\n", ms.shape,"\n", ms)
################################################################
ns = ns.unsqueeze(1)
ms = ms.unsqueeze(0)
print("ns.shape & ns->unsqueeze:\n", ns.shape,"\n", ns)
print("ms.shape & ms->unsqueeze:\n", ms.shape,"\n", ms)

n = ns
m = ms
print("n.shape & n:\n", n.shape,"\n", n)
print("m.shape & m:\n",m.shape,"\n",m)
tmp = (n - m) ** 2
print("tmp.shape:\n",tmp.shape)
print("tmp1->(n - m) ** 2:\n",tmp)
#tmp = tmp[0:-1:2] + tmp[1:-1:2]
#print(tmp[:,:,0::2],"\n", tmp[:,:,1::2])
tmps = tmp[:,:,0::2] + tmp[:,:,1::2]
print("tmps and tmps.shape:\n",tmps,"\n",tmps.shape)
#tmp = np.sqrt(tmps)               #######################-> to get square root
tmp = tmps.sqrt()
print("tmp3->tmps-square root:\n",tmp,"\n",tmp.shape)
#tmp = tmp.mean(axis=2, keepdim=False)/4
tmp = torch.mean(tmp,dim = 2,keepdim = False)/4
print("tmp->mean:\n",tmp,"\n",tmp.shape)

# get DIoU+ #
print("DIoU+ and DIoU.shape:\n",tmp/XYs,"\n", (tmp/XYs).shape)

