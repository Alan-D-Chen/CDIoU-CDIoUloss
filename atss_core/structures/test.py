# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site:
# @File: test.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,十月 29
# ---
import numpy as np
import time
import torch
import torch.nn.functional as F
#from sqrtm import sqrtm


n = torch.rand([3, 1, 4]) + 2
m = torch.rand([1, 5, 4])
print("n.shape & n:\n", n.shape,"\n", n)
print("m.shape & m:\n",m.shape,"\n",m)
tmp = (n - m) ** 2
print("tmp.shape:\n",tmp.shape)
print("tmp1:\n",tmp)
#tmp = tmp[0:-1:2] + tmp[1:-1:2]
print(tmp[:,:,0::2],"\n", tmp[:,:,1::2])
tmps = tmp[:,:,0::2] + tmp[:,:,1::2]
print("tmps:\n",tmps)

tmp = np.sqrt(tmps)
print("tmp3->tmps:\n",tmp)
print(n[:,:,0])
print(n[:,:,1])

ns = torch.cat((n,n[:,:[0,3]],n[:,:[1,2]]),dim = 1)
print("ns:\n",ns)

