#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   metrics.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, 2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/18 0:32   Bot Zhao      1.0         None
"""

# import lib
import numpy as np


def offset_MSE(data1, data2):
    data1_ca = np.reshape(data1,(-1,data1.shape[-1]))
    data2_ca = np.reshape(data2,(-1,data2.shape[-1]))
    MSE_12 = np.sum((data1_ca-data2_ca)**2,axis=1)/data2_ca.shape[0]
    MSE_1 = np.sum(data1_ca**2,axis=1)/data2_ca.shape[0]
    return np.mean(MSE_12/MSE_1)


def cc(data):
    data = np.reshape(data,(-1,data.shape[-1]))
    return np.mean(np.corrcoef(data.T))

#     pass
test_data = np.random.rand(60,60,56)
print(cc(test_data))