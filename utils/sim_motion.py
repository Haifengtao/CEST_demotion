#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   sim_motion.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, 2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/18 0:29   Bot Zhao      1.0         None
"""

# import lib
import numpy as np
import cv2


class simulatecestMotion2D(object):
    def __init__(self, offset, shift, degree):
        self.offset = offset
        self.shift = shift
        self.degree = degree
        self.new_image = None

    def simulated_cest(self, data):
        self.new_image = np.zeros(data.shape)
        for i in range(data.shape[-1]):
            temp_img = data[:, :, i]
            self.new_image[:, :, i] = self.rotate(self.translate(temp_img, self.shift[i, :]), self.degree[i])
        return self.new_image

    def translate(self, image, shift):
        M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return shifted

    def rotate(self, image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]
        # if the center is None, initialize it as the center of the image
        if center is None:
            center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from read_data import *

    rat01 = r"D:\research\CEST\cest_data\C6_glioma_CEST\20200614-19\20200614_092412_CEST_rat_20200614rat1_1_1\\my_study\\"
    rat02 = r"D:\research\CEST\cest_data\C6_glioma_CEST\20200614-19\20200615_112220_CEST_rat_20200614rat2_1_6\\my_study\\"
    rat03 = r"D:\research\CEST\cest_data\C6_glioma_CEST\20200614-19\20200615_084515_CEST_rat_20200614rat3_1_5\\my_study\\"
    rat04 = r"D:\research\CEST\cest_data\C6_glioma_CEST\20200614-19\20200615_145508_CEST_rat_20200614rat4_1_8\\my_study\\"
    rat05 = r"D:\research\CEST\cest_data\C6_glioma_CEST\20200614-19\20200616_092217_CEST_rat_20200616rat5_1_9\\my_study\\"

    cest01 = 24
    cest02 = 16
    cest03 = 15
    cest04 = 17
    cest05 = 38

    data_m, data_s0, par = get_data(rat02, cest02)
    offset = par['dim'][-1]
    rotate_angle = np.random.normal(0,1,offset)
    shift_x = np.expand_dims(np.random.normal(0,1,offset),1)
    shift_y = np.expand_dims(np.random.normal(0,1,offset),1)
    shift = np.concatenate((shift_x,shift_y),axis=1)

    plt.plot(rotate_angle,label="rotate")
    plt.plot(shift_x,label='shift_x')
    plt.plot(shift_y,label='shift_y')
    plt.legend()
    sm = simulatecestMotion2D(offset,shift,rotate_angle)
    simulated_data_m = sm.simulated_cest(data_m)