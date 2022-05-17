#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   B0_correct.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, 2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/18 0:37   Bot Zhao      1.0         None
"""

# import lib
import math
import numpy as np
from scipy import interpolate
from tqdm import tqdm


def get_data_z(data0, s0):
    data = np.reshape(data0, (-1, data0.shape[-1]))
    s0 = np.reshape(s0, (-1, 1))
    data_z = data / s0
    # data_z = cv2.GaussianBlur(data_z,(1,3),0)
    data_z = np.reshape(data_z, data0.shape)
    return data_z


def b0_correction(data0, par):
    """
    self-adapting B0 correction

    """
    data = np.reshape(data0, (-1, data0.shape[-1]))
    voxel_roi_num = par['dim'][0] * par['dim'][1]
    B0_shift = np.zeros(par['dim'][:-1])
    new_data_z = np.zeros(data.shape)
    for i in range(voxel_roi_num):
        offset = np.array(par['ppgFreqList1'])
        f = interpolate.interp1d(offset, data[i, :], kind="cubic")
        offset_new = np.linspace(np.min(offset), np.max(offset), 10000)
        z_new = f(offset_new)
        B0_shift[math.floor(i / par['dim'][1]), i % par['dim'][1]] = offset_new[np.argmin(z_new)]
        f_new = interpolate.interp1d(offset - offset_new[np.argmin(z_new)], data[i, :], kind="cubic",
                                     bounds_error=False, fill_value="extrapolate")
        b0crrected_Z = f_new(offset)
        new_data_z[i, :] = b0crrected_Z
    new_data_z = np.reshape(new_data_z, data0.shape)
    return B0_shift, new_data_z


def b0_correction_inter(data0, mask, offset, mode="2D", interp_num=5000):
    """
    self-adapting B0 correction

    """
    if mode == "2D":
        x, y, _ = data0.shape
        voxel_roi_num = x * y
        B0_shift = np.zeros((x, y))
    elif mode == "3D":
        x, y, z, _ = data0.shape
        voxel_roi_num = x * y * z
        B0_shift = np.zeros((x, y, z))
    data = np.reshape(data0, (-1, data0.shape[-1]))
    new_data_z = np.zeros(data0.shape)
    for i in tqdm(range(voxel_roi_num)):
        if mode == "3D":
            temp_x = i // (y*z)
            temp_y = (i-temp_x*y*z)//z
            temp_z = i-temp_x*y*z-temp_y*z
            if mask[temp_x, temp_y, temp_z] == 0:
                continue
            else:
                f = interpolate.interp1d(offset, data[i, :], kind="linear")
                offset_new = np.linspace(np.min(offset), np.max(offset), interp_num)
                z_new = f(offset_new)
                B0_shift[temp_x, temp_y, temp_z] = offset_new[np.argmin(z_new)]
                f_new = interpolate.interp1d(offset - offset_new[np.argmin(z_new)], data[i, :], kind="linear",
                                             bounds_error=False, fill_value="extrapolate")
                b0crrected_Z = f_new(offset)
                new_data_z[temp_x, temp_y, temp_z, :] = b0crrected_Z
        if mode == "2D":
            if mask[math.floor(i / y), i % y] == 0:
                continue
            f = interpolate.interp1d(offset, data[i, :], kind="linear")
            offset_new = np.linspace(np.min(offset), np.max(offset), interp_num)
            z_new = f(offset_new)
            B0_shift[math.floor(i / y), i % y] = offset_new[np.argmin(z_new)]
            f_new = interpolate.interp1d(offset - offset_new[np.argmin(z_new)], data[i, :], kind="linear",
                                         bounds_error=False, fill_value="extrapolate")
            b0crrected_Z = f_new(offset)
            new_data_z[math.floor(i / y), i % y, :] = b0crrected_Z
    return B0_shift, new_data_z, mask


