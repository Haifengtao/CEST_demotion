#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   quantifi_cest.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, 2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/18 0:54   Bot Zhao      1.0         None
"""

# import lib
from scipy import interpolate
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm


def get_MTasy(ppm, data_z, mask, offset):
    cest_effect = np.zeros(mask.shape)
    for x in tqdm(range(mask.shape[0])):
        for y in range(mask.shape[1]):
            for z in range(mask.shape[2]):
                if mask[x, y, z] == 0:
                    continue
                f = interpolate.interp1d(offset, data_z[x, y, z, :], kind="cubic")
                if ppm != 0:
                    cest_effect[x, y, z] = f(-ppm) - f(ppm)
                else:
                    cest_effect[x, y, z] = f(0)
    return cest_effect
