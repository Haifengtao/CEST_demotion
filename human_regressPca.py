#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   human_regressPca.py
@Contact :   760320171@qq.com
@License :   (C)Copyright, 2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/18 0:21   Bot Zhao      1.0         None
"""

# import lib
import numpy as np
from utils import *
import glob


name = "NC21275_NC21275"  # NC21262_NC21262
root_dir = r".\data\NC21275_NC21275\RESEARCH_NORMAL_COHORT_20210702_103648_548000"
cest_data = r".\data\NC21275_NC21275\RESEARCH_NORMAL_COHORT_20210702_103648_548000\APT_0_8_ORIG_0039"
out_data = r".\data\NC21275_NC21275\RESEARCH_NORMAL_COHORT_20210702_103648_548000\APT_0_8_ORIG_0039_nii"


move_data = np.loadtxt(out_data+"//rp_01_"+name+".txt")

offset = [-50.0, -25.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.5, -4.0, -3.75, -3.5, -3.25, -3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 25.0, 50.0]
print("running:--------------Get_z_spectrum!!-------------------------")
regis_mask_dir = root_dir + "/mask.nii.gz"
mask, mask_array = read_data.read_array_nii(regis_mask_dir)
imgs = glob.glob(out_data+"/r*nii")
data_z, data_m, data_s0 = read_data.get_data(imgs, dims=3, offset=len(offset))

print("running:--------------B0 Correction!!-------------------------")
B0_shift, new_data_z, test_mask = B0_correct.b0_correction_inter(data_z, mask_array, offset, mode="3D", interp_num=5000)

demotioned_z = demotion_methods.regress_PCA(offset, new_data_z, mask_array, move_data)
APT_asy = quantifi_cest.get_MTasy(3.5, new_data_z, mask_array, offset)
APT_asy_PCA_reg = quantifi_cest.get_MTasy(3.5, demotioned_z, mask_array, offset)