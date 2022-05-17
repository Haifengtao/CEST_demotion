#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   read_data.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, 2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/18 0:21   Bot Zhao      1.0         None
"""

# import lib
import os
import numpy as np
import nibabel as nib
from bruker2nifti.converter import Bruker2Nifti
import SimpleITK as sitk


def readpaths(rootdir):
    list_file = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    pathes = []
    for i in list_file:
        if i[0] != 'c':  # 列出文件名不以c开头的文件
            path = os.path.join(rootdir, i)
            pathes.append(path)
    return pathes


def convert(path):
    pfo_study_in = path
    pfo_study_out = path
    bru = Bruker2Nifti(pfo_study_in, pfo_study_out, study_name='my_study')

    # select the options (attributes) you may want to change - the one shown below are the default one:
    bru.verbose = 2
    bru.correct_slope = True
    bru.get_acqp = True
    bru.get_method = True
    bru.get_reco = False
    bru.nifti_version = 1
    bru.qform_code = 1
    bru.sform_code = 2
    bru.save_human_readable = True
    bru.save_b0_if_dwi = False
    # Check that the list of scans and the scans names automatically selected makes some sense:
    print(bru.scans_list)
    print(bru.list_new_name_each_scan)
    # call the function convert, to convert the study:
    bru.convert()


def get_data_mouse(studypath, cest06):
    cest_data = studypath + "my_study_" + str(cest06) + "\\my_study_" + str(cest06) + ".nii.gz"
    par_data = studypath + "my_study_" + str(cest06) + "\\my_study_" + str(cest06) + "_method.npy"
    par = {}
    data = nib.load(cest_data).dataobj

    data_m = data[:, :, :-1]
    data_s0 = data[:, :, -1]
    par_data = np.load(par_data, allow_pickle=True).item()
    par['ppgFreqList1'] = par_data['ppgFreqList1'][:-1]
    par['FrqWork'] = par_data['FrqWork'][0]
    par['dim'] = data_m.shape
    return data_m, data_s0, par


def get_data_z(data0, s0):
    data = np.reshape(data0, (-1, data0.shape[-1]))
    s0 = np.reshape(s0, (-1, 1))
    data_z = data/s0
    # data_z = cv2.GaussianBlur(data_z, (1,3), 0)
    data_z = np.reshape(data_z, data0.shape)
    return data_z


def get_data_human(cest_data, dims, offset=1):
    """
    :return:
    """
    if dims == 2:
        data = nib.load(cest_data[0]).dataobj
        data_m = data[:, :, 1:]
        data_s0 = data[:, :, 0]
    elif dims == 3:
        cases = cest_data
        name = "_".join(cases[0].split("_")[1:])
        assert offset + 1 == len(cases), "Check your offset or cest images number"
        data_s0 = nib.load(cases[0]).dataobj[:]
        x, y, z = data_s0.shape
        data_m = np.zeros((x, y, z, offset))
        for idx, i in enumerate(cases[1:]):
            temp = nib.load(i).dataobj
            data_m[:, :, :, idx] = temp
    else:
        raise Exception("check the dimention!")
    data_s0 = data_s0.astype("float32")
    data_s0[data_s0 == 0] = 0.001
    data_z = get_data_z(data_m, data_s0)
    return data_z, data_m, data_s0



def trans_4d_image2nii(root_dir, out_dir, name="zbt", offset=1):
    """
    convert the SIMENS's dicom (.IMA file) to nii.gz
    :param root_dir:
    :param out_dir:
    :param name:
    :param offset: the number of 3d file. The default value is 1, it's meaning that there is one 3d file
     under dictionary.
    :return: None
    """

    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(os.path.join(root_dir, os.listdir(root_dir)[0]))
    print(series_id)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(root_dir, series_id[0])

    z_nums = len(series_file_names) // offset
    for i in range(0, len(series_file_names) + 1, z_nums):
        if i == 0:
            continue
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        series_reader = sitk.ImageSeriesReader()  # 读取数据端口
        series_reader.SetFileNames(series_file_names[i - z_nums:i])
        images = series_reader.Execute()  # 读取数据
        sitk.WriteImage(images, out_dir + "/" + str(i // z_nums) + "_" + name + ".nii.gz")  # 保存为nii


def dicom2nii(rootdir, cur_name, new_name):
    if not os.path.isdir(os.path.join(rootdir, os.listdir(rootdir)[0])):
        outdir = rootdir.replace(cur_name, new_name)
        print(os.listdir(rootdir)[0])
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        os.system("D: && cd D:\jupyter\cest_pipeline\external tools && dcm2niix -f " +
                  rootdir + " -m n -p y -z y -o " + outdir)
    else:
        for i in os.listdir(rootdir):
            print(i)
            dicom2nii(os.path.join(rootdir, i), cur_name, new_name)


def read_array_nii(input_dir):
    image = sitk.ReadImage(input_dir)
    array = sitk.GetArrayFromImage(image)
    array = array.transpose((2, 1, 0))
    return image, array

pathes = [r'D:\research\CEST\cest_data\C6_glioma_CEST\20200614-19\20200614_092412_CEST_rat_20200614rat1_1_1',
          r'D:\research\CEST\cest_data\C6_glioma_CEST\20200614-19\20200615_112220_CEST_rat_20200614rat2_1_6',
          r'D:\research\CEST\cest_data\C6_glioma_CEST\20200614-19\20200615_084515_CEST_rat_20200614rat3_1_5',
          r"D:\research\CEST\cest_data\C6_glioma_CEST\20200614-19\20200615_145508_CEST_rat_20200614rat4_1_8",
          r'D:\research\CEST\cest_data\C6_glioma_CEST\20200614-19\20200616_092217_CEST_rat_20200616rat5_1_9']

ID = ['rat01', 'rat02', 'rat03', "rat04", 'rat05']
if __name__ == '__main__':
    print(pathes)
    for (path, idx) in zip(pathes, ID):
        print(path)
        print(idx)
        try:
            convert(path)
        except FileExistsError:
            continue
