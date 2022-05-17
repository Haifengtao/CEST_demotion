#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   demotion_methods.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, 2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/18 0:33   Bot Zhao      1.0         None
"""

# import lib
import numpy as np
from pystackreg import StackReg
from tqdm import tqdm
from numpy import linalg as la
from metrics import *
import ants
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def reg_pystackreg(moved, fixed, mode):
    if mode == 'Affine':
        sr = StackReg(StackReg.AFFINE)
        out = sr.register_transform(fixed, moved)
    elif mode == 'Rigid':
        sr = StackReg(StackReg.RIGID_BODY)
        out = sr.register_transform(fixed, moved)
    else:
        raise('ERROR REGISTRATION MODE!')
    return out


def reg_ants(moved, fixed, mode='SyN'):
    moved = ants.from_numpy(moved)
    fixed = ants.from_numpy(fixed)
    if mode == 'SyN':
        mytx = ants.registration(fixed=fixed , moving=moved, type_of_transform='SyN' )
        out = mytx['warpedmovout'].numpy()
    else:
        raise Exception('ERROR REGISTRATION MODE!')
    return out


class reg2one_ref(object):
    def __init__(self, data_m, ref, mode='Rigid'):
        self.shape = data_m.shape
        self.source = data_m
        self.ref = ref
        self.mode = mode
        self.moved_img = np.zeros(self.shape)

    def reg2ref(self):
        for omega in range(self.shape[-1]):
            if self.mode == 'Affine':
                # Affine transformation
                sr = StackReg(StackReg.AFFINE)
                self.moved_img[:, :, omega] = sr.register_transform(self.ref, self.source[:, :, omega])
            #                 plt.imshow(moved_img[:,:,omega])
            elif self.mode == 'Rigid':
                sr = StackReg(StackReg.RIGID_BODY)
                self.moved_img[:, :, omega] = sr.register_transform(self.ref, self.source[:, :, omega])
        return self.moved_img


class reg_by_svd(object):
    def __init__(self, data, img_raw=None, mode='ANTs'):
        self.data = data
        self.raw = img_raw
        self.offset = data.shape[-1]
        self.raw_shape = data.shape
        self.casorati_M = np.reshape(data, (-1, self.offset))
        self.registrated_data = np.zeros(data.shape)
        self.mode = mode

    def reg_iters(self, iters, cal_mse=False):
        temp_data = self.data
        temp_mse = [offset_MSE(self.raw, temp_data)]
        for i in tqdm(range(iters), total=iters, leave=True, ascii=True):
            casorati_M = np.reshape(temp_data, (-1, self.offset))
            denoised = self.svd_denoise(casorati_M, 0.2)
            temp_data = self.ref2denoised(denoised, self.casorati_M)
            if cal_mse:
                mse = offset_MSE(self.raw, temp_data)
                temp_mse.append(mse)
        return temp_data, temp_mse

    def svd_denoise(self, data, threshold):
        u, sigma, vt = la.svd(data)
        de_sigma = np.zeros(data.shape)
        i = 0
        value_sum = 0.0
        temp_sigma = sigma / np.sum(sigma)
        for i in range(sigma.shape[0]):
            if abs(temp_sigma[i]) > threshold:
                de_sigma[i, i] = sigma[i]
        return np.dot(np.dot(u, de_sigma), vt)

    def ref2denoised(self, ref_data, moving_data):
        moved_data = np.zeros(self.raw_shape)
        ref_data = np.reshape(ref_data, self.raw_shape)
        moving_data = np.reshape(moving_data, self.raw_shape)
        for i in range(self.offset):
            if self.mode == 'ANTs':
                moved_data[:, :, i] = reg_ants(moving_data[:, :, i], ref_data[:, :, i])
            elif self.mode == 'StackReg':
                sr = StackReg(StackReg.RIGID_BODY)
                moved_data[:, :, i] = sr.register_transform(ref_data[:, :, i], moving_data[:, :, i])
            else:
                raise Exception('ERROR REGISTRATION MODE!')
        return moved_data


class denosiy_pca(object):
    """
    Denoise by PCA for cest data
    """

    def __init__(self, data, normalize=True):
        """
        Parameters:
            data: 3D-array(x,y,offset)
            normalize: if true, do normalization:data-mean(),v.v.
        """
        self.data_casorati_allpic = np.reshape(data, (-1, data.shape[-1]))
        self.data_casorati = np.reshape(data[data != 0], (-1, data.shape[-1]))
        self.data_shape = data.shape
        self.voxel_num = self.data_casorati.shape[0]
        self.offset = self.data_casorati.shape[1]
        self.normalize = normalize
        self.data_casorati_normal = self.data_casorati - np.mean(self.data_casorati, axis=0)
        self.denoisy_voxel = np.zeros((self.voxel_num, self.offset))
        self.denoisy_img = np.zeros(data.shape)

    def test_denoisy(self, x, voxel):
        """
        Parameters:
            x: The coordinate of offset
            voxel: the voxel that you want test
        """
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.plot(x, self.denoisy_voxel[voxel, :])
        plt.xlabel("de noise z at:%d" % (voxel))
        plt.vlines(-3.5, 0, 1, color='r')
        plt.vlines(3.5, 0, 1, color='g')
        plt.vlines(2, 0, 1, color='b')
        plt.xlim([-4, 4])
        plt.ylim([0, 1])
        plt.subplot(132)
        plt.plot(x, self.data_casorati[voxel, :])
        plt.xlabel("raw z at:%d" % (voxel))
        plt.vlines(-3.5, 0, 1, color='r')
        plt.vlines(3.5, 0, 1, color='g')
        plt.vlines(2, 0, 1, color='b')
        plt.xlim([-4, 4])
        plt.ylim([0, 1])
        plt.subplot(133)
        plt.plot(x, (self.data_casorati[voxel, :] - self.denoisy_voxel[voxel, :]))
        plt.xlabel("raw_z-denoise_z at:%d" % (voxel))
        plt.xlim([-4, 4])
        plt.ylim([-0.1, 0.1])
        plt.show()

    def roi2img(self):
        de_noise_data = np.zeros(self.data_casorati_allpic.shape)
        for i in range(self.offset):
            de_noise_data[:, i][self.data_casorati_allpic[:, i] != 0] = self.denoisy_voxel[:, i]
        self.denoisy_img = np.reshape(de_noise_data, self.data_shape)

    def median_cri(self, pca_model):

        main_component = pca_model.components_
        lamda = pca_model.explained_variance_
        num = lamda.shape[0]
        if num % 2:
            media = lamda[num // 2]
        else:
            media = (lamda[num // 2] + lamda[(num // 2) - 1]) / 2

        pre_lamda = []
        for i in lamda:
            if i >= 2 * media:
                continue
            else:
                pre_lamda.append(i)

        num2 = len(pre_lamda)
        print(num2)
        if num2 % 2:
            media2 = lamda[num2 // 2]
        else:
            media2 = (lamda[num2 // 2] + lamda[(num2 // 2) - 1]) / 2

        print(media2 * 1.29 * 1.29)
        for idx, i in enumerate(lamda):
            if i < media2 * 1.29 * 1.29:
                break

        print("median K (number of components)", idx)
        return idx

    #         print(media2)
    #         print(main_component.shape)
    #         print(main_component)
    #         print(lamda)
    #         pass

    def get_denoised_img(self, threshold):
        """
        Parameters:
           threshold: the n_components of PCA,
           see the detail: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        """
        if self.normalize:
            pca = PCA().fit(self.data_casorati_normal)
        else:
            print("unnormalize")
            pca = PCA().fit(self.data_casorati)
        k_num = self.median_cri(pca)
        main_component = pca.components_[:k_num, :]
        mean_z = np.mean(self.data_casorati, axis=0)
        print("主成分的个数：%d" % (main_component.shape[0]))

        for voxel in range(self.voxel_num):
            bias = np.zeros((self.offset))
            for i in range(main_component.shape[0]):
                inner_product = np.dot(np.reshape(main_component[i, :], (self.offset, 1)),
                                       np.reshape(main_component[i, :], (1, self.offset)))
                bias += np.dot((self.data_casorati[voxel, :] - mean_z), inner_product)
            #             z_j = mean_z + bias/main_component.shape[0]
            z_j = mean_z + bias
            self.denoisy_voxel[voxel, :] = z_j
        #         print(bias)
        #         print(bias/main_component.shape[0])
        #         print(mean_z)
        self.roi2img()


class denoise_regress(object):
    def __init__(self, data, mask, covariance, normalize=True):
        self.data = data
        self.mask = mask
        self.regressed_data = np.zeros(data.shape)
        self.cov = np.concatenate((np.ones((self.data.shape[-1], 1)), covariance), axis=1)
        self.x = self.mask.shape[0]
        self.y = self.mask.shape[1]
        self.z = self.mask.shape[2]
        print(self.data.shape)
        print(self.mask.shape)
        print(self.cov.shape)

    def regress(self):
        for x in tqdm(range(self.x)):
            for y in range(self.y):
                for z in range(self.z):
                    if self.mask[x, y, z] == 0:
                        continue
                    else:
                        Y = self.data[x, y, z, :]
                        model = LinearRegression(copy_X=True, fit_intercept=False, n_jobs=1, normalize=False)
                        model.fit(self.cov, Y)  # 线性回归建模
                        self.regressed_data[x, y, z, :] = model.coef_[0] + Y - np.sum(self.cov * model.coef_, axis=1)


class denoise_regress_v2(object):
    def __init__(self, data, mask, b0shift, covariance, normalize=True):
        self.data = data
        self.mask = mask
        self.b0shift = b0shift
        self.regressed_data = np.zeros(data.shape)
        self.cov = np.concatenate((np.ones((self.data.shape[-1], 1)), covariance), axis=1)
        #         print(self.cov )
        self.x = self.mask.shape[0]
        self.y = self.mask.shape[1]
        self.z = self.mask.shape[2]
        print(self.data.shape)
        print(self.mask.shape)
        print(self.cov.shape)

    def regress(self):
        for x in tqdm(range(self.x)):
            for y in range(self.y):
                for z in range(self.z):
                    if self.mask[x, y, z] == 0:
                        continue
                    else:
                        Y = self.data[x, y, z, :]
                        X = np.concatenate((np.ones((self.data.shape[-1], 1)) * self.b0shift[x, y, z], self.cov),
                                           axis=1)
                        #                         print(X.shape)
                        model = LinearRegression(copy_X=True, fit_intercept=False, n_jobs=1, normalize=False)
                        model.fit(X, Y)  # 线性回归建模
                        #                         print(model.coef_[1])
                        #                         print(model.coef_.shape)
                        self.regressed_data[x, y, z, :] = model.coef_[1] + Y - np.sum(X * model.coef_, axis=1)


def regress_PCA(offset, new_data_z, mask_array, move_data):
    for i in range(new_data_z.shape[-1]):
        new_data_z[:, :, :, i][mask_array != 1] = 0
    mean_z = np.reshape(new_data_z[new_data_z != 0], (-1, new_data_z.shape[-1]))
    mean_z = np.mean(mean_z, axis=0)
    print((new_data_z - mean_z).shape)
    model = denoise_regress(new_data_z - mean_z, mask_array, move_data[1:, :])
    model.regress()
    regressed_data_z = model.regressed_data
    regressed_data_z = regressed_data_z + mean_z
    for i in range(regressed_data_z.shape[-1]):
        regressed_data_z[:, :, :, i][mask_array != 1] = 0
    model = denosiy_pca(regressed_data_z)
    model.get_denoised_img(-1)
    model.test_denoisy(offset, 12000)
    pca_regress_data_z = model.denoisy_img
    return pca_regress_data_z