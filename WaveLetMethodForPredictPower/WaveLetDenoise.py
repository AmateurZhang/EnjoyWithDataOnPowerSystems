# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:51:03 2017

@author: thuzhang
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pywt   # python 小波变换的包 

def wt(index_list,wavefunc,lv,m,n):   # 打包为函数，方便调节参数。  lv为分解层数；data为最后保存的dataframe便于作图；index_list为待处理序列；wavefunc为选取的小波函数；m,n则选择了进行阈值处理的小波系数层数
   
    # 分解
    coeff = pywt.wavedec(index_list,wavefunc,mode='sym',level=lv)   # 按 level 层分解，使用pywt包进行计算， cAn是尺度系数 cDn为小波系数

    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0 # sgn函数

    # 去噪过程
    for i in range(m,n+1):   # 选取小波系数层数为 m~n层，尺度系数不需要处理
        cD = coeff[i]
        for j in range(len(cD)):
            Tr = np.sqrt(2*np.log(len(cD)))  # 计算阈值
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) - Tr  # 向零收缩
            else:
                coeff[i][j] = 0   # 低于阈值置零

    # 重构
    denoised_index = pywt.waverec(coeff,wavefunc)
    plt.figure(figsize=(15,5))
    plt.plot(index_list, 'blue')
    plt.plot(denoised_index,'red')
    plt.show()
    return denoised_index

