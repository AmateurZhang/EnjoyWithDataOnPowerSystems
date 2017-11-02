# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 20:31:58 2017

@author: thuzhang
"""


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pywt   # python 小波变换的包 
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR 
from statsmodels.tsa.arima_model import ARIMA 
from statsmodels.tsa.arima_model import ARMA

# 函数打包
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
    
    
    
# 打包为函数
def preTest(Length=4,sts=0,end=1000,draw='False'):
    Data=[]
    OriginData=pd.read_table(r'Light1.csv',sep=",")
    Data=OriginData.as_matrix().astype(np.float32)
    data = Data[24*sts:24*end,0]
    
    index_list=data[:-Length]
    dataNew=wt(index_list,'db4',2,1,2)
    index_list=dataNew
    #date_list1 = np.array(data['tradeDate'])[:-10]

    index_for_predict =data[-Length:]  # 预测的真实值序列
    #date_list2 = np.array(data['tradeDate'])[-10:]

    # 分解 
    A2,D2,D1 = pywt.wavedec(index_list,'db4',mode='sym',level=2)  # 分解得到第4层低频部分系数和全部4层高频部分系数
    coeff = [A2,D2,D1]
    #print(coeff)
    
    # 对每层小波系数求解模型系数
    order_A2 = sm.tsa.arma_order_select_ic(A2,ic=['aic', 'bic'])['aic_min_order']   # AIC准则求解模型阶数p,q
    order_D2 = sm.tsa.arma_order_select_ic(D2,ic='aic')['aic_min_order']   # AIC准则求解模型阶数p,q
    order_D1 = sm.tsa.arma_order_select_ic(D1,ic='aic')['aic_min_order']   # AIC准则求解模型阶数p,q
    
    #print(order_A2,order_D2,order_D1)
    # 对每层小波系数构建ARMA模型
    # 值得注意的是，有时候用AIC准则求解的模型参数来建模会报错，这时候请调节数据时间长度。 
    model_A2 =  ARMA(A2,order=order_A2)   # 建立模型
    model_D2 =  ARMA(D2,order=order_D2)
    model_D1 =  ARMA(D1,order=order_D1)

    results_A2 = model_A2.fit()
    results_D2 = model_D2.fit()
    results_D1 = model_D1.fit()
    # 画出每层的拟合曲线
    if draw!='False':
        plt.figure(figsize=(10,15))
        plt.subplot(3, 1, 1)
        plt.plot(A2, 'blue')
        plt.plot(results_A2.fittedvalues,'red')
        plt.title('model_A2')
        plt.subplot(3, 1, 2)
        plt.plot(D2, 'blue')
        plt.plot(results_D2.fittedvalues,'red')
        plt.title('model_D2')

        plt.subplot(3, 1, 3)
        plt.plot(D1, 'blue')
        plt.plot(results_D1.fittedvalues,'red')
        plt.title('model_D1')
    
    A2_all,D2_all,D1_all = pywt.wavedec(data,'db4',mode='sym',level=2) # 对所有序列分解
    delta = [len(A2_all)-len(A2),len(D2_all)-len(D2),len(D1_all)-len(D1)] # 求出差值，则delta序列对应的为每层小波系数ARMA模型需要预测的步数
    #print(delta)
    # 预测小波系数 包括in-sample的和 out-sample的需要预测的小波系数
    pA2 = model_A2.predict(params=results_A2.params,start=1,end=len(A2)+delta[0])
    pD2 = model_D2.predict(params=results_D2.params,start=1,end=len(D2)+delta[1])
    pD1 = model_D1.predict(params=results_D1.params,start=1,end=len(D1)+delta[2])

    # 重构
    coeff_new = [pA2,pD2,pD1]
    denoised_index = pywt.waverec(coeff_new,'db4')
   # print(denoised_index,len(denoised_index))   
    # 输出10个预测值
    temp_data_wt = {'real_value':index_for_predict,'pre_value_wt':denoised_index[-Length:],'err_wt':denoised_index[-Length:]-index_for_predict,'err_rate_wt/%':(denoised_index[-Length:]-index_for_predict)/index_for_predict*100}
    predict_wt = pd.DataFrame(temp_data_wt,index = None,columns=['real_value','pre_value_wt','err_wt','err_rate_wt/%'])
    
    if draw!='False':
        # 画出重构后的原序列预测图
        plt.figure(figsize=(15,5))
        plt.plot(data, 'blue')
        plt.plot(denoised_index,'red')
        plt.show()
        #
        plt.figure(figsize=(8,10))
        plt.plot(data[-Length:], '-g')
        plt.plot(denoised_index[-Length:],'red')
        plt.show()
        print(predict_wt)
    
    AverErr=sum(abs((denoised_index[-Length:]-index_for_predict)/index_for_predict*100))/Length    
    print(AverErr)
    return AverErr
    
    #predict_wt.to_csv('WaveLets.csv')

#Result=[]
#for i in range(340,350):
   # tp=0
   # try:
    #    tp=preTest(end=i) 
   # except:
   #     pass
   # if tp>0:
    #    Result.append(tp)

#plt.figure(figsize=(15,5))
#plt.plot(Result, 'blue')
#plt.show()
preTest()
    