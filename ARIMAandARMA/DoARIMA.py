# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 17:20:54 2017

@author: thuzhang
"""

# ARIMA

from statsmodels.tsa.arima_model import ARMA 
from statsmodels.tsa.arima_model import ARMAResults
import numpy as np 
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from DeNoise import Denoise
from ParseData import ParseData
from statsmodels.tsa.seasonal import seasonal_decompose

def Sigmoid(X,A,B,r):
    return A*B*np.exp(r*X)/(A+B*(np.exp(r*X)-1)) 

StartTime=48
LengthOfData=720
LengthOfPredict=24

OriginData=(ParseData(File='7576.csv',Sts=StartTime,Length=LengthOfData+LengthOfPredict+1))

SigmaUp=np.average(OriginData)+2*np.sqrt(np.var(OriginData))
SigmaDown=np.average(OriginData)-2*np.sqrt(np.var(OriginData))
MeanData=np.average(OriginData)
RelativeValue=np.log(SigmaUp/SigmaDown-1)/MeanData



print(np.average(OriginData),np.sqrt(np.var(OriginData)))

def Logistic(x):
    return Sigmoid(x,SigmaUp,SigmaDown,RelativeValue)

def test_stationarity(timeseries):
    # 决定起伏统计
    rolmean = pd.rolling_mean(timeseries, window=100)    # 对size个数据进行移动平均
    rol_weighted_mean = pd.ewma(timeseries, span=36)    # 对size个数据进行加权移动平均
    rolstd = pd.rolling_std(timeseries, window=100)      # 偏离原始值多少
    # 画出起伏统计
    plt.figure(figsize=(15,5))
    #orig = plt.plot(timeseries, color='blue', label='Original')
    #mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    weighted_mean = plt.plot(rol_weighted_mean, color='green', label='weighted Mean')
    #std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    # 进行df测试
    print ('Result of Dickry-Fuller test')
    dftest = ts.adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical value(%s)' % key] = value
    print (dfoutput)


print(np.var(OriginData))
OriginData=Logistic(OriginData)

try:
    Data=Denoise(OriginData[:LengthOfData],'db4',2,1,2)
except:
    Data=OriginData[:LengthOfData]
    print('[Denoise] Denoise failed.')
    

test_stationarity(Data)

# estimating
ts_log_diff = np.diff(OriginData,1)

test_stationarity(ts_log_diff)

Transformed=np.fft.fft(ts_log_diff)
plt.figure(figsize=(15,10))
plt.plot(Transformed,'red')
plt.show()

DataModel=Data[0:LengthOfData]
DataTest=OriginData[LengthOfData:]

# 分解decomposing
decomposition = seasonal_decompose(ts_log_diff,freq=24)

trend = decomposition.trend  # 趋势
seasonal = decomposition.seasonal  # 季节性
residual = decomposition.resid  # 剩余的


plt.figure(figsize=(15,5))
plt.plot(trend,'g')
plt.plot(seasonal,'red')

plt.show()

#order_D2 = sm.tsa.arma_order_select_ic(DataModel,ic='aic')['aic_min_order'] 

Tp=ARMA(DataModel,(24,3))
result=Tp.fit()
print(Tp.fit().params)
Output=result.forecast(LengthOfPredict)[0]


plt.figure(figsize=(10,10))
plt.plot(DataTest, color='blue',label='True Value')
plt.plot(Output,color='red',label='Predict')
plt.legend(loc='best')
plt.show()

print('Time:',StartTime,LengthOfData+StartTime)

Outs=np.c_[DataTest,Output]    
#Output
Save=pd.DataFrame(Outs.astype(float))
Save.to_csv(r'PredictWindSpdData.csv',header=None,index=None)
