# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:59:23 2017

@author: thuzhang
"""

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
import math
from sklearn import linear_model

def Sigmoid(X,A,B,r):
    return A*B*np.exp(r*X)/(A+B*(np.exp(r*X)-1)) 

def ISigmoid(Y,A,B,r):
    return 1/r *np.log((A-B)*Y/((A-Y)*B))

def Logistic(x,SigmaUp,SigmaDown,RelativeValue):
    return Sigmoid(x,SigmaUp,SigmaDown,RelativeValue)

def InverseLogistic(y,SigmaUp,SigmaDown,RelativeValue):
    return ISigmoid(y,SigmaUp,SigmaDown,RelativeValue)

StartTime=360
LengthOfData=720
LengthOfPredict=168

OriginData=(ParseData(File='2014_hourly_ME.csv',Sts=StartTime,Length=LengthOfData+LengthOfPredict))

#Logistic Transform Index
SigmaUp=np.average(OriginData)+1*np.sqrt(np.var(OriginData))
SigmaDown=np.average(OriginData)-1*np.sqrt(np.var(OriginData))
MeanData=np.average(OriginData)
RelativeValue=np.log(SigmaUp/SigmaDown-1)/MeanData

print(SigmaUp,SigmaDown,RelativeValue)

#Logistic Transform
#OriginData=Logistic(OriginData,SigmaUp,SigmaDown,RelativeValue)
print('Ori')
plt.figure(figsize=(15,6))
plt.title('Decomposion')
plt.plot(OriginData, color='blue',label='Trend')
plt.show()
#Denoise
try:
    Data=Denoise(OriginData[:LengthOfData],'db4',2,1,2)
except:
    Data=OriginData[:LengthOfData]
    print('[Denoise] Denoise failed.')
    
#decompose DataSet
DataModel=Data[0:LengthOfData]
DataTest=OriginData[LengthOfData:]

#decompose
decomposition = seasonal_decompose(DataModel, model="additive",freq=24)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

#Result Plot
plt.figure(figsize=(15,6))
plt.title('Decomposion')
plt.plot(trend, color='blue',label='Trend')
plt.plot(seasonal,color='red',label='Seasonal')
plt.plot(residual,color='black',label='ERROR')
plt.legend(loc='best')
plt.show()

def FormSeasonalData(seasonal,freq,length):
    DataOri=seasonal[-freq:]
    Result=[]
    for i in range(0,length):
        Result.append(DataOri[np.mod(i,freq)])
    return Result


def CleanNaNs(Data):
    Out=[]
    for i in range(1,len(Data)):
        if math.isnan(Data[i])==0:
            Out.append(Data[i])
    return Out

TPTrends=CleanNaNs(trend)


decomposition = seasonal_decompose(TPTrends, model="additive",freq=168)
trendtrends = decomposition.trend
seasonaltrends = decomposition.seasonal
residualtrends = decomposition.resid

#plt.plot(FormSeasonalData(seasonaltrends,168,168*2))

def MergeDataSets(a,b):
    c=[]
    for i in range(0,min(len(a),len(b))):
        c.append(a[i]+b[i])
    return c

MergeMix=MergeDataSets(FormSeasonalData(seasonaltrends,168,168*2),FormSeasonalData(seasonal,24,168*2))        
plt.figure(figsize=(15,6))
plt.title('MixTwoTrends')
plt.plot(FormSeasonalData(seasonal,24,168*2),label='24H Seasonal')
plt.plot(FormSeasonalData(seasonaltrends,168,168*2),label='7*24H Seasonal')
plt.plot(MergeMix,'r',label='Mixed')
plt.legend(loc='best')
plt.show()


#Result Plot
plt.figure(figsize=(15,6))
plt.title('Decomposion')
plt.plot(trendtrends, color='blue',label='Trend')
plt.plot(seasonaltrends,color='red',label='Seasonal')
plt.plot(residualtrends,color='black',label='ERROR')
plt.legend(loc='best')
plt.show()

def Regression(DataSet,NaN,Length):
    Data=CleanNaNs(DataSet)
    DataX=[]
    for i in range(1,len(Data)+1):
        DataX.append(i)
    reg = linear_model.LinearRegression()
    reg.fit(np.reshape(DataX,(-1,1)),np.reshape(Data[:len(Data)],(-1,1)))
    Result=[]
    for j in range(0,Length):
        Result.append(reg.predict(j+NaN)[0])
    return Result

MainTrend=Regression(trendtrends[-336:],84,LengthOfPredict)

plt.figure(figsize=(15,6))
plt.title('Results')
plt.plot(MergeDataSets(MergeMix,MainTrend),label='Predict')
plt.plot(DataTest,label='Actual')
plt.legend(loc='best')
plt.show()


#Output
Mode='TESTIO'
Outs=np.c_[trend,seasonal,residual] 
OutsTrends=np.c_[trendtrends,seasonaltrends,residualtrends] 
Save=pd.DataFrame(Outs.astype(float))
Save.to_csv('PredictWindSpdData%s.csv'%Mode,header=None,index=None)

SaveTs=pd.DataFrame(OutsTrends.astype(float))
SaveTs.to_csv('PredictWindSpdDataTrends%s.csv'%Mode,header=None,index=None)


