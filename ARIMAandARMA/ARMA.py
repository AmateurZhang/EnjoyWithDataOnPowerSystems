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

def Sigmoid(X,A,B,r):
    return A*B*np.exp(r*X)/(A+B*(np.exp(r*X)-1)) 

def ISigmoid(Y,A,B,r):
    return 1/r *np.log((A-B)*Y/((A-Y)*B))

def Logistic(x,SigmaUp,SigmaDown,RelativeValue):
    return Sigmoid(x,SigmaUp,SigmaDown,RelativeValue)

def InverseLogistic(y,SigmaUp,SigmaDown,RelativeValue):
    return ISigmoid(y,SigmaUp,SigmaDown,RelativeValue)

StartTime=00
LengthOfData=720
LengthOfPredict=24

OriginData=(ParseData(File='2014_hourly_ME.csv',Sts=StartTime,Length=LengthOfData+LengthOfPredict))

#Logistic Transform Index
SigmaUp=np.average(OriginData)+1*np.sqrt(np.var(OriginData))
SigmaDown=np.average(OriginData)-1*np.sqrt(np.var(OriginData))
MeanData=np.average(OriginData)
RelativeValue=np.log(SigmaUp/SigmaDown-1)/MeanData

print(SigmaUp,SigmaDown,RelativeValue)

#Logistic Transform
OriginData=Logistic(OriginData,SigmaUp,SigmaDown,RelativeValue)

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
plt.figure(figsize=(20,9))
plt.title('Decomposion')
plt.plot(trend, color='blue',label='Trend')
plt.plot(seasonal,color='red',label='Seasonal')
plt.plot(residual,color='black',label='ERROR')
plt.legend(loc='best')
plt.show()

def FormSeasonalData(seasonal,freq,length):
    DataOri=seasonal[:freq]
    Result=[]
    for i in range(0,length):
        Result.append(DataOri[np.mod(i,length)])
        

#ARMA
Tp=ARMA(seasonal,(24,0))
result=Tp.fit()
print(Tp.fit().params)
Output=result.forecast(LengthOfPredict)[0]


#Result Plot
plt.figure(figsize=(10,10))
plt.title('Results in Logistic Fields')
#plt.plot(DataTest, color='blue',label='True Value')
plt.plot(Output,color='red',label='Predict')
#plt.plot(Output-DataTest,color='black',label='ERROR')
plt.legend(loc='best')
plt.show()

#Reverse Logistic
DataTestAct=InverseLogistic(DataTest,SigmaUp,SigmaDown,RelativeValue)
OutputAct=InverseLogistic(Output,SigmaUp,SigmaDown,RelativeValue)

#Result Plot
plt.figure(figsize=(10,10))
plt.title('Results in Real Fields')
plt.plot(DataTestAct, color='blue',label='True Value')
plt.plot(OutputAct,color='red',label='Predict')
#plt.plot(OutputAct-DataTestAct,color='black',label='ERROR')
plt.legend(loc='best')
plt.show()

#Output
Mode='TESTIO'
Outs=np.c_[DataTest,Output,DataTestAct,OutputAct]  
Save=pd.DataFrame(Outs.astype(float))
Save.to_csv('PredictWindSpdData%s.csv'%Mode,header=None,index=None)