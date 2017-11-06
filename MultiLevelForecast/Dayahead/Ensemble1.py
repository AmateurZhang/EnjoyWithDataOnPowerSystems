# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 23:30:08 2017

@author: thuzhang
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
#Ensemble

LengthOfTrainDataInDays=365*2
LengthOfTrainDataInHours=LengthOfTrainDataInDays*24
TheStartTimePointInDays=60
TheStartTimePointInHours=60*24


def PkLoadForecast(sts,length):
    File='DataBase/PkLoadPredict.csv'
    OriginData=pd.read_table(File,sep=",",index_col=False)
    DataOri=OriginData.as_matrix().astype(np.float32)


    #Get the properties and the target
    Data=DataOri[-sts-length:-sts+1]
    y_train=Data[:-1,1]
    X_train=Data[:-1,2:]
    X_test=Data[-1,2:].reshape(1,7)
    y_test=Data[-1,1]
    
    #print(X_test,y_test)
    method=GradientBoostingRegressor()
    method.fit(X_train,y_train.ravel())

    Y_Predict=method.predict(X_test)
    Y_Error=Y_Predict-y_test

    _Expection=np.mean(Y_Error)
    print(Y_Predict[0],y_test,_Expection)
    return Y_Predict[0],y_test,_Expection



def SeasonalForecast(sts,length):
    File='DataBase/SeasonalPredict.csv'
    OriginData=pd.read_table(File,sep=",",index_col=False)
    DataOri=OriginData.as_matrix().astype(np.float32)

    
    #Get the properties and the target
    Data=DataOri[-sts-length:-sts+1]
    y_train=Data[:-1,1]
    X_train=Data[:-1,2:]
    X_test=Data[-1,2:].reshape(1,6)
    y_test=Data[-1,1]
    
    #print(X_test,y_test)
    method=RandomForestRegressor()
    method.fit(X_train,y_train.ravel())

    Y_Predict=method.predict(X_test)
    Y_Error=Y_Predict-y_test
    
    _Expection=np.mean(Y_Error)
    print(Y_Predict[0],y_test,_Expection)
    return Y_Predict[0],y_test,_Expection

_YPre=[]
_YTest=[]
_Error=[]

for i in range(0,30*24):
    PkY_Pre,Pky_test,Pk_Error=PkLoadForecast(np.int(TheStartTimePointInDays-i/24),LengthOfTrainDataInDays)
    SeY_Pre,Sey_test,Se_Error=SeasonalForecast(TheStartTimePointInDays*24-i,LengthOfTrainDataInDays*24)
    AllY_Pre=PkY_Pre-SeY_Pre
    Ally_test=Pky_test-Sey_test
    AllError=Pk_Error-Se_Error
    _YPre.append(AllY_Pre)
    _YTest.append(Ally_test)
    _Error.append(AllError)
 
print('---------------------------')    
print(np.mean(_Error)/np.mean(_YTest))
plt.figure(figsize=(18,8))
plt.plot(_YTest,'green',label='Test')
plt.plot(_YPre, 'blue',label='Predict')
plt.plot(_Error,'orange',label='Error')
plt.legend(loc='best')
plt.show()

_Result=np.c_[_YPre,_YTest,_Error]
df = pd.DataFrame(data=_Result)
df.to_csv('EnsembleResults.csv')
    