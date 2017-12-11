# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 19:43:07 2017

@author: thuzhang
"""
import random
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def Resample(a,resample=True):
    Length=len(a)
    if resample==False:
        return random.sample(a,Length)
    else:
        sample=[]
        for i in range(Length):
            sample.append(a[random.randint(0,Length-1)])
        return sample
    
def Resampletwo(a,b,resample=True):
    Length=len(a)
    if resample==False:
        return random.sample(a,Length),random.sample(b,Length)
    else:
        sample=pd.DataFrame(columns=a.columns)
        sampleb=pd.DataFrame(columns=a.columns)
        for i in range(Length):
            sample.loc[i]=a.loc[random.randint(0,Length-1)]
            sampleb.loc[i]=b.loc[random.randint(0,Length-1)]
        return sample    
    
def traintestsplit(X,Y,size=0.8):
    length=int(size*len(X))
    X_train=X[0:length]
    X_test=X[length:]
    y_train=Y[0:length]
    y_test=Y[length:]
    return X_train, X_test, y_train, y_test 

    
File="DataBase.csv"
OriginData=pd.read_table(File,sep=",",index_col=False)
Data=OriginData.as_matrix().astype(np.float32)

Y=OriginData["LOAD"]
X=OriginData.iloc[:,1:]

params = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}

X_train, X_test, y_train, y_test = traintestsplit(X,Y,0.8)

clfset=[]
for i in range(1,20):
    Resampletwo(X_train,y_train)
    paramss=params
    paramss['max_depth']=6
    clf=GradientBoostingRegressor(**paramss)
    clf.fit(X_train,y_train.ravel())
    mse = mean_squared_error(y_test, clf.predict(X_test))
    print("MSE: %.4f" % mse)
    clfset.append(clf)
    

def Bootitem():   
    _resamplemodels=Resample(clfset,resample=True)
    predicted=[]
    error=[]
    for i,model in enumerate(_resamplemodels):
        _res=model.predict(X_test[10:11])
        predicted.append(_res[0])
        error.append(model.loss_(y_test[10:11],_res))
    return np.mean(predicted),np.mean(np.sqrt(error))

bootres=[]
eeres=[]
for i in range(0,50):
    mem,eer=Bootitem()
    bootres.append(mem)
    eeres.append(eer)
    

np.mean(bootres)
np.var(bootres)
    

