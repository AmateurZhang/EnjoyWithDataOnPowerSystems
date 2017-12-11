# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 19:43:07 2017

@author: thuzhang
"""
import random
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
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
    
def Resampletwo(a,resample=True):
    if resample==False:
        return a.sample(n=len(a))
    else:
        return a.sample(n=len(a),replace=True)
    
def traintestsplit(X,size=0.8):
    length=int(size*len(X))
    train=X[0:length]
    test=X[length:]
    return train, test

def XandYsplit(X):
    return X.iloc[:,1:],X.iloc[:,0]
    
File="DataBase.csv"
OriginData=pd.read_table(File,sep=",",index_col=False)
Data=OriginData.as_matrix().astype(np.float32)



params = {'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 2,
        }

_train,_test= traintestsplit(OriginData,0.8)
X_test,y_test=XandYsplit(_test)
clfsetRandomForest=[]
for i in range(0,25):
    _resampledtrain=Resampletwo(_train,resample=True)
    X_train,y_train=XandYsplit(_resampledtrain)    
    paramss=params
    paramss['max_depth']=6
    clf=RandomForestRegressor(**paramss)
    clf.fit(X_train,y_train)
    mse = mean_squared_error(y_test, clf.predict(X_test))
    print("MSE: %.4f" % mse)
    clfsetRandomForest.append(clf)
    

def BootitemRF():   
    _resamplemodels=Resample(clfsetRandomForest,resample=True)
    predicted=[]
    for i,model in enumerate(_resamplemodels):
        _res=model.predict(X_test[100:101])
        predicted.append(_res[0])
    return np.mean(predicted)

bootresRandomForest=[]
eeres=[]
for i in range(0,500):
    mem=BootitemRF()
    bootresRandomForest.append(mem)

    
print(bootresRandomForest,"\n")
np.mean(bootresRandomForest)
np.var(bootresRandomForest)

plt.hist(bootresRandomForest)
np.percentile(bootresRandomForest,95)
