# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:54:13 2017

@author: thuzhang
"""

import random
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# public functions
def MAPE(X,Y):
    count=len(X)
    Error=[]
    for i in range(0,count):
        Error.append(1-X[i]/Y[i])
    return np.mean(np.abs(Error))

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
    
def traintestsplit(X,size=0.8,asint=False):
    length=0
    if(asint==False):
        length=int(size*len(X))
    else:
        length=size
    train=X[0:length]
    test=X[length:]
    return train, test

def XandYsplit(X):
    return X.iloc[:,1:],X.iloc[:,0]



# Algorithm
class RandomForest:
    #Initialize
    def __init__(self,traindata,testdata):
        self._train=traindata
        self._test=testdata
        self.X_test,self.y_test=XandYsplit(testdata)
    # Build models   
    def BuildModels(self,n):
        clfsetRandomForest=[]
        # params
        paramsRF = {'n_estimators': 200, 'max_depth': 6, 'min_samples_split': 2}
        for i in range(0,n):
            _resampledtrain=Resampletwo(self._train,resample=True)
            X_train,y_train=XandYsplit(_resampledtrain)    
            paramss=paramsRF
            clf=RandomForestRegressor(**paramss)
            clf.fit(X_train,y_train)
            mse = mean_squared_error(y_train, clf.predict(X_train))
            print("MSE: %.4f" % mse)
            clfsetRandomForest.append(clf)
            #Self
        self.clfset=clfsetRandomForest
        return clfsetRandomForest
    
    #Private Class
    def BootitemRF(self,ListAlgorithm,Point):   
        _resamplemodels=Resample(ListAlgorithm,resample=True)
        predicted=[]
        for i,model in enumerate(_resamplemodels):
            _res=model.predict(self.X_test[Point:Point+1])
            predicted.append(_res[0])
        return np.mean(predicted)
    #Public Class
    def Bootstrap(self,times,Point,hist=True):
        bootresRandomForest=[]
        Truey=self.y_test[Point:Point+1]
        for i in range(0,times):
            mem=self.BootitemRF(self.clfset,Point)
            bootresRandomForest.append(mem)
        if(hist==True):
            plt.figure(figsize=(5,5))
            plt.hist(bootresRandomForest,normed=True)
            plt.vlines(Truey,0,.1)
            plt.show()
        return bootresRandomForest

class GBRT:
    #Initialize
    def __init__(self,traindata,testdata):
        self._train=traindata
        self._test=testdata
        self.X_test,self.y_test=XandYsplit(testdata)
    # Build models
    def BuildModels(self,n):
        paramsGBRT = {'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 2,
          'learning_rate': 0.1, 'loss': 'ls'}
        clfsetGBRT=[]
        for i in range(0,n):
            _resampledtrain=Resampletwo(self._train,resample=True)
            X_train,y_train=XandYsplit(_resampledtrain)    
            paramss=paramsGBRT
            paramss['max_depth']=6
            clf=GradientBoostingRegressor(**paramss)
            clf.fit(X_train,y_train)
            mse = mean_squared_error(y_train, clf.predict(X_train))
            print("MSE: %.4f" % mse)
            clfsetGBRT.append(clf)
        self.clfset=clfsetGBRT
        return clfsetGBRT
    def BootitemGBRT(self,ListAlgorithm,Point):   
        _resamplemodels=Resample(ListAlgorithm,resample=True)
        predicted=[]
        for i,model in enumerate(_resamplemodels):
            _res=model.predict(self.X_test[Point:Point+1])
            predicted.append(_res[0])
        return np.mean(predicted)
    def Bootstrap(self,times,Point,hist=True):
        bootresGBRT=[]
        Truey=self.y_test[Point:Point+1]
        for i in range(0,times):
            mem=self.BootitemGBRT(self.clfset,Point)
            bootresGBRT.append(mem)
        if(hist==True):
            plt.figure(figsize=(5,5))
            plt.hist(bootresGBRT,normed=True) 
            plt.vlines(Truey,0,.1)
            plt.show()
        return bootresGBRT

class AlgorithmBoots:
    def __init__(self,a,b,testdata):
        self.a=a
        self.b=b
        self.clfset=self.MergeAlgorithms(self.a,self.b)
        self.X_test,self.y_test=XandYsplit(testdata)
        
    def MergeAlgorithms(self,a,b):
        _clfsetRandomForest=a
        _clfsetGBRT=b
        _clfsetRandomForest.extend(_clfsetGBRT)
        clfset=_clfsetRandomForest
        self.clfset=clfset
        return clfset
    def BootitemEns(self,ListAlgorithm,Point):   
        _resamplemodels=Resample(ListAlgorithm,resample=True)
        predicted=[]
        for i,model in enumerate(_resamplemodels):
            _res=model.predict(self.X_test[Point:Point+1])
            predicted.append(_res[0])
        return predicted
    # public
    def PointForecast(self,times,Point,hist=True):
        bootresens=[]
        Truey=self.y_test[Point:Point+1]
        for i in range(0,times):
            mem=self.BootitemEns(self.clfset,Point)
            bootresens.extend(mem) 
        if(hist==True):
            plt.figure(figsize=(5,5))
            plt.hist(bootresens,normed=True) 
            plt.vlines(Truey,0,.02)
            plt.show()
        print(y_test.iloc[Point],np.mean(bootresens),np.percentile(bootresens,2.5),np.percentile(bootresens,97.5))
        return y_test.iloc[Point],np.mean(bootresens),np.percentile(bootresens,2.5),np.percentile(bootresens,97.5)
    
# IO, DATA SPLIT
File="DataBase.csv"
OriginData=pd.read_table(File,sep=",",index_col=False)

#split Data
# test and train
_train,_test= traintestsplit(OriginData,365*48,True)
_X_train,_y_train=XandYsplit(_train)
X_test,y_test=XandYsplit(_test)


_RF=RandomForest(_train,_test)
RandomForestModels=_RF.BuildModels(20)
for i in range(24,28):
    _RF.Bootstrap(20,i)
    
_GBRT=GBRT(_train,_test)
GBRTModels=_GBRT.BuildModels(40)
for j in range(24,28):
    _GBRT.Bootstrap(500,j)

_Ens=AlgorithmBoots(RandomForestModels,GBRTModels,_test)
for j in range(0,24):
    _Ens.PointForecast(200,j)








