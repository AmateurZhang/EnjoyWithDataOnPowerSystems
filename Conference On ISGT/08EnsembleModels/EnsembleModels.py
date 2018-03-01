# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 16:28:55 2018

@author: thuzhang
"""

import random
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

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
    
def traintestsplit(X,size=0.8,asint=False,start=0):
    length=0
    if(asint==False):
        length=int(size*len(X))
    else:
        length=size
    train=X[start:length]
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
        paramsRF = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 2}
        for i in range(0,n):
            print(i)
            _resampledtrain=Resampletwo(self._train,resample=True)
            X_train,y_train=XandYsplit(_resampledtrain)    
            clf=RandomForestRegressor()
            clf.fit(X_train,y_train)
            clfsetRandomForest.append(clf)
        self.clfset=clfsetRandomForest
        return clfsetRandomForest
    
class GBRT:
    #Initialize
    def __init__(self,traindata,testdata):
        self._train=traindata
        self._test=testdata
        self.X_test,self.y_test=XandYsplit(testdata)
    # Build models
    def BuildModels(self,n):
        #Params
        paramsGBRT = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 2,
          'learning_rate': 0.1, 'loss': 'ls'}
        clfsetGBRT=[]
        for i in range(0,n):
            print(i)
            _resampledtrain=Resampletwo(self._train,resample=True)
            X_train,y_train=XandYsplit(_resampledtrain)    
            paramss=paramsGBRT
            clf=GradientBoostingRegressor()
            clf.fit(X_train,y_train)
            clfsetGBRT.append(clf)
        self.clfset=clfsetGBRT
        return clfsetGBRT
    
class LSSVM:
    #Initialize
    def __init__(self,traindata,testdata):
        self._train=traindata
        self._test=testdata
        self.X_test,self.y_test=XandYsplit(testdata)
    # Build models   
    def BuildModels(self,n):
        clfsetRandomForest=[]
        # params        
        for i in range(0,n):
            print(i)
            _resampledtrain=Resampletwo(self._train,resample=True)
            X_train,y_train=XandYsplit(_resampledtrain)    
            clf=LinearSVR()
            clf.fit(X_train,y_train)
            clfsetRandomForest.append(clf)
        self.clfset=clfsetRandomForest
        return clfsetRandomForest

class MLPRegress:
    #Initialize
    def __init__(self,traindata,testdata):
        self._train=traindata
        self._test=testdata
        self.X_test,self.y_test=XandYsplit(testdata)
    # Build models   
    def BuildModels(self,n):
        clfsetRandomForest=[]
        # params        
        for i in range(0,n):
            print(i)
            _resampledtrain=Resampletwo(self._train,resample=True)
            X_train,y_train=XandYsplit(_resampledtrain)    
            clf=MLPRegressor(hidden_layer_sizes=(22, 1),)
            clf.fit(X_train,y_train)
            clfsetRandomForest.append(clf)
        self.clfset=clfsetRandomForest
        return clfsetRandomForest
 
    
class AlgorithmBoots:
    def __init__(self,a,b,c,testdata):
        self.a=a
        self.b=b
        self.c=c
        self.clfset=self.MergeAlgorithms(self.a,self.b,self.c)
        self.X_test,self.y_test=XandYsplit(testdata)
        
    def MergeAlgorithms(self,a,b,c):
        _clfsetRandomForest=a
        _clfsetGBRT=b
        _clfsetLSSVM=c
        _clfsetRandomForest.extend(_clfsetGBRT)
        _clfsetRandomForest.extend(_clfsetLSSVM)
        clfset=_clfsetRandomForest
        self.clfset=clfset
        return clfset

    # public     
    def PointForecastBootOnData(self,times,Point,hist=False):
        predicted=[]
        RealPoint=self.y_test.iloc[Point]
        MeanPoint=[]
        LowQuantile=[]
        HighQuantile=[]
        
        for i,model in enumerate(self.clfset):
            _res=model.predict(self.X_test[Point:Point+1])
            predicted.append(_res[0])
            
        for j in range(0,times):
            _resample=Resample(predicted)
            MeanPoint.append(np.percentile(_resample,50))
            LowQuantile.append(np.percentile(_resample,5))
            HighQuantile.append(np.percentile(_resample,95))
        return RealPoint,MeanPoint,LowQuantile,HighQuantile

# IO, DATA SPLIT
File="NH.csv"
OriginData=pd.read_table(File,sep=",",index_col=False)

_train,_test= traintestsplit(OriginData,20928,True,3408)
_X_train,_y_train=XandYsplit(_train)
X_test,y_test=XandYsplit(_test)

_RF=RandomForest(_train,_test)
RandomForestModels=_RF.BuildModels(50)

_GBRT=GBRT(_train,_test)
GBRTModels=_GBRT.BuildModels(50)

_LSSVM=LSSVM(_train,_test)
LSSVMModels=_LSSVM.BuildModels(50)


_Ens=AlgorithmBoots(RandomForestModels[:],GBRTModels[:],LSSVMModels[:],_test)

Times=100
PredictHour=0
RRt=90*24
RealValue=[]
MQ=[]
lowQ=[]
hiQ=[]
for i in range(PredictHour,PredictHour+RRt):
    realValue,_MQ,_lowQ,_hiQ=_Ens.PointForecastBootOnData(Times,i)
    RealValue.append(realValue)
    MQ.extend(_MQ)
    lowQ.extend(_lowQ)
    hiQ.extend(_hiQ)
    
MQ=np.reshape(MQ,(RRt,Times))
lowQ=np.reshape(lowQ,(RRt,Times))
hiQ=np.reshape(hiQ,(RRt,Times))

timestamp=20
plt.figure(figsize=(8,5))
plt.ylim([0,0.30])
plt.vlines(np.mean(RealValue[timestamp]),0,0.22,'green',label='Real Value')
plt.hist(MQ[timestamp,],normed=True)
plt.vlines(np.mean(MQ[timestamp,]),0,0.22, color='blue',label="Point Forecast")
plt.hist(lowQ[timestamp,],normed=True,color='gray')
plt.vlines(np.percentile(lowQ[timestamp,],2.5),0,0.22,'gray',label='Lower Bound')
plt.hist(hiQ[timestamp,],normed=True,color='orange')
plt.vlines(np.percentile(hiQ[timestamp,],97.5),0,0.22,'orange',label='Upper Bound')
plt.xlabel('Power Demand/MW')
plt.ylabel('Probability')
plt.legend(loc='best')
#plt.savefig("C:/Users/thuzhang/Documents/MyFiles/00Seminar/07PROJECT7/probegg.eps")
plt.show()



Mid=[]
for i in range(0,len(MQ)):
    Mid.append(np.mean(MQ[i]))
    
lowQlow=[]
for i in range(0,len(lowQ)):
    lowQlow.append(np.percentile(lowQ[i],2.5))

hiQhi=[]
for i in range(0,len(hiQ)):
    hiQhi.append(np.percentile(hiQ[i],97.5))

plottime=90*24
ststime=0
plt.figure(figsize=(36,5))
plt.xlabel('Hour/h')
plt.ylabel('Power demand/MW')
#plt.ylim([700,2000])
plt.plot(RealValue[ststime:plottime],'green',label='Real Value')
plt.plot(Mid[ststime:plottime], color='blue',label="Point Forecast")
plt.plot(lowQlow[ststime:plottime],'gray',label='Lower Bound')
plt.plot(hiQhi[ststime:plottime],'orange',label='Upper Bound')
plt.xticks(np.arange(0,24*7+1,24))
plt.legend(loc='best')
plt.show()

print(MAPE(RealValue[ststime:plottime],Mid[ststime:plottime]))

def Accept(a,b,c,alpha):
    res=0
    for i in range(0,len(a)):
        if (a[i]-b[i]*(1-alpha)>=0.0) and (a[i]-c[i]*(1+alpha)<=0.0):
            res=res+1
    return res/len(a)
Accept(RealValue[ststime:plottime],lowQlow[ststime:plottime],hiQhi[ststime:plottime],0.043)
ahypo=[]
index=[]
for i in np.arange(0,0.05,0.001):
    a=Accept(RealValue[ststime:plottime],lowQlow[ststime:plottime],hiQhi[ststime:plottime],i)
    ahypo.append(a)   
    index.append(i)
plt.figure(figsize=(8,5))
plt.xlabel('Alpha')
plt.ylabel('Cover Capability')
plt.plot(index,ahypo,label='Cover Capability')
plt.legend(loc='best')
plt.show()

def Width(b,c,alpha):
    res=[]
    for i in range(0,len(b)):
        r=b[i]*(1+alpha)-c[i]*(1-alpha)
        res.append(r)
    return res
np.mean(Width(hiQhi[ststime:plottime],lowQlow[ststime:plottime],0.02))

plt.figure(figsize=(8,5))
plt.xlabel('Bound Width/MW')
plt.ylabel('Probability')
plt.hist(Width(hiQhi[ststime:plottime],lowQlow[ststime:plottime],0.03),label='Width Distribution', normed=1)
plt.vlines(np.mean(Width(hiQhi[ststime:plottime],lowQlow[ststime:plottime],0.03)),0,0.02,label='Average Interval Width: %d'%np.mean(Width(hiQhi[ststime:plottime],lowQlow[ststime:plottime],0.03)))
plt.legend(loc='best')
plt.show()



