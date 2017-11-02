# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 10:03:23 2017

@author: thuzhang
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

File='USA_ALL.csv'
OriginData=pd.read_table(File,sep=",")
Data=OriginData.as_matrix().astype(np.float32)

Start=0
End=8000
OBSPWR=Data[Start:End,0]
HR_FCSR=Data[Start:End,1]
OBS_DIFF=Data[Start:End,2]
HR_DIFF=Data[Start:End,3]
HR_ERROR=Data[Start:End,4]
DIFF_ERROR=Data[Start:End,5]

def ClearData(Ori,Standard,index=0.6,lo=0.1):
    Max=np.max(Standard)
    Thru=index*Max
    loo=lo*Max
    NewOri=[]
    NewStan=[]
    DropOri=[]
    DropStan=[]
    for i in range(0,min(len(Standard),len(Ori))):
        if Standard[i]<=Thru and Standard[i]>loo:
            NewOri.append(Ori[i])
            NewStan.append(Standard[i])
        else:
            DropOri.append(Ori[i])
            DropStan.append(Standard[i])
    return NewOri,NewStan,DropOri,DropStan
        
    

def Normalization(List):
    Max=np.max(List)
    Min=np.min(List)
    for i in range(0,len(List)):
        List[i]=(List[i]-Min)/(Max-Min)
#sns.distplot(HR_ERROR)
#
#Normalization(HR_FCSR)
def Predict(Data1,Data2,index,perc=.1):
    NewHR,NewOBS,DropHR,DropOBS=ClearData(Data1,OBSPWR,index)     
    NewOBSD,NewOBS,DropOBSD,DropOBS=ClearData(Data2,OBSPWR,index)    
        
    #Normalization(NewHR)
    #Normalization(NewOBSD)

    X=np.c_[NewHR]
    Y=NewOBSD
    Length=int(perc*len(Y))

     # training and testing sets
    X_train, X_test, y_train, y_test = X[:Length],X[Length:],Y[:Length],Y[Length:]  #20 test 80 train

    # data training
    clf=linear_model.LinearRegression()
    clf.fit(X_train, y_train)

    #predict
    y_pre=clf.predict(X_test)

    plt.figure(figsize=(15,5))
    plt.plot(y_test,'g')
    plt.plot(y_pre,'r')
    plt.show()
    #clf.decision_function()
    result=evalmape(y_pre,y_test)
    print(index,evalmape(y_pre,y_test))
    return y_pre,y_test,X_test


def evalmape(preds, dtrain):
    tp=[]
    for i in range(0,min(len(preds),len(dtrain))):
        tp.append((abs(preds[i]-dtrain[i])**2))
    #print(tp)
    a=0
    for j in range(0,len(tp)):
        a=a+tp[j]
    return np.sqrt(a/len(tp))
    
Result=[]
ResultDiff=[]
ResT=[]
ResX=[]
for i in range(80,81,5):
    Result,ResT,ResX=(Predict(HR_FCSR,OBSPWR,i/100))
for i in range(80,81,5):
    ResultDiff,ResDT,ResXd=(Predict(HR_DIFF,OBS_DIFF,i/100))

RTD1=ResultDiff[1:]
RTD1=np.append(RTD1,[0])

RTD2=ResultDiff[:-1]
RTD2=np.append([0],RTD2)

ResultUP=Result+1.5*RTD1
ResultDown=Result-1.5*RTD2
plt.figure(figsize=(15,5))
plt.plot(Result[000:300],'g',label='Result')
plt.plot(ResultUP[000:300],'r',label='ResultUP')
plt.plot(ResultDown[000:300],'b',label='ResultDown')
plt.plot(ResT[000:300],'y',label='RealValue')
plt.plot(ResX[000:300],'orange',label='ResultTest')
plt.legend(loc='best')
plt.show()    


