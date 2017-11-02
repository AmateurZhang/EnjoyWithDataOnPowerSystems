# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 21:12:39 2017

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

File='USA_ALL.csv'
OriginData=pd.read_table(File,sep=",")
Data=OriginData.as_matrix().astype(np.float32)

Start=0
End=8800
OBSPWR=Data[Start:End,0]
HR_FCSR=Data[Start:End,1]
OBS_DIFF=Data[Start:End,2]
HR_DIFF=Data[Start:End,3]
HR_ERROR=Data[Start:End,4]
DIFF_ERROR=Data[Start:End,5]

def Normalization(List):
    Max=np.max(List)
    Min=np.min(List)
    for i in range(0,len(List)):
        List[i]=(List[i]-Min)/(Max-Min)
#sns.distplot(HR_ERROR)
#
Normalization(HR_FCSR)
Normalization(HR_DIFF)
Normalization(OBS_DIFF)

X=np.c_[HR_FCSR,HR_DIFF]
Y=OBS_DIFF

 # training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)  #20 test 80 train

# data training
clf=GradientBoostingRegressor()
clf.fit(X_train, y_train.ravel())

#predict
y_pre=clf.predict(X_test)

plt.figure(figsize=(15,5))
plt.plot(y_test,'g')
plt.plot(y_pre,'r')
plt.show()
#clf.decision_function()

def evalmape(preds, dtrain):
    tp=[]
    for i in range(0,min(len(preds),len(dtrain))):
        tp.append((abs(preds[i]-dtrain[i])**2))
    #print(tp)
    a=0
    for j in range(0,len(tp)):
        a=a+tp[j]
    return np.sqrt(a/len(tp))
    
        

print(evalmape(y_pre,y_test)/np.average(y_test))
#report
#cl_report=metrics.classification_report(y_test,y_pre)

#print(ac_score,cl_report)