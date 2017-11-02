# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 22:39:11 2017

@author: thuzhang
"""

#Predict WindSpeed(WINDSPD)

#packages
import numpy as np
import pandas as pd
import sklearn.svm as svm
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

def PredictWindSpd(NumberOfDays=2):
    Data=[]
    OriginData=pd.read_table(r'FormerDataModel.csv',sep=",",header=None)
    Data=OriginData.as_matrix().astype(np.float32)
    
    #Get the properties and the target
    Y=Data[:,1].astype(np.float32)
    X=Data[:,2:2+3*NumberOfDays]
    
    #Machine Learning model builds up
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    Length=len(Y)-1
    method=GradientBoostingRegressor()
    method.fit(X[0:Length,:],Y[0:Length].ravel())

    DataForecast=[]
    OriginDataFC=pd.read_table(r'FormerDataForecast.csv',sep=",",header=None)
    DataForecast=OriginDataFC.as_matrix().astype(np.float32)
    
    X_Pre=DataForecast[(len(DataForecast)-25):(len(DataForecast)-1),2:2+3*NumberOfDays]
    Y_Pre=method.predict(X_Pre)
    
    print(Y_Pre)
    return Y_Pre
    
    
    #y_pre=method.predict(X[Length])
    #y_all=Y[Length]
    
    #print(y_pre[0])
    #print(y_all)
    
    #Result=[]
    #Result.append(y_pre)
    #Result.append(y_all) 
    #Score=metrics.accuracy_score(y_pre,Y[Length:])
    #print(Score)
    
    #plt.figure()
    #set the size of subplots
    #left,width=0.1,2.5
    #bottom,height=0.11,1
    #bottom_h=bottom+height+0.04
    #rect_line1=[left,bottom,width,height]
    #axs=plt.axes(rect_line1)
    #plot1=axs.plot(y_pre[0:LengthOfTime],'-ob')
    #plot2=axs.plot(y_all[0:LengthOfTime],'-og',ms=1)
    #plt.show()
    
    #Output
    #Save=pd.DataFrame(y_all.astype(float))
    #Save.to_csv(r'PredictWindSpdData.csv',header=None,index=None)
    
#PredictWindSpd(1)    