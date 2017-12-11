# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 19:43:07 2017

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
    
def traintestsplit(X,size=0.8):
    length=int(size*len(X))
    train=X[0:length]
    test=X[length:]
    return train, test

def XandYsplit(X):
    return X.iloc[:,1:],X.iloc[:,0]
    
File="DataBase.csv"
OriginData=pd.read_table(File,sep=",",index_col=False)

_train,_test= traintestsplit(OriginData,0.8)
_X_train,_y_train=XandYsplit(_train)
X_test,y_test=XandYsplit(_test)

# RF
clfsetRandomForest=[]
paramsRF = {'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 2,
        }
for i in range(0,25):
    _resampledtrain=Resampletwo(_train,resample=True)
    X_train,y_train=XandYsplit(_resampledtrain)    
    paramss=paramsRF
    clf=RandomForestRegressor(**paramss)
    clf.fit(X_train,y_train)
    mse = mean_squared_error(y_train, clf.predict(X_train))
    print("MSE: %.4f" % mse)
    clfsetRandomForest.append(clf)
    

def BootitemRF(Point):   
    _resamplemodels=Resample(clfsetRandomForest,resample=True)
    predicted=[]
    for i,model in enumerate(_resamplemodels):
        _res=model.predict(X_test[Point:Point+1])
        predicted.append(_res[0])
    return np.mean(predicted)

bootresRandomForest=[]
eeres=[]
for i in range(0,500):
    mem=BootitemRF(100)
    bootresRandomForest.append(mem)
plt.hist(bootresRandomForest)

#GBRT
paramsGBRT = {'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 2,
          'learning_rate': 0.1, 'loss': 'ls'}
clfsetGBRT=[]
for i in range(0,25):
    _resampledtrain=Resampletwo(_train,resample=True)
    X_train,y_train=XandYsplit(_resampledtrain)    
    paramss=paramsGBRT
    paramss['max_depth']=6
    clf=GradientBoostingRegressor(**paramss)
    clf.fit(X_train,y_train)
    mse = mean_squared_error(y_train, clf.predict(X_train))
    print("MSE: %.4f" % mse)
    clfsetGBRT.append(clf)
    

def BootitemGBRT():   
    _resamplemodels=Resample(clfsetGBRT,resample=True)
    predicted=[]
    for i,model in enumerate(_resamplemodels):
        _res=model.predict(X_test[100:101])
        predicted.append(_res[0])
    return np.mean(predicted)

bootresGBRT=[]

for i in range(0,500):
    mem=BootitemGBRT()
    bootresGBRT.append(mem)
plt.hist(bootresGBRT)


# Bootstrap

_clfsetRandomForest=clfsetRandomForest
_clfsetGBRT=clfsetGBRT
_clfsetRandomForest.extend(clfsetGBRT)
clfset=_clfsetRandomForest


def BootitemEns(Point):   
    _resamplemodels=Resample(clfset,resample=True)
    predicted=[]
    for i,model in enumerate(_resamplemodels):
        _res=model.predict(X_test[Point:Point+1])
        predicted.append(_res[0])
    return np.mean(predicted)



def PointForecast(Point):
    bootresens=[]
    for i in range(0,1):
        mem=BootitemEns(Point)
        bootresens.append(mem)       
    return y_test.iloc[Point],np.mean(bootresens),np.percentile(bootresens,5),np.percentile(bootresens,95)

# RF
def PointForecastRF(Point):
    bootresens=[]
    for i in range(0,10):
        mem=BootitemRF(Point)
        bootresens.append(mem)       
    return y_test.iloc[Point],np.mean(bootresens),np.percentile(bootresens,5),np.percentile(bootresens,95)    
Y_real_RF=[]
Y_mean_RF=[]
Y_5_RF=[]
Y_95_RF=[] 
for i in range(24*7,14*24):
    y_real_RF,y_mean_RF,y_5_RF,y_95_RF=PointForecastRF(i)
    print(y_real_RF,y_mean_RF,y_5_RF,y_95_RF)
    Y_real_RF.append(y_real_RF)
    Y_mean_RF.append(y_mean_RF)
    Y_5_RF.append(y_5_RF)
    Y_95_RF.append(y_95_RF)

print(MAPE(Y_mean_RF,Y_real_RF)) 
mape=MAPE(Y_mean_RF,Y_real_RF)   
plt.figure(figsize=(18,8))
plt.ylim([0,1500])
plt.plot(Y_real_RF,'green',label='Real')
plt.plot(Y_mean_RF, 'blue',label='Predict MAPE=%f'%mape)
plt.plot(Y_5_RF,'gray',label='5%-Percentile')
plt.plot(Y_95_RF,'yellow',label='95%-Percentile')
plt.legend(loc='best')
plt.show()





#
Y_real=[]
Y_mean=[]
Y_5=[]
Y_95=[] 
Y_error=[]  

for i in range(0,7*24):
    y_real,y_mean,y_5,y_95=PointForecast(i)
    print(y_real,y_mean,y_5,y_95)
    Y_real.append(y_real)
    Y_mean.append(y_mean)
    Y_5.append(y_5)
    Y_95.append(y_95)
    Y_error.append(y_mean-y_real)

print(MAPE(Y_mean,Y_real)) 
mape=MAPE(Y_mean,Y_real)   
plt.figure(figsize=(18,8))
plt.ylim([0,1600])
plt.plot(Y_real,'green',label='Real')
plt.plot(Y_mean, 'blue',label='Predict MAPE=%f'%mape)
plt.plot(Y_5,'gray',label='5%-Percentile')
plt.plot(Y_95,'yellow',label='95%-Percentile')
plt.legend(loc='best')
plt.show()


#Residual Bootstrap (先验)
def BootitemRB(Point):   
    _resamplemodels=Resample(clfset,resample=True)
    predicted=[]
    for i,model in enumerate(_resamplemodels):
        _res=model.predict(_X_train[Point:Point+1])
        predicted.append(_res[0])
    return np.mean(predicted)

def PointForecastRB(Point):
    bootresens=[]
    for i in range(0,1):
        mem=BootitemRB(Point)
        bootresens.append(mem)       
    return _y_train.iloc[Point],np.mean(bootresens),np.percentile(bootresens,5),np.percentile(bootresens,95),_X_train.iloc[Point]

#
Y_real_RB=[]
Y_mean_RB=[]
Y_error_RB=[]
Y_5_RB=[]
Y_95_RB=[]
X_Origin_Data=pd.DataFrame(columns=_X_train.columns)

for i in range(len(_y_train)-1-30*24,len(_y_train)-1):
    y_real_RB,y_mean_RB,y_5_RB,y_95_RB,x_Origin_Data=PointForecastRB(i)
    print(y_real_RB,y_mean_RB,y_5_RB,y_95_RB)
    Y_real_RB.append(y_real_RB)
    Y_mean_RB.append(y_mean_RB)
    Y_error_RB.append(y_mean_RB-y_real_RB)
    Y_5_RB.append(y_5_RB)
    Y_95_RB.append(y_95_RB)
    #X_Origin_Data.append(x_Origin_Data)
    _X=X_Origin_Data.append(x_Origin_Data)
    X_Origin_Data=_X
    
plt.hist(Y_error_RB)    
Centre_Y_error_RB=Y_error_RB-np.mean(Y_error_RB)


#Errors
X_Residual=X_Origin_Data.loc[:,['WEEKDAY','HOUR','TP1']]
_Y_error_Seri=pd.DataFrame(Y_error_RB,columns=['ERROR'])
_Y_mean_Seri=pd.DataFrame(Y_mean_RB,columns=['Predicted'])

pd.merge(_Y_error_Seri,_Y_mean_Seri,left_index=True,right_index=True,how='outer')
_x1=X_Residual.reset_index(drop=True)
X_Residual=_x1
_Merged_Error_Esti=pd.merge(_Y_error_Seri,X_Residual,left_index=True,right_index=True,how='outer')

X_trainl,y_trainl=XandYsplit(_Merged_Error_Esti)    
lclf=Lasso(alpha=0.05)    
lclf.fit(X_trainl,y_trainl)

lclf.predict(X_trainl)

plt.figure(figsize=(10,8))
plt.plot(y_trainl,color='red')
plt.plot(lclf.predict(X_trainl),color='green')
plt.show()

for j in range(20,50):
    Linearclf=[]
    _ResLr=[]
    _lpoint=j
    for i in range(0,500):
        _resampledtrain=Resampletwo(_Merged_Error_Esti,resample=True)
        X_train,y_train=XandYsplit(_resampledtrain)    
        clf=RandomForestRegressor()
        clf.fit(X_train,y_train)
        mse = mean_squared_error(y_train, clf.predict(X_train))
        _ResLr.append(clf.predict(X_train.iloc[_lpoint:_lpoint+1])[0])
        #print("MSE: %.4f" % mse)
        Linearclf.append(clf)

    plt.hist(_ResLr)
    y_trainl.iloc[_lpoint:_lpoint+1]
    np.percentile(_ResLr,95)

    plt.figure(figsize=(5,5))
    plt.hist(_ResLr)
    plt.vlines(y_trainl.iloc[_lpoint:_lpoint+1],0,100)
    plt.show()
 
#
Linearclf=[]

for i in range(0,500):
   _resampledtrain=Resampletwo(_Merged_Error_Esti,resample=True)
   X_train,y_train=XandYsplit(_resampledtrain)    
   clf=LinearRegression()
   clf.fit(X_train,y_train)
   Linearclf.append(clf)
   
X_test_RB=X_test
_X_Residual_BP=X_test_RB.loc[:,['WEEKDAY','HOUR','TP1']]
_Y_error_Seri_BP=pd.DataFrame(Y_error,columns=['ERROR'])
_Y_mean_Seri_BP=pd.DataFrame(Y_mean,columns=['Predicted'])

pd.merge(_Y_error_Seri_BP,_Y_mean_Seri_BP,left_index=True,right_index=True,how='outer')
_x1=_X_Residual_BP.reset_index(drop=True)
X_Residual_BP=_x1
_Merged_Error_Esti_BP=pd.merge(_Y_error_Seri_BP,X_Residual_BP,left_index=True,right_index=True,how='outer')        

for i in range(24,60):
    _Point_BP=i
    _Res_BP=[]
    X_testlbp,y_testlbp=XandYsplit(_Merged_Error_Esti_BP)    
    for i,model in enumerate(Linearclf):
        _res=(model.predict(X_testlbp[_Point_BP:_Point_BP+1]))[0]
        _Res_BP.append(_res)
    
    plt.figure(figsize=(5,5))
    plt.hist(_Res_BP)
    plt.vlines(y_testlbp.iloc[_Point_BP:_Point_BP+1],0,100)
    plt.show()
