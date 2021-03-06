# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 13:59:01 2017

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


File='DataBase/PkLoadPredict.csv'
OriginData=pd.read_table(File,sep=",",index_col=False)
Data=OriginData.as_matrix().astype(np.float32)

    
_LengthOfFile=1095
#Get the properties and the target

Y=Data[:_LengthOfFile,1]
X=Data[:_LengthOfFile,2:]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

method=GradientBoostingRegressor()
method.fit(X_train,y_train.ravel())

Y_Predict=method.predict(X_test)
Y_Error=Y_Predict-y_test

_Result=np.c_[Y_Predict,y_test,Y_Error]
df = pd.DataFrame(data=_Result)
df.to_csv('PkLoadPredictResult.csv')

_Expection=np.mean(Y_Error)

print(_Expection)
print(1-np.sqrt(np.mean((Y_Predict- y_test)**2))/np.mean(y_test))
print(np.sqrt(np.mean((Y_Predict- y_test)**2))/np.mean(y_test))


plt.figure(figsize=(18,8))
plt.plot(Y_Predict, 'blue',label='Predict')
plt.plot(y_test,'red',label='Test')
plt.plot(Y_Error,'orange',label='Error')
plt.legend(loc='best')
plt.show()