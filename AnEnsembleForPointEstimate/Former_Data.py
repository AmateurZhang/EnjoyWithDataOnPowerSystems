# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 20:50:15 2017

@author: thuzhang
"""

# The Former Data
# In this part, we will calculate the Data from N to 1 day(s) before the 
# forecasting time. The Value we get are R(t-24N)-R(t-24N-1), R(T-24N) and 
# R(T-24N-1)-R(T-24N-2).
# In the test before, we are informed that the Delta Data are helpful to 
# gain a more precise predict. However, if we just input the Data like 
# R(t-24N-i), i=0,1,2 into the estimate machine, the constrains on the Data
# are too much for the model or in another word, this method provides us too
# much information, which can misleading the forecast machine.
# And, significantly, the origin Dataset is from ReducedNoiseData.csv

#packages
import numpy as np
import pandas as pd

def FormerData(NumOfFormerDays=30,model='Model'):
    Data=[]
    OriginData=pd.read_table(r'ReducedNoiseData.csv',sep=",",header=None)
    Data=OriginData.as_matrix().astype(np.float32)
    DataNew=[]
    DataOutput=[]
    #model or forecast
    if model=='Model':
         #Expansion
        for Num in range(1,2+3*NumOfFormerDays-1):
            TempOnes=np.ones(len(Data[:,0]))
            Data=np.c_[Data,TempOnes]
        #function  
        for i in range(0,len(Data[:,0])):
            for j in range(1,NumOfFormerDays+1):
                if i-24*j-2>=0:
                    Data[i,-2+(j)*3]=Data[i-24*(j)-2,0]
                if i-24*j-1>=0:
                    Data[i,(j)*3-1]=Data[i-24*(j)-1,0]
                if i-24*j>=0:
                    Data[i,1+(j)*3-1]=Data[i-24*(j),0]
        DataOutput=Data[(24*(NumOfFormerDays)+2):,:]
    else:
        for Num in range(1,2+3*NumOfFormerDays-1):
            TempOnes=np.ones(len(Data[:,0]))
            Data=np.c_[Data,TempOnes]
        #function  
        for i in range(0,len(Data[:,0])):
            for j in range(0,NumOfFormerDays):
                if i-24*j-2>=0:
                    Data[i,-2+(j+1)*3]=Data[i-24*(j)-2,0]
                if i-24*j-1>=0:
                    Data[i,-1+(j+1)*3]=Data[i-24*(j)-1,0]
                if i-24*j>=0:
                    Data[i,(j+1)*3]=Data[i-24*(j),0]
        DataOutput=Data[(24*(NumOfFormerDays-1)+2):,:]
    
    #Output
    Save=pd.DataFrame(DataOutput.astype(float))
    Save.to_csv('FormerData%s.csv'%model,header=None,index=None)

#FormerData(2,model='Model')
#FormerData(2,model='Forecast')
        
