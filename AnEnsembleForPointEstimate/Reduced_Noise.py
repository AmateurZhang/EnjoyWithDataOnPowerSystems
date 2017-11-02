# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 20:16:35 2017

@author: thuzhang
"""
# Reduce Noise for the WINDSPEED in Origin7679
# In the former test, the main frequency in the WINDSPEED is the 24-hour 
#Repeatation, or 365 in the FFT result.
#In this function, we remove the frequency more than 400 to eliminate the 
#posibility of overestimation in the later forecast.

#packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def NoiseReduce(LengthOfData=500,mode='Actual'):
    Data=[]
    OriginData=pd.read_table(r'Origin7679.csv',sep=",")
    Data=OriginData.as_matrix().astype(np.float32)
    WindSpeed=Data[:,0]
    if mode=='Actual':
        LengthOfData=len(WindSpeed)
    WindSpeed=WindSpeed[0:LengthOfData]
    
    #FFT
    Transformed=np.fft.fft(WindSpeed)
    
    #Reduce Noise 
    MaxThreshold=2000
    for i in range(1,len(Transformed)):
        if i>MaxThreshold:
            Transformed[i]=0
            
    #Inverse FFT
    IrSignal=abs(np.fft.ifft(Transformed))
    
    #Plot for Reduced Noise Data
    #LengthOfReducedData=LengthOfData
    #plt.figure()
    #set the size of subplots
    #left,width=0.1,2.5
    #bottom,height=0.11,1
    #rect_line1=[left,bottom,width,height]
    #axe=plt.axes(rect_line1)
    #axe.plot(IrSignal[0:LengthOfReducedData],'-b') 
   # axe.plot(WindSpeed[0:LengthOfReducedData],'-g')      
    #plt.show()
    
    #Renew the Data
    DataNew=[]
    DataNew=WindSpeed
    DataNew=np.c_[Data[0:LengthOfData,0],DataNew]

    
    
    #Output
    Save=pd.DataFrame(Data[0:LengthOfData,:].astype(float))
    Save.to_csv('ReducedNoiseData%s.csv'%mode,header=None,index=None)
    
    return DataNew
    
    

#NoiseReduce()
    