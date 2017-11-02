# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:00:38 2017

@author: thuzhang
"""

# Ensemble


# packages
from PredictWindSpd import PredictWindSpd
from Reduced_Noise import NoiseReduce
from Former_Data import FormerData
#from ReadTrueValue import ReadTrueValue
import matplotlib.pyplot as plt

#Consts
StartTime=8024
ForecastDays=5
DaysBack=7

#Actual Load
#NoiseReduce(mode='Actual')

Result=[]
TrueResult=[]
#Needed
ActResSts=NoiseReduce(StartTime,mode='')
FormerData(DaysBack,model='Model')
FormerData(DaysBack,model='Forecast')
#for i in range(0,ForecastDays*24):            
Result=PredictWindSpd(DaysBack)
ActResTp=NoiseReduce(StartTime+24,mode='')
ActRes=ActResTp[len(ActResTp)-25:len(ActResTp)-1,1]


plt.figure()
#set the size of subplots
left,width=0.1,2.5
bottom,height=0.11,1
bottom_h=bottom+height+0.04
rect_line1=[left,bottom,width,height]
axs=plt.axes(rect_line1)
plot1=axs.plot(Result,'-ob')
plot2=axs.plot(ActRes,'-og',ms=1)
plot3=axs.plot()
plt.show()


