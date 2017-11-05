# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 17:05:49 2017

@author: thuzhang
"""
import numpy as np
import pandas as pd

File='DataBase/DataBaseNECA.csv'
OriginData=pd.read_table(File,sep=",")
for i in range(0,int(len(OriginData)/24)):
    _DailyData=OriginData["SYSLoad"][24*i:24*i+24]
    _DryData=OriginData["DryBulb"][24*i:24*i+24]
    _DewData=OriginData["DewPnt"][24*i:24*i+24]
    
    _MaxValue=np.max(_DailyData)
    _MaxValuePoint=np.where(_DailyData==_MaxValue)[0][0]
    
    _MeanDryData=np.mean(_DryData)
    
    _MeanDewData=np.mean(_DewData)
        
    print(_MaxValue,_MaxValuePoint)
    print(_MeanDryData,_MeanDewData)
    for j in range(0,24):
        OriginData["PkLoad"][24*i+j]=_MaxValue
        OriginData["PkLoadPoint"][24*i+j]=_MaxValuePoint
        
        OriginData["MeanDry"][24*i+j]=_MeanDryData
        OriginData["MeanDew"][24*i+j]=_MeanDewData
        
    print('Row%d'%i)

OriginData.to_csv('DataBase/DataBaseNECAOutput2.csv')