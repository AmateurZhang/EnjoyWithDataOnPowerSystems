# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 00:30:24 2017

@author: thuzhang
"""

import numpy as np
import pandas as pd


File='DataBase/DataBaseNECA.csv'
OriginData=pd.read_table(File,sep=",")
for i in range(0,int(len(OriginData)/24)):
    _DailyData=OriginData[24*i:24*i+24]
    _MaxValue=np.max(_DailyData)[0]
    _MaxValuePoint=np.where(_DailyData==_MaxValue)[0][0]
    print(_MaxValue)
    print(_MaxValuePoint)
    for j in range(0,24):
        OriginData["PkLoad"][24*i+j]=_MaxValue
        OriginData["PkLoadPoint"][24*i+j]=_MaxValuePoint
    print('Row%d'%i)

OriginData.to_csv('DataBase/DataBaseNECAOutput.csv')