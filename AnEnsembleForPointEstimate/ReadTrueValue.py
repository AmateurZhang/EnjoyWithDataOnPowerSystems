# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 14:13:15 2017

@author: thuzhang
"""

# Read True Value
import numpy as np
import pandas as pd

def ReadTrueValue(time=7000):
    Data=[]
    OriginData=pd.read_table(r'ReducedNoiseDataActual.csv',sep=",",header=None)
    Data=OriginData.as_matrix().astype(np.float32)
    #print(Data[time,1])
    return Data[time,1]

ReadTrueValue(7000)