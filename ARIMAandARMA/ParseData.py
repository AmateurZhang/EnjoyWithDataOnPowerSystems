# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 17:06:55 2017

@author: thuzhang
"""

# Parse Data
# parse DataSet Here ...
import numpy as np
import pandas as pd

def ParseData(File='',Sts=10,Length=100):
    Data=[]
    try:
        OriginData=pd.read_table(File,sep=",",header=None)
    except:
        print('[ParseData] Open file failed.')
        return -1
    Data=OriginData.as_matrix().astype(np.float32) #Read as Numpy Array
    if Length-Sts>=len(Data[:,0]):
        print('[ParseData] Overflow!')
        return -1        
    else:
        print('[ParseData] Return Data with Length %d'%Length)
        return Data[Sts:Sts+Length,0]
        
    
#ParseData(File=r'2014_hourly_ME.csv',Length=100) 