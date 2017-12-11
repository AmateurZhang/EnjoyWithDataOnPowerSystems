# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:33:00 2017

@author: thuzhang
"""

import random
import numpy as np

def Resample(a,resample=True):
    Length=len(a)
    if resample==False:
        print(0)
        return random.sample(a,Length)
    else:
        sample=[]
        print(1)
        for i in range(Length):
            sample.append(a[random.randint(0,Length-1)])
        return sample
    
_list=[3,4,55,66,33,2,4]
print(Resample(_list,resample=True))

    
            
            
    