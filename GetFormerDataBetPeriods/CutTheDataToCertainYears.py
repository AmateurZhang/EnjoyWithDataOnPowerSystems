# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 13:10:59 2017

@author: thuzhang
"""

# this program is used for merge xls files in the same formats
import xlrd

import xlwt

output=xlwt.Workbook()

# consts
lowpage=2013
highpage=2017
LengthofItems=30

Outputnrow=[1]*LengthofItems
Outputncol=[0]*LengthofItems

for index in range(lowpage,highpage):
    data =xlrd.open_workbook('%d_smd_hourly.xls'%index)
    names=data.sheet_names()
    for nameindex in range(1,len(names)):
        outputsheet=output
        if index==lowpage:
            outputsheet=output.add_sheet('%s'%names[nameindex],cell_overwrite_ok=True)
            table =data.sheets()[nameindex]
            nrows=table.nrows
            ncols=table.ncols
            for i in range(nrows-49,nrows):
                value= table.row_values(i)
                for item in range(0,len(value)):
                    outputsheet.write(Outputnrow[nameindex],item,value[item])
                Outputnrow[nameindex]+=1
            continue    
    
        
        outputsheet=output.get_sheet(nameindex-1)
        table =data.sheets()[nameindex]
        nrows=table.nrows
        ncols=table.ncols
        for i in range(1,nrows):
            value= table.row_values(i)
            for item in range(0,len(value)):
                #print(Outputnrow[nameindex])
                outputsheet.write(Outputnrow[nameindex],item,value[item])
            Outputnrow[nameindex]+=1
          
                
output.save('smd_hourly_ME_Output_14_16.xls')            
        
        
    