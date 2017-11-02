# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import xlrd

import xlwt


output=xlwt.Workbook()


data =xlrd.open_workbook(r'smd_hourly_ME_2014-16.xlsx')
names=data.sheet_names()
for index in range(0,len(names)):
    outputsheet=output.add_sheet('%s1'%names[index])
    table =data.sheets()[index]
    nrows=table.nrows
    ncols=table.ncols
    currentsheet=1
    for i in range(1,nrows):
        if i==65535:
            currentsheet+=1
            outputsheet=output.add_sheet("%s%d"%names[index]%currentsheet) #自动编号
            
        rowValues=table.row_values(i)
        hour=int(rowValues[1])
        x=xlrd.xldate_as_tuple(table.cell_value(i,0),data.datemode)
        y=xlrd.xldate.xldate_as_datetime(table.cell_value(i,0),data.datemode).weekday()
        year=x[0]
        month=x[1]
        day=x[2]
        weekday=y
        DEMAND=rowValues[3]
    
        t24=0
        t25=0
        t26=0
        t47=0
        t48=0
        t49=0
    
        if i-49>0:
            t49=table.cell_value(i-49,3)
        
        if i-48>0:
            t48=table.cell_value(i-48,3)
        
        if i-47>0:
            t47=table.cell_value(i-47,3)
    
        if i-26>0:
            t26=table.cell_value(i-26,3)
        
        if i-25>0:
            t25=table.cell_value(i-25,3)  
    
        if i-24>0:
            t24=table.cell_value(i-24,3)
        
       # print(DEMAND,year,month,day,weekday,hour,t24,t25,t26,t47,t48,t49)
        
        k=i%65535
        
        outputsheet.write(k,0,DEMAND)
        outputsheet.write(k,1,year)
        outputsheet.write(k,2,month)
        outputsheet.write(k,3,day)
        outputsheet.write(k,4,weekday)
        outputsheet.write(k,5,hour)
        outputsheet.write(k,6,t24)
        outputsheet.write(k,7,t25)
        outputsheet.write(k,8,t26)
        outputsheet.write(k,9,t47)
        outputsheet.write(k,10,t48)
        outputsheet.write(k,11,t49)
        
        #以下为新版本兼容
        """
        outputsheet.cell(i,1,DEMAND)
        outputsheet.cell(i,1,year)
        outputsheet.cell(i,2,month)
        outputsheet.cell(i,3,day)
        outputsheet.cell(i,4,weekday)
        outputsheet.cell(i,5,hour)
        outputsheet.cell(i,6,t24)
        outputsheet.cell(i,7,t25)
        outputsheet.cell(i,8,t26)
        outputsheet.cell(i,9,t47)
        outputsheet.cell(i,10,t48)
        outputsheet.cell(i,11,t49)
        """
    

output.save(r'smd_hourly_ME_Output_2.xls')
    
  


