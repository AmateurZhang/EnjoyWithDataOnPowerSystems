CutTheDataToCertainYears.py

Parse the Data from 20xx_smd_hourly.xls to 20xx_smd_hourly.xls

`lowpage=2013 `

`highpage=2017`

with output file smd_hourly_ME_Output\_14\_16.xls

GetTheFormerDataForAllSheets.py

From smd_hourly_ME_Output\_14\_16.xls

Convert the data to the following form.

```python
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
    outputsheet.write(k,11,t49)`
```
with output smd_hourly_Output\_13\_Merge.xls

