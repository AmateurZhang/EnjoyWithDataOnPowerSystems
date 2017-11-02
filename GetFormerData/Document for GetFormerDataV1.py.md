Document for GetFormerDataV1.py

Parse the Data in form of 

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
    outputsheet.write(k,11,t49)
```
from the Database ISO-NEWENGLAND: $smd\_hourly\_ME\_2014-16.xlsx$.

$t24$ means the value at $t-24(hour)$. 

The Data can be used as point estimate for power loads.