# This is a Temp README

###  PkLoadPredict.py

- We decompose the Power Load into Peak Value and Others.
- The Original Database is DataBaseNECA.xlsx, which includes the Electric Market Information within the ISO-NE Control Area. i.e. SYSLoad, Temperature etc.
- We choose these items to predict PkLoad (Vector)
  - PkLoad, T24PkLoad, T24PkPoint, T48PkLoad, T48PkPoint, MeanDry, MeanDew, WeekDay, Month
  - eg. 18116,18435,18,18294,18,29.5 ,15.6 ,3,1
- The Error is <3% by cross validation. Result is in the File PkLoadPredictResult.csv.

### SeasonalPredict.py

- We use Random forest Method to train our model.

- The Original Database is SeasonalPredict.csv, which includes the features to predict.

- We choose these items to predict Seasonal

  - Seasonal, T24Seasonal, T24Dry, T24Dew, Hour, Date, WeekDay, Month.
  - eg. 6632, 6457, 47, 45, 1,15, 3, 1

- The error is about 15-17% according to the Seasonal Data, 5% or less according to the total Dataset.

- Note that We DON'T use the 7*24 ahead data as a feature. 

  - The Dataset using the mentioned feature is presented in SeasonalPredict2.csv.

  - The Error is Larger than the former model about 8-10%.

    ​

