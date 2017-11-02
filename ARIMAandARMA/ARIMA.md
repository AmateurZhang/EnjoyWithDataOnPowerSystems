# ARMA/ARIMA

- Decomposition

  - ```python
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(ts_log_diff,freq=24)
    trend = decomposition.trend  # 趋势
    seasonal = decomposition.seasonal  # 季节性
    residual = decomposition.resid  # 剩余的
    ```

- 24h period ARMA

  - `Tp=ARMA(DataModel,(24,3))`

- Data: 2014_hourly_ME.csv from ISO-NEWENGLAND 