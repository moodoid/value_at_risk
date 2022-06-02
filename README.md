# value_at_risk

### Calculate Value-at-Risk (VaR) of a portfolio through historical and parametric methods


```buildoutcfg
import datetime as dt
import numpy as np
import pandas as pd

from value_at_risk import VAR 

idx_len = 10000
mu = 0
sigma = .005

# 50 bps daily volatility (8% Ann Vol)
data = pd.Series(data=np.random.normal(loc=mu, scale=sigma, size=idx_len),
                 index=[pd.date_range(dt.datetime.today(), periods=idx_len).tolist()])
                 
data = data.apply(lambda x: x + 1).cumprod()

val_at_risk = VAR(data=data, mu=mu, sigma=sigma, alpha=.05, smooth_factor=1, pct=True)

print(val_at_risk.historical_var)
print(val_at_risk.parametric_var)
```