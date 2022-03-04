import datetime as dt
import numpy as np
import pandas as pd

from resources.var import VAR

if __name__ == '__main__':
    idx_len = 200
    mu = 0
    sigma = .005

    # 50 bps daily volatility (8% Vol)
    data = pd.Series(data=np.random.normal(loc=mu, scale=sigma, size=idx_len),
                     index=[pd.date_range(dt.datetime.today(), periods=idx_len).tolist()]).apply(
        lambda x: x + 1).cumprod()

    val_at_risk = VAR(data=data, mu=mu, sigma=sigma, smooth_factor=1, alpha=.02, pct=True)
