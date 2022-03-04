import pandas as pd
import numpy as np

from typing import Union, Optional, NoReturn
from scipy.stats import norm

from value_at_risk.resources.exceptions import VARMethodsError, HistoricalVARMethodError


class ValueAtRiskDefaultAttrs:

    def __init__(self, data: Optional[Union[pd.Series, str], type(None)] = None,
                 mu: Optional[Union[int, float], type(None)] = None,
                 sigma: Optional[Union[int, float], type(None)] = None,
                 mkt_val: Optional[Union[int, float], type(None)] = None):

        self.trading_days = 252

        if isinstance(mkt_val, (int, float)):
            self.mkt_val = mkt_val
        else:
            self.mkt_val = 1

        if not isinstance(data, pd.Series):
            self._mu = mu
            self._sigma = sigma
        else:
            self._returns = data.sort_index(ascending=True).pct_change().dropna()

    @property
    def returns(self) -> Union[pd.Series, NoReturn]:
        if hasattr(self, '_returns'):
            return self._returns
        else:
            if not isinstance(self._mu, (int, float)) or not isinstance(self._sigma, (int, float)):
                raise VARMethodsError()
            else:
                return HistoricalVARMethodError()

    @property
    def roll_len(self) -> int:
        if hasattr(self, '_returns'):
            return len(self.returns) if len(self.returns) < self.trading_days else self.trading_days
        else:
            return 1

    @property
    def ann_factor(self) -> Union[float, int]:
        return np.sqrt(self.trading_days / self.roll_len)

    @property
    def mu(self) -> Union[float, int]:
        if hasattr(self, '_mu'):
            return self._mu
        else:
            return self.returns.rolling(window=self.roll_len).mean().dropna().mean().iloc[0]

    @property
    def sigma(self) -> Union[float, int]:
        if hasattr(self, '_sigma'):
            return self._sigma
        else:
            return self.returns.rolling(window=self.roll_len).dropna().std().iloc[0]


class ParametricValueAtRisk(ValueAtRiskDefaultAttrs):

    def __init__(self, data: Optional[Union[pd.Series, str], type(None)] = None,
                 mu: Optional[Union[int, float], type(None)] = None,
                 sigma: Optional[Union[int, float], type(None)] = None,
                 mkt_val: Optional[Union[int, float], type(None)] = None):

        ValueAtRiskDefaultAttrs.__init__(self, data=data, mu=mu, sigma=sigma, mkt_val=mkt_val)

    def calculate_parametric_var(self, alpha: float = .01, smooth_factor: float = 1.0, pct: bool = True) -> Union[
        float, int]:
        """
        Calculate the value at risk (VaR) from
        :param alpha: float -> Confidence level which translates to the return threshold above the inverse CDF assuming
        our return distribution is normal or Gaussian (Default cutoff at .01)
        :param smooth_factor: float -> Alpha or smoothing factor to exponentially adjust weights in moving frame.
        Note that if the class's sigma and mean were given then the smoothing factor will not be available (default is 1)
        :param pct: bool -> Set to False if notional value of asset is to be returned  (default is True)

        :return: float, int -> Notional value of asset at risk (VaR) or percentage based if param percent is True
        """

        if smooth_factor == 1:
            sigma = self.sigma
        else:
            assert not isinstance(self.returns, type(None))
            sigma = self.returns.iloc[-self.roll_len:, ].ewm(alpha=smooth_factor, adjust=True).std().iloc[0]

        var = sigma * norm.ppf(1 - alpha) * self.ann_factor

        if pct:
            return var * 100
        else:
            return var * self.mkt_val


class HistoricalValueAtRisk(ValueAtRiskDefaultAttrs):

    def __init__(self, data: Optional[Union[pd.Series, str], type(None)] = None,
                 mu: Optional[Union[int, float], type(None)] = None,
                 sigma: Optional[Union[int, float], type(None)] = None,
                 mkt_val: Optional[Union[int, float], type(None)] = None):

        ValueAtRiskDefaultAttrs.__init__(self, data=data, mu=mu, sigma=sigma, mkt_val=mkt_val)

    def calculate_historical_var(self, alpha: float = .01, iter: int = 10000, pct: bool = True) -> Union[
        float, int, HistoricalVARMethodError]:
        """
        Calculate the value at risk (VaR) from random samples (default sample number set to 10000) of historical returns
        :param alpha: float -> Confidence level which translates to the quantile of returns corresponding to the highest
        available datapoint within the interval of data points that the specified quantile lies between
        :param iter: int -> Number of iterations to draw random samples from historical data (default at 1000)
        :param pct: bool -> Set to False if notional value of asset is to be returned (default is True)

        :return: float, int -> Notional value of asset at risk (VaR) or percentage based if param percent is True
        """

        returns = self.returns

        if returns == HistoricalVARMethodError:
            return HistoricalVARMethodError()

        func_vec = np.vectorize(
            lambda: np.array([np.random.choice(returns, self.trading_days, replace=True) for _ in range(iter)]),
            otypes=[int, float])

        simulations = np.apply_along_axis(lambda x: np.quantile(x, 1 - alpha, interpolation='higher'), 0, func_vec())

        var = np.mean(simulations)

        if pct:
            return var * 100
        else:
            return var * self.mkt_val
