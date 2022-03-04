import pandas as pd

from typing import Union, Optional

from value_at_risk.resources.support import ParametricValueAtRisk, HistoricalValueAtRisk


class ValueAtRisk(ParametricValueAtRisk, HistoricalValueAtRisk):
    """
    Calculate Value at Risk through historical and parametric methods
    """

    def __init__(self, data: Optional[Union[pd.Series, str], type(None)] = None,
                 mu: Optional[Union[int, float], type(None)] = None,
                 sigma: Optional[Union[int, float], type(None)] = None,
                 mkt_val: Optional[Union[int, float], type(None)] = None,
                 alpha: Optional[Union[int, float], type(None)] = None,
                 smooth_factor: Optional[Union[int, float], type(None)] = None,
                 pct: Optional[Union[int, float], type(None)] = None):
        """
        :param data: pd.Series, str -> n x 1 dataframe with price data and datetime as an index
        :param mu: int, float -> Given expected mean of portfolio returns to calculate the VaR using
        the parametric VaR method
        :param sigma: int, float -> Given expected standard deviation of portfolio returns to calculate the VaR using
        the parametric VaR method
        :param mkt_val: int, float -> The specified notional value of the portfolio (Default is 1)
        :param alpha: float -> Confidence level which translates to the return threshold above the inverse CDF assuming
        our return distribution is normal or Gaussian (Default cutoff at .01)
        :param smooth_factor: float -> Alpha or smoothing factor to exponentially adjust weights in moving frame.
        Note that if the class's sigma and mean were given then the smoothing factor will not be available (default is 1)
        :param pct: bool -> Set to False if notional value of asset is to be returned  (default is True)

        :return NoneType (Sets historical_var and parametric_var attrs)
        """

        kwargs = locals()

        ParametricValueAtRisk.__init__(self, **kwargs)
        HistoricalValueAtRisk.__init__(self, **kwargs)

    @property
    def historical_var(self):
        return self.calculate_historical_var()

    @property
    def parametric_var(self):
        return self.calculate_parametric_var()


class VAR(ValueAtRisk):

    def __init__(self, data: Optional[Union[pd.Series, str], type(None)] = None,
                 mu: Optional[Union[int, float], type(None)] = None,
                 sigma: Optional[Union[int, float], type(None)] = None,
                 mkt_val: Optional[Union[int, float], type(None)] = None):
        super().__init__(**locals())
