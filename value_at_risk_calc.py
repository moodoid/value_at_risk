import re
import pandas as pd
import numpy as np

from typing import Union
from scipy.stats import norm


class ValueAtRisk:
    '''
    Calculate Value at Risk through historical and parametric methods
    '''

    def __init__(self, data: pd.Series, market_value: Union[int, float], mu: Union[int, float],
                 sigma: Union[int, float]):
        '''
        :param data: pd.DataFrame -> N x 1 dataframe with price data and datetime as an index
        :param market_value: int, float -> The specified notional value of the portfolio
        :param sigma: int, float -> Given expected standard deviation of portfolio returns to calculate the VaR using
        the parametric VaR method
        :param mu: int, float -> Given expected mean of portfolio returns to calculate the VaR using
        the parametric VaR method
        '''

        self.__kwargs = locals()

        # setting attributes to kwargs
        self.__market_value = market_value
        self.__prices = data

        # setting default attributes and formatting dataframe and initializing rolling means and sigmas
        self.__year = 252
        self.__annualizing_factor = 1


        self.__returns = self.__prices.pct_change(periods=1).dropna()
        self.__rolling_window = len(self.__returns) if len(self.__returns) < self.__year else self.__year

        self.__mu = mu
        self.__sigma = sigma



        self.__mu = self.__returns.rolling(window=self.__rolling_window).mean().dropna().iloc[-1, 0]
            self.__sigma = self.__returns.rolling(window=self.__rolling_window).std().dropna().iloc[-1, 0]
            self.__annualizing_factor = np.sqrt(self.__year / self.__rolling_window)
        else:
            pass

    # getters
    @property
    def year(self):
        return self.__year

    @property
    def annualizing_factor(self):
        return self.__annualizing_factor

    @property
    def prices(self):
        try:
            return self.__prices
        except AttributeError:
            return 'No data given--only parametric VaR method available'

    @property
    def returns(self):
        try:
            return self.__returns
        except AttributeError:
            return 'No data given--neither method of VaR methods available'

    @property
    def mu(self):
        return self.__mu

    @property
    def sigma(self):
        return self.__sigma

    def var_historical(self, alpha: float = .01, market_value: int = 1, iter: int = 500, pct: bool = True):
        '''
        Calculate the value at risk (VaR) from random samples (default sample number set to 500) of historical returns
        :param alpha: float -> Confidence level which translates to the quantile of returns corresponding to the highest
        available datapoint within the interval of data points that the specified quantile lies between
        :param market_value: int|float -> Nominal value (normally in EUR) of current exposure to asset
        :param iter: int -> Number of iterations to draw random samples from historical data
        :return: float -> Nominal value of asset (EUR) at risk (VaR)
        '''

        if not isinstance(self.__market_value, type(np.nan)):
            market_value = self.__market_value

        simulations = []
        for _ in range(iter):
            simulation = self.returns.sample(min(len(self.returns), self.__year), replace=True)
            simulations.append(simulation.quantile((1 - alpha), interpolation='higher') * self.annualizing_factor)
        var = np.mean(simulations)

        if np.isnan(var):
            var = 0.0

        if pct:
            return var * 100
        else:
            return var * market_value

    def var_parametric(self, alpha: float = .01, smoothing_factor: float = 1.0, percent: bool = True):
        '''
        Calculate the value at risk (VaR) from
        :param alpha: float -> Confidence level which translates to the return threshold above the inverse CDF assuming
        our return distribution is normal or Gaussian
        :param market_value: int|float -> Nominal value (normally in EUR) of current exposure to asset
        :param smoothing_factor: float -> Alpha or smoothing factor to exponentially adjust weights in moving frame.
        Note that if the class's sigma and mean were given then the smoothing factor will not be available
        :return: float -> Nominal value of asset (EUR) at risk (VaR)
        '''

        if not getattr(self, 'market_value', None) == None:
            market_value = self.market_value
        else:
            market_value = 1

        if smoothing_factor == 1:
            pass
        else:
            if 'mu' not in self.__kwargs or 'mean' not in self.__kwargs:
                pass
            else:
                self.__sigma = pd.DataFrame(self.__returns.iloc[len(self.__returns) - self.__rolling_window:
                                                                len(self.__returns), 0]).ewm(alpha=smoothing_factor,
                                                                                             adjust=True).std().iloc[
                    -1, 0]

        var = self.sigma * norm.ppf(1 - alpha) * self.annualizing_factor

        if np.isnan(var):
            var = 0.0

        if percent:
            return var * 100
        else:
            return var * market_value


class VAR(ValueAtRisk):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ComputeVAR:
    def __init__(self, notional_val_df: pd.DataFrame, hist_returns_df: pd.DataFrame, var_calc_map: dict = None):
        self.__var_calc_map = var_calc_map
        weighted_tickers = notional_val_df.index[np.where(notional_val_df.notnull())[0]]
        self.__notional_val_df = notional_val_df.loc[weighted_tickers, :]
        self.__hist_returns_df = hist_returns_df.loc[:, weighted_tickers]

        self.portfolio_var = self.get_portfolio_var()

    @property
    def hist_returns_df(self):
        return self.__hist_returns_df

    @property
    def notional_val_df(self):
        return self.__notional_val_df

    @property
    def var_calc_map(self):
        return self.__var_calc_map

    def get_portfolio_var(self):
        return self.compute_var(absolute='Portfolio')

    def get_incremental_var_dict(self):
        return self.structure_var_data()

    @staticmethod
    def analytical_std(returns_matrix: pd.DataFrame, weights_matrix: pd.DataFrame):
        if weights_matrix.T.to_numpy().shape == (1, 1):
            weights_matrix_cus = weights_matrix.T.to_numpy()
        else:
            weights_matrix_cus = np.squeeze(weights_matrix.T.to_numpy())

        portfolio_std = np.sqrt(
            weights_matrix_cus.dot(np.squeeze(returns_matrix.fillna(returns_matrix.mean(axis=1))
                                              .cov().to_numpy())).dot(
                weights_matrix.to_numpy())[0]) * np.sqrt(248 / len(returns_matrix))

        return portfolio_std

    @staticmethod
    def analytical_mean(returns_matrix: pd.DataFrame, weights_matrix: pd.DataFrame):
        portfolio_mean = weights_matrix.T.to_numpy().dot(returns_matrix.mean().to_numpy())[0] * \
                         np.sqrt(248 / len(returns_matrix))

        return portfolio_mean

    def structure_var_data(self):
        incremental_var_dict = {}
        for ticker in self.var_calc_map['Portfolio']:
            incremental_var_dict[ticker] = self.compute_var(incremental=ticker)

        for bucket in ['EUR', 'USD', 'Equity', 'Spac']:
            if re.search('|'.join(list(self.var_calc_map.keys())), bucket, re.I):
                incremental_var_dict[bucket] = self.compute_var(incremental=bucket)

        return incremental_var_dict

    def compute_var(self, **kwargs):
        col_filt, var_type = None, None
        for kwarg, arg in kwargs.items():
            if re.search('incremental|absolute', kwarg, re.I):
                var_type = kwarg
            else:
                raise (Exception('Specify kwarg as incremental or absolute'))
            col_filt = self.var_calc_map[arg]

        # filter prices and weights dataframe by tickers
        if re.search('absolute', var_type, re.I):
            col_list = [col for col in self.hist_returns_df.columns if
                        re.search('|'.join(col_filt), str(col), re.I)]
        elif re.search('incremental', var_type, re.I):
            col_list = [col for col in self.hist_returns_df.columns if col not in col_filt]
        else:
            col_list = None

        hist_returns_df = self.hist_returns_df.loc[:, col_list]
        notional_val_df = self.notional_val_df.loc[[_ for _ in col_list], :]

        weights_df = notional_val_df / notional_val_df.apply(lambda x: abs(x)).sum(axis=0)

        # re-creation of historical series of invested amount for historical VaR method or parametric VaR method
        # notional_val_df = notional_val_df.loc[[_[0] for _ in hist_returns_df.columns], :]
        hist_returns_cumprod_df = hist_returns_df.replace(np.nan, 0).apply(lambda x: x + 1
        if np.sign(notional_val_df.iloc[notional_val_df.index.get_loc(x.name), 0]) > 0 else 1 - x).cumprod()

        # multiply notional value dataframe by cumulative product (return series)
        portfolio_df = pd.DataFrame(hist_returns_cumprod_df.apply(lambda x: np.sum(np.asarray(x)
                                                                                   * np.asarray(
            notional_val_df.iloc[:, 0].apply(lambda y: abs(y)))), axis=1))

        # portfolio expected sigma to be used for parametric VaR calculations
        portfolio_std = self.analytical_std(returns_matrix=hist_returns_df, weights_matrix=weights_df)

        mkt_val = np.sum(notional_val_df).item()
        portfolio_var_obj = VAR(df=portfolio_df, market_value=mkt_val)
        analytical_var_obj = VAR(sigma=portfolio_std, market_value=mkt_val)

        if re.search('absolute', var_type, re.I):
            return pd.DataFrame(columns=['Portfolio VaR Price Series', 'Portfolio VaR Analytical'],
                                data=[[portfolio_var_obj.var_parametric(percent=True),
                                       analytical_var_obj.var_parametric(percent=True)],
                                      [portfolio_var_obj.var_parametric(percent=False),
                                       analytical_var_obj.var_parametric(percent=False)
                                       ]], index=['Percent VaR', 'Notional Value VaR'])
        elif re.search('incremental', var_type, re.I):
            incremental_portfolio = self.portfolio_var.loc['Notional Value VaR', 'Portfolio VaR Price Series'].item() - \
                                    portfolio_var_obj.var_parametric(percent=False)
            incremental_analytical = self.portfolio_var.loc['Notional Value VaR', 'Portfolio VaR Analytical'].item() - \
                                     analytical_var_obj.var_parametric(percent=False)
            return pd.DataFrame(
                columns=['Security Incremental VaR Price Series', 'Security Incremental VaR Analytical'],
                data=[[incremental_portfolio / self.portfolio_var.loc['Notional Value VaR',
                                                                      'Portfolio VaR Price Series'].item() * 100,
                       incremental_analytical / self.portfolio_var.loc['Notional Value VaR',
                                                                       'Portfolio VaR Analytical'].item() * 100],
                      [incremental_portfolio, incremental_analytical]], index=['Percent VaR', 'Notional Value VaR'])
