import unittest
import pandas as pd
import numpy as np
import datetime as dt

from value_at_risk.resources.var import VAR
from value_at_risk.resources.exceptions import HistoricalVARMethodError, VARMethodsError


class ValueAtRiskTests(unittest.TestCase):
    mu, sigma = 0, .005
    idx_len = 500
    accept_df = pd.Series(data=np.random.normal(loc=mu, scale=sigma, size=idx_len),
                          index=[pd.date_range(dt.datetime.today(), periods=idx_len).tolist()]).apply(
        lambda x: x + 1).cumprod()

    acceptable_numerical_types = (int, np.int32, np.int64, np.float32, np.float64, float)

    accept_test_cases = [dict(data=accept_df, mu=mu, sigma=sigma)]

    partial_test_cases = [dict(mu=mu, sigma=sigma)]

    reject_test_case = [dict()]

    def test_result(self):
        for test_case in self.accept_test_cases:
            var_obj = VAR(**test_case)
            self.assertIsInstance(type(var_obj.parametric_var), np.float64)
            self.assertIsInstance(type(var_obj.historical_var),
                                  (int, np.int32, np.int64, np.float32, np.float64, float))

        for test_case in self.partial_test_cases:
            var_obj = VAR(**test_case)
            self.assertIsInstance(type(var_obj.parametric_var), self.acceptable_numerical_types)
            self.assertRaises(HistoricalVARMethodError, var_obj.historical_var)

        for test_case in self.reject_test_case:
            var_obj = VAR(**test_case)
            self.assertRaises(VARMethodsError, var_obj.parametric_var)
            self.assertRaises(HistoricalVARMethodError, var_obj.historical_var)
