import unittest
import pandas as pd
import numpy as np
import datetime as dt

from value_at_risk.var import VAR
from value_at_risk.exceptions import HistoricalVARMethodError, VARMethodsError


class ValueAtRiskTestCases:
    def __init__(self):
        self.mu, self.sigma = 0, .005
        idx_len = 500
        self.accept_df = pd.Series(data=np.random.normal(loc=self.mu, scale=self.sigma, size=idx_len),
                                   index=[pd.date_range(dt.datetime.today(), periods=idx_len).tolist()]).apply(
            lambda x: x + 1).cumprod()

    @property
    def acceptable_numerical_types(self) -> tuple:
        return int, type(np.int32), type(np.int64), type(np.float32), type(np.float64), float

    @property
    def accept_cases(self) -> list:
        return [dict(data=self.accept_df, mu=self.mu, sigma=self.sigma)]

    @property
    def partial_cases(self) -> list:
        return [dict(mu=self.mu, sigma=self.sigma)]

    @property
    def reject_cases(self) -> list:
        return [dict()]


class ValueAtRiskTest(unittest.TestCase):

    def setUp(self) -> None:
        test_cases = ValueAtRiskTestCases()

        self.accept_test_cases = test_cases.accept_cases
        self.partial_test_cases = test_cases.partial_cases
        self.reject_test_case = test_cases.reject_cases
        self.acceptable_numerical_types = test_cases.acceptable_numerical_types

    def test_results(self):
        for test_case in self.accept_test_cases:
            var_obj = VAR(**test_case)
            self.assertIsInstance(type(var_obj.parametric_var), self.acceptable_numerical_types)

            self.assertIsInstance(type(var_obj.historical_var), self.acceptable_numerical_types)

        for test_case in self.partial_test_cases:
            var_obj = VAR(**test_case)
            self.assertIsInstance(type(var_obj.parametric_var), self.acceptable_numerical_types)
            self.assertIsInstance(type(var_obj.historical_var), type(HistoricalVARMethodError))

        for test_case in self.reject_test_case:
            var_obj = VAR(**test_case)
            self.assertIsInstance(type(var_obj.parametric_var), type(VARMethodsError))
            self.assertIsInstance(type(var_obj.historical_var), type(VARMethodsError))
