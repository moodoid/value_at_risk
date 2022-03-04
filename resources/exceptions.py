class ValueAtRiskError(Exception):
    pass


class HistoricalVARMethodError(ValueAtRiskError):

    def __str__(self):
        return "No data kwargs given--only parametric VaR calculations method available"


class VARMethodsError(ValueAtRiskError):

    def __str__(self):
        return "No data kwarg or mu and sigma kwargs specified--neither method of VaR calculations methods available"
