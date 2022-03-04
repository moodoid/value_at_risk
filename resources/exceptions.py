class ValueAtRiskErrors(Exception):
    pass


class HistoricalVARMethodError(ValueAtRiskErrors):

    def __str__(self):
        return "No data kwargs given--only parametric VaR calculations method available"


class VARMethodsError(ValueAtRiskErrors):

    def __str__(self):
        return "No data or mu and sigma kwargs given--neither method of VaR methods available"
