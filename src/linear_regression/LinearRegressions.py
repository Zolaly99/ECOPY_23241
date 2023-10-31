import pandas as pd
import statsmodels.api as sm


class LinearRegressionSM:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self._model = None

    def fit(self):
        right_hand_side = sm.add_constant(self.right_hand_side)
        _model = sm.OLS(self.left_hand_side, right_hand_side).fit()
        self._model = _model

    def get_params(self):
        params = self._model.params
        return pd.Series(params, name='Beta coefficients')

    def get_pvalues(self):
        p_values = self._model.pvalues
        return pd.Series(p_values, name='P-values for the corresponding coefficients')

    def get_wald_test_result(self, restr_matrix):
        wald = self._model.wald_test(restr_matrix)
        f_stat = float(wald.statistic)
        p_val = float(wald.pvalue)
        return f'F-value: {f_stat:.3}, p-value: {p_val:.3}'

    def get_model_goodness_values(self):
        adj_r2 = self._model.rsquared_adj
        aic = self._model.aic
        bic = self._model.bic
        return f'Adjusted R-squared: {adj_r2:.3}, Akaike IC: {aic:.3}, Bayes IC: {bic:.3}'