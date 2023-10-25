class LinearRegressionSM:

    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side

    def fit(self):
        left = self.left_hand_side
        right = self.right_hand_side
        comp = pd.concat([left_hand_side, right_hand_side], axis=1)
        self._model = sm.OLS(endog=left, exog=right, data=comp).fit()

    def get_params(self):
        params = self._model.params
        params.name = "Beta coefficients"
        return pd.Series(params)

    def get_pvalues(self):
        pvalues = self._model.pvalues
        pvalues.name = "P-values for the corresponding coefficients"
        return pvalues
