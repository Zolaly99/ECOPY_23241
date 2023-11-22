import pandas as pd
import statsmodels.api as sm
import numpy as np
import scipy
from scipy.stats import norm
from scipy.optimize import minimize



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


class LinearRegressionNP:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.right_hand_side = sm.add_constant(self.right_hand_side)
        self._model = None


    def fit(self):
        _model = sm.OLS(self.left_hand_side, self.right_hand_side).fit()

    def get_params(self):
        self.xTx = np.dot(np.transpose(self.right_hand_side), self.right_hand_side)
        self.xTy = np.dot(np.transpose(self.right_hand_side), self.left_hand_side)
        self.params = np.dot(np.linalg.inv(self.xTx), self.xTy)
        return pd.Series(self.params, name='Beta coefficients')

    def get_pvalues(self):
        self.xTx_inv = np.linalg.inv(self.xTx)
        self.errors = self.left_hand_side-np.dot(self.right_hand_side, self.params)
        self.n=len(self.left_hand_side)
        self.p=len(self.right_hand_side.columns)
        self.var = np.dot(np.transpose(self.errors), self.errors)/(self.n-self.p)
        self.se_sq = self.var*np.diag(self.xTx_inv)
        self.se = np.sqrt(self.se_sq)
        self.t_stats = np.divide(self.params, self.se)
        term = np.minimum(scipy.stats.t.cdf(self.t_stats, self.n-self.p),1-scipy.stats.t.cdf(self.t_stats, self.n-self.p))
        p_values = (term)*2
        return pd.Series(p_values, name='P-values for the corresponding coefficients')

    def get_wald_test_result(self, restr_matrix):

        r = np.transpose(np.zeros((len(restr_matrix))))
        term_1=np.dot(restr_matrix,self.params)-r
        term_2 = np.dot(np.dot(restr_matrix, self.xTx_inv), np.transpose(restr_matrix))
        f_stat = (np.dot(np.transpose(term_1), np.dot(np.linalg.inv(term_2), term_1))/len(restr_matrix))/self.var
        p_value = (1-scipy.stats.f.cdf(f_stat, len(restr_matrix), self.n-self.p))
        f_stat.astype(float)
        p_value.astype(float)
        return f'Wald: {round(f_stat,3)}, p-value: {round(p_value,3)}'

    def get_model_goodness_values(self):
        y_demean = self.left_hand_side-sum(self.left_hand_side)/len(self.left_hand_side)
        yTy = np.dot(np.transpose(y_demean), y_demean)
        SSE = np.dot(np.transpose(self.errors), self.errors)
        SST = yTy
        r2=round(1-SSE/SST,3)
        adj_r2 = round(1-(self.n-1)/(self.n-self.p)*(1-r2),3)
        return f'Centered R-squared: {r2:.3f}, Adjusted R-squared: {adj_r2:.3f}'








class LinearRegressionGLS:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.right_hand_side = sm.add_constant(self.right_hand_side)
        self._model = None


    def fit(self):
        _model = sm.GLS(self.left_hand_side, self.right_hand_side).fit()

    def get_params(self):
        self.xTx = np.dot(np.transpose(self.right_hand_side), self.right_hand_side)
        self.xTy = np.dot(np.transpose(self.right_hand_side), self.left_hand_side)
        self.params = np.dot(np.linalg.inv(self.xTx), self.xTy)
        self.resid = self.left_hand_side-np.dot(self.right_hand_side, self.params)



        self.resid_sq=self.resid**2
        self.log_resid_sq = np.log(self.resid_sq)
        self.xTx_residsq_regr = np.dot(np.transpose(self.right_hand_side), self.right_hand_side)
        self.xTy_residsq_regr= np.dot(np.transpose(self.right_hand_side), self.log_resid_sq)        #lognegyzetes reziduum ~ eredeti X
        self.params_residsq_regr = np.dot(np.linalg.inv(self.xTx_residsq_regr), self.xTy_residsq_regr)


        self.resid_3rd = np.dot(self.right_hand_side, self.params_residsq_regr)
        self.resid_3rd_unlogged = np.sqrt(np.exp(self.resid_3rd))
        self.resid_3rd_unlogged_inv = 1/self.resid_3rd_unlogged
        self.Vinv = (np.diag(self.resid_3rd_unlogged_inv))  #ez jó sztem
        self.FGLS_xTx = np.dot(np.dot(np.transpose(self.right_hand_side), self.Vinv), self.right_hand_side)
        self.FGLS_xTy = np.dot(np.dot(np.transpose(self.right_hand_side), self.Vinv), self.left_hand_side)

        self.FGLS_params = np.dot(np.linalg.inv(self.FGLS_xTx), self.FGLS_xTy)

        return pd.Series(self.FGLS_params, name='Beta coefficients')

    def get_pvalues(self):
        self.get_params()
        self.xTx_inv = np.linalg.inv(self.FGLS_xTx)
        self.errors = self.left_hand_side - np.dot(self.right_hand_side, self.FGLS_params)
        self.n = len(self.left_hand_side)
        self.p = len(self.right_hand_side.columns)
        self.var = np.dot(np.transpose(self.errors), self.errors) / (self.n - self.p)
        self.se_sq = self.var * np.diag(self.xTx_inv)
        self.se = np.sqrt(self.se_sq)
        self.t_stats = np.divide(self.FGLS_params, self.se)
        term = np.minimum(scipy.stats.t.cdf(self.t_stats, self.n - self.p),
                      1 - scipy.stats.t.cdf(self.t_stats, self.n - self.p))
        p_values = (term) * 2
        return pd.Series(p_values, name='P-values for the corresponding coefficients')

    def get_wald_test_result(self, restr_matrix):
        self.get_params()
        self.get_pvalues()
        r = np.transpose(np.zeros((len(restr_matrix))))
        term_1=np.dot(restr_matrix,self.FGLS_params)-r
        term_2 = np.dot(np.dot(restr_matrix, self.xTx_inv), np.transpose(restr_matrix))
        f_stat = (np.dot(np.transpose(term_1), np.dot(np.linalg.inv(term_2), term_1))/len(restr_matrix))/self.var
        p_value = (1-scipy.stats.f.cdf(f_stat, len(restr_matrix), self.n-self.p))
        f_stat.astype(float)
        p_value.astype(float)
        return f'Wald: {f_stat:.3f}, p-value: {p_value:.3f}'

    def get_model_goodness_values(self):
        self.get_params()
        self.get_pvalues()
        self.errors = self.left_hand_side-np.dot(self.right_hand_side, self.FGLS_params)

        y_demean = self.left_hand_side
        yTy = np.dot(np.dot(np.transpose(y_demean), self.Vinv), y_demean)
        SSE_1 = np.dot(np.dot(np.transpose(self.left_hand_side), self.Vinv), self.right_hand_side)
        SSE_2 = np.dot(np.dot(np.transpose(self.right_hand_side), self.Vinv), self.right_hand_side)
        SSE_3 = np.dot(np.dot(np.transpose(self.right_hand_side), self.Vinv), self.left_hand_side)
        SSE = np.dot(np.dot(SSE_1, np.linalg.inv(SSE_2)), np.transpose(SSE_1))
        SST = yTy
        r2 = 1 - SSE / SST
        adj_r2 = 1 - (self.n - 1) / (self.n - self.p) * (1 - r2)
        return f'Centered R-squared: {r2:.3f}, Adjusted R-squared: {adj_r2:.3f}'




class LinearRegressionML:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.right_hand_side = sm.add_constant(self.right_hand_side)
        self._model = None

    def fit(self):
        def MLE_Norm(parameters):
            # extract parameters
            beta0, beta1, beta2, beta3, std_dev = parameters
            # predict the output
            pred = beta0 * self.right_hand_side.iloc[:, 0] + beta1 * self.right_hand_side.iloc[:,
                                                                     1] + beta2 * self.right_hand_side.iloc[:,
                                                                                  2] + beta3 * self.right_hand_side.iloc[
                                                                                               :, 3]
            # Calculate the log-likelihood for normal distribution
            LL = np.sum(scipy.stats.norm.logpdf(self.left_hand_side, pred, std_dev))
            # Calculate the negative log-likelihood
            neg_LL = -1 * LL
            return neg_LL

        self.model_ = minimize(MLE_Norm, np.array([0.1, 0.1, 0.1, 0.1, 0.1]), method='Nelder-Mead', tol=1e-6)

    def get_params(self):
        self.params = self.model_.x[0:4]
        return pd.Series(self.model_.x[0:4], name='Beta coefficients')

    def get_pvalues(self):
        self.get_params()
        self.xTx = np.dot(np.transpose(self.right_hand_side), self.right_hand_side)

        self.xTx_inv = np.linalg.inv(self.xTx)
        self.n = len(self.left_hand_side)
        self.p = len(self.right_hand_side.columns)
        self.var = (self.model_.x[4])**2*self.n/(self.n-self.p)
        self.se_sq = self.var * np.diag(self.xTx_inv)
        self.se = np.sqrt(self.se_sq)
        self.t_stats = np.divide(self.params, self.se)
        term = np.minimum(scipy.stats.t.cdf(self.t_stats, self.n - self.p),
                          1 - scipy.stats.t.cdf(self.t_stats, self.n - self.p))
        p_values = (term) * 2
        return pd.Series(p_values, name='P-values for the corresponding coefficients')

    def get_model_goodness_values(self):
        self.get_params()
        self.get_pvalues()
        self.errors = self.left_hand_side - np.dot(self.right_hand_side, self.params)
        y_demean = self.left_hand_side - sum(self.left_hand_side) / len(self.left_hand_side)
        yTy = np.dot(np.transpose(y_demean), y_demean)
        SSE = np.dot(np.transpose(self.errors), self.errors)
        SST = yTy
        r2 = round(1 - SSE / SST, 3)
        adj_r2 = round(1 - (self.n - 1) / (self.n - self.p) * (1 - r2), 3)
        return f'Centered R-squared: {r2:.3f}, Adjusted R-squared: {adj_r2:.3f}'















