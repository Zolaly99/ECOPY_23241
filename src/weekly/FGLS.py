import numpy as np
import pandas as pd


class FGLS:
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
        self.resid = self.left_hand_side-np.dot(self.right_hand_side, self.params)
        self.resid_sq=self.resid**2
        self.xTx_residsq_regr = np.dot(np.transpose(self.right_hand_side), self.right_hand_side)
        self.xTy_residsq_regr= np.dot(np.transpose(self.right_hand_side), self.resid_sq)
        self.params_residsq_regr = np.dot(np.linalg.inv(self.xTx_residsq_regr), self.xTy_residsq_regr)
        self.resid_2nd_regr = self.left_hand_side-np.dot(self.right_hand_side, self.params_residsq_regr)
        self.Vhat = np.dot(self.resid_2nd_regr, np.transpose(self.resid_2nd_regr))
        self.Vhat_inv = np.linalg.inv(self.Vhat)
        self.C = np.linalg.cholesky(self.Vhat_inv)
        self.Y_transformed = np.dot(self.C, self.left_hand_side)
        self.X_transformed = np.dot(self.C, self.right_hand_side)
        self.xTx_FGLS = np.dot(np.transpose(self.X_transformed), self.X_transformed)
        self.xTy_FGLS = np.dot(np.transpose(self.X_transformed), self.Y_transformed)
        self.params_FGLS = np.dot(np.linalg.inv(self.xTx_FGLS), self.xTy_FGLS)



        self.FGLS_var = np.dot(np.transpose(self.errors), self.errors)/(self.n-self.p)
        self.FGLS_sd = np.sqrt(self.FGLS_var)
        self.FGLS_condvar = self.FGLS_var*np.linalg.inv(np.dot(np.dot(np.transpose(self.right_hand_side), self.Vhat_inv), self.right_hand_side))
        return pd.Series(..., name='FGLS coefficients')
