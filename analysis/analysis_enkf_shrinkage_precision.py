# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import Ridge

from analysis.analysis import Analysis


class AnalysisEnKFShrinkagePrecision(Analysis):
    """Analysis EnKF Modified Cholesky decomposition
    
    Attributes:
        model (Model object): An object that has all the methods and attributes of the model
        r (int): Value used in the process of removing correlations

    Methods:
        get_precision_matrix(DX, regularization_factor=0.01): Returns the computed precision matrix
        perform_assimilation(background, observation): Perform assimilation step given background and observations
        get_analysis_state(): Returns the computed column mean of ensemble Xa
        get_ensemble(): Returns ensemble Xa
        get_error_covariance(): Returns the computed covariance matrix of the ensemble Xa
        inflate_ensemble(inflation_factor): Computes new ensemble Xa given the inflation factor
    """

    def __init__(self, model, r=1, **kwargs):
        """
        Initialize an instance of AnalysisEnKFModifiedCholesky.

        Parameters:
            model (Model object): An object that has all the methods and attributes of the model given
            r (int, optional): Value used in the process of removing correlations
        """
        self.model = model
        self.r = r

    def get_pseudo_inverse_background(self, DX, rtol_pesudo_inverse=0.01):
        n, _ = DX.shape
        U, s, _ = np.linalg.svd(DX, full_matrices=False)
        k = len(s)

        A_pinv_truncated = np.zeros((n, n))
        for i in range(k):
            if s[i]/s[0]>rtol_pesudo_inverse:
                A_pinv_truncated += (1 / (s[i] ** 2)) * np.outer(U[:, i], U[:, i])
            else:
                break

        return A_pinv_truncated

    def get_target_precision_matrix(self, DX, regularization_factor=0.01):
        """
        Perform calculations to get the precision matrix given the deviation matrix.

        Parameters:
            DX (ndarray): Deviation matrix
            regularization_factor (float, optional): Value used as alpha in the ridge model

        Returns:
            precision_matrix (ndarray): Precision matrix
        """
        n, _ = DX.shape
        lr = Ridge(fit_intercept=False, alpha=regularization_factor)
        L = np.eye(n)
        D = np.zeros((n, n))
        D[0, 0] = 1 / np.var(DX[0, :])  # We are estimating D^{-1}
        for i in range(1, n):
            ind_prede = self.model.get_pre(i, self.r)
            y = DX[i, :]
            X = DX[ind_prede, :].T
            lr_fit = lr.fit(X, y)
            err_i = y - lr_fit.predict(X)
            D[i, i] = 1 / np.var(err_i)
            L[i, ind_prede] = -lr_fit.coef_

        return L.T @ (D @ L)
    
    def tr(self, A):
        return np.matrix.trace(A)
    
    def fr(self, A):
        return np.linalg.norm(A, 'fro')
    
    def get_shrinkage_precision_matrix(self, DX, 
                                       regularization_factor=0.01, 
                                       rtol_pesudo_inverse=0.01):
        

        n, N = DX.shape

        #print(n, N)

        Binv = self.get_target_precision_matrix(DX, regularization_factor=regularization_factor)

        Pseu = self.get_pseudo_inverse_background(DX, 
                                                  rtol_pesudo_inverse=rtol_pesudo_inverse)
        

        #plt.figure()
        #plt.subplot(1,2,1);sns.heatmap(Binv)
        #plt.subplot(1,2,2);sns.heatmap(Pseu)
        #plt.show()

        Si_tr = self.tr(Pseu)
        Si_fr = self.fr(Pseu)
        Bi_fr = self.fr(Binv)
        Si_Bi_tr = self.tr(Pseu @ Binv)

        gamma = Si_Bi_tr/n

        alpha_ = (gamma/(1+gamma)) * min(1, N/n)

        #print(alpha_, 1-alpha_)

        Binv_shrunk =  alpha_ * Binv + (1-alpha_) * Pseu

        return Binv_shrunk

    def perform_assimilation(self, background, observation):
        """Perform assimilation step of ensemble Xa given the background and the observations

        Parameters:
            background (Background Object): The background object defined in the class background
            observation (Observation Object): The observation object defined in the class observation
        
        Returns:
            Xa (Matrix): Matrix of ensemble
        """
        Xb = background.get_ensemble()
        y = observation.get_observation()
        H = observation.get_observation_operator()
        R = observation.get_data_error_covariance()
        n, ensemble_size = Xb.shape
        Ys = np.random.multivariate_normal(y, R, size=ensemble_size).T
        xb = np.mean(Xb, axis=1)
        DX = Xb - np.outer(xb, np.ones(ensemble_size))
        Binv_shrunk = self.get_shrinkage_precision_matrix(DX, regularization_factor=0.01, rtol_pesudo_inverse=0.01)
        D = Ys - H @ Xb
        Rinv = np.diag(np.reciprocal(np.diag(R)))
        IN = Binv_shrunk + H.T @ (Rinv @ H)
        Z = np.linalg.solve(IN, H.T @ (Rinv @ D))
        self.Xa = Xb + Z
        return self.Xa

    def get_analysis_state(self):
        """Compute column-wise mean vector of Matrix of ensemble Xa

        Parameters:
            None

        Returns:
            mean_vector (ndarray): Mean vector
        """
        return np.mean(self.Xa, axis=1)

    def get_ensemble(self):
        """Returns ensemble Xa

        Parameters:
            None

        Returns:
            ensemble_matrix (ndarray): Ensemble matrix
        """
        return self.Xa

    def get_error_covariance(self):
        """Returns the computed covariance matrix of the ensemble Xa

        Parameters:
            None

        Returns:
            covariance_matrix (ndarray): Covariance matrix of the ensemble Xa
        """
        return np.cov(self.Xa)

    def inflate_ensemble(self, inflation_factor):
        """Computes ensemble Xa given the inflation factor

        Parameters:
            inflation_factor (int): Double number indicating the inflation factor

        Returns:
            None
        """
        n, ensemble_size = self.Xa.shape
        xa = self.get_analysis_state()
        DXa = self.Xa - np.outer(xa, np.ones(ensemble_size))
        self.Xa = np.outer(xa, np.ones(ensemble_size)) + inflation_factor * DXa
