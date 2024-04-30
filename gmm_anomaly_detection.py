import numpy as np
import streamlit as st
import pandas as pd
from sklearn.mixture import GaussianMixture


class gmmAnomalyDetector(GaussianMixture):
  """
    Gaussian Mixture Model (GMM) based Anomaly Detector.

    This class inherits from the GaussianMixture class in scikit-learn and extends it
    to provide methods for anomaly detection using GMM.

    Parameters:
    ----------
    n_components : int, default=1
        The number of mixture components.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        The type of covariance parameters to use.

    tol : float, default=1e-3
        The convergence threshold.

    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance.

    max_iter : int, default=100
        The maximum number of EM iterations.

    n_init : int, default=2
        The number of initializations to perform.

    init_params : {'kmeans', 'random'}, default='kmeans'
        The method used to initialize the weights, means, and precisions.

    random_state : int, RandomState instance, default=42
        Controls the random number generator.

    Methods:
    ----------
    predict(X, outlier_fraction=0.001):
        Predict anomalies in the input data X.

        Parameters:
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        outlier_fraction : float, default=0.001
            The fraction of samples considered as anomalies.

        Returns:
        ----------
        y_pred : array-like of shape (n_samples,)
            An array indicating whether each sample is an anomaly (1) or not (0).

    predict_score(X):
        Compute the anomaly scores for the input data X.

        Parameters:
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        ----------
        score : array-like of shape (n_samples,)
            An array containing the anomaly scores for each sample.
    """
    def __init__(self,
                 n_components=1,
                 covariance_type='full',
                 tol=1e-3,
                 reg_covar=1e-6,
                 max_iter=100,
                 n_init=2,
                 init_params='kmeans++',
                 random_state=42
                 ):
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state
        )

    def predict(self, X, outlier_fraction=0.001):
        """
        Predict anomalies in the input data X.

        Parameters:
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        outlier_fraction : float, default=0.001
            The fraction of samples considered as anomalies.

        Returns:
        ----------
        y_pred : array-like of shape (n_samples,)
            An array indicating whether each sample is an anomaly (1) or not (0).
        """
        # Use the base class predict to get the scores
        score = self.score_samples(X)

        # Get the score threshold for anomaly
        pct_threshold = np.percentile(score, outlier_fraction)

        # Create a DataFrame with original data and scores
        data_gmm = pd.DataFrame()
        data_gmm['score'] = score

        # Create binary predictions based on the threshold using custom function
        y_pred = data_gmm.apply(lambda row: 1 if row['score'] < pct_threshold else 0, axis=1)

        return y_pred.values

    def predict_score(self, X):
        """
        Compute the anomaly scores for the input data X.

        Parameters:
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        ----------
        score : array-like of shape (n_samples,)
            An array containing the anomaly scores for each sample.
        """
        # Use the base class predict to get the scores
        score = self.score_samples(X)
        return score
