r"""
Marginal transform for multivariate extremes

The scaling method is applied to data whose margins are regularly varying
with index :math:`\alpha` and unit scale,
:math:`n P(n^{-1/\alpha} X > z) \to z^{-\alpha}`.

References
----------
Klüppelberg, C. and M. Krali (2021) Estimating an extreme Bayesian network via scalings. 
Journal of Multivariate Analysis 181, 104672.
Klüppelberg, C. and M. Krali (2025) Causal analysis of extreme risk in a
network of industry portfolios. arXiv:2504.00523.
"""

import numpy as np
import pandas as pd

__all__ = ["to_frechet"]


def _ranks(data):
    # ties get the largest rank, as R's ecdf in the reference code
    return pd.DataFrame(np.asarray(data, dtype=float)).rank(
        method="max").to_numpy()


def _like(data, values):
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(values, index=data.index, columns=data.columns)
    return values


def to_frechet(data, alpha=2.0):
    r"""
    Transform the margins to standard Fréchet

    Parameters
    ----------
    data : array_like (nobs x d)
        Larger values are more extreme.
    alpha : float
        Tail index of the target margins.

    Notes
    -----
    Empirical integral transform, eq. (25) of Klüppelberg and Krali (2025)
    and eq. (38) of the 2021 paper,

    .. math:: \tilde X_{\ell i} = \Big\{-\log\Big(\frac{1}{n+1}
              \sum_{j=1}^n 1\{X_{ji} \leq X_{\ell i}\}\Big)\Big\}^{-1/\alpha}

    so that :math:`P(\tilde X \leq x) \approx \exp(-x^{-\alpha})`.

    Returns
    -------
    ndarray or DataFrame, same shape as data
    """
    nobs = np.shape(data)[0]
    u = _ranks(data) / (nobs + 1)
    return _like(data, (-np.log(u)) ** (-1.0 / alpha))
