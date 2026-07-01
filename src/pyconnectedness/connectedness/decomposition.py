r"""
Forecast error variance decompositions for connectedness analysis.

Two schemes are provided:

* :func:`generalized_fevd` — the generalized decomposition of Pesaran and
  Shin (1998), invariant to variable ordering (Diebold-Yilmaz 2012/2014).
* :func:`orthogonalized_fevd` — the Cholesky decomposition of Diebold and
  Yilmaz (2009), which depends on variable ordering.

References
----------
Pesaran and Shin (1998) Generalized impulse response analysis in linear
multivariate models. Economics Letters, 58, 17-29.

Diebold and Yilmaz (2009) Measuring financial asset return and volatility
spillovers, with application to global equity markets. The Economic Journal,
119, 158-171.
"""

from __future__ import annotations

import numpy as np


def generalized_fevd(ma_coefficients: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    r"""
    Generalized forecast error variance decomposition (Pesaran-Shin 1998).

    Parameters
    ----------
    ma_coefficients : ndarray (H x k x k)
        Moving-average matrices :math:`\Phi_0, \ldots, \Phi_{H-1}`.
    sigma : ndarray (k x k)
        Residual covariance matrix :math:`\Sigma_u`.

    Notes
    -----
    The share of the H-step forecast error variance of variable i due to
    shocks in variable j is

    .. math::

        \theta_{ij}(H) = \frac{\sigma_{jj}^{-1}
            \sum_{h=0}^{H-1} (e_i' \Phi_h \Sigma e_j)^2}
            {\sum_{h=0}^{H-1} e_i' \Phi_h \Sigma \Phi_h' e_i}

    where :math:`e_i` is a selection vector and :math:`\sigma_{jj}` the j-th
    diagonal element of :math:`\Sigma`. Rows do not sum to one under this
    scheme; apply :func:`normalize_fevd`.

    Returns
    -------
    ndarray (k x k)
        Matrix theta with theta[i, j] the contribution of j to i.
    """
    ma = np.asarray(ma_coefficients)
    sigma = np.asarray(sigma)
    H, k, _ = ma.shape

    sigma_jj = np.diag(sigma)
    numerator = np.zeros((k, k))
    denominator = np.zeros(k)
    for h in range(H):
        phi = ma[h]
        phi_sigma = phi @ sigma          # entry [i, j] = e_i' Phi_h Sigma e_j
        numerator += phi_sigma ** 2
        denominator += np.diag(phi_sigma @ phi.T)

    return (numerator / sigma_jj[np.newaxis, :]) / denominator[:, np.newaxis]


def orthogonalized_fevd(ma_coefficients: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    r"""
    Orthogonalized (Cholesky) forecast error variance decomposition (DY 2009).

    Parameters
    ----------
    ma_coefficients : ndarray (H x k x k)
        Moving-average matrices :math:`\Phi_0, \ldots, \Phi_{H-1}`.
    sigma : ndarray (k x k)
        Residual covariance matrix :math:`\Sigma_u`.

    Notes
    -----
    Uses the Cholesky factor :math:`P` with :math:`\Sigma = P P'`:

    .. math::

        \theta_{ij}(H) = \frac{\sum_{h=0}^{H-1} (e_i' \Phi_h P e_j)^2}
            {\sum_{h=0}^{H-1} e_i' \Phi_h \Sigma \Phi_h' e_i}

    Rows sum to one by construction. The result depends on the column order
    of the data — the known ordering sensitivity of the DY-2009 method.

    Returns
    -------
    ndarray (k x k)
        Matrix theta with theta[i, j] the contribution of j to i.
    """
    ma = np.asarray(ma_coefficients)
    sigma = np.asarray(sigma)
    H, k, _ = ma.shape

    P = np.linalg.cholesky(sigma)        # Sigma = P P', lower triangular
    numerator = np.zeros((k, k))
    denominator = np.zeros(k)
    for h in range(H):
        phi = ma[h]
        phi_P = phi @ P
        numerator += phi_P ** 2
        denominator += np.diag(phi @ sigma @ phi.T)

    return numerator / denominator[:, np.newaxis]


def normalize_fevd(theta: np.ndarray) -> np.ndarray:
    """
    Normalize a decomposition row-wise so each row sums to one.

    Parameters
    ----------
    theta : ndarray (k x k)

    Notes
    -----
    Required after :func:`generalized_fevd`. A no-op after
    :func:`orthogonalized_fevd`, whose rows already sum to one.

    Returns
    -------
    ndarray (k x k)
    """
    theta = np.asarray(theta)
    return theta / theta.sum(axis=1, keepdims=True)