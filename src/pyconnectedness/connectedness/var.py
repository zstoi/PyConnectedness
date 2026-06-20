"""
VAR-Wrapper 

Vector autoregression (VAR) for connecctedness analysis

wrapper around statsmodels.tsa.api.VAR that exposes
the quantities consumed from the Diebold-Yilmaz cocnnectedness
the moving-average coefficient matrices of the VAR - the residula covariance matrix

References
----------
Lütkepohl (2005) New Introduction to Multiple Time Series Analysis.
 
Pesaran and Shin (1998) Generalized impulse response analysis in linear
multivariate models. Economics Letters, 58, 17-29.
 
Diebold and Yilmaz (2009) Measuring financial asset return and volatility
spillovers, with application to global equity markets. The Economic Journal,
119, 158-171.

"""


from __future__ import annotations
 
from dataclasses import dataclass
 
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
 
 
@dataclass
class VARFit:
    r"""
    Estimated VAR(p) specification.
 
    Holds the fitted statsmodels results together with the quantities the
    connectedness routines consume. The MA(\infty) representation
 
    .. math:: y_t = \sum_{i=0}^\infty \Phi_i u_{t-i}
 
    and the residual covariance :math:`\Sigma_u` are the two inputs to the
    forecast error variance decomposition.
 
    Parameters
    ----------
    names : list of str
        Variable names, taken from the columns of the input data.
    lag_order : int
        Selected lag order p.
    sigma : ndarray (k x k)
        Residual covariance matrix :math:`\Sigma_u`.
    results : VARResults
        The underlying fitted statsmodels object.
    """
 
    names: list[str]
    lag_order: int
    sigma: np.ndarray
    results: object
 
    @property
    def k(self) -> int:
        """int : Number of variables in the system."""
        return len(self.names)
 
    def ma_coefficients(self, horizon: int) -> np.ndarray:
        r"""
        Moving-average coefficients of the MA(\infty) representation.
 
        Parameters
        ----------
        horizon : int
            Forecast horizon H. The number of returned matrices equals
            ``horizon``.
 
        Notes
        -----
        Returns the matrices :math:`\Phi_0, \ldots, \Phi_{H-1}` with
        :math:`\Phi_0 = I_k`. This matches an H-step forecast error variance
        decomposition that sums over horizons :math:`h = 0, \ldots, H-1`.
 
        Returns
        -------
        ndarray (horizon x k x k)
        """
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        # ma_rep(maxn=n) returns n + 1 matrices Phi_0, ..., Phi_n.
        return np.asarray(self.results.ma_rep(maxn=horizon - 1))
 
 
def fit_var(
    data: pd.DataFrame,
    lags: int | None = None,
    ic: str = "aic",
    max_lags: int = 10,
    trend: str = "c",
) -> VARFit:
    """
    Estimate a VAR(p) model on a pandas DataFrame.
 
    Parameters
    ----------
    data : DataFrame
        Multivariate time series, one column per variable. The index is
        assumed to be ordered in time.
    lags : int, optional
        Fixed lag order p. If None, the order is selected by information
        criterion.
    ic : {"aic", "bic", "hqic", "fpe"}
        Information criterion used for automatic lag selection.
    max_lags : int
        Maximum lag order considered during automatic selection.
    trend : str
        Deterministic term passed through to statsmodels ("c", "n", "ct", ...).
 
    Returns
    -------
    VARFit
 
    Raises
    ------
    TypeError
        If ``data`` is not a DataFrame.
    ValueError
        If ``data`` contains missing values.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.isnull().any().any():
        raise ValueError("data contains missing values; clean it first")
 
    model = VAR(data)
    if lags is None:
        results = model.fit(maxlags=max_lags, ic=ic, trend=trend)
    else:
        results = model.fit(lags, trend=trend)
 
    return VARFit(
        names=list(data.columns),
        lag_order=int(results.k_ar),
        sigma=np.asarray(results.sigma_u),
        results=results,
    )