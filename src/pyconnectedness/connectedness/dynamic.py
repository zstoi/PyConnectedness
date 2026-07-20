"""
Rolling-window (dynamic) connectedness measures

References
----------
Diebold and Yilmaz (2009) Measuring financial asset return and volatility
spillovers, with application to global equity markets. The Economic Journal,
119, 158-171.
Diebold and Yilmaz (2012) Better to give than to receive: predictive
directional measurement of volatility spillovers. International Journal of
Forecasting, 28, 57-66.
"""

from dataclasses import dataclass
 
import numpy as np
import pandas as pd
 
from .static import static_connectedness
from .var import fit_var
 
 
@dataclass
class DynamicConnectednessResult:
    """
    Rolling-window connectedness measures.
 
    Parameters
    ----------
    total : Series (T,)
        Total connectedness (spillover) index per window, in percent.
    directional_to : DataFrame (T x k)
        Contribution of each variable to the forecast error variance of all
        others, off-diagonal column sums.
    directional_from : DataFrame (T x k)
        Contribution received by each variable from all others, off-diagonal
        row sums.
    net : DataFrame (T x k)
        TO minus FROM. Sums to zero across variables in every window.
    fevd : ndarray (T x k x k)
        The per-window variance decompositions, stacked in window order. Kept
        so that pairwise measures and heatmaps can be derived without refitting.
    names : list of str
    window, horizon, lag_order : int
    method : str
 

    """
 
    total: pd.Series
    directional_to: pd.DataFrame
    directional_from: pd.DataFrame
    net: pd.DataFrame
    fevd: np.ndarray
    names: list
    window: int
    horizon: int
    lag_order: int
    method: str = "generalized"
 
    @property
    def k(self):
        return len(self.names)
 
    @property
    def n_windows(self):
        return len(self.total)
 
 
def dynamic_connectedness(
    data, window, horizon=10, *, method="generalized", lags=None, **fit_kwargs
):
    r"""
    Connectedness measures over a rolling estimation window
 
    Parameters
    ----------
    data : DataFrame (n x k)
        Observations in rows, variables in columns. For DY-2012 the columns
        must already be log volatilities.
    window : int
        Number of observations per estimation window. 200 weeks in DY-2009,
        200 days in DY-2012, 150 days in DY-2014.
    horizon : int
        Forecast horizon H of the variance decomposition. The papers use
        H = 10 (2009, 2012) and H = 12 (2014); the default follows the two
        earlier papers, where the measures are already flat in H.
    method : {"generalized", "orthogonalized"}
        Decomposition applied in every window, passed on to
        static_connectedness. DY-2009 is Cholesky, DY-2012 onward generalized.
    lags : int or None
        VAR order, held fixed across windows. If None it is selected once on
        the full sample and then frozen; see Notes.
    **fit_kwargs
        Further arguments for fit_var, e.g. ``ic="aic"``, ``trend="c"``.
 
    
 
    Returns
    -------
    DynamicConnectednessResult
 
    Raises
    ------
    TypeError
        If data is not a DataFrame.
    ValueError
        If the window is longer than the sample.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a DataFrame")
 
    nobs, k = data.shape
    if window > nobs:
        raise ValueError(f"window ({window}) exceeds the sample length ({nobs})")
 
    if lags is None:
        lags = fit_var(data, **fit_kwargs).lag_order
 
    index = data.index[window - 1:]  # right-aligned, as in the papers
    n_windows = len(index)
 
    theta = np.empty((n_windows, k, k))
    to = np.empty((n_windows, k))
    frm = np.empty((n_windows, k))
    total = np.empty(n_windows)
 
    for t in range(n_windows):
        res = static_connectedness(
            data.iloc[t:t + window],
            horizon,
            method=method,
            lags=lags,
            **fit_kwargs,
        )
        theta[t] = res.fevd
        to[t] = np.asarray(res.directional_to)
        frm[t] = np.asarray(res.directional_from)
        total[t] = res.total
 
    names = list(data.columns)
 
    return DynamicConnectednessResult(
        total=pd.Series(total, index=index, name="total"),
        directional_to=pd.DataFrame(to, index=index, columns=names),
        directional_from=pd.DataFrame(frm, index=index, columns=names),
        net=pd.DataFrame(to - frm, index=index, columns=names),  # = res.net
        fevd=theta,
        names=names,
        window=window,
        horizon=horizon,
        lag_order=lags,
        method=method,
    )
 