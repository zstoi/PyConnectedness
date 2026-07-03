r"""
Static connectedness measures (Diebold-Yilmaz 2009, 2012).

From a normalized variance decomposition :math:`D = [\tilde\theta_{ij}]` the
spillover measures are derived: the total connectedness index (TCI),
directional "from" (shocks received), directional "to" (shocks transmitted), and
net (to minus from).

References
----------
Diebold and Yilmaz (2009) Measuring financial asset return and volatility
spillovers, with application to global equity markets. The Economic Journal,
119, 158-171.

Diebold and Yilmaz (2012) Better to give than to receive: predictive
directional measurement of volatility spillovers. International Journal of
Forecasting, 28, 57-66.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .decomposition import generalized_fevd, normalize_fevd, orthogonalized_fevd
from .var import VARFit, fit_var


@dataclass
class ConnectednessResult:
    """
    Result of a static connectedness computation.

    Parameters
    ----------
    table : DataFrame
        Spillover table in Diebold-Yilmaz layout: the normalized decomposition
        (in percent) with "FROM" column and "TO"/"NET" rows.
    fevd : DataFrame (k x k)
        Normalized variance decomposition, in percent.
    total : float
        Total connectedness index, in percent.
    directional_to : Series
        "To others" spillover per variable.
    directional_from : Series
        "From others" spillover per variable.
    net : Series
        Net spillover per variable (to minus from).
    """

    table: pd.DataFrame
    fevd: pd.DataFrame
    total: float
    directional_to: pd.Series
    directional_from: pd.Series
    net: pd.Series

    def __repr__(self) -> str:
        return f"ConnectednessResult(TCI={self.total:.2f}%, n={self.fevd.shape[0]})"


def _measures(theta_norm: np.ndarray) -> dict:
    """Compute directional and total measures from a normalized decomposition."""
    k = theta_norm.shape[0]
    own = np.diag(theta_norm)
    to_others = (theta_norm.sum(axis=0) - own) * 100.0      #  sums, off-diagonal
    from_others = (theta_norm.sum(axis=1) - own) * 100.0    #  sums, off-diagonal
    net = to_others - from_others
    total = (theta_norm.sum() - np.trace(theta_norm)) / k * 100.0
    return {"to": to_others, "from": from_others, "net": net, "total": total}


def static_connectedness(
    data: pd.DataFrame | None = None,
    horizon: int = 10,
    *,
    method: str = "generalized",
    var_fit: VARFit | None = None,
    **fit_kwargs,
) -> ConnectednessResult:
    """
    Compute static Diebold-Yilmaz connectedness.

    Provide either ``data`` (a VAR is estimated internally) or a pre-fitted
    ``var_fit``.

    Parameters
    ----------
    data : DataFrame, optional
        Multivariate time series, one column per variable.
    horizon : int
        Forecast horizon H for the variance decomposition.
    method : {"generalized", "orthogonalized"}
        Decomposition scheme - "generalized" (Pesaran-Shin, order-invariant,
        reproduces DY-2012/2014) or "orthogonalized" (Cholesky, order-
        dependent, reproduces DY-2009). Under "orthogonalized" the column
        order of the data determines the result.
    var_fit : VARFit, optional
        Pre-fitted VAR model; skips the internal estimation.
    **fit_kwargs
        Passed through to :func:`pyconnectedness.connectedness.var.fit_var`
        (e.g. ``lags``, ``ic``, ``max_lags``).

    Returns
    -------
    ConnectednessResult

    Raises
    ------
    ValueError
        If neither ``data`` nor ``var_fit`` is given, or ``method`` is unknown.
    """
    if var_fit is None:
        if data is None:
            raise ValueError("provide either 'data' or 'var_fit'")
        var_fit = fit_var(data, **fit_kwargs)

    ma = var_fit.ma_coefficients(horizon)
    if method == "generalized":
        theta = generalized_fevd(ma, var_fit.sigma)
    elif method == "orthogonalized":
        theta = orthogonalized_fevd(ma, var_fit.sigma)
    else:
        raise ValueError(
            f"unknown method '{method}'; use 'generalized' or 'orthogonalized'"
        )
    theta_norm = normalize_fevd(theta)

    names = var_fit.names
    m = _measures(theta_norm)
    fevd_df = pd.DataFrame(theta_norm * 100.0, index=names, columns=names)

    # Spillover table in DY layout: FEVD + FROM column, then TO and NET rows.
    table = fevd_df.copy()
    table["FROM"] = m["from"]
    to_row = pd.Series(dict(zip(names, m["to"])), name="TO")
    to_row["FROM"] = np.nan
    net_row = pd.Series(dict(zip(names, m["net"])), name="NET")
    net_row["FROM"] = m["total"]   # corner entry = total connectedness index
    table = pd.concat([table, to_row.to_frame().T, net_row.to_frame().T])

    return ConnectednessResult(
        table=table,
        fevd=fevd_df,
        total=float(m["total"]),
        directional_to=pd.Series(m["to"], index=names, name="TO"),
        directional_from=pd.Series(m["from"], index=names, name="FROM"),
        net=pd.Series(m["net"], index=names, name="NET"),
    )