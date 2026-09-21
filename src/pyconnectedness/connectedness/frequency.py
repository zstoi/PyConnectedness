r"""
Frequency connectedness (Baruník-Křehlík 2018).

References
----------
Baruník and Křehlík (2018) Measuring the frequency dynamics of financial
connectedness and systemic risk. Journal of Financial Econometrics, 16, 271-296.

Diebold and Yilmaz (2012) Better to give than to receive: predictive directional 
measurement of volatility spillovers. International Journal of
Forecasting, 28, 57-66.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .static import _build_result
from .var import VARFit, fit_var


@dataclass
class FrequencyConnectednessResult:
    """
    Connectedness by frequency band.

    Parameters
    ----------
    bands : dict
        Band label -> ConnectednessResult, the frequency connectedness
    within : dict
        Band label -> ConnectednessResult, the band rescaled as if it were the
        whole system.
    share : Series
        Share of each band in the forecast error variance, in percent. Sums
        to 100.
    total : Series
        Total connectedness per band, in percent
    horizon : int
    """

    bands: dict
    within: dict
    share: pd.Series
    total: pd.Series
    horizon: int

    def __repr__(self) -> str:
        return (f"FrequencyConnectednessResult(TCI={self.total.sum():.2f}%, "
                f"bands={self.total.round(2).to_dict()})")


def frequency_connectedness(
    data: pd.DataFrame | None = None,
    horizon: int = 100,
    *,
    periods: tuple = (5, 20),
    method: str = "generalized",
    var_fit: VARFit | None = None,
    **fit_kwargs,
) -> FrequencyConnectednessResult:
    r"""
    Compute Baruník-Křehlík frequency connectedness.

    Provide either `data`` (a VAR is estimated internally) or a pre-fitted
    `var_fit`

    Parameters
    ----------
    data : DataFrame, optional
        Multivariate time series, one column per variable.
    horizon : int
        Forecast horizon H, also the number of points of the frequency grid.
    periods : tuple of int
        Band cutoffs as cycle lengths in observations, increasing and larger
        than 2 
    method : {"generalized", "orthogonalized"}
        Decomposition scheme, as in static_connectedness.
    var_fit : VARFit, optional
        Pre-fitted VAR model; skips the internal estimation.
    **fit_kwargs
        Passed through to :func:`pyconnectedness.connectedness.var.fit_var`
        (e.g. ``lags``, ``ic``, ``max_lags``).

    Notes
    -----
    The share of the forecast error variance of variable i due to shocks in
    j on the frequency band d is

    .. math::

        \tilde\theta_{ij}(d) = \frac{\sigma_{jj}^{-1} \int_d
            |(\Psi(e^{-i\omega}) \Sigma)_{ij}|^2 d\omega}
            {\sum_l \sigma_{ll}^{-1} \int_{-\pi}^{\pi}
            |(\Psi(e^{-i\omega}) \Sigma)_{il}|^2 d\omega}

    with :math:`\Psi(e^{-i\omega}) = \sum_h \Phi_h e^{-i\omega h}`.

    A cutoff of p observations is the frequency :math:`2\pi / p`. As the R
    package frequencyConnectedness takes the cutoffs in radians, so
    `periods=(5, 20)` corresponds to
    `c(pi + 0.00001, 2*pi/5, 2*pi/20, 0)`.

    Returns
    -------
    FrequencyConnectednessResult

    Raises
    ------
    ValueError
        If neither `data` nor `var_fit` is given, `method` is unknown,
        `periods` is not increasing or not larger than 2, or a band holds no
        frequency at the given horizon.
    """
    if var_fit is None:
        if data is None:
            raise ValueError("provide either 'data' or 'var_fit'")
        var_fit = fit_var(data, **fit_kwargs)

    if min(periods) <= 2:
        raise ValueError("periods must be larger than 2")
    if (np.diff(periods) <= 0).any():
        raise ValueError("periods must be increasing")
   
    names = var_fit.names
    sigma = var_fit.sigma
    k = var_fit.k

    # 1) MA matrices Phi_0, ..., Phi_{H-1}, H x k x k
    ma = var_fit.ma_coefficients(horizon)

    # 2) psi[s] = Psi(exp(-i omega_s)) = sum_h Phi_h exp(-i omega_s h)
    psi = np.fft.fft(ma, axis=0)

    # 3) omega_s = 2 pi s / H; rows above H/2 are the negative frequencies and
    # fold onto their positive twins
    omega = 2 * np.pi * np.abs(np.fft.fftfreq(horizon))

    # 4) contribution of shocks in j to variable i at each frequency
    if method == "generalized":
        numerator = np.abs(psi @ sigma) ** 2 / np.diag(sigma)
    elif method == "orthogonalized":
        numerator = np.abs(psi @ np.linalg.cholesky(sigma)) ** 2
    else:
        raise ValueError(
            f"unknown method '{method}'; use 'generalized' or 'orthogonalized'"
        )

    # 5) normalize row i over all frequencies and all shocks
    theta = numerator / numerator.sum(axis=(0, 2))[np.newaxis, :, np.newaxis]

    # 6) band edges as frequencies, high to low
    tol = 1e-10
    edges = [np.inf] + [2 * np.pi / p for p in periods] + [0.0]

    labels = [f"<={periods[0]}"]
    for i in range(1, len(periods)):
        labels.append(f"{periods[i - 1]}-{periods[i]}")
    labels.append(f">{periods[-1]}")

    # 7)sum theta over the frequencies of each band
    bands = {}
    within = {}
    share = {}
    total = {}
    for b, label in enumerate(labels):
        in_band = (omega >= edges[b + 1] - tol) & (omega < edges[b] - tol)
        if not in_band.any():
            raise ValueError(f"band '{label}' is empty at horizon {horizon}")
        theta_band = theta[in_band].sum(axis=0)
        bands[label] = _build_result(theta_band, names)
        within[label] = _build_result(theta_band * k / theta_band.sum(), names)
        share[label] = theta_band.sum() / k * 100.0
        total[label] = bands[label].total

    return FrequencyConnectednessResult(
        bands=bands,
        within=within,
        share=pd.Series(share, name="share"),
        total=pd.Series(total, name="total"),
        horizon=horizon,
    )