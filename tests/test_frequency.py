from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyconnectedness import (
    fit_var,
    frequency_connectedness,
    static_connectedness,
)

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"

raw_vola = pd.read_excel(DATA / "dy2012_vola.xlsx").set_index("date")
log_vola = np.log(raw_vola)
fit = fit_var(log_vola, lags=4)


@pytest.mark.parametrize("horizon", [100, 101])
@pytest.mark.parametrize("method", ["generalized", "orthogonalized"])
def test_bands_add_up_to_static(method, horizon):
    freq = frequency_connectedness(var_fit=fit, horizon=horizon, method=method)
    static = static_connectedness(var_fit=fit, horizon=horizon, method=method)

    fevd = sum(band.fevd for band in freq.bands.values())
    net = sum(band.net for band in freq.bands.values())
    assert np.allclose(fevd, static.fevd, atol=1e-10)
    assert np.allclose(net, static.net, atol=1e-10)
    assert freq.total.sum() == pytest.approx(static.total, abs=1e-10)
    assert freq.share.sum() == pytest.approx(100.0)
    for label in freq.bands:
        within = freq.within[label].total * freq.share[label] / 100.0
        assert within == pytest.approx(freq.total[label])


# R, frequencyConnectedness:
#   est <- vars::VAR(log_vola, p = 4, type = "const")
#   spilloverBK12(est, n.ahead = 99, no.corr = FALSE,
#                 partition = c(pi + 0.00001, 2*pi/5, 2*pi/20, 0))
def test_matches_r_package():
    freq = frequency_connectedness(var_fit=fit, horizon=100, periods=(5,20), method="generalized")
    within = [freq.within[label].total for label in freq.bands]
    net = freq.bands[">20"].net / 4
    row = freq.bands["<=5"].fevd.iloc[0] / 100

    assert np.allclose(freq.total, [2.3918, 1.2252, 12.4753], atol=1e-4)
    assert np.allclose(within, [7.5543, 8.1375, 23.4134], atol=1e-4)
    assert np.allclose(net, [1.751695, 0.102064, -0.671167, -1.182592],
                       atol=1e-6)
    assert np.allclose(row, [0.223809, 0.016779, 0.001437, 0.006480],
                       atol=1e-6)


def test_periods_not_increasing():
    with pytest.raises(ValueError, match="increasing"):
        frequency_connectedness(var_fit=fit, horizon=100, periods=(20, 5))


def test_period_of_two_rejected():
    with pytest.raises(ValueError, match="larger than 2"):
        frequency_connectedness(var_fit=fit, horizon=100, periods=(2, 20))


def test_empty_band_rejected():
    with pytest.raises(ValueError, match="'101-102' is empty"):
        frequency_connectedness(var_fit=fit, horizon=100, periods=(101, 102))