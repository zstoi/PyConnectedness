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

raw_vola = pd.read_excel(DATA / "dy2009_vola.xlsx").set_index("date")
log_vola = np.log(raw_vola)
fit = fit_var(log_vola, lags=2)


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



def test_periods_not_increasing():
    with pytest.raises(ValueError, match="increasing"):
        frequency_connectedness(var_fit=fit, horizon=100, periods=(20, 5))


def test_period_of_two_rejected():
    with pytest.raises(ValueError, match="larger than 2"):
        frequency_connectedness(var_fit=fit, horizon=100, periods=(2, 20))


def test_empty_band_rejected():
    with pytest.raises(ValueError, match="'101-102' is empty"):
        frequency_connectedness(var_fit=fit, horizon=100, periods=(101, 102))
