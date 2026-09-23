from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyconnectedness import dynamic_connectedness, static_connectedness

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"

returns = pd.read_excel(DATA / "dy2009_returns.xlsx").set_index("date")
volatility = pd.read_excel(DATA / "dy2009_vola.xlsx").set_index("date")
raw_vola = pd.read_excel(DATA / "dy2012_vola.xlsx").set_index("date")
log_vola = np.log(raw_vola)

conn = static_connectedness(returns, horizon=10, method="orthogonalized", lags=2)
dyn = dynamic_connectedness(returns, window=200, horizon=10, method="orthogonalized", lags=2)


def test_dy2009_returns():
    assert conn.total == pytest.approx(35.53, abs=0.01)


def test_dy2009_volatility():
    vol = static_connectedness(volatility, horizon=10, method="orthogonalized", lags=2)
    assert vol.total == pytest.approx(39.45, abs=0.01)


def test_dy2012_log_volatility():
    log = static_connectedness(log_vola, horizon=10, method="generalized", lags=4)
    assert log.total == pytest.approx(12.59, abs=0.01)


def test_dy2012_needs_logs():
    raw = static_connectedness(raw_vola, horizon=10, method="generalized", lags=4)
    assert raw.total < 20.0


def test_fevd_rows_sum_to_hundred():
    assert np.allclose(conn.fevd.sum(axis=1), 100.0)


def test_net_sums_to_zero():
    assert conn.net.sum() == pytest.approx(0.0, abs=1e-9)


def test_pairwise_is_antisymmetric():
    pw = conn.pairwise_net.to_numpy()
    assert np.allclose(pw, -pw.T)


def test_pairwise_rows_equal_net():
    assert np.allclose(conn.pairwise_net.sum(axis=1), conn.net)


def test_table_has_summary_rows():
    assert list(conn.table.index[-3:]) == ["TO", "TO_incl_own", "NET"]
    assert conn.table.loc["TO_incl_own", "FROM"] == pytest.approx(conn.total)


def test_cholesky_is_lower_bound():
    gen = static_connectedness(returns, horizon=10, method="generalized", lags=2)
    assert gen.total > conn.total


def test_window_count():
    assert dyn.n_windows == len(returns) - 200 + 1


def test_last_window_equals_static():
    last = static_connectedness(returns.iloc[-200:], horizon=10, method="orthogonalized", lags=2)
    assert dyn.total.iloc[-1] == pytest.approx(last.total)
    assert np.allclose(dyn.pairwise_net[-1], last.pairwise_net)


def test_dynamic_pairwise_rows_equal_net():
    assert np.allclose(dyn.pairwise_net.sum(axis=2), dyn.net)


def test_missing_values_rejected():
    broken = returns.copy()
    broken.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        static_connectedness(broken, horizon=10, lags=2)
