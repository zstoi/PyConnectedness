
import warnings

import numpy as np
import pandas as pd

from pyconnectedness import fit_var


def _sample_data(index):
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "x": rng.normal(size=len(index)),
            "y": rng.normal(size=len(index)),
        },
        index=index,
    )


def test_fit_var_infers_regular_datetime_frequency():
    # Regular weekly dates, but deliberately without freq metadata
    dates = pd.date_range("2020-01-03", periods=40, freq="W-FRI")
    index = pd.DatetimeIndex(dates.values)

    assert index.freq is None
    assert pd.infer_freq(index) == "W-FRI"

    data = _sample_data(index)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_var(data, lags=1)

    frequency_warnings = [
        warning
        for warning in caught
        if "frequency" in str(warning.message).lower()
    ]

    assert frequency_warnings == []


def test_fit_var_does_not_modify_irregular_datetime_index():
    index = pd.DatetimeIndex(
        [
            "2020-01-03",
            "2020-01-10",
            "2020-01-17",
            "2020-01-31",  # 24 Jan missing -> irregular
            "2020-02-07",
            "2020-02-14",
            "2020-02-21",
            "2020-02-28",
            "2020-03-06",
            "2020-03-13",
            "2020-03-20",
            "2020-03-27",
        ]
    )

    assert pd.infer_freq(index) is None

    data = _sample_data(index)
    original_index = data.index.copy()

    fit_var(data, lags=1)

    pd.testing.assert_index_equal(data.index, original_index)
