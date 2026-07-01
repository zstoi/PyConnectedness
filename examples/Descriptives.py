"""
Reproduce Diebold-Yilmaz (2009) Table 1 — descriptive statistics.

# put into sandbox

"""
from pyconnectedness import fit_var

import pandas as pd
from scipy import stats

# load the weekly return data (German decimal commas -> dots) ---
data = pd.read_excel("examples/data/dy2009_returns.xlsx")
data_vol = pd.read_excel("examples/data/dy2009_vola.xlsx")
data["date"] = pd.to_datetime(data["date"], format="%m.%d.%y")
data_vol["date"] = pd.to_datetime(data["date"], format="%d.%m.%y")
data = data.set_index("date")
data_vol = data_vol.set_index("date")
for col in data.columns:
    if data[col].dtype == object:
        data[col] = data[col].str.replace(",", ".", regex=False)
    data[col] = pd.to_numeric(data[col], errors="coerce")
for col in data_vol.columns:
    if data_vol[col].dtype == object:
        data_vol[col] = data_vol[col].str.replace(",", ".", regex=False)
    data_vol[col] = pd.to_numeric(data_vol[col], errors="coerce")


def describe(df):
    """Descriptive statistics DY_2009

    Kurtosis is raw (normal = 3), not excess; 
    skewness and kurtosis use the population (biased) estimator, matching the paper
    """
    rows = ["Mean", "Median", "Maximum", "Minimum", "Std.Dev.", "Skewness", "Kurtosis"]
    out = pd.DataFrame(index=rows)
    for col in df.columns:
        x = df[col].dropna().values
        out[col] = [
            x.mean(),
            pd.Series(x).median(),
            x.max(),
            x.min(),
            x.std(ddof=1),
            stats.skew(x, bias=True),
            stats.kurtosis(x, fisher=False, bias=True),  # raw kurtosis (normal = 3)
        ]
    return out


table1 = describe(data)
table2 = describe(data_vol)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
print("Reproducing Diebold-Yilmaz (2009) Table 1 - Descriptive Statistics\n")
print(table1.round(5).to_string())

print ("**" * 50)

print("Reproducing Diebold-Yilmaz (2009) Table 2 - Descriptive Statistics\n")
print(table2.round(5).to_string())


# Zusatz: Test var-wrapper


print("Markets:", list(data.columns))
print("Observations:", len(data))
print("Missing after conversion:", int(data.isnull().sum().sum()))

fit = fit_var(data, lags=2)
print("Lag order:", fit.lag_order, "| k:", fit.k, "| Sigma:", fit.sigma.shape)
print("MA shape:", fit.ma_coefficients(10).shape)   # expect (10, 19, 19)

# Success

