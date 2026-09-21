<p align="center">
  <img src="https://raw.githubusercontent.com/zstoi/PyConnectedness/main/logos/logo.svg" alt="PyConnectedness logo" width="180">
</p>

<h1 align="center">PyConnectedness</h1>

<p align="center">
  <img src="https://img.shields.io/badge/status-active%20development-orange.svg" alt="Status: active development">
  <img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPLv3">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
</p>

> ⚠️ **Active development.** This library is currently under active development. The API and
> project structure may still change as additional connectedness and dependence methods are added. 

## About

**[PyConnectedness](https://www.prototypefund.de/projects/pyconnectedness)** is a Python library for 
analysing connectedness, spillovers and dependence in multivariate (time series) data. The current implementation focuses on variance-decomposition-based connectedness measures in the spirit of Diebold and Yilmaz (2009, 2012, 2014), including static and rolling-window dynamic connectedness, directional spillovers, net spillovers, net pairwise directional connectedness, and graphical representations of connectedness networks. A max-linear Bayesian network module is in progress, aiming to extend the package towards modelling extremal dependence and causal structures between extreme events. Additional methods, including frequency-domain connectedness for analysing spillovers across different time horizons, are planned. The goal is to bring methods that are well established in econometrics and statistics, but still scattered or missing in Python, into one open-source package.


## Project status

![Implemented](https://img.shields.io/badge/status-implemented-brightgreen) 

- **Connectedness and spillover analysis**  
  Static and rolling-window dynamic connectedness, directional `TO` and `FROM`, net and net pairwise directional connectedness.

- **Visualisation** 
  Spillover heatmaps and directed connectedness networks.

![In progress](https://img.shields.io/badge/status-in%20progress-orange)  

- **Max-linear Bayesian networks** 
  Modelling extremal dependence and causal structures between extreme events.

- **Frequency-domain connectedness**  
  Analysis of spillovers across different time horizons.

## Installation

PyConnectedness requires Python 3.10 or newer.

A standard PyPI installation will be available once the package is released there:

```bash
pip install pyconnectedness
```

## Usage

PyConnectedness provides a compact interface for estimating VAR-based connectedness measures.

A static connectedness estimate can be computed directly from a pandas `DataFrame` using a simple one-liner:

```python
from pyconnectedness import static_connectedness

result = static_connectedness(
    data,
    horizon=10,
    method="generalized",
    lags=2,
)

print(result.table)
print(result.total)
print(result.directional_to)
print(result.directional_from)
print(result.net)
print(result.pairwise_net)
```

For Diebold-Yilmaz (2009) - an orthogonalized Cholesky variance decomposition can be used:

```python
result = static_connectedness(
    data,
    horizon=10,
    method="orthogonalized",
    lags=2,
)
```

Rolling-window connectedness is available through:

```python
from pyconnectedness import dynamic_connectedness

dynamic = dynamic_connectedness(
    data,
    window=200,
    horizon=10,
    method="generalized",
    lags=2,
)

print(dynamic.total)
print(dynamic.net)
```

The package currently provides:

- VAR estimation and moving-average representations
- orthogonalized forecast error variance decomposition
- generalized forecast error variance decomposition
- static connectedness measures
- rolling-window dynamic connectedness
- total connectedness
- directional `TO` and `FROM` measures
- net directional connectedness
- net pairwise directional connectedness
- spillover heatmaps
- directed connectedness networks

For complete examples, see the replication notebooks below.

## Replication of Diebold-Yilmaz papers

The `examples/` directory contains end-to-end replications of the main Diebold-Yilmaz connectedness frameworks using the `PyConnectedness` package, providing reproducible examples and validation against published results.

### Diebold-Yilmaz (2009)

The DY-2009 replication uses the orthogonalized, Cholesky-based variance decomposition and reproduces the static and dynamic spillover analysis using the original framework.

Source notebook:

[`examples/replicate_dy2009.ipynb`](https://github.com/zstoi/PyConnectedness/blob/main/examples/replicate_dy2009.ipynb)

Or with Colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zstoi/PyConnectedness/blob/main/examples/replicate_dy2009.ipynb)


### Diebold-Yilmaz (2012)

The DY-2012 replication uses the generalized forecast error variance decomposition and the corresponding row-normalized connectedness framework.


Source notebook:

[`examples/replicate_dy2012.ipynb`](https://github.com/zstoi/PyConnectedness/blob/main/examples/replicate_dy2012.ipynb)



These notebooks also serve as reproducible usage examples for the package.




## Repository structure

```
pyconnectedness/
├── README.md
├── LICENSE                     # GPLv3
├── pyproject.toml              # packaging & dependencies
├── .gitignore
├── .github/
│   └── workflows/
│       └── tests.yml           # continuous integration
├── src/
│   └── pyconnectedness/        # the library source code
│       ├── __init__.py
│       ├── connectedness/      # Diebold-Yilmaz spillover measures
│       ├── causality/         # max-linear Bayesian networks
│       └── viz/                # network plotting
├── tests/                      # unit tests
├── examples/                   # example notebooks
├── logos/                      # project & funding logos
└── docs/                       # documentation
```

## References

The methods draw on, among others:

- Diebold and Yilmaz (2009, 2012, 2014) for *variance-decomposition-based connectedness and spillover measures*
- Baruník and Křehlík (2018) for *frequency-domain connectedness*
- Gissibl and Klüppelberg (2018) for *max-linear models on directed acyclic graphs* 

(Reference list to be completed as the modules are implemented.)

## Data

The datasets used for the Diebold-Yilmaz connectedness analysis in this project are obtained from Mendeley Data:
Nguyen, Viet Hoang; Kocenda, Evzen; Greenwood-Nimmo, Matthew (2024), “Detecting Statistically Significant Changes in Connectedness: A Bootstrap-based Technique”, Mendeley Data, V1, doi: 10.17632/rtwsfgpgmf.1

Doan, Tom (2025), "DIEBOLDYILMAZ_IJF2012: RATS program to replicate Diebold and Yilmaz(2012) spillover calculations", [EconPapers](https://econpapers.repec.org/software/bocbocode/rtz00199.htm)


## Funding

Developed with support from the **Prototype Fund** (Software Sprint), funded by the German Federal Ministry of Research, Technology and Space (BMFTR) and supported by the Open Knowledge Foundation Deutschland.

<table align="center">
  <tr>
    <td align="center" valign="middle" width="320">
      <a href="https://www.prototypefund.de/projects/pyconnectedness">
        <img src="https://raw.githubusercontent.com/zstoi/PyConnectedness/main/logos/prototypefund.svg" alt="Prototype Fund" height="130">
      </a>
    </td>
    <td align="left" valign="middle" width="320">
      <img src="https://raw.githubusercontent.com/zstoi/PyConnectedness/main/logos/bmftr.svg" alt="BMFTR" height="130">
    </td>
  </tr>
</table>


## License

Released under the [GNU General Public License v3.0](https://github.com/zstoi/PyConnectedness/blob/main/LICENSE).
