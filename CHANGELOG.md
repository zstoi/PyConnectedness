# Changelog

## Unreleased

Animations: heatmaps, networks and time series that build up step by step, or play back one rolling window per frame.

## 0.1.0 — 26 September 2026

First public release. It brings the connectedness framework of Diebold and Yilmaz to Python: VAR estimation with lag selection, generalized and orthogonalized forecast error variance decompositions, and the spillover tables of the 2009 and 2012 frameworks. The package provides total connectedness together with directional `TO` and `FROM` measures, `NET` connectedness, and net pairwise directional connectedness. These measures are also available in rolling-window form for dynamic connectedness analysis. The release further implements the frequency decomposition of Baruník and Křehlík (2018), whose frequency-band contributions add up to the corresponding Diebold-Yilmaz connectedness measures. Diebold-Yilmaz replication examples and numerical consistency tests are included to validate the implementation.

Visualisation tools include spillover heatmaps for connectedness matrices and directed network representations of pairwise connectedness.