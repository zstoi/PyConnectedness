# Changelog

## 0.2.0 — 13 November 2026

This release adds animated heatmaps, connectedness networks and spillover time series. Animations show how connectedness evolves across rolling windows or build up individual figures step by step. They are enabled with `animation=True` and can be saved as GIF files.

Visualisation improvements include:

- A new `plot_spillovers` function for total, directional and net spillover time series.
- A `kind` option in `spillover_heatmap` for displaying the normalized variance decomposition or net pairwise flows.

The DY-2009 replication notebook now includes animation examples and GIF exports. Additional tests cover the visualisation functions.

The minimum pandas version 2.2 supports frequency aliases such as `"QE"` and `"YE"`.


## 0.1.0 — 26 September 2026

First public release. It brings the connectedness framework of Diebold and Yilmaz to Python: VAR estimation with lag selection, generalized and orthogonalized forecast error variance decompositions, and the spillover tables of the 2009 and 2012 frameworks. The package provides total connectedness together with directional `TO` and `FROM` measures, `NET` connectedness, and net pairwise directional connectedness. These measures are also available in rolling-window form for dynamic connectedness analysis. The release further implements the frequency decomposition of Baruník and Křehlík (2018), whose frequency-band contributions add up to the corresponding Diebold-Yilmaz connectedness measures. Diebold-Yilmaz replication examples and numerical consistency tests are included to validate the implementation.

Visualisation tools include spillover heatmaps for connectedness matrices and directed network representations of pairwise connectedness.

