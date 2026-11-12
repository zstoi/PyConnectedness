from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.animation import FuncAnimation

from pyconnectedness import dynamic_connectedness, static_connectedness
from pyconnectedness.viz import (
    connectedness_graph,
    plot_connectedness_graph,
    plot_spillovers,
    spillover_heatmap,
)
#
matplotlib.use("Agg")

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"

returns = pd.read_excel(DATA / "dy2009_returns.xlsx").set_index("date")
data = returns.iloc[:260, :5]

conn = static_connectedness(data, horizon=10, method="generalized", lags=2)
dyn = dynamic_connectedness(data, window=200, horizon=10,
                            method="generalized", lags=2)


@pytest.fixture(autouse=True)
def close_figures():
    # every test opens figures; matplotlib starts warning at twenty
    yield
    plt.close("all")


def _frames(anim):
    return len(list(anim.new_frame_seq()))


def _render(anim, tmp_path):
    # draws every frame, so the figure ends on the last one
    anim.save(tmp_path / "anim.gif", writer="pillow", fps=5)


def _image(ax):
    # imshow masks NaN cells; fill them back in to compare with numpy
    return np.ma.filled(ax.images[0].get_array().astype(float), np.nan)


def _widths(ax):
    return np.array([arrow.get_linewidth() for arrow in ax.patches])


def test_still_image_by_default():
    assert isinstance(spillover_heatmap(conn), plt.Axes)
    assert isinstance(spillover_heatmap(dyn), plt.Axes)
    assert isinstance(plot_connectedness_graph(conn), plt.Axes)
    assert isinstance(plot_connectedness_graph(dyn), plt.Axes)
    assert isinstance(plot_spillovers(dyn.net), plt.Axes)


def test_heatmap_pairwise_net_is_centred():
    ax = spillover_heatmap(conn, kind="pairwise_net")
    low, high = ax.images[0].get_clim()
    assert low == pytest.approx(-high)
    assert ax.images[0].get_cmap().name == "RdBu_r"
    assert ax.get_ylabel() == "net transmitter"
    assert np.allclose(_image(ax), conn.pairwise_net)


def test_heatmap_rolling_shows_last_window():
    last = static_connectedness(data.iloc[-200:], horizon=10,
                                method="generalized", lags=2)
    ax = spillover_heatmap(dyn)
    assert np.allclose(_image(ax), last.fevd)
    assert dyn.total.index[-1].strftime("%Y-%m-%d") in ax.get_title()


def test_heatmap_build_up(tmp_path):
    _, ax = plt.subplots()
    anim = spillover_heatmap(conn, ax=ax, animation=True, n_frames=6,
                             hold=2)
    assert isinstance(anim, FuncAnimation)
    assert _frames(anim) == 6 + 2

    values = conn.fevd.to_numpy()
    shown = ~np.isnan(_image(ax))
    assert shown.sum() == np.ceil(values.size / 6)
    # the strongest cells come first
    assert np.abs(values[shown]).min() >= np.abs(values[~shown]).max()

    _render(anim, tmp_path)
    assert np.allclose(_image(ax), values)
    assert [text.get_text() for text in ax.texts[:2]] == [
        f"{value:.1f}" for value in values[0, :2]
    ]


def test_heatmap_rolling_animation(tmp_path):
    _, ax = plt.subplots()
    anim = spillover_heatmap(dyn, kind="pairwise_net", ax=ax,
                             animation=True, n_frames=5, hold=1)
    assert _frames(anim) == 5 + 1
    assert np.allclose(_image(ax), dyn.pairwise_net[0])

    _render(anim, tmp_path)
    assert np.allclose(_image(ax), dyn.pairwise_net[-1])


def test_heatmap_scale_covers_every_frame(tmp_path):
    _, ax = plt.subplots()
    anim = spillover_heatmap(dyn, kind="pairwise_net", ax=ax,
                             animation=True, n_frames=dyn.n_windows,
                             hold=0)
    span = np.abs(dyn.pairwise_net).max()
    # the peak lies before the last window, so a scale taken from the
    # still image of the last window would be too narrow
    assert span > np.abs(dyn.pairwise_net[-1]).max()
    assert ax.images[0].get_clim() == pytest.approx((-span, span))
    _render(anim, tmp_path)


def test_heatmap_one_frame_per_quarter(tmp_path):
    anim = spillover_heatmap(dyn, animation=True, freq="QE", hold=0)
    assert _frames(anim) == dyn.total.index.to_period("Q").nunique()
    _render(anim, tmp_path)


def test_heatmap_unknown_kind():
    with pytest.raises(ValueError, match="unknown kind"):
        spillover_heatmap(conn, kind="total")


def test_freq_needs_dates():
    plain = dynamic_connectedness(data.reset_index(drop=True), window=200,
                                  horizon=10, lags=2)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        spillover_heatmap(plain, animation=True, freq="QE")


def test_graph_from_result_equals_graph_from_builder():
    direct = plot_connectedness_graph(conn, threshold=1.0)
    built = plot_connectedness_graph(connectedness_graph(conn, threshold=1.0))
    assert np.allclose(direct.collections[0].get_sizes(),
                       built.collections[0].get_sizes())
    assert np.allclose(_widths(direct), _widths(built))


def test_rolling_graph_uses_last_window():
    last = static_connectedness(data.iloc[-200:], horizon=10,
                                method="generalized", lags=2)
    graph = connectedness_graph(dyn)
    assert set(graph.edges) == set(connectedness_graph(last).edges)
    net = dict(graph.nodes(data="net"))
    assert np.allclose([net[name] for name in dyn.names], last.net)


def test_graph_build_up(tmp_path):
    _, ax = plt.subplots()
    anim = plot_connectedness_graph(conn, ax=ax, animation=True,
                                    n_frames=4, hold=1)
    k = len(conn.net)
    n_edges = connectedness_graph(conn).number_of_edges()
    assert _frames(anim) == k + min(4, n_edges) + 1
    assert len(ax.collections[0].get_offsets()) == 1
    assert len(ax.patches) == 0

    # matplotlib has no public call that draws a single frame, three nodes
    # are visible here and no edge may be drawn before all nodes are
    anim._draw_frame(2)
    assert len(ax.collections[0].get_offsets()) == 3
    assert len(ax.patches) == 0

    _render(anim, tmp_path)
    assert len(ax.collections[0].get_offsets()) == k
    assert len(ax.patches) == n_edges


def test_graph_rolling_animation(tmp_path):
    _, ax = plt.subplots()
    anim = plot_connectedness_graph(dyn, ax=ax, animation=True, n_frames=5,
                                    hold=1, edge_scale=4.0)
    assert _frames(anim) == 5 + 1
    # one width scale for all frames: the first window's strongest flow is
    # drawn relative to the strongest flow of the whole animation
    windows = np.unique(np.linspace(0, dyn.n_windows - 1, 5).round())
    strongest = dyn.pairwise_net[windows.astype(int)].max()
    assert dyn.pairwise_net[0].max() < strongest
    expected = 4.0 * dyn.pairwise_net[0].max() / strongest
    assert _widths(ax).max() == pytest.approx(expected)

    _render(anim, tmp_path)
    assert dyn.total.index[-1].strftime("%Y-%m-%d") in ax.get_title()
    assert len(ax.patches) == connectedness_graph(dyn).number_of_edges()


def test_reference_weight_thins_the_edges():
    graph = connectedness_graph(conn)
    weight = max(w for _, _, w in graph.edges(data="weight"))
    plain = plot_connectedness_graph(graph)
    halved = plot_connectedness_graph(graph, max_weight=2 * weight)
    assert len(plain.patches) == graph.number_of_edges()
    assert np.allclose(_widths(halved) * 2, _widths(plain))


def test_lines_sequential(tmp_path):
    _, ax = plt.subplots()
    anim = plot_spillovers(dyn.net, ax=ax, animation=True,
                           frames_per_line=5, hold=2)
    assert _frames(anim) == dyn.k * 5 + 2
    drawn = [np.isfinite(line.get_ydata()).sum() for line in ax.lines[:2]]
    assert drawn == [np.ceil(dyn.n_windows / 5), 0]
    # the legend 
    assert len(ax.get_legend().get_texts()) == 1

    _render(anim, tmp_path)
    assert np.allclose(ax.lines[0].get_ydata(), dyn.net.iloc[:, 0])
    assert len(ax.get_legend().get_texts()) == dyn.k


def test_lines_parallel(tmp_path):
    anim = plot_spillovers(dyn.net, animation=True, sequential=False,
                           frames_per_line=5, hold=2)
    assert _frames(anim) == 5 + 2
    _render(anim, tmp_path)


def test_lines_accept_series(tmp_path):
    ax = plot_spillovers(dyn.total, ylabel="total connectedness (%)")
    assert len(ax.lines) == 1
    assert ax.get_legend() is None
    anim = plot_spillovers(dyn.total, animation=True, frames_per_line=4,
                           hold=1)
    _render(anim, tmp_path)


def test_lines_reject_plain_array():
    with pytest.raises(TypeError):
        plot_spillovers(dyn.net.to_numpy())


def test_lines_reject_zero_frames():
    with pytest.raises(ValueError, match="frames_per_line"):
        plot_spillovers(dyn.total, animation=True, frames_per_line=0)


def test_zero_frames_rejected():
    with pytest.raises(ValueError, match="n_frames"):
        spillover_heatmap(dyn, animation=True, n_frames=0)