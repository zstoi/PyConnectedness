r"""
Network representation of connectedness results.

The variables are nodes and the net pairwise measures are
the edges. Because the net pairwise matrix is asymmetric, each pair of
variables contributes a single edge, pointing from the net transmitter to the
net receiver

With ``animation=True`` a rolling result is played back one window per
frame, and a single network is built up node by node and then edge by edge,
strongest flow first.

"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from ..connectedness.dynamic import DynamicConnectednessResult
from ..connectedness.static import ConnectednessResult
from _frames import reveal_counts, window_positions, window_title

def _window_graph(result, t, threshold):
    """Graph of window t of a rolling result, with the node measures."""
    names = result.names
    pairwise = pd.DataFrame(result.pairwise_net[t], index=names,
                            columns=names)
    graph = connectedness_graph(pairwise, threshold=threshold)

    to = result.directional_to.iloc[t].to_dict()
    frm = result.directional_from.iloc[t].to_dict()
    net = result.net.iloc[t].to_dict()
    nx.set_node_attributes(graph, to, "to")
    nx.set_node_attributes(graph, frm, "from")
    nx.set_node_attributes(graph, net, "net")
    return graph

def connectedness_graph(result, *, threshold: float = 0.0) -> nx.DiGraph:
    r"""
    Build a directed graph from net pairwise directional connectedness.

    Parameters
    ----------
    result : ConnectednessResult or DataFrame
        A static connectedness result, or a net pairwise matrix such as
        ``ConnectednessResult.pairwise_net``.
    threshold : float
        Keep only net flows above this value, in percent. The default keeps
        every pair that is not exactly balanced.

    Note
    -----
    An edge from i to j carries weight :math:`C_{ij} = \tilde\theta_{ji} -
    \tilde\theta_{ij} > 0`, the part of j's forecast error variance that i
    explains in excess of what j explains of i's. Keeping only the positive
    entries of the antisymmetric matrix leaves one edge per pair and drops
    the zero diagonal, so no self-loops arise.

    When a :class:`ConnectednessResult` is passed, the directional measures
    are attached to the nodes as the attributes ``to``, ``from`` and ``net``.

    Returns
    -------
    networkx.DiGraph
        Edge weights under the attribute ``weight``.

    Raises
    ------
    ValueError
        If ``threshold`` is negative, which would add the diagonal as
        self-loops.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    if isinstance(result, DynamicConnectednessResult):
        return _window_graph(result, -1, threshold)
    if isinstance(result, ConnectednessResult):
        pairwise = result.pairwise_net
    else:
        pairwise = pd.DataFrame(result)

    names = list(pairwise.index)
    values = pairwise.to_numpy(dtype=float)

    graph = nx.DiGraph()
    graph.add_nodes_from(names)
    senders, receivers = np.nonzero(values > threshold)
    graph.add_weighted_edges_from(
        (names[i], names[j], float(values[i, j]))
        for i, j in zip(senders, receivers, strict=True)
    )

    if isinstance(result, ConnectednessResult):
        nx.set_node_attributes(graph, result.directional_to.to_dict(), "to")
        nx.set_node_attributes(graph, result.directional_from.to_dict(), "from")
        nx.set_node_attributes(graph, result.net.to_dict(), "net")

    return graph

def _draw_graph(graph, ax, pos, node_scale, edge_scale, labels, max_net,
                max_weight):
    """Draw one network  - a frame can be removed"""
    net = np.array([graph.nodes[name].get("net", 0.0) for name in graph])

    if max_net is None:
        max_net = np.abs(net).max()

    if max_net == 0:
        sizes = np.full(len(net), node_scale * 0.25)
    else:
        sizes = node_scale * (0.25 + 0.75 * np.abs(net) / max_net)

    colors = np.where(net >= 0.0, "tab:red", "tab:blue")

    nodes = nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=sizes,
                                   node_color=colors, alpha=0.85)
    artists = [nodes]

    if graph.number_of_edges() > 0:
        weights = np.array([w for _, _, w in graph.edges(data="weight")])
        if max_weight is None:
            max_weight = weights.max()
        widths = edge_scale * weights / max_weight
        # node_size lets networkx stop the arrows at the node boundary
        arrows = nx.draw_networkx_edges(graph, pos, ax=ax, node_size=sizes,
                                        width=widths, edge_color="0.45",
                                        arrowsize=12,
                                        connectionstyle="arc3,rad=0.08")
        artists.extend(arrows)

    if labels:
        names = nx.draw_networkx_labels(graph, pos, ax=ax, font_size=9)
        artists.extend(names.values())

    return artists

def plot_connectedness_graph(
    graph,
    *,
    ax=None,
    pos=None,
    node_scale: float = 1600.0,
    edge_scale: float = 4.0,
    labels: bool = True,
    max_net: float | None = None,
    max_weight: float | None = None,
    threshold: float = 0.0,
    animation: bool = False,
    n_frames: int = 40,
    freq: str | None = None,
    hold: int = 10,
    interval: int = 150,
):
    """
    Draw a connectedness graph.

    Parameters
    ----------
    graph : networkx.DiGraph
        As returned by :func:`connectedness_graph`.
    ax : matplotlib Axes, optional
        Axes to draw on. A new figure is created if omitted.
    pos : dict, optional
        Node positions. Defaults to a circular layout, which keeps the
        picture readable and comparable across periods; a spring layout
        moves the nodes whenever the data change.
    node_scale : float
        Area of the largest node, in points squared.
    edge_scale : float
        Line width of the strongest edge, in points.
    labels : bool
        Draw the variable names.

    Notes
    -----
    Node area is proportional to the absolute net position and node colour to
    its sign, red for net transmitters and blue for net receivers. Edge width
    is proportional to the net flow. Both are scaled relative to the largest
    value in the graph, so sizes are comparable within a plot but not across
    plots.

    Returns
    -------
    matplotlib Axes
    """
    rolling = isinstance(graph, DynamicConnectednessResult)
    if rolling:
        result = graph
        graph = _window_graph(result, -1, threshold)
    elif isinstance(graph, ConnectednessResult):
        graph = connectedness_graph(graph, threshold=threshold)

    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 7.0))
    if pos is None:
        pos = nx.circular_layout(graph)
    ax.set_axis_off()

    if not animation:
        _draw_graph(graph, ax, pos, node_scale, edge_scale, labels, max_net,
                    max_weight)
        if rolling:
            ax.set_title(window_title(result, -1))
        return ax    

    if rolling:
        windows = window_positions(result.total.index, n_frames, freq)
        frames = []
        for t in windows:
            frames.append(_window_graph(result, t, threshold))
        n_steps = len(windows)
        net = result.net.to_numpy(dtype=float)[windows]
        weights = np.asarray(result.pairwise_net, dtype=float)[windows]
    else:
        names = list(graph)
        edges = sorted(graph.edges(data="weight"), key=lambda e: e[2],
                       reverse=True)
        counts = reveal_counts(len(edges), n_frames)
        n_steps = len(names) + len(counts)
        net = np.array([graph.nodes[name].get("net", 0.0) for name in names])
        weights = np.array([w for _, _, w in edges])

  #  net = np.array([graph.nodes[name].get("net", 0.0) for name in graph])

   # max_abs_net = np.abs(net).max()
    if max_net is None:
        max_net = np.abs(net).max()
    if max_weight is None and weights.size > 0:
        max_weight = weights.max()

    # networkx rescales the view whenever something is drawn; fixed limits
    # keep the picture from breathing while nodes and edges come and go
    xy = np.array(list(pos.values()))
    lim = (xy.min() - 0.2, xy.max() + 0.2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")

    artists = []

    def update(i):
        step = min(i, n_steps - 1)
        # remove the previous frame before the next one is drawn
        for artist in artists:
            artist.remove()
        artists.clear()
        if rolling:
            frame = frames[step]
            ax.set_title(window_title(result, windows[step]))
        else:
            # a fresh graph per step; a subgraph of the full graph would
            # already carry every edge between the nodes shown
            frame = nx.DiGraph()
            for name in names[:step + 1]:
                frame.add_node(name, **graph.nodes[name])
            if step >= len(names):
                n_edges = counts[step - len(names)]
                frame.add_weighted_edges_from(edges[:n_edges])
        artists.extend(_draw_graph(frame, ax, pos, node_scale, edge_scale,
                                   labels, max_net, max_weight))
        return artists

    update(0)
    return FuncAnimation(ax.figure, update, frames=n_steps + hold,
                         interval=interval)
