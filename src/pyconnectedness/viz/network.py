r"""
Network representation of connectedness results.

Diebold and Yilmaz (2014) read the variance decomposition as a weighted,
directed network: the variables are nodes and the net pairwise measures are
the edges. Because the net pairwise matrix is antisymmetric, each pair of
variables contributes a single edge, pointing from the net transmitter to the
net receiverr

References
----------
Diebold and Yilmaz (2014) On the network topology of variance decompositions:
measuring the connectedness of financial firms. Journal of Econometrics, 182, 119-134.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from ..connectedness.static import ConnectednessResult


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
    the zero diagonal, so no self-loops arise. -->

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


def plot_connectedness_graph(
    graph,
    *,
    ax=None,
    pos=None,
    node_scale: float = 1600.0,
    edge_scale: float = 4.0,
    labels: bool = True,
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
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 7.0))
    if pos is None:
        pos = nx.circular_layout(graph)

    net = np.array([graph.nodes[name].get("net", 0.0) for name in graph])
    sizes = node_scale * (0.25 + 0.75 * np.abs(net) / np.abs(net).max())
    colors = np.where(net >= 0.0, "tab:red", "tab:blue")

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=sizes,
                           node_color=colors, alpha=0.85)

    if graph.number_of_edges() > 0:
        weights = np.array([w for _, _, w in graph.edges(data="weight")])
        # node_size lets networkx stop the arrows at the node boundary
        nx.draw_networkx_edges(graph, pos, ax=ax, node_size=sizes,
                               width=edge_scale * weights / weights.max(),
                               edge_color="0.45", arrowsize=12,
                               connectionstyle="arc3,rad=0.08")

    if labels:
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=9)

    ax.set_axis_off()
    return ax