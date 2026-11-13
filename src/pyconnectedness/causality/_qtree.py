r"""
Root-directed spanning trees for extremes (QTree)

The tree is learned from pairwise quantile scores of log-differences and
the minimum directed spanning tree of Chu-Liu/Edmonds. Data are on the log
scale, where the max-linear tree with noise reads

.. math:: X_i = \bigvee_{j \to i} (c_{ij} + X_j) \vee Z_i + \varepsilon_i

Adapted from the GPL-3.0 reference implementation by Ngoc Mai Tran
(https://github.com/princengoc/qtree); 

Adjacency matrices are boolean with entry [j, i] for the edge j -> i,
score matrices hold the edge j -> i in entry [i, j].

References
----------
Gissibl, N. and C. Klüppelberg (2018) Max-linear models on directed
    acyclic graphs. Bernoulli 24(4A), 2693-2720.
Tran, N. M., J. Buck and C. Klüppelberg (2024) Estimating a directed tree
    for extremes. Journal of the Royal Statistical Society Series B 86(3),
    771-792.
"""

import networkx as nx
import numpy as np


def quantile_scores(x, threshold=0.8, lower=0.05):
    r"""
    Quantile-to-mean gap for every ordered pair of nodes

    Parameters
    ----------
    x : ndarray (nobs x d)
        Log data, NaN allowed.
    threshold : float
        Quantile level :math:`\alpha` above which the source node must
        lie.
    lower : float
        Small quantile level :math:`\underline{r}`.

    Notes
    -----
    For the candidate edge :math:`j \to i`, eq. (7) of the paper,

    .. math:: w_{ij} = \frac{1}{n_{ij}} \left(\mathbb{E}\,
              \mathcal{X}_{ij}(\alpha)
              - Q_{\mathcal{X}_{ij}(\alpha)}(\underline{r})\right)^2

    where :math:`\mathcal{X}_{ij}(\alpha) = \{x_i - x_j : x_j \geq
    Q_{X_j}(\alpha)\}`

    As in the reference code, the gap is divided by the number of
    non-missing values of node i rather than by the size of
    :math:`\mathcal{X}_{ij}(\alpha)`. Ties at the cutoff make the subset
    sizes differ across sources, and this changes the tree.

    Returns
    -------
    scores : ndarray (d x d)
        The diagonal is inf.
    """
    d = x.shape[1]
    scores = np.full((d, d), np.inf)
    counts = (~np.isnan(x)).sum(axis=0)

    for j in range(d):
        xj = x[:, j]
        cutoff = np.nanquantile(xj, threshold)
        # >= instead of > keeps every observation when threshold is 0
        sub = x[xj >= cutoff]
        diff = sub - sub[:, [j]]
        for i in range(d):
            if i == j:
                continue
            gap = diff[:, i]
            gap = gap[~np.isnan(gap)]
            if gap.size < 2:
                continue
            scores[i, j] = (gap.mean() - np.quantile(gap, lower))**2
            scores[i, j] /= counts[i]

    return scores


def spanning_tree(scores, minimize=True):
    """
    Optimal root-directed spanning tree for a score matrix

    Parameters
    ----------
    scores : ndarray (d x d)
    minimize : bool
        If False, the maximum tree is returned (used for the centroid).

    Notes
    -----
    In a root-directed tree every node except the root has exactly one
    child. networkx finds arborescences, where every node has one parent,
    so the edges are flipped before and after Chu-Liu/Edmonds. The root
    is chosen by the algorithm.

    Returns
    -------
    adjacency : ndarray (d x d) of bool
    """
    d = scores.shape[0]
    sign = 1.0 if minimize else -1.0
    graph = nx.DiGraph()
    graph.add_nodes_from(range(d))
    for i in range(d):
        for j in range(d):
            if i != j and np.isfinite(scores[i, j]):
                # flipped: i -> j stands for the edge j -> i
                graph.add_edge(i, j, weight=sign * scores[i, j])

    arb = nx.minimum_spanning_arborescence(graph)
    adjacency = np.zeros((d, d), dtype=bool)
    for i, j in arb.edges():
        adjacency[j, i] = True
    return adjacency


def reachability(adjacency):
    """
    Directed paths of a graph

    Returns
    -------
    reach : ndarray (d x d) of bool
        reach[j, i] is True if there is a path from j to i of length >= 1.
    """
    adj = np.asarray(adjacency, dtype=bool)
    reach = adj.copy()
    for _ in range(adj.shape[0]):
        new = reach | ((reach.astype(int) @ adj.astype(int)) > 0)
        if (new == reach).all():
            break
        reach = new
    return reach


def centroid(trees):
    """
    Centroid tree and variability of a set of trees

    Parameters
    ----------
    trees : list of ndarray (d x d) of bool

    Notes
    -----
    Algorithm 2 of the paper. The centroid is the maximum spanning tree of
    the edge frequencies. The variability adds the average mismatch in
    edges and in reachability, each relative to the centroid, as in the
    reference code.

    Returns
    -------
    tree : ndarray (d x d) of bool
    variability : float
    """
    freq = np.mean(trees, axis=0)
    tree = spanning_tree(freq.T, minimize=False)
    reach = reachability(tree)

    edge_miss = np.mean([(tree != t).sum() for t in trees])
    reach_miss = np.mean([(reach != reachability(t)).sum() for t in trees])
    return tree, reach_miss / reach.sum() + edge_miss / tree.sum()


def tree_metrics(estimated, true):
    """
    nSHD, FDR, FPR and TPR of an estimated graph

    Parameters
    ----------
    estimated, true : ndarray (d x d) of bool

    Notes
    -----
    Definitions of eq. (1) in Tran, Buck and Klüppelberg (2024). A
    reversed edge counts once in the structural Hamming distance.

    Returns
    -------
    metrics : dict
    """
    est = np.asarray(estimated, dtype=bool)
    tru = np.asarray(true, dtype=bool)
    d = tru.shape[0]

    correct = (est & tru).sum()
    reverse = (est & ~tru & tru.T).sum()
    extra = (est & ~tru & ~tru.T).sum()
    missing = (tru & ~est & ~est.T).sum()
    shd = extra + missing + reverse

    n_est, n_true = est.sum(), tru.sum()
    wrong = n_est - correct
    return {
        "nshd": shd / max(n_est + n_true, 1),
        "fdr": wrong / max(n_est, 1),
        "fpr": wrong / max(d * (d - 1) - n_true, 1),
        "tpr": correct / max(n_true, 1),
    }


def simulate_tree(nobs, d, noise=0.1, coef_range=(0.2, 1.0), seed=None):
    """
    Draw log data from a random max-linear root-directed tree

    Parameters
    ----------
    nobs : int
    d : int
        Number of nodes.
    noise : float
        Noise-to-signal ratio; Gaussian noise with this fraction of the
        median standard deviation is added after the recursion.
    coef_range : tuple
        Edge weights are uniform on this range (original scale).
    seed : {None, int, Generator}

    Notes
    -----
    Mirrors the simulation study in Section 5 of the paper: standard
    Fréchet innovations, i.e. Gumbel on the log scale.

    Returns
    -------
    x : ndarray (nobs x d)
    adjacency : ndarray (d x d) of bool
    """
    rng = np.random.default_rng(seed)

    # every node drains into one node with a smaller label, node 0 is
    # the root; the labels are shuffled afterwards
    child = np.array([-1] + [rng.integers(0, i) for i in range(1, d)])
    perm = rng.permutation(d)
    adjacency = np.zeros((d, d), dtype=bool)
    logc = np.zeros((d, d))
    for j in range(1, d):
        adjacency[perm[j], perm[child[j]]] = True
        logc[perm[j], perm[child[j]]] = np.log(rng.uniform(*coef_range))

    x = rng.gumbel(size=(nobs, d))
    # sources first: a node is finished once all its parents are
    for i in nx.topological_sort(nx.DiGraph(adjacency)):
        for j in np.flatnonzero(adjacency[:, i]):
            x[:, i] = np.maximum(x[:, i], logc[j, i] + x[:, j])

    scale = noise * np.median(x.std(axis=0))
    return x + rng.normal(scale=scale, size=x.shape), adjacency
