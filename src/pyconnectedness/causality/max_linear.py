r"""
Recursive max-linear models on directed acyclic graphs

Every node variable is the largest of its weighted parents and an own innovation,

.. math:: X_i = \bigvee_{k \in pa(i)} c_{ik} X_k \vee c_{ii} Z_i,
          \qquad i = 1, \dots, d,

with the unique solution :math:`X = A \times_{\max} Z`. The max-linear
coefficient :math:`a_{ij}` is the weight of the strongest path from j to
i, so A holds the channels along which extreme shocks propagate.

Two estimation methods are available through :meth:`MaxLinear.fit`. The
scaling method estimates a causal order and A for a general DAG. QTree estimates the graph 
when it is a root-directed tree, as in a river network.

References
----------
Gissibl, N. and C. Klüppelberg (2018) Max-linear models on directed acyclic graphs. 
Bernoulli 24(4A), 2693-2720.
Klüppelberg, C. and M. Krali (2021) Estimating an extreme Bayesian network via scalings. 
Journal of Multivariate Analysis 181, 104672.
Klüppelberg, C. and M. Krali (2025) Causal analysis of extreme risk in a network of industry portfolios. 
arXiv:2504.00523.
Tran, N. M., J. Buck and C. Klüppelberg (2024) Estimating a directed tree for extremes. 
Journal of the Royal Statistical Society Series B 86(3), 771-792.
"""

import networkx as nx
import numpy as np
import pandas as pd

from . import _qtree, _scaling
from .transforms import to_frechet

__all__ = [
    "MaxLinear",
    "MaxLinearResults",
    "KSelectionResults",
    "QTreeResults",
    "ml_coef_matrix",
    "nshd",
    "simulate_max_linear",
]

# rescaling factor a and tolerance eps of the two papers
_ORDER_DEFAULTS = {"generations": (1.3, 0.1),
                   "sequential": (1.01, (0.0045, 0.0045))}


def ml_coef_matrix(weights):
    r"""
    Max-linear coefficient matrix of a weighted DAG

    Parameters
    ----------
    weights : array_like (d x d)
        Entry (i, k) is the edge weight :math:`c_{ik}` of k -> i, the
        diagonal holds :math:`c_{ii}`. Zeros are missing edges.

    Notes
    -----
    Theorem 2.2 of Gissibl and Klüppelberg (2018),

    .. math:: a_{ij} = \bigvee_{p \in P_{ji}} c_{jj} \prod_{(k, l)
              \in p} c_{lk}, \qquad a_{ii} = c_{ii}

    with :math:`P_{ji}` the paths from j to i, and :math:`a_{ij} = 0` if
    j is no ancestor of i.

    Returns
    -------
    coefs : ndarray (d x d)
    """
    c = np.asarray(weights, dtype=float)
    d = c.shape[0]
    edges = c * (1 - np.eye(d))
    if not nx.is_directed_acyclic_graph(nx.DiGraph(edges.T)):
        raise ValueError("weights do not define a DAG")

    # paths[i, j]: largest product of edge weights from j to i
    paths = np.eye(d)
    for _ in range(d - 1):
        longer = (edges[:, :, None] * paths[None, :, :]).max(axis=1)
        paths = np.maximum(paths, longer)
    return paths * np.diag(c)[None, :]


def simulate_max_linear(coefs, nobs, seed=None):
    r"""
    Simulate a max-linear model with standard Fréchet innovations

    Parameters
    ----------
    coefs : array_like (d x d)
        Max-linear coefficient matrix A, rows are effects.
    nobs : int
    seed : {None, int, Generator}

    Notes
    -----
    .. math:: X_i = \bigvee_j a_{ij} Z_j, \qquad
              P(Z_j \leq z) = \exp(-z^{-2})

    Returns
    -------
    data : ndarray (nobs x d)
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(coefs, dtype=float)
    z = (-np.log(rng.uniform(size=(nobs, a.shape[1])))) ** -0.5
    return (a[None, :, :] * z[:, None, :]).max(axis=2)


def nshd(adj1, adj2):
    r"""
    Normalised structural Hamming distance between two directed graphs

    Parameters
    ----------
    adj1, adj2 : array_like (d x d)
        Adjacency matrices, nonzero entries are edges.

    Notes
    -----
    .. math:: nSHD(G_1, G_2) = \frac{SHD(G_1, G_2)}{|E(G_1)| + |E(G_2)|}

    where SHD counts the edge additions, deletions and reversals that turn one graph into the other, 
    one for every node pair whose edges differ (Klüppelberg and Krali, 2025, Section 7.1).

    Returns
    -------
    float
    """
    e1 = np.asarray(adj1) != 0
    e2 = np.asarray(adj2) != 0
    shd = np.triu((e1 != e2) | (e1.T != e2.T), 1).sum()
    nedges = e1.sum() + e2.sum()
    return shd / nedges if nedges else 0.0


class MaxLinear:
    r"""
    Recursive max-linear model on a directed acyclic graph

    Parameters
    ----------
    data : array_like (nobs x d)
        Extremes in the upper tail, e.g. losses ``max(-returns, 0)``.
    transform : {"frechet", None}
        Standardisation of the margins for the scaling method, see Notes.
    names : sequence of str, optional
        Variable names, taken from ``data`` if it is a DataFrame.

    Notes
    -----
    The scaling method needs margins that are regularly varying with index 2 and 
    unit scale (Assumption A1 of Klüppelberg and Krali, 2025).
    ``"frechet"`` is the empirical integral transform of eq. (25)  
    With None the data are taken as already standardised.

    QTree does not use the transform. It works with the logarithms of the
    data and allows missing values.

    The observations are treated as independent replicates.

    References
    ----------
    Klüppelberg and Krali (2021), Sections 4 to 7.
    Klüppelberg and Krali (2025), Sections 4 to 7.
    Tran, Buck and Klüppelberg (2024), Algorithms 1 and 2.
    """

    def __init__(self, data, transform="frechet", names=None):
        if isinstance(data, pd.DataFrame):
            self.index = data.index
            if names is None:
                names = [str(c) for c in data.columns]
        else:
            self.index = None
        values = np.asarray(data, dtype=float)
        if values.ndim != 2:
            raise ValueError("data must be 2-dimensional (nobs x d)")

        self.data = values
        self.nobs, self.neqs = values.shape
        if names is None:
            names = [f"x{i}" for i in range(self.neqs)]
        self.names = list(names)

        if transform is None:
            std = values
        elif transform == "frechet":
            std = to_frechet(values)
        else:
            raise ValueError(f"transform '{transform}' not supported")
        self.transform = transform
        self.std_data = std

    def fit(self, k=None, method="scaling", *, k_order=None, order=None,
            order_method="generations", a=None, eps=None, threshold=0.8,
            lower=0.05, log=True):
        """
        Estimate the model

        Parameters
        ----------
        k : int
            Number of largest radii for the coefficient matrix. Needed for
            method "scaling".
        method : {"scaling", "qtree"}
            "scaling" estimates a causal order and the max-linear
            coefficients of a general DAG from scalings of
            max-projections (Klüppelberg and Krali, 2021, 2025). 
            "qtree" estimates a root-directed spanning tree (Tran, Buck and Klüppelberg, 2024).
        k_order : int, optional
            Number of largest radii for the causal order, defaults to k.
            The 2025 paper orders with k_order=250 and estimates with k between 50 and 98.     
        order : sequence, optional
            Known causal order, sources first, as names or positions.
            Skips the ordering; every node is then a step of its own.
        order_method : {"generations", "sequential"}
            "generations" is Algorithm 1 of Klüppelberg and Krali (2025),
            which orders a whole generation per step. "sequential" is
            Algorithms 4 and 5 of Klüppelberg and Krali (2021): the source
            nodes, then one node per step.
        a : float, optional
            Rescaling factor, a > 1. Defaults to 1.3 for "generations"
            and 1.01 for "sequential", the values of the papers.
        eps : float or tuple, optional
            Tolerance. For "generations" a float, default 0.1. For
            "sequential" the bounds (eps1, eps2) of Algorithm 4, default
            (0.0045, 0.0045); a float sets both.
        threshold : float
            QTree: quantile level above which the source node must lie.
        lower : float
            QTree: small quantile level of the score.
        log : bool
            QTree: set to False if the data are already in logs.

        Notes
        -----
        Coefficients between nodes ordered in the same step are set to zero for "generations" 
        (Theorem 4(b) of the 2025 paper), and the
        matrix is standardised to unit row norm. The 2021 paper does
        neither, so "sequential" estimates all coefficients and reports
        them as they are.

        Returns
        -------
        MaxLinearResults or QTreeResults
        """
        if method == "qtree":
            x = self._log_data(log)
            scores = _qtree.quantile_scores(x, threshold, lower)
            adjacency = _qtree.spanning_tree(scores)
            return QTreeResults(self, adjacency, scores, threshold, lower)
        if method != "scaling":
            raise ValueError(f"method '{method}' not supported")
        if k is None:
            raise ValueError("k is needed for method 'scaling'")
        k_order = k if k_order is None else k_order
        for val in (k, k_order):
            if not 1 < val < self.nobs:
                raise ValueError("k must lie strictly between 1 and nobs")

        internal, gens, a, eps = self._causal_order(k_order, order,
                                                    order_method, a, eps)
        if order is not None:
            k_order = None
        return self._fit_coefs(k, internal, gens, k_order=k_order,
                               order_method=order_method, a=a, eps=eps,
                               known=order is not None)

    def _log_data(self, log):
        if not log:
            return self.data
        if np.nanmin(self.data) <= 0:
            raise ValueError("data must be positive, or pass log=False")
        return np.log(self.data)

    def _causal_order(self, k_order, order, order_method, a, eps):
        if order_method not in _ORDER_DEFAULTS:
            raise ValueError(f"order_method '{order_method}' not supported")
        if np.isnan(self.std_data).any():
            raise ValueError("data contains missing values")
        a0, eps0 = _ORDER_DEFAULTS[order_method]
        a = a0 if a is None else a
        eps = eps0 if eps is None else eps
        if a <= 1:
            raise ValueError("a must be larger than 1")

        if order is not None:
            pos = [self.names.index(o) if isinstance(o, str) else int(o)
                   for o in order]
            if sorted(pos) != list(range(self.neqs)):
                raise ValueError("order must contain every variable once")
            return pos[::-1], [[p] for p in pos], a, eps

        if order_method == "sequential":
            if np.isscalar(eps):
                eps = (eps, eps)
            internal, gens = _scaling.causal_order_sequential(
                self.std_data, k_order, a, eps)
        else:
            internal, gens = _scaling.causal_order(self.std_data, k_order,
                                                   a, eps)
        return internal, gens, a, eps

    def _fit_coefs(self, k, internal, gens, *, k_order, order_method, a,
                   eps, known):
        # the zeros and the unit row norm belong to the 2025 paper
        paper25 = order_method == "generations"
        coefs, coefs2 = _scaling.coef_matrix(
            self.std_data, internal, gens, k,
            zero_within=paper25 and not known, standardize=paper25)
        return MaxLinearResults(self, coefs, coefs2, internal, gens, k=k,
                                k_order=k_order, method="scaling",
                                order_method=order_method, a=a, eps=eps,
                                known=known)

    def select_k(self, ks=range(50, 100, 2),
                 deltas=(0.0, 0.025, 0.05, 0.1), *, block=5, k_order=250,
                 order=None, order_method="generations", a=None,
                 eps=None):
        """
        Choose the number of exceedances by the stability of the DAG

        Parameters
        ----------
        ks : sequence of int
            Numbers of exceedances, split into consecutive blocks.
        deltas : sequence of float
            Thresholds of the minimum max-linear DAG.
        block : int
            Block size; the paper uses blocks {k, k+2, ..., k+8}.
        k_order : int
            Number of exceedances for the causal order, estimated once
            for all ks.
        order, order_method, a, eps
            See :meth:`fit`.

        Notes
        -----
        Within each block, the DAG with the smallest summed nSHD to the
        other DAGs of the block is its centroid, eq. (28). The selected
        pair (delta, k) minimises this sum over all blocks and deltas
        (Klüppelberg and Krali, 2025, Section 7.1).

        Returns
        -------
        KSelectionResults
        """
        ks = sorted(int(r) for r in ks)
        if len(ks) % block:
            raise ValueError("the number of ks must be a multiple of block")
        internal, gens, a, eps = self._causal_order(k_order, order,
                                                    order_method, a, eps)
        if order is not None:
            k_order = None
        fits = {r: self._fit_coefs(r, internal, gens, k_order=k_order,
                                   order_method=order_method, a=a,
                                   eps=eps, known=order is not None)
                for r in ks}
        return KSelectionResults(fits, deltas, block)

    def select_threshold(self, thresholds=(0.0, 0.2, 0.4, 0.6, 0.7, 0.75,
                                           0.8, 0.85, 0.9),
                         lowers=(0.05,), *, nrep=100, frac=0.75, log=True,
                         seed=None):
        """
        Choose the QTree parameters by the variability of the tree

        Parameters
        ----------
        thresholds, lowers : sequence of float
            Grid for ``threshold`` and ``lower`` of :meth:`fit`.
        nrep : int
            Number of subsamples.
        frac : float
            Subsample size as fraction of nobs, drawn without replacement.
        log : bool
            Set to False if the data are already in logs.
        seed : {None, int, Generator}

        Notes
        -----
        Algorithm 2 of Tran, Buck and Klüppelberg (2024). For every grid
        point the tree is estimated on each subsample; the centroid of
        the grid point with the smallest tree variability is returned.

        Returns
        -------
        QTreeResults
        """
        x = self._log_data(log)
        rng = np.random.default_rng(seed)
        size = int(frac * self.nobs)
        # the same subsamples for every grid point, otherwise the
        # variabilities are not comparable
        subsamples = [rng.choice(self.nobs, size, replace=False)
                      for _ in range(nrep)]

        best = None
        grid = []
        for r in lowers:
            for q in thresholds:
                trees = [_qtree.spanning_tree(
                    _qtree.quantile_scores(x[idx], q, r))
                    for idx in subsamples]
                tree, var = _qtree.centroid(trees)
                grid.append((q, r, var))
                if best is None or var < best[2]:
                    best = (q, r, var, tree)

        q, r, var, tree = best
        grid = pd.DataFrame(grid,
                            columns=["threshold", "lower", "variability"])
        scores = _qtree.quantile_scores(x, q, r)
        return QTreeResults(self, tree, scores, q, r, variability=var,
                            grid=grid)


class MaxLinearResults:
    """
    Estimation results of a recursive max-linear model

    Parameters
    ----------
    model : MaxLinear
    coefs : ndarray (d x d)
    coefs2 : ndarray (d x d)
    order : list of int
        Causal order in the paper convention, sources last.
    generations : list of list of int
    k, k_order : int
    method, order_method : str
    a, eps : float
    known : bool
        Whether the causal order was given.

    Attributes
    ----------
    coefs : DataFrame (d x d)
        Max-linear coefficient matrix, rows are effects and columns
        causes; entry (i, j) is the weight of the strongest path from j
        to i. Unit row norm for order_method "generations".
    coefs2 : DataFrame (d x d)
        Squared coefficients from Algorithm 2 before clipping at zero.
    causal_order : list of str
        Sources first.
    generations : list of list of str
        Nodes ordered in each step of the ordering, sources first.
    names : list of str
    nobs, neqs : int
    k, k_order : int
        Exceedances for the coefficients and for the order.
    """

    def __init__(self, model, coefs, coefs2, order, generations, *, k,
                 k_order, method, order_method, a, eps, known=False):
        self.model = model
        self.names = model.names
        self.nobs = model.nobs
        self.neqs = model.neqs
        self.k = k
        self.k_order = k_order
        self.method = method
        self.order_method = "known" if known else order_method
        self.a = a
        self.eps = eps
        self._order = order
        self._generations = generations

        names = self.names
        self.coefs = pd.DataFrame(coefs, index=names, columns=names)
        self.coefs2 = pd.DataFrame(coefs2, index=names, columns=names)
        self.causal_order = [names[i] for i in order[::-1]]
        self.generations = [[names[i] for i in g] for g in generations]

    def __repr__(self):
        return (f"<MaxLinearResults: d={self.neqs}, k={self.k}, "
                f"k_order={self.k_order}, steps={len(self.generations)}>")

    @property
    def zero_diagonal(self):
        """
        Nodes whose own coefficient was estimated as zero

        Such a node removes every edge it could carry from the thresholded
        DAG, see :meth:`min_dag`.
        """
        diag = np.diag(self.coefs.to_numpy())
        return [n for n, v in zip(self.names, diag, strict=True) if v == 0]

    def min_dag(self, delta=0.0):
        r"""
        Weighted adjacency matrix of the thresholded minimum max-linear DAG

        Parameters
        ----------
        delta : float
            Hard threshold.

        Notes
        -----
        The edge j -> i is kept when :math:`\hat a_{ij}` exceeds the
        weight of every path through an intermediate node by more than
        delta (Klüppelberg and Krali, 2025, Section 7.1).

        Returns
        -------
        DataFrame (d x d)
            Entry (i, j) is :math:`\hat a_{ij}` for an edge j -> i, else
            zero.
        """
        adj = _scaling.min_dag(self.coefs.to_numpy(), self._order, delta)
        return pd.DataFrame(adj, index=self.names, columns=self.names)

    def edges(self, delta=0.0):
        """
        Edges of the thresholded DAG, strongest first

        Returns
        -------
        DataFrame
            Columns "cause", "effect" and "coef".
        """
        adj = self.min_dag(delta).to_numpy()
        effect, cause = np.nonzero(adj)
        out = pd.DataFrame({
            "cause": [self.names[j] for j in cause],
            "effect": [self.names[i] for i in effect],
            "coef": adj[effect, cause],
        })
        return out.sort_values("coef", ascending=False, ignore_index=True)

    def to_networkx(self, delta=0.0):
        """
        Thresholded DAG as a networkx graph, edges from cause to effect

        Edges carry the coefficient as "weight"; nodes carry the step of
        the ordering as "generation", 0 for the sources.
        """
        graph = nx.DiGraph()
        for s, gen in enumerate(self.generations):
            graph.add_nodes_from(gen, generation=s)
        edges = self.edges(delta)
        graph.add_weighted_edges_from(
            edges[["cause", "effect", "coef"]].itertuples(index=False))
        return graph

    def summary(self, delta=0.0, top=10):
        """
        Text summary of the fit

        Parameters
        ----------
        delta : float
            Threshold for the reported edges.
        top : int
            Number of edges to list.

        Returns
        -------
        str
        """
        edges = self.edges(delta)
        width = 72
        lines = [
            "Recursive max-linear model".center(width),
            "=" * width,
            f"Method:        {self.method:<20} Variables:    {self.neqs}",
            f"Order:         {self.order_method:<20} "
            f"k (order):    {self.k_order}",
            f"Transform:     {str(self.model.transform):<20} "
            f"Observations: {self.nobs}",
            f"k:             {self.k:<20} Edges:        {len(edges)}",
            f"a:             {self.a:<20} eps:          {self.eps}",
            f"delta:         {delta}",
            "-" * width,
            "Causal order, steps of the ordering, sources first:",
        ]
        lines += [f"  {s}: {', '.join(g)}"
                  for s, g in enumerate(self.generations)]
        lines += ["-" * width, f"Strongest edges (top {top}):"]
        lines += [f"  {r.cause:>8} -> {r.effect:<8} {r.coef:.3f}"
                  for r in edges.head(top).itertuples()]
        if self.zero_diagonal:
            lines += ["-" * width,
                      "Zero diagonal: " + ", ".join(self.zero_diagonal)]
        lines.append("=" * width)
        return "\n".join(lines)


class KSelectionResults:
    """
    Stability of the minimum max-linear DAG over the number of exceedances

    Parameters
    ----------
    fits : dict
        Maps k to MaxLinearResults with a common causal order.
    deltas : sequence of float
    block : int

    Attributes
    ----------
    nshd : DataFrame
        Summed nSHD of each DAG to the other DAGs of its block, eq. (28);
        rows k, columns delta.
    delta, k : float, int
        Selected threshold and number of exceedances.
    block_ks : list of int
        The block of the selected k.
    results : MaxLinearResults
        The fit at the selected k.
    """

    def __init__(self, fits, deltas, block):
        self.fits = fits
        self.ks = sorted(fits)
        self.deltas = list(deltas)
        self.blocks = [self.ks[i:i + block]
                       for i in range(0, len(self.ks), block)]

        table = {}
        for delta in self.deltas:
            adj = {r: fits[r].min_dag(delta).to_numpy() for r in self.ks}
            table[delta] = {r: sum(nshd(adj[r], adj[s]) for s in blk
                                   if s != r)
                            for blk in self.blocks for r in blk}
        self.nshd = pd.DataFrame(table)

        self.k, self.delta = self.nshd.stack().idxmin()
        self.block_ks = next(b for b in self.blocks if self.k in b)
        self.results = fits[self.k]

    def __repr__(self):
        return (f"<KSelectionResults: k={self.k}, delta={self.delta}, "
                f"block={self.block_ks}>")

    def stability(self, delta=None, ks=None):
        """
        Stability score, eq. (29)

        Parameters
        ----------
        delta : float, optional
            Defaults to the selected delta.
        ks : sequence of int, optional
            Defaults to the block of the selected k.

        Returns
        -------
        DataFrame (d x d)
            Entry (i, j) counts the DAGs of ks with the edge j -> i.
        """
        delta = self.delta if delta is None else delta
        ks = self.block_ks if ks is None else ks
        return sum((self.fits[r].min_dag(delta) > 0).astype(int)
                   for r in ks)


class QTreeResults:
    """
    Estimation results of QTree

    Parameters
    ----------
    model : MaxLinear
    adjacency : ndarray (d x d) of bool
    scores : ndarray (d x d)
    threshold, lower : float
    variability : float, optional
    grid : DataFrame, optional

    Attributes
    ----------
    adjacency : DataFrame (d x d) of bool
        Rows are sources and columns targets: entry (j, i) is True for
        the edge j -> i.
    scores : DataFrame (d x d)
        Scores on the full sample; entry (i, j) belongs to the edge
        j -> i.
    root : str
    threshold, lower : float
        Parameters used (or selected).
    variability : float or None
        Tree variability, only after :meth:`MaxLinear.select_threshold`.
    grid : DataFrame or None
        Variability at every grid point, only after
        :meth:`MaxLinear.select_threshold`.
    """

    def __init__(self, model, adjacency, scores, threshold, lower,
                 variability=None, grid=None):
        self.model = model
        self.names = model.names
        self.nobs = model.nobs
        self.neqs = model.neqs
        self.method = "qtree"
        self.threshold = threshold
        self.lower = lower
        self.variability = variability
        self.grid = grid
        self._adjacency = adjacency

        names = self.names
        self.adjacency = pd.DataFrame(adjacency, index=names, columns=names)
        self.scores = pd.DataFrame(scores, index=names, columns=names)
        # the root is the only node without an outgoing edge
        self.root = names[np.flatnonzero(~adjacency.any(axis=1))[0]]

    def __repr__(self):
        return (f"<QTreeResults: d={self.neqs}, root={self.root}, "
                f"threshold={self.threshold}, lower={self.lower}>")

    def edges(self):
        """
        Edges of the tree

        Returns
        -------
        DataFrame
            Columns "cause" and "effect", e.g. upstream and downstream.
        """
        cause, effect = np.nonzero(self._adjacency)
        return pd.DataFrame({"cause": [self.names[j] for j in cause],
                             "effect": [self.names[i] for i in effect]})

    @property
    def reachability(self):
        """
        Directed paths of the tree

        Returns
        -------
        DataFrame (d x d) of bool
            Entry (j, i) is True if there is a path from j to i.
        """
        return pd.DataFrame(_qtree.reachability(self._adjacency),
                            index=self.names, columns=self.names)

    def to_networkx(self):
        """
        Tree as a networkx graph, edges from cause to effect
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(self.names)
        graph.add_edges_from(self.edges().itertuples(index=False))
        return graph

    def compare(self, true):
        """
        Metrics against a known tree, for the tree and its reachability

        Parameters
        ----------
        true : array_like (d x d)
            Adjacency matrix, entry (j, i) nonzero for the edge j -> i.

        Returns
        -------
        DataFrame
            nSHD, FDR, FPR and TPR as defined in eq. (1) of Tran, Buck and
            Klüppelberg (2024).
        """
        true = np.asarray(true) != 0
        tree = _qtree.tree_metrics(self._adjacency, true)
        reach = _qtree.tree_metrics(_qtree.reachability(self._adjacency),
                                    _qtree.reachability(true))
        return pd.DataFrame({"tree": tree, "reachability": reach}).T

    def summary(self):
        """
        Text summary of the fit

        Returns
        -------
        str
        """
        width = 72
        lines = [
            "Max-linear root-directed tree".center(width),
            "=" * width,
            f"Method:        {self.method:<20} Variables:    {self.neqs}",
            f"Root:          {self.root:<20} Observations: {self.nobs}",
            f"threshold:     {self.threshold:<20} "
            f"lower:        {self.lower}",
        ]
        if self.variability is not None:
            lines.append(f"variability:   {self.variability:.4f}")
        lines += ["-" * width, "Edges:"]
        lines += [f"  {r.cause:>8} -> {r.effect}"
                  for r in self.edges().itertuples()]
        lines.append("=" * width)
        return "\n".join(lines)
