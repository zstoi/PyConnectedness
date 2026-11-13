r"""
Structure learning and estimation of recursive max-linear models via scalings

Adapted from code of the authors (CJS_functions_code.R) 

Node comes before its parents, so the sources are last and the max-linear coefficient matrix 
is upper triangular in that order.

References
----------
Klüppelberg, C. and M. Krali (2021) Estimating an extreme Bayesian network via scalings. 
Journal of Multivariate Analysis 181, 104672.
Klüppelberg, C. and M. Krali (2025) Causal analysis of extreme risk in a network of industry portfolios. 
arXiv:2504.00523.
Krali, M. (2025) Causal discovery in heavy-tailed linear structural equation models via scalings. 
arXiv:2502.13762.
"""

import numpy as np


def _angles(y, k):
    """
    Squared angles of the observations with the k largest radii

    Ties at the k-th radius are all kept, as in the reference code.
    """
    r2 = (y**2).sum(axis=1)
    top = r2 >= np.sort(r2)[-k]
    return y[top] ** 2 / r2[top, None]


def scaling(z, cols, k):
    r"""
    Squared scaling of the maximum over a set of components

    Parameters
    ----------
    z : ndarray (nobs x d)
        Data with standard margins of index 2
    cols : sequence of int
        Components I of the max-projection.
    k : int
        Number of largest radii.

    Notes
    -----
    Empirical version of Proposition 1(i), based on eq. (24) and (26),

    .. math:: \hat\sigma^2_{M_I} = \frac{|I|}{k} \sum_{\ell=1}^n
              \bigvee_{i \in I} \omega_{\ell i}^2
              1\{R_\ell \geq R^{(k)}\}

    with the angles and radii of the subvector :math:`X_I` only. The
    factor :math:`|I|` is the mass of its angular measure.

    Returns
    -------
    sigma2 : float
    """
    w2 = _angles(z[:, cols], k)
    return len(cols) * w2.max(axis=1).mean()


def _delta_sources(z, i, j, a, k):
    r"""
    Entry (i, j) of :math:`\Delta_\emptyset` in the first step of Algorithm 1

    Equation (9): j is a source node iff the entry is zero for all i. The rescaled and 
    the plain maximum are estimated from their own exceedances, and
    :math:`\sigma_j^2 = 1` enters by its theoretical value.
    """
    scaled = _angles(np.c_[z[:, i], a * z[:, j]], k)
    plain = _angles(z[:, [i, j]], k)
    return ((1 + a**2) * scaled.max(axis=1).mean()
            - 2 * plain.max(axis=1).mean() - (a**2 - 1))


def _delta_ordered(z, i, j, ordered, a, k):
    r"""
    Entry (i, j) of :math:`\Delta_O` once the nodes in O are ordered

    Equation (13),

    .. math:: (\Delta_O)_{ij} = \sigma^2_{M_{i,aj,aO}} - \sigma^2_{M_{i,j,O}}
              - (a^2 - 1)\, \sigma^2_{M_{j,O}}

    All three scalings are taken from the exceedances of the rescaled vector 
    :math:`(X_i, aX_j, aX_O)`, as in ``sigma_i_aj_aI_diff`` 
    Sharing one set of extremes keeps the difference
    stable.
    """
    rest = [j] + ordered
    w2 = _angles(np.c_[z[:, i], a * z[:, rest]], k)
    mass = 1 + a**2 * len(rest)
    # angles of j and O with the rescaling undone
    unscaled = w2[:, 1:].max(axis=1) / a**2
    s_iajao = w2.max(axis=1).mean()
    s_ijo = np.maximum(w2[:, 0], unscaled).mean()
    s_jo = unscaled.mean()
    return mass * (s_iajao - s_ijo - (a**2 - 1) * s_jo)


def causal_order(z, k, a=1.3, eps=0.1):
    r"""
    Estimate a causal order, Algorithm 1

    Parameters
    ----------
    z : ndarray (nobs x d)
        Data with standard margins of index 2.
    k : int
        Number of largest radii.
    a : float
        Rescaling factor, a > 1.
    eps : float
        Relative tolerance; nodes whose column minimum of
        :math:`\Delta_O` lies within it of the largest one are ordered in
        the same step.

    Notes
    -----
    Theorem 4(a) finds the source nodes, Theorem 4(b) then adds one generation of 
    descendants per step. Nodes ordered in the same step
    are causally unrelated

    Returns
    -------
    order : list of int
        Paper convention, a node before its parents, sources last.
    generations : list of list of int
        Nodes ordered in each step, the sources first.
    """
    d = z.shape[1]
    order = []
    generations = []
    while len(order) < d:
        rest = [j for j in range(d) if j not in order]
        if len(rest) == 1:
            generations.append(rest)
            order = rest + order
            break

        delta = np.full((d, d), np.inf)
        for j in rest:
            for i in rest:
                if i == j:
                    continue
                if order:
                    delta[i, j] = _delta_ordered(z, i, j, order, a, k)
                else:
                    delta[i, j] = _delta_sources(z, i, j, a, k)

        colmin = delta.min(axis=0)[rest]
        dev = colmin - colmin.max()
        tol = eps * abs(colmin.max())
        pick = [rest[p] for p in np.argsort(dev, kind="stable")
                if abs(dev[p]) <= tol]
        generations.append(pick)
        order = pick + order

    return order, generations


def _scaling_k(y, k, mass):
    # eq. (37) of the 2021 paper divides by k, not by the number of
    # exceedances; the two differ only when radii are tied
    return mass / k * _angles(y, k).max(axis=1).sum()


def source_matrix(z, k, a=1.01):
    r"""
    Pairwise scaling differences of Algorithm 4 (2021)

    Parameters
    ----------
    z : ndarray (nobs x d)
        Data with standard margins of index 2.
    k : int
        Number of largest radii.
    a : float
        Rescaling factor, a > 1.

    Notes
    -----
    Theorem 2 of Klüppelberg and Krali (2021): m is a source node iff

    .. math:: \sigma^2_{M_{i,am}} - \sigma^2_{M_{i,m}} - (a^2 - 1) = 0

    for all i. The entry (i, m) holds the estimated left-hand side.

    Returns
    -------
    delta : ndarray (d x d)
        The diagonal is NaN.
    """
    d = z.shape[1]
    delta = np.full((d, d), np.nan)
    for m in range(d):
        for i in range(d):
            if i == m:
                continue
            pair = z[:, [i, m]]
            delta[i, m] = (_scaling_k(pair * [1.0, a], k, 1 + a**2)
                           - _scaling_k(pair, k, 2) - (a**2 - 1))
    return delta


def causal_order_sequential(z, k, a=1.01, eps=(0.0045, 0.0045)):
    r"""
    Estimate a causal order, Algorithms 4 and 5 (2021)

    Parameters
    ----------
    z : ndarray (nobs x d)
        Data with standard margins of index 2.
    k : int
        Number of largest radii.
    a : float
        Rescaling factor, a > 1.
    eps : tuple of float
        Bounds (eps1, eps2) of Algorithm 4.

    Notes
    -----
    Algorithm 4 takes m as a source node if every entry of its column in
    :func:`source_matrix` lies in [-eps2, eps1]. Algorithm 5 then adds one
    node per step. With O the nodes ordered so far, the next node is the
    i that maximises (Theorem 3)

    .. math:: \sigma^2_{M_{aO, ai, V \setminus (O \cup i)}}
              - \sigma^2_{M_V} - (a^2 - 1)\, \sigma^2_{M_{O \cup i}}

    which is zero iff all ancestors of i are in O, and negative otherwise.

    Returns
    -------
    order : list of int
        Paper convention, a node before its parents, sources last.
    generations : list of list of int
        The source nodes, then one node per step.

    Raises
    ------
    ValueError
        If no column passes the bounds of Algorithm 4.
    """
    d = z.shape[1]
    eps1, eps2 = eps
    delta = source_matrix(z, k, a)
    upper = np.nanmax(delta, axis=0)
    lower = np.nanmin(delta, axis=0)
    sources = [m for m in range(d)
               if upper[m] <= eps1 and lower[m] >= -eps2]
    if not sources:
        raise ValueError("no source node found, widen eps or change k")

    ordered = list(sources)
    generations = [list(sources)]
    plain = _scaling_k(z, k, d)
    while len(ordered) < d:
        best, best_diff = None, -np.inf
        for i in range(d):
            if i in ordered:
                continue
            sub = ordered + [i]
            lift = np.ones(d)
            lift[sub] = a
            mass = d + (a**2 - 1) * len(sub)
            diff = (_scaling_k(z * lift, k, mass) - plain
                    - (a**2 - 1) * _scaling_k(z[:, sub], k, len(sub)))
            if diff > best_diff:
                best, best_diff = i, diff
        ordered.append(best)
        generations.append([best])

    return ordered[::-1], generations


def coef_matrix(z, order, generations, k, zero_within=True,
                standardize=True):
    r"""
    Estimate the max-linear coefficient matrix, Algorithm 2

    This is Algorithm 1 of the 2021 paper with the zeros of Theorem 4(b)
    added.

    Parameters
    ----------
    z : ndarray (nobs x d)
        Data with standard margins of index 2.
    order : list of int
        Causal order in the paper convention, sources last.
    generations : list of list of int
        Steps of Algorithm 1. Coefficients between nodes of one step are
        set to zero.
    k : int
        Number of largest radii.
    zero_within : bool
        Set coefficients between nodes of one step to zero, Theorem 4(b)
        The 2021 paper estimates them
    standardize : bool
        Rescale the clipped matrix to unit row norm. The 2021 paper
        reports it as it is.

    Notes
    -----
    With the nodes relabelled along the order, eq. (15)-(17) give

    .. math::

        a_{ii}^2 = \sigma^2_{M_{i,\dots,d}} - \sigma^2_{M_{i+1,\dots,d}},
        \qquad
        a_{ij}^2 = \sigma^2_{M_{i,j+1,\dots,d}}
        - \sigma^2_{M_{j+1,\dots,d}} - \sum_{\ell=i}^{j-1} a_{i\ell}^2

    The zeros for nodes of one step are set inside the recursion, as in
    ``A_generations`` of the reference code, so that their estimated mass
    moves on to the next coefficient of the row. Finally
    :math:`\hat A_{0+} = \max(\hat A_0^2, 0)^{1/2}` is standardised to
    unit row norm (Section 7.1).

    Returns
    -------
    coefs : ndarray (d x d)
        Coefficients in the original labels, rows are effects and
        columns causes.
    coefs2 : ndarray (d x d)
        Squared coefficients before clipping and standardisation, same
        labels. Negative entries show estimation noise.
    """
    d = len(order)
    zp = z[:, order]
    tail = np.zeros(d + 1)
    for j in range(1, d):
        tail[j] = scaling(zp, list(range(j, d)), k)

    pos = {node: p for p, node in enumerate(order)}
    step = np.empty(d, dtype=int)
    for s, gen in enumerate(generations):
        step[[pos[node] for node in gen]] = s

    a2 = np.zeros((d, d))
    for i in range(d):
        if i < d - 1:
            a2[i, i] = scaling(zp, list(range(i, d)), k) - tail[i + 1]
        else:
            a2[i, i] = scaling(zp, [i], k)
        for j in range(i + 1, d):
            if zero_within and step[i] == step[j]:
                continue
            if j < d - 1:
                cols = [i] + list(range(j + 1, d))
                cum = scaling(zp, cols, k) - tail[j + 1]
            else:
                cum = scaling(zp, [i], k)
            a2[i, j] = cum - a2[i, i:j].sum()

    a = np.sqrt(np.clip(a2, 0.0, None))
    if standardize:
        norm = np.sqrt((a**2).sum(axis=1))
        norm[norm == 0] = 1.0
        a /= norm[:, None]

    coefs = np.zeros((d, d))
    coefs2 = np.zeros((d, d))
    coefs[np.ix_(order, order)] = a
    coefs2[np.ix_(order, order)] = a2
    return coefs, coefs2


def min_dag(coefs, order, delta=0.0):
    r"""
    Thresholded minimum max-linear DAG

    Parameters
    ----------
    coefs : ndarray (d x d)
        Max-linear coefficient matrix, rows are effects.
    order : list of int
        Causal order in the paper convention, sources last.
    delta : float
        Hard threshold.

    Notes
    -----
    Section 7.1: the edge j -> i is kept when

    .. math:: \hat a_{ij} > \bigvee_{k \in de(j) \cap pa(i)}
              \frac{\hat a_{ik} \hat a_{kj}}{\hat a_{kk}} + \delta

    As in ``mwp_gr_eps`` of the reference code, an intermediate k with an
    estimated diagonal of zero gives an infinite path weight and removes
    every edge it could carry.

    Returns
    -------
    adj : ndarray (d x d)
        ``adj[i, j] = a_ij`` for an edge j -> i, else zero.
    """
    d = len(order)
    ap = coefs[np.ix_(order, order)]
    diag = np.diag(ap)
    keep = np.zeros((d, d), dtype=bool)
    for i in range(d):
        for j in range(i + 1, d):
            via = 0.0
            for m in range(i + 1, j):
                through = ap[i, m] * ap[m, j]
                if through > 0:
                    w = through / diag[m] if diag[m] > 0 else np.inf
                    via = max(via, w)
            keep[i, j] = ap[i, j] > via + delta

    adj = np.zeros((d, d))
    adj[np.ix_(order, order)] = np.where(keep, ap, 0.0)
    return adj
