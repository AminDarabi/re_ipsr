# This file is developed by Amin Darabi as part of the project of
# the course "Geometric Modeling and shape analysis" (IFT6113) at the
# Université de Montréal (UdeM) in Fall 2024.
"""
Screened Poisson Surface Reconstruction in pure NumPy / SciPy.

A compact re-implementation of Screened Poisson Surface Reconstruction
(Kazhdan & Hoppe, 2013), written to be read and compared against the
official C++ implementation (PoissonRecon, used through Open3D), whose
discretisation it follows (first-order elements, Neumann boundary):

    E(chi) = int |grad chi - V|^2
             + alpha * spn * Area / N * sum_d 2^d * sum_{cells c of depth d}
                                          W_c (chi(p_c) - 1/2)^2

* ``chi`` is the indicator function (1 inside); ``V`` is the inward normal
  field. Every sample is splatted with quadratic B-splines (as PoissonRecon,
  ``NORMAL_DEGREE = 2``) at the depth where an octree cell holds about
  ``samples_per_node`` samples (``spn``), interpolating between the two
  nearest depths; the octree is not refined beyond that depth.
* ``chi`` is represented in a *hierarchical* basis: the sum, over octree
  depths, of trilinear (first-order B-spline) hat functions on the grid of
  spacing ``2^-d``. Depths ``<= full_depth`` are complete grids; finer depths
  only keep nodes near the samples (adaptive).
* Screening: at depth ``d`` the samples are gathered per cell ``c`` (their
  centroid ``p_c`` and total weight ``W_c``) and weighted by ``2^d``, as in
  PoissonRecon.
* Like PoissonRecon's cascadic multigrid, the depths are solved coarse to
  fine; each depth solves (with conjugate gradients) for the correction left
  over by the coarser depths.
* The iso-value is the weighted average of ``chi`` at the samples, and the
  surface is extracted on the leaves of the octree (as in PoissonRecon,
  surfaces far from the samples are extracted too).

The SciPy KD-tree gives the sampling density (the area ``Area/N`` covered by
each sample, from its nearest neighbours), from which the splatting depths
are derived, and the per-vertex density returned along with the mesh. All
sparse data (octree nodes, normal field) are sorted integer keys of the grid
of each depth; the B-spline operations are separable, one axis at a time.

Known differences from the C++ code: the density comes from k-NN instead of
a splatted kernel density, linear systems are solved by conjugate gradients
instead of Gauss-Seidel, the active nodes are those within two cells of the
samples, and the surface is extracted with marching tetrahedra (vertices
refined by one regula falsi step) instead of marching cubes: about 3x more
triangles, and possibly small cracks between leaves of different depths.

Example
-------
>>> from repsr.psr import poisson_reconstruction
>>> vertices, triangles, densities = poisson_reconstruction(
...     points, normals, depth=8, point_weight=10.0)
"""

from itertools import permutations

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import cg
from scipy.spatial import cKDTree

# corner c of a cell sits at offset ((c >> 0) & 1, (c >> 1) & 1, (c >> 2) & 1)
_CORNERS = np.array([[(c >> a) & 1 for a in range(3)] for c in range(8)])
# the 27 offsets {-1, 0, 1}^3
_OFFSETS = np.stack(np.meshgrid(
    [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij'), -1).reshape(-1, 3)
# Freudenthal (Kuhn) split of a cube into 6 tetrahedra; conforming on a grid
_TETS = np.array([
    [0, 1 << a, (1 << a) | (1 << b), 7] for a, b in permutations(range(3), 2)
])
# refinement of a quadratic B-spline: cell j -> cells 2j + shift
_REFINE = np.array([-1, 0, 1, 2]), np.array([1.0, 3.0, 3.0, 1.0]) / 4


def _quadratic(t: np.ndarray) -> np.ndarray:
    """Quadratic B-spline centred at 0 (support ``(-1.5, 1.5)``)."""
    t = np.abs(t)
    return np.where(t < 0.5, 0.75 - t ** 2,
                    np.where(t < 1.5, 0.5 * (1.5 - t) ** 2, 0.0))


def _hat_quadratic_integrals():
    """
    On the unit grid, the integrals of the hat function of node 0 (and of its
    derivative) against the quadratic B-spline of cell ``o`` (centred at
    ``o + 1/2``), for the overlapping cells ``o = -2 .. 1``. Exact (3-point
    Gauss quadrature on each unit interval of the hat).
    """
    gx, gw = np.polynomial.legendre.leggauss(3)
    x = np.concatenate([(gx - 1) / 2, (gx + 1) / 2])
    w = np.concatenate([gw, gw]) / 2
    cells = np.arange(-2, 2)
    b = _quadratic(x[None] - cells[:, None] - 0.5) * w
    return cells, b @ (1 - np.abs(x)), b @ -np.sign(x)


_HAT_QUADRATIC = _hat_quadratic_integrals()


def _refined(table: np.ndarray) -> np.ndarray:
    """``table`` (over cells -2 .. 1) for a quadratic B-spline of the coarser
    grid, refined on the fly: node ``m`` and coarse cell ``j`` interact when
    ``m - 2 j`` is in -2 .. 4."""
    out = np.zeros(7)
    for shift, weight in zip(*_REFINE):
        for cell, value in zip(_HAT_QUADRATIC[0], table):
            out[shift - cell + 2] += weight * value
    return out


def _tet_table() -> np.ndarray:
    """Marching tetrahedra table: code -> 2 triangles x 3 edges x 2 ends."""
    table = -np.ones((16, 2, 3, 2), dtype=np.int64)
    for code in range(16):
        ins = [i for i in range(4) if (code >> i) & 1]
        out = [i for i in range(4) if not (code >> i) & 1]
        if len(ins) in (1, 3):
            a = ins[0] if len(ins) == 1 else out[0]
            table[code, 0] = [[a, o] for o in range(4) if o != a]
        elif len(ins) == 2:
            (a, b), (c, d) = ins, out
            table[code, 0] = [[a, c], [a, d], [b, d]]
            table[code, 1] = [[a, c], [b, d], [b, c]]
    return table


_TET_TABLE = _tet_table()


def _encode(ijk: np.ndarray, m: int) -> np.ndarray:
    """Integer key of grid coordinates ``ijk`` on an ``m^3`` grid."""
    ijk = ijk.astype(np.int64, copy=False)
    return (ijk[..., 0] * m + ijk[..., 1]) * m + ijk[..., 2]


def _decode(keys: np.ndarray, m: int) -> np.ndarray:
    """Inverse of :func:`_encode`."""
    return np.stack([keys // (m * m), (keys // m) % m, keys % m], -1)


def _lookup(sorted_keys: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Index of every ``query`` key in ``sorted_keys``, -1 if absent."""
    if len(sorted_keys) == 0:
        return -np.ones(query.shape, dtype=np.int64)
    idx = np.minimum(np.searchsorted(sorted_keys, query), len(sorted_keys) - 1)
    return np.where(sorted_keys[idx] == query, idx, -1)


def _sort(keys: np.ndarray):
    """Sorting order of ``keys`` and the start of every run of equal keys
    (faster than ``np.unique`` with inverse)."""
    order = np.argsort(keys)
    keys = keys[order]
    start = np.r_[True, keys[1:] != keys[:-1]][:len(keys)]
    return order, np.flatnonzero(start)


def _unique(keys: np.ndarray) -> np.ndarray:
    """Sorted unique keys."""
    keys = np.sort(keys)
    return keys[np.r_[True, keys[1:] != keys[:-1]]] if len(keys) else keys


def _merge(keys: np.ndarray, vals: np.ndarray):
    """Sum the values of duplicated keys (sparse grid data, sorted keys)."""
    if len(keys) == 0:
        return keys, vals
    order, start = _sort(keys)
    return keys[order[start]], np.add.reduceat(vals[order], start, axis=0)


# Sparse grid data of depth d: values at sorted keys of the grid of
# (2^d + 1)^3 nodes (the cells of the depth use the same keys as their
# lowest corner).

def _convolve(keys, vals, m, axis, shifts, weights, size, refine=False):
    """
    One axis of a separable convolution of sparse data on the grid ``m^3``:
    the value at a grid point is added, times ``weights[k]``, ``shifts[k]``
    further along ``axis`` (after doubling the coordinate when ``refine``);
    the points outside ``[0, size)`` are dropped.
    """
    stride = m ** (2 - axis)
    coord = keys // stride % m
    if refine:
        keys, coord = keys + coord * stride, 2 * coord
    out_keys, out_vals = [], []
    for shift, weight in zip(shifts, weights):
        ok = (coord + shift >= 0) & (coord + shift < size)
        out_keys.append(keys[ok] + shift * stride)
        out_vals.append(vals[ok] * weight)
    return _merge(np.concatenate(out_keys), np.concatenate(out_vals))


def _splat(x: np.ndarray, v: np.ndarray, depth: int):
    """Splat the vectors ``v`` at ``x`` with the quadratic B-splines of the
    cells of a depth, normalised to a unit integral."""
    res = 1 << depth
    g = x * res
    cell = np.clip(np.floor(g).astype(np.int64), 0, res - 1)
    t = g - cell - 0.5
    w = np.stack([0.5 * (0.5 - t) ** 2, 0.75 - t ** 2, 0.5 * (0.5 + t) ** 2])
    keys = _encode(cell, res + 1)[None] + _encode(_OFFSETS, res + 1)[:, None]
    weight = (w[_OFFSETS[:, 0] + 1, :, 0] * w[_OFFSETS[:, 1] + 1, :, 1] *
              w[_OFFSETS[:, 2] + 1, :, 2]).reshape(-1, 1)
    vals = np.tile(v * res ** 3, (len(_OFFSETS), 1)) * weight
    moved = cell[None] + _OFFSETS[:, None]
    ok = np.all((moved >= 0) & (moved < res), axis=2).ravel()
    return _merge(keys.ravel()[ok], vals[ok])


def _divergence(keys: np.ndarray, field: np.ndarray, res: int,
                refine: bool = False):
    """
    Project the quadratic B-spline field (coefficients ``field`` on the cells
    ``keys`` of the grid ``res``, or of the grid ``res / 2`` when ``refine``)
    on the gradients of the hat functions of the grid ``res``: returns the
    node keys and ``<grad phi_m, V>``. Separable:
    ``(G x M x M) V_x + (M x G x M) V_y + (M x M x G) V_z``.
    """
    cells, mass, grad = _HAT_QUADRATIC
    shifts = -cells
    if refine:
        shifts, mass, grad = np.arange(-2, 5), _refined(mass), _refined(grad)
    for axis in range(3):
        weights = np.where(np.arange(3) == axis, grad[:, None],
                           mass[:, None] / res)
        keys, field = _convolve(keys, field, res + 1, axis, shifts, weights,
                                res + 1, refine)
    return keys, field.sum(1)


def _restrict(keys: np.ndarray, vals: np.ndarray, res: int):
    """
    Restrict values on the hat functions of the grid ``2 res`` to the hats of
    the grid ``res``: a coarse hat is the sum of the fine hats with weights
    1/2, 1, 1/2 along every axis.
    """
    m = 2 * res + 1
    for axis in range(3):
        stride = m ** (2 - axis)
        coord = keys // stride % m
        half = keys - (coord - coord // 2) * stride
        odd = coord % 2 == 1
        keys, vals = _merge(
            np.concatenate([half, half[odd] + stride]),
            np.concatenate([np.where(odd, 0.5, 1.0) * vals, 0.5 * vals[odd]]))
    return _encode(_decode(keys, m), res + 1), vals


def _basis(x: np.ndarray, res: int):
    """
    Hat functions of a depth that are non-zero at the points ``x``.

    Returns the keys (on the node grid ``res + 1``) of the 8 corners of the
    cell containing each point and the values of their hat functions, both
    ``(n, 8)``.
    """
    g = x * res
    cell = np.clip(np.floor(g).astype(np.int64), 0, res - 1)
    f = g - cell
    w = np.stack([1 - f, f], -1)
    values = (w[:, 0, _CORNERS[:, 0]] * w[:, 1, _CORNERS[:, 1]] *
              w[:, 2, _CORNERS[:, 2]])
    return _encode(cell, res + 1)[:, None] + _encode(_CORNERS, res + 1), values


def _dilate(cells: np.ndarray, m: int, shifts) -> np.ndarray:
    """Sorted keys (on the grid ``m``) of the cells ``cells + s`` for every
    ``s`` in ``shifts^3``, e.g. ``range(-1, 2)`` for the cells within one
    cell (separable: one axis at a time)."""
    keys = _unique(_encode(cells, m))
    for axis, stride in enumerate((m * m, m, 1)):
        coord = _decode(keys, m)[:, axis]
        keys = _unique(np.concatenate([
            keys[(coord + s >= 0) & (coord + s < m)] + s * stride
            for s in shifts]))
    return keys


def _gather(x: np.ndarray, w: np.ndarray, res: int, region=None):
    """Centroids and total weights of the samples per cell of a depth,
    keeping only the cells of ``region`` (sorted keys) when given."""
    keys = _encode(np.clip(np.floor(x * res).astype(np.int64), 0, res - 1),
                   res)
    if region is not None:
        keep = _lookup(region, keys) >= 0
        x, w, keys = x[keep], w[keep], keys[keep]
    _, inv = np.unique(keys, return_inverse=True)
    total = np.bincount(inv, w)
    centroid = np.stack([np.bincount(inv, w * x[:, i]) for i in range(3)], 1)
    return centroid / total[:, None], total


class _Depth:
    """Active nodes (sorted keys) and solved coefficients of one depth."""

    def __init__(self, depth: int, node_keys: np.ndarray):
        self.depth = depth
        self.res = 1 << depth
        self.keys = node_keys
        self.coeffs = np.zeros(len(node_keys))
        self.complete = len(node_keys) == (self.res + 1) ** 3

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Value of this depth's part of the indicator function at ``x``."""
        idx, values = _basis(x, self.res)
        if not self.complete:
            idx = _lookup(self.keys, idx)
        return (np.where(idx >= 0, self.coeffs[idx], 0.0) * values).sum(1)


def _evaluate(depths: list, x: np.ndarray) -> np.ndarray:
    """Value of the indicator function (sum over ``depths``) at ``x``."""
    out = np.zeros(len(x))
    for dep in depths:
        out += dep.evaluate(x)
    return out


def _stiffness(nodes: np.ndarray, offset: np.ndarray, res: int) -> np.ndarray:
    """
    Entries ``<grad phi_n, grad phi_{n + offset}>`` of the trilinear stiffness
    matrix on the grid of spacing ``h = 1 / res``: tensor products of the 1D
    mass and stiffness matrices, whose diagonal is halved on the domain
    boundary (Neumann condition).
    """
    h = 1.0 / res
    end = (nodes == 0) | (nodes == res)
    m1 = np.where(offset == 0, np.where(end, 2.0, 4.0) * h / 6, h / 6)
    k1 = np.where(offset == 0, np.where(end, 1.0, 2.0) / h, -1.0 / h)
    return (k1[:, 0] * m1[:, 1] * m1[:, 2] + m1[:, 0] * k1[:, 1] * m1[:, 2] +
            m1[:, 0] * m1[:, 1] * k1[:, 2])


def _solve_depth(
    dep: _Depth, coarser, points: np.ndarray, weights: np.ndarray,
    divergence: tuple, cg_tol: float, cg_iters: int
) -> _Depth:
    """
    Solve for the coefficients of ``dep``, with the screening ``points``
    (cell centroids) and ``weights`` of the depth, given ``coarser``: the
    indicator function of the coarser depths, as a trilinear function of
    the previous depth (``None`` for depth 0). Returns the indicator
    function up to this depth, on the nodes around the active ones.
    """
    res, keys = dep.res, dep.keys
    n = len(keys)
    nodes = _decode(keys, res + 1)
    ring = _dilate(nodes, res + 1, range(-1, 2))
    chi_ring = np.zeros(len(ring)) if coarser is None else \
        coarser.evaluate(_decode(ring, res + 1) / res)
    in_ring = _lookup(ring, keys)
    active = -np.ones(len(ring), dtype=np.int32)
    active[in_ring] = np.arange(n)

    # Stiffness matrix (built row by row: at most 27 entries per row, so the
    # CSR arrays need no sorting) and the coarser solution's gradient term.
    # Only the nodes on the domain boundary miss neighbours or have other
    # stencil values.
    cols = np.empty((n, len(_OFFSETS)), dtype=np.int32)
    vals = np.empty((n, len(_OFFSETS)))
    rhs = np.zeros(n)
    edge = np.flatnonzero(np.any((nodes == 0) | (nodes == res), axis=1))
    for o, offset in enumerate(_OFFSETS):
        if offset[2] == -1:
            # The ring holds whole z-runs: one lookup for the three offsets
            # (dx, dy, -1 / 0 / 1), whose keys are consecutive.
            middle = _lookup(ring, keys + _encode(offset + (0, 0, 1), res + 1))
        k = np.full(n, _stiffness(np.ones((1, 3)), offset, res)[0])
        k[edge] = _stiffness(nodes[edge], offset, res)
        nb = nodes[edge] + offset
        k[edge[np.any((nb < 0) | (nb > res), axis=1)]] = 0
        nb_ring = np.where(k != 0, middle + offset[2], 0)
        rhs -= k * chi_ring[nb_ring]
        cols[:, o] = np.where(k != 0, active[nb_ring], -1)
        vals[:, o] = k
    stored = cols >= 0
    matrix = sps.csr_matrix(
        (vals[stored], cols[stored],
         np.concatenate([[0], np.cumsum(stored.sum(1))])), shape=(n, n))

    # Divergence of the normal field <grad phi_n, V> (given on the nodes).
    idx = _lookup(keys, divergence[0])
    rhs += np.bincount(idx[idx >= 0], divergence[1][idx >= 0], minlength=n)

    # Screening matrix and screening targets.
    corners, values = _basis(points, res)
    idx = _lookup(keys, corners)
    assert np.all(idx >= 0), "samples must be covered by the active nodes"
    screening = sps.coo_matrix((
        (weights[:, None, None] * values[:, :, None] * values[:, None, :]
         ).ravel(),
        (np.repeat(idx, 8, axis=1).ravel(), np.tile(idx, (1, 8)).ravel())),
        shape=(n, n))
    chi = 0.0 if coarser is None else coarser.evaluate(points)
    target = (weights * (0.5 - chi))[:, None] * values
    rhs += np.bincount(idx.ravel(), target.ravel(), minlength=n)

    matrix = matrix + screening.tocsr()
    diag = matrix.diagonal()
    precond = sps.diags(1.0 / np.where(diag > 0, diag, 1.0))
    dep.coeffs, _ = cg(matrix, rhs, rtol=cg_tol, maxiter=cg_iters, M=precond)
    total = _Depth(dep.depth, ring)
    total.coeffs = chi_ring
    total.coeffs[in_ring] += dep.coeffs
    return total


def _marching_tetrahedra(cell_keys: np.ndarray, res: int, nodal, function,
                         iso: float):
    """
    Iso-surface in the given cells of a depth of a function given by its
    values at the nodes (``nodal``: node keys -> values) and at any point
    (``function``: positions -> values).
    """
    corners = _encode(_decode(cell_keys, res), res + 1)[:, None] + \
        _encode(_CORNERS, res + 1)
    val = nodal(corners) - iso
    crossing = (val > 0).any(1) & (val <= 0).any(1)
    corners, val = corners[crossing], val[crossing]
    inside = val[:, _TETS] > 0
    cell, tet = np.nonzero(inside.any(2) & ~inside.all(2))
    tets, inside = _TETS[tet], inside[cell, tet]
    code = (inside * (1 << np.arange(4))).sum(1)

    # The edges of the tetrahedra go from a corner to a corner above it:
    # an edge is its lower node and the bits of the corner offset.
    edges, values, outward = [], [], []
    for slot in range(2):
        e = _TET_TABLE[code, slot]
        has = e[:, 0, 0] >= 0
        t, ins, c = tets[has], inside[has], cell[has, None]
        u = np.take_along_axis(t, e[has][..., 0], axis=1)
        w = np.take_along_axis(t, e[has][..., 1], axis=1)
        low, high = np.minimum(u, w), np.maximum(u, w)
        edges.append(corners[c, low] * 8 + (low ^ high))
        values.append(np.stack([val[c, low], val[c, high]], -1))
        p = _CORNERS[t]
        c_in = (p * ins[..., None]).sum(1) / ins.sum(1, keepdims=True)
        c_out = (p * ~ins[..., None]).sum(1) / (~ins).sum(1, keepdims=True)
        outward.append(c_out - c_in)
    edges = np.concatenate(edges).ravel()
    order, start = _sort(edges)
    first = order[start]
    triangles = np.empty(len(edges), dtype=np.int64)
    triangles[order] = np.cumsum(np.r_[True, np.diff(edges[order]) != 0]) - 1
    edges, triangles = edges[first], triangles.reshape(-1, 3)
    va, vb = np.concatenate(values).reshape(-1, 2)[first].T
    pa = _decode(edges // 8, res + 1) / res
    step = _CORNERS[edges % 8] / res

    # Linear interpolation, then one regula falsi step with the function
    # itself, which is not linear along the diagonals of the cells.
    t = va / (va - vb)
    vt = function(pa + t[:, None] * step) - iso
    left = np.sign(vt) != np.sign(va)
    t = np.where(left, t * va / (va - vt), t + (1 - t) * vt / (vt - vb))
    vertices = pa + np.nan_to_num(t, nan=0.5)[:, None] * step

    tri = vertices[triangles]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = np.einsum('ij,ij->i', normal, np.concatenate(outward)) < 0
    triangles[flip] = triangles[flip][:, ::-1]
    return vertices, triangles


def poisson_reconstruction(
    points: np.ndarray,
    normals: np.ndarray,
    depth: int = 8,
    full_depth: int = 5,
    samples_per_node: float = 1.5,
    point_weight: float = 2.0,
    scale: float = 1.1,
    k_area: int = 16,
    cg_tol: float = 1e-3,
    cg_iters: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Screened Poisson surface reconstruction of an oriented point cloud.

    Parameters
    ----------
    points : np.ndarray
        ``(n, 3)`` sample positions
    normals : np.ndarray
        ``(n, 3)`` outward normals (need not be unit length)
    depth : int, optional
        Maximum octree depth (grid of at most ``2^depth`` cells per axis),
        by default 8. As in PoissonRecon, sparse regions stop refining earlier.
    full_depth : int, optional
        Depths up to this one are complete grids, by default 5
    samples_per_node : float, optional
        Number of samples per octree cell at the splatting depth, by default
        1.5 (as Open3D)
    point_weight : float, optional
        Screening weight alpha, by default 2.0 (as Open3D / PoissonRecon
        with first-order elements; iPSR uses 10)
    scale : float, optional
        Ratio between the reconstruction cube and the bounding box,
        by default 1.1
    k_area : int, optional
        Neighbours used to estimate the area of each sample, by default 16
    cg_tol, cg_iters : float, int, optional
        Relative tolerance and maximum iterations of the conjugate gradient
        solve of every depth (PoissonRecon also solves approximately, with 8
        Gauss-Seidel sweeps per depth)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Vertices ``(v, 3)``, triangles ``(f, 3)`` with outward orientation,
        and the number of input samples within two finest cells of every
        vertex (to trim low density regions, as Open3D's densities).
    """

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)

    # Map the points into the unit cube.
    low, high = points.min(0), points.max(0)
    size = (high - low).max() * scale
    origin = (low + high) / 2 - size / 2
    x = (points - origin) / size

    # Area of each sample from its k nearest neighbours (sampling density).
    tree = cKDTree(x)
    k = min(k_area, len(x) - 1)
    if k > 0:
        area = np.pi * tree.query(x, k + 1, workers=-1)[0][:, -1] ** 2 / k
    else:
        area = np.ones(len(x))
    unit = normals / np.maximum(
        np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    # Merge the samples falling in the same finest cell (as PoissonRecon):
    # summed normal direction, magnitude spn * area.
    res = 1 << depth
    cells = np.clip(np.floor(x * res).astype(np.int64), 0, res - 1)
    _, group = np.unique(_encode(cells, res), return_inverse=True)
    count = np.bincount(group).astype(np.float64)
    samples = np.stack([np.bincount(group, x[:, i]) for i in range(3)], 1)
    samples /= count[:, None]
    direction = np.stack([np.bincount(group, unit[:, i]) for i in range(3)], 1)
    length = np.linalg.norm(direction, axis=1)
    sample_area = samples_per_node * np.bincount(group, area)
    inward = -direction / np.maximum(length, 1e-12)[:, None] * \
        sample_area[:, None] * (length > 1e-12)[:, None]
    screen = point_weight * samples_per_node * area.mean()

    # Splatting depth: where a cell holds samples_per_node samples, split
    # between the two nearest depths; no refinement beyond it.
    level = np.clip(-np.log(sample_area / count) / np.log(4), 0, depth)
    top = np.maximum(np.ceil(level).astype(np.int64), 0)
    frac = np.where(top > 0, 1 - (top - level), 1.0)
    finest = int(top.max())
    full_depth = min(full_depth, finest)

    # Active nodes of every depth: complete up to full_depth, then the
    # corners of the cells within two cells of a sample refined to the depth
    # (whose right-hand side needs the normal field of the coarser depth on
    # the parent cells within three: ``feed``).
    regions, feed, depths = {}, {}, []
    for d in range(finest + 1):
        r = 1 << d
        if d <= full_depth:
            regions[d] = np.arange(r ** 3, dtype=np.int64)
            node_keys = np.arange((r + 1) ** 3, dtype=np.int64)
        else:
            occupied = np.clip(np.floor(samples[top >= d] * r), 0, r - 1)
            regions[d] = _dilate(occupied, r, range(-2, 3))
            feed[d] = _dilate(occupied // 2, r // 2 + 1, range(-3, 4))
            node_keys = _dilate(_decode(regions[d], r), r + 1, (0, 1))
        depths.append(_Depth(d, node_keys))

    # Normal field: the sum of the quadratic splats of every depth. The
    # right-hand side of depth d, <grad phi, V>, adds the projection of the
    # splats of finer depths (restricted depth by depth) and of the coarser
    # ones (refined depth by depth, only near the active cells of the depth).
    # Both are exact.
    splats = []
    for d in range(finest + 1):
        here = np.concatenate([top == d, top == d + 1])
        weight = np.concatenate([frac, 1 - frac])[here]
        splats.append(_splat(
            np.concatenate([samples, samples])[here],
            np.concatenate([inward, inward])[here] * weight[:, None], d))
    divergence = [None] * (finest + 1)
    for d in range(finest, -1, -1):
        part = _divergence(*splats[d], 1 << d)
        if d < finest:
            finer = _restrict(*divergence[d + 1], 1 << d)
            part = _merge(np.concatenate([part[0], finer[0]]),
                          np.concatenate([part[1], finer[1]]))
        divergence[d] = part
    field = splats[0]
    for d in range(1, finest + 1):
        r = 1 << d
        if d > full_depth:
            keep = _lookup(feed[d], field[0]) >= 0
            field = field[0][keep], field[1][keep]
        field = _encode(_decode(field[0], r // 2 + 1), r + 1), field[1]
        part = _divergence(*field, r, refine=True)
        keep = _lookup(depths[d].keys, part[0]) >= 0
        divergence[d] = _merge(
            np.concatenate([divergence[d][0], part[0][keep]]),
            np.concatenate([divergence[d][1], part[1][keep]]))
        if d < finest:
            for axis in range(3):
                field = _convolve(*field, r + 1, axis, *_REFINE, r, True)
            field = _merge(np.concatenate([field[0], splats[d][0]]),
                           np.concatenate([field[1], splats[d][1]]))

    # Cascadic solve, coarse to fine; ``totals[d]`` is the indicator function
    # up to depth d, as a trilinear function of that depth near its active
    # nodes (the complete depths are thus collapsed into one).
    totals = []
    for d, dep in enumerate(depths):
        centroid, weight = _gather(samples, count, dep.res,
                                   regions[d] if d > full_depth else None)
        totals.append(_solve_depth(
            dep, totals[-1] if d else None, centroid,
            screen * dep.res * weight, divergence[d], cg_tol, cg_iters))
    del splats, divergence, field
    solved = [totals[full_depth]] + depths[full_depth + 1:]
    iso = np.average(_evaluate(solved, samples), weights=count)

    # Extract the surface on the leaves of the octree: on a leaf of depth d,
    # the indicator function is totals[d] plus the finer depths.
    verts, tris, offset = [], [], 0
    leaves = regions[full_depth]
    for d in range(full_depth, finest + 1):
        r = 1 << d
        if d < finest:
            parents = _encode(_decode(regions[d + 1], 2 * r) // 2, r)
            split = np.isin(leaves, parents)
            children = _decode(leaves[split], r)[:, None, :] * 2 + _CORNERS
            next_leaves = _encode(children, 2 * r).ravel()
            leaves = leaves[~split]
        else:
            next_leaves = None

        def function(x, d=d):
            return totals[d].evaluate(x) + _evaluate(depths[d + 1:], x)

        def nodal(k, d=d, r=r):
            # corners 4..7 are above corners 0..3: the next keys of the ring
            idx = _lookup(totals[d].keys, k[:, :4])
            values = totals[d].coeffs[np.concatenate([idx, idx + 1], 1)]
            if d < finest:
                nodes = _unique(k.ravel())
                values += _evaluate(depths[d + 1:], _decode(
                    nodes, r + 1) / r)[_lookup(nodes, k)]
            return values

        v, t = _marching_tetrahedra(leaves, r, nodal, function, iso)
        verts.append(v)
        tris.append(t + offset)
        offset += len(v)
        leaves = next_leaves

    vertices = np.concatenate(verts) if verts else np.zeros((0, 3))
    triangles = np.concatenate(tris) if tris else np.zeros((0, 3), np.int64)
    densities = tree.query_ball_point(
        vertices, r=2.0 / (1 << finest), return_length=True,
        workers=-1).astype(np.float64)

    return vertices * size + origin, triangles, densities
