# This file is developed by Amin Darabi as part of the project of
# the course "Geometric Modeling and shape analysis" (IFT6113) at the
# Université de Montréal (UdeM) in Fall 2024.
"""
Screened Poisson Surface Reconstruction in pure NumPy / SciPy.

A compact re-implementation of Screened Poisson Surface Reconstruction
(Kazhdan & Hoppe, 2013), written to be read and compared against the
official C++ implementation (PoissonRecon, used through Open3D).

Formulation (same conventions as PoissonRecon / iPSR, Neumann boundary):

    E(chi) = int |grad chi - V|^2
             + alpha * (Area / N) * sum_i w_i (chi(p_i) - 1/2)^2

* ``chi`` is the indicator function (1 inside, 0 outside); ``V`` is the
  inward normal field: as in PoissonRecon, every sample splats ``-a_i n_i``
  (``a_i``: surface area represented by sample ``i``) into the trilinear
  basis of the finest depth, ``V = sum_m V_m phi_m``.
* ``chi`` is represented in a *hierarchical* basis: the sum, over octree
  depths ``d = 0 .. depth``, of trilinear (first-order B-spline) hat
  functions on the grid of spacing ``2^-d``. Depths ``<= full_depth`` are
  complete grids; finer depths only keep nodes near the samples (adaptive).
* Like PoissonRecon's cascadic multigrid, the depths are solved coarse to
  fine; each depth solves (with conjugate gradients) for the correction left
  over by the coarser depths.
* The iso-value is the weighted average of ``chi`` at the samples, and the
  surface is extracted with marching tetrahedra on the leaves of the octree
  (so, as in PoissonRecon, surfaces far away from the samples are extracted
  too, at a coarse resolution).

The SciPy KD-tree is used to estimate the area ``a_i`` of each sample from
its k nearest neighbours (density normalisation) and to compute the
per-vertex sample density returned along with the mesh.

Known differences from the C++ code: first-order (trilinear) elements only,
normals are always splatted at the finest depth (PoissonRecon adapts the
splatting depth to the density), the sample area comes from k-NN instead of
a splatted kernel density, the surface is extracted with marching
tetrahedra instead of marching cubes, and it may have small cracks between
octree leaves of different depths.

Example
-------
>>> import numpy as np
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
# the 27 node offsets of the support of a trilinear hat function
_OFFSETS = np.stack(np.meshgrid(
    [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij'), -1).reshape(-1, 3)
# Freudenthal (Kuhn) split of a cube into 6 tetrahedra; conforming on a grid
_TETS = np.array([
    [0, 1 << a, (1 << a) | (1 << b), 7] for a, b in permutations(range(3), 2)
])


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


def _dilate(cells: np.ndarray, res: int) -> np.ndarray:
    """Sorted keys of the cells within one cell (Chebyshev) of ``cells``."""
    nb = (cells[:, None, :] + _OFFSETS[None]).reshape(-1, 3)
    nb = nb[np.all((nb >= 0) & (nb < res), axis=1)]
    return np.unique(_encode(nb, res))


def _basis(x: np.ndarray, res: int):
    """
    Hat functions of a depth that are non-zero at the points ``x``.

    Returns the node coordinates of the 8 corners of the cell containing each
    point ``(n, 8, 3)`` and the values of their hat functions ``(n, 8)``.
    """
    g = x * res
    cell = np.clip(np.floor(g).astype(np.int64), 0, res - 1)
    f = g - cell
    factors = np.where(_CORNERS[None], f[:, None, :], 1 - f[:, None, :])
    values = factors[..., 0] * factors[..., 1] * factors[..., 2]
    return cell[:, None, :] + _CORNERS[None], values


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
        corners, values = _basis(x, self.res)
        idx = _encode(corners, self.res + 1)
        if not self.complete:
            idx = _lookup(self.keys, idx)
        return (np.where(idx >= 0, self.coeffs[idx], 0.0) * values).sum(1)


def _evaluate(depths: list, x: np.ndarray) -> np.ndarray:
    """Value of the indicator function (sum over ``depths``) at ``x``."""
    out = np.zeros(len(x))
    for dep in depths:
        out += dep.evaluate(x)
    return out


def _factors(nodes: np.ndarray, offset: np.ndarray, res: int):
    """
    Per-axis 1D integrals between the hat of every node and the hat of
    ``node + offset`` on the grid of spacing ``h = 1 / res``: mass
    ``int phi phi``, stiffness ``int phi' phi'`` and ``int phi' phi``.
    Nodes on the domain boundary only keep half of their support (Neumann).
    """
    h = 1.0 / res
    low, high = nodes == 0, nodes == res
    end = low | high
    m1 = np.where(offset == 0, np.where(end, 2.0, 4.0) * h / 6, h / 6)
    k1 = np.where(offset == 0, np.where(end, 1.0, 2.0) / h, -1.0 / h)
    g1 = np.where(offset == 0, 0.5 * high - 0.5 * low, -0.5 * offset)
    return m1, k1, g1


def _stiffness(nodes: np.ndarray, offset: np.ndarray, res: int) -> np.ndarray:
    """Entries ``<grad phi_n, grad phi_{n + offset}>`` (tensor products)."""
    m1, k1, _ = _factors(nodes, offset, res)
    return (k1[:, 0] * m1[:, 1] * m1[:, 2] + m1[:, 0] * k1[:, 1] * m1[:, 2] +
            m1[:, 0] * m1[:, 1] * k1[:, 2])


def _divergence(nodes: np.ndarray, field: np.ndarray, res: int):
    """
    Project the vector field ``sum_q field_q phi_q`` (nodes ``q`` of the grid
    of spacing ``1 / res``) on the hat gradients: returns the nodes ``m`` and
    the values ``<grad phi_m, V>``, i.e. the right-hand side of the Poisson
    equation at that depth.
    """
    keys, vals = [], []
    for offset in _OFFSETS:
        target = nodes - offset
        ok = np.all((target >= 0) & (target <= res), axis=1)
        m1, _, g1 = _factors(target[ok], offset, res)
        grad_mass = np.stack([g1[:, 0] * m1[:, 1] * m1[:, 2],
                              m1[:, 0] * g1[:, 1] * m1[:, 2],
                              m1[:, 0] * m1[:, 1] * g1[:, 2]], 1)
        keys.append(_encode(target[ok], res + 1))
        vals.append((grad_mass * field[ok]).sum(1))
    keys, inv = np.unique(np.concatenate(keys), return_inverse=True)
    return _decode(keys, res + 1) / res, np.bincount(inv, np.concatenate(vals))


def _solve_depth(
    dep: _Depth, coarser: list, samples: np.ndarray, weights: np.ndarray,
    chi_samples: np.ndarray, screen: float, divergence: tuple,
    cg_tol: float, cg_iters: int
) -> None:
    """Solve for the coefficients of ``dep`` given the coarser solution."""
    res, keys = dep.res, dep.keys
    n = len(keys)
    nodes = _decode(keys, res + 1)

    # Stiffness matrix and the coarser solution's gradient term.
    neighbours = (nodes[:, None, :] + _OFFSETS[None]).reshape(-1, 3)
    neighbours = neighbours[np.all((neighbours >= 0) &
                                   (neighbours <= res), axis=1)]
    ring = np.unique(_encode(neighbours, res + 1))
    chi_ring = _evaluate(coarser, _decode(ring, res + 1) / res)
    rows, cols, vals = [], [], []
    rhs = np.zeros(n)
    for offset in _OFFSETS:
        nb = nodes + offset
        inside = np.all((nb >= 0) & (nb <= res), axis=1)
        k = _stiffness(nodes[inside], offset, res)
        nb_keys = _encode(nb[inside], res + 1)
        rhs[inside] -= k * chi_ring[_lookup(ring, nb_keys)]
        col = _lookup(keys, nb_keys)
        ok = col >= 0
        rows.append(np.nonzero(inside)[0][ok])
        cols.append(col[ok])
        vals.append(k[ok])

    # Divergence of the normal field: the finest-depth projection restricted
    # to this depth (a coarse hat is exactly a sum of finest hats).
    corners, values = _basis(divergence[0], res)
    idx = _lookup(keys, _encode(corners, res + 1))
    ok = idx >= 0
    rhs += np.bincount(idx[ok], (values * divergence[1][:, None])[ok],
                       minlength=n)

    # Screening matrix and screening targets.
    corners, values = _basis(samples, res)
    idx = _lookup(keys, _encode(corners, res + 1))
    assert np.all(idx >= 0), "samples must be covered by the active nodes"
    sw = screen * weights
    for a in range(8):
        for b in range(8):
            rows.append(idx[:, a])
            cols.append(idx[:, b])
            vals.append(sw * values[:, a] * values[:, b])
    target = sw[:, None] * values * (0.5 - chi_samples[:, None])
    rhs += np.bincount(idx.ravel(), target.ravel(), minlength=n)

    matrix = sps.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n)
    )
    diag = matrix.diagonal()
    precond = sps.diags(1.0 / np.where(diag > 0, diag, 1.0))
    dep.coeffs, _ = cg(matrix, rhs, rtol=cg_tol, maxiter=cg_iters, M=precond)


def _marching_tetrahedra(cell_keys: np.ndarray, res: int, depths: list,
                         iso: float):
    """Iso-surface of the indicator function in the given cells of a depth."""
    cells = _decode(cell_keys, res)
    corner_keys = _encode(cells[:, None, :] + _CORNERS[None], res + 1)
    nodes, inv = np.unique(corner_keys, return_inverse=True)
    inv = inv.reshape(-1, 8)
    pos = _decode(nodes, res + 1) / res
    val = _evaluate(depths, pos) - iso

    tets = inv[:, _TETS].reshape(-1, 4)
    inside = val[tets] > 0
    crossing = inside.any(1) & ~inside.all(1)
    tets, inside = tets[crossing], inside[crossing]
    code = (inside * (1 << np.arange(4))).sum(1)

    ends_u, ends_w, outward = [], [], []
    for slot in range(2):
        edges = _TET_TABLE[code, slot]
        has = edges[:, 0, 0] >= 0
        t, e, ins = tets[has], edges[has], inside[has]
        ends_u.append(np.take_along_axis(t, e[..., 0], axis=1))
        ends_w.append(np.take_along_axis(t, e[..., 1], axis=1))
        p = pos[t]
        c_in = (p * ins[..., None]).sum(1) / ins.sum(1, keepdims=True)
        c_out = (p * ~ins[..., None]).sum(1) / (~ins).sum(1, keepdims=True)
        outward.append(c_out - c_in)
    u, w = np.concatenate(ends_u), np.concatenate(ends_w)
    outward = np.concatenate(outward)

    edge_keys = np.minimum(u, w) * len(nodes) + np.maximum(u, w)
    unique_edges, triangles = np.unique(edge_keys, return_inverse=True)
    triangles = triangles.reshape(-1, 3)
    a, b = unique_edges // len(nodes), unique_edges % len(nodes)
    t = val[a] / (val[a] - val[b])
    vertices = pos[a] + t[:, None] * (pos[b] - pos[a])

    tri = vertices[triangles]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = np.einsum('ij,ij->i', normal, outward) < 0
    triangles[flip] = triangles[flip][:, ::-1]
    return vertices, triangles


def poisson_reconstruction(
    points: np.ndarray,
    normals: np.ndarray,
    depth: int = 8,
    full_depth: int = 5,
    point_weight: float = 2.0,
    scale: float = 1.1,
    k_area: int = 8,
    cg_tol: float = 1e-6,
    cg_iters: int = 500
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
        by default 8. Memory and time grow ~4x per depth; 9 is practical.
    full_depth : int, optional
        Depths up to this one are complete grids, by default 5
    point_weight : float, optional
        Screening weight alpha, by default 2.0 (as Open3D / PoissonRecon
        with first-order elements; iPSR uses 10)
    scale : float, optional
        Ratio between the reconstruction cube and the bounding box,
        by default 1.1
    k_area : int, optional
        Neighbours used to estimate the area of each sample, by default 8
    cg_tol, cg_iters : float, int, optional
        Relative tolerance and maximum iterations of the conjugate gradient
        solve of every depth

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Vertices ``(v, 3)``, triangles ``(f, 3)`` with outward orientation,
        and the number of input samples within two finest cells of every
        vertex (to trim low density regions, as Open3D's densities).
    """

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    full_depth = min(full_depth, depth)

    # Map the points into the unit cube.
    low, high = points.min(0), points.max(0)
    size = (high - low).max() * scale
    origin = (low + high) / 2 - size / 2
    x = (points - origin) / size

    # Area of each sample from its k nearest neighbours (density weights).
    tree = cKDTree(x)
    k = min(k_area, len(x) - 1)
    radius = tree.query(x, k + 1)[0][:, -1] if k > 0 else np.ones(len(x))
    area = np.pi * radius ** 2 / max(k, 1)
    unit = normals / np.maximum(
        np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    # Merge the samples falling in the same finest cell (as PoissonRecon).
    res = 1 << depth
    cells = np.clip(np.floor(x * res).astype(np.int64), 0, res - 1)
    _, group = np.unique(_encode(cells, res), return_inverse=True)
    count = np.bincount(group).astype(np.float64)
    samples = np.stack([np.bincount(group, x[:, i]) for i in range(3)], 1)
    samples /= count[:, None]
    inward = -np.stack(
        [np.bincount(group, area * unit[:, i]) for i in range(3)], 1)
    screen = point_weight * area.sum() / len(x)

    # Splat the inward normals into the finest hats (unit integral kernel)
    # and project the field on the hat gradients.
    corners, values = _basis(samples, res)
    field_keys, inv = np.unique(_encode(corners, res + 1), return_inverse=True)
    field = np.stack([np.bincount(
        inv.ravel(), (values * inward[:, None, i]).ravel()) for i in range(3)],
        1) * res ** 3
    divergence = _divergence(_decode(field_keys, res + 1), field, res)

    # Active nodes of every depth: complete up to full_depth, then the
    # corners of the cells within one cell of a sample.
    regions = {}
    depths = []
    for d in range(depth + 1):
        r = 1 << d
        if d <= full_depth:
            regions[d] = np.arange(r ** 3, dtype=np.int64)
            node_keys = np.arange((r + 1) ** 3, dtype=np.int64)
        else:
            occupied = np.unique(np.clip(
                np.floor(samples * r).astype(np.int64), 0, r - 1), axis=0)
            regions[d] = _dilate(occupied, r)
            node_keys = np.unique(_encode(
                _decode(regions[d], r)[:, None, :] + _CORNERS[None], r + 1))
        depths.append(_Depth(d, node_keys))

    # Cascadic solve, coarse to fine. Once the complete depths are solved,
    # their sum (a trilinear function on the full_depth grid) is collapsed
    # into one depth to speed up the evaluations.
    chi_samples = np.zeros(len(samples))
    solved = []
    for d, dep in enumerate(depths):
        _solve_depth(dep, solved, samples, count, chi_samples, screen,
                     divergence, cg_tol, cg_iters)
        chi_samples += dep.evaluate(samples)
        solved.append(dep)
        if d == full_depth:
            dense = _Depth(d, dep.keys)
            dense.coeffs = _evaluate(solved, _decode(dep.keys, dep.res + 1) /
                                     dep.res)
            solved = [dense]
    depths = solved
    iso = np.average(chi_samples, weights=count)

    # Extract the surface on the leaves of the octree.
    verts, tris, offset = [], [], 0
    leaves = regions[full_depth]
    for d in range(full_depth, depth + 1):
        r = 1 << d
        if d < depth:
            parents = _encode(_decode(regions[d + 1], 2 * r) // 2, r)
            split = np.isin(leaves, parents)
            children = _decode(leaves[split], r)[:, None, :] * 2 + _CORNERS
            next_leaves = _encode(children, 2 * r).ravel()
            leaves = leaves[~split]
        else:
            next_leaves = None
        if len(leaves):
            v, t = _marching_tetrahedra(leaves, r, depths, iso)
            verts.append(v)
            tris.append(t + offset)
            offset += len(v)
        leaves = next_leaves

    vertices = np.concatenate(verts) if verts else np.zeros((0, 3))
    triangles = np.concatenate(tris) if tris else np.zeros((0, 3), np.int64)
    densities = tree.query_ball_point(
        vertices, r=2.0 / res, return_length=True).astype(np.float64)

    return vertices * size + origin, triangles, densities
