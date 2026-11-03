# This file is developed by Amin Darabi as part of the project of
# the course "Geometric Modeling and shape analysis" (IFT6113) at the
# Université de Montréal (UdeM) in Fall 2024.
"""
Fused iterative Poisson surface reconstruction (iPSR inside the PSR solver).

iPSR [Hou et al. 2022] treats screened PSR as a black box: every iteration
rebuilds the octree, the density estimate, the linear systems and solves them
from scratch, although only the normals change. Here the iPSR loop runs
inside :mod:`repsr.psr`:

* Everything that depends only on the point positions is built once: the
  samples and their density, the octree (active nodes of every depth), the
  stiffness + screening matrices, the interpolation between depths and the
  KD-trees. An iteration only rebuilds the normal field (the right-hand
  side, linear in the normals), re-solves with conjugate gradients started
  from the previous solution, extracts the surface and updates the normals.
* Coarse to fine: the iPSR iterations start at a coarse depth, where they
  are cheap and converge easily, on the samples gathered per cell of that
  depth; each finer stage inherits the normals of the coarser one, so it
  only needs a few iterations.
* As iPSR, the indicator function vanishes on the domain boundary
  (Dirichlet) by default, which fixes the global sign of the normals.

The discretisation is the one of :func:`repsr.psr.poisson_reconstruction`
(the same matrices for every depth, whatever the stage), and the normal
update and stopping rule are the ones of iPSR: every face adds its
area-weighted normal to its ``k`` nearest samples, and a stage stops when
the average of the 0.1% largest normal changes is below ``threshold``.

Example
-------
>>> import open3d as o3d
>>> import repsr
>>> points = o3d.io.read_point_cloud("bunny.ply")
>>> meshes, normals = repsr.fused_iterative_reconstruction(points)
"""

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import cg
from scipy.spatial import cKDTree
import open3d as o3d
import tqdm

from repsr import psr as P
from repsr.utility import initial_normal


def _rows(cols: np.ndarray, vals: np.ndarray, shape: tuple):
    """CSR matrix from per-row columns and values (columns -1 skipped)."""
    stored = (cols >= 0) & (vals != 0)
    indptr = np.concatenate([[0], np.cumsum(stored.sum(1))])
    return sps.csr_matrix((vals[stored], cols[stored], indptr), shape=shape)


class _Level:
    """
    The linear system of one depth, which does not depend on the normals:
    stiffness + screening matrix on the active nodes, the stiffness between
    the active nodes and the other nodes of their ring, the screening points
    and the interpolation of the coarser solution; and the current solution.
    """

    def __init__(self, dep: P._Depth, ring: np.ndarray, coarser,
                 points: np.ndarray, weights: np.ndarray):
        res, keys = dep.res, dep.keys
        n = len(keys)
        nodes = P._decode(keys, res + 1)
        self.in_ring = P._lookup(ring, keys)
        active = -np.ones(len(ring), dtype=np.int64)
        active[self.in_ring] = np.arange(n)
        self.dep, self.total = dep, P._Depth(dep.depth, ring)

        # Stiffness rows over the ring (as in repsr.psr._solve_depth).
        cols = np.empty((n, len(P._OFFSETS)), dtype=np.int64)
        vals = np.empty((n, len(P._OFFSETS)))
        edge = np.flatnonzero(np.any((nodes == 0) | (nodes == res), axis=1))
        for o, offset in enumerate(P._OFFSETS):
            if offset[2] == -1:
                middle = P._lookup(
                    ring, keys + P._encode(offset + (0, 0, 1), res + 1))
            k = np.full(n, P._stiffness(np.ones((1, 3)), offset, res)[0])
            k[edge] = P._stiffness(nodes[edge], offset, res)
            nb = nodes[edge] + offset
            k[edge[np.any((nb < 0) | (nb > res), axis=1)]] = 0
            cols[:, o] = np.where(k != 0, middle + offset[2], -1)
            vals[:, o] = k
        inner = np.where(cols >= 0, active[np.maximum(cols, 0)], -1)
        stiffness = _rows(inner, vals, (n, n))
        # stiffness towards the ring nodes that are not active
        self.outer = _rows(np.where(inner < 0, cols, -1), vals,
                           (n, len(ring)))

        # Screening matrix (the hats of removed boundary nodes are zero).
        corners, values = P._basis(points, res)
        idx = P._lookup(keys, corners)
        pair = (idx[:, :, None] >= 0) & (idx[:, None, :] >= 0)
        self.screening = sps.coo_matrix((
            (weights[:, None, None] * values[:, :, None] *
             values[:, None, :])[pair],
            (np.broadcast_to(idx[:, :, None], pair.shape)[pair],
             np.broadcast_to(idx[:, None, :], pair.shape)[pair])),
            shape=(n, n)).tocsr()
        self.screen = idx, values, weights
        self.matrix = (stiffness + self.screening).tocsr()
        diag = self.matrix.diagonal()
        self.precond = sps.diags(1.0 / np.where(diag > 0, diag, 1.0))

        # The coarser solution, trilinear on the coarser grid near its
        # active nodes, interpolated at the ring and the screening points.
        self.coarser = coarser
        if coarser is not None:
            self.from_coarser = coarser.interpolation(
                P._decode(ring, res + 1) / res)
            self.points_from_coarser = coarser.interpolation(points)
        self.x = np.zeros(n)

    def interpolation(self, x: np.ndarray):
        """Sparse matrix evaluating the solution up to this depth at ``x``
        (exact where the corners of the cells of ``x`` are in the ring)."""
        idx, values = P._basis(x, self.dep.res)
        idx = P._lookup(self.total.keys, idx)
        assert np.all(idx >= 0), "points outside the ring of the depth"
        return sps.csr_matrix(
            (values.ravel(), idx.ravel(), np.arange(0, idx.size + 1, 8)),
            shape=(len(x), len(self.total.keys)))

    def solve(self, divergence: tuple, cg_tol: float, cg_iters: int):
        """Solve for this depth given the right-hand side ``divergence``
        (node keys, values) and the coarser solution (already solved),
        starting from the previous solution."""
        keys = self.dep.keys
        n = len(keys)
        if self.coarser is None:
            chi_ring, chi_points = np.zeros(len(self.total.keys)), 0.0
        else:
            chi_ring = self.from_coarser @ self.coarser.total.coeffs
            chi_points = self.points_from_coarser @ self.coarser.total.coeffs

        # gradient term of the coarser solution: -K chi (ring columns)
        chi_active = chi_ring[self.in_ring]
        rhs = -(self.matrix @ chi_active - self.screening @ chi_active)
        rhs -= self.outer @ chi_ring
        idx = P._lookup(keys, divergence[0])
        rhs += np.bincount(idx[idx >= 0], divergence[1][idx >= 0],
                           minlength=n)
        idx, values, weights = self.screen
        target = (weights * (0.5 - chi_points))[:, None] * values
        rhs += np.bincount(idx[idx >= 0], target[idx >= 0], minlength=n)

        if n:
            self.x, _ = cg(self.matrix, rhs, x0=self.x, rtol=cg_tol,
                           maxiter=cg_iters, M=self.precond)
        self.dep.coeffs = self.x
        self.total.coeffs = chi_ring
        self.total.coeffs[self.in_ring] += self.x


class _Problem:
    """Everything of :func:`repsr.psr.poisson_reconstruction` that depends
    only on the point positions, for all the depths."""

    def __init__(self, points: np.ndarray, depth: int, full_depth: int,
                 samples_per_node: float, point_weight: float, scale: float,
                 k_area: int, boundary: str = 'dirichlet'):
        low, high = points.min(0), points.max(0)
        self.size = (high - low).max() * scale
        self.origin = (low + high) / 2 - self.size / 2
        x = (points - self.origin) / self.size

        tree = cKDTree(x)
        k = min(k_area, len(x) - 1)
        area = np.pi * tree.query(x, k + 1, workers=-1)[0][:, -1] ** 2 / k \
            if k > 0 else np.ones(len(x))

        # samples: the points merged per finest cell
        res = 1 << depth
        cells = np.clip(np.floor(x * res).astype(np.int64), 0, res - 1)
        _, self.group = np.unique(P._encode(cells, res), return_inverse=True)
        self.count = np.bincount(self.group).astype(np.float64)
        self.samples = np.stack(
            [np.bincount(self.group, x[:, i]) for i in range(3)], 1) / \
            self.count[:, None]
        self.area = samples_per_node * np.bincount(self.group, area)
        self.screen = point_weight * samples_per_node * area.mean()
        self.level = np.clip(-np.log(self.area / self.count) / np.log(4),
                             0, depth)
        top = np.maximum(np.ceil(self.level).astype(np.int64), 0)
        self.finest = int(top.max())
        self.full_depth = min(full_depth, self.finest)

        # Active nodes of every depth (as repsr.psr).
        self.regions, self.feed, self.levels = {}, {}, []
        for d in range(self.finest + 1):
            r = 1 << d
            if d <= self.full_depth:
                self.regions[d] = np.arange(r ** 3, dtype=np.int64)
                node_keys = np.arange((r + 1) ** 3, dtype=np.int64)
            else:
                occupied = np.clip(np.floor(self.samples[top >= d] * r),
                                   0, r - 1)
                self.regions[d] = P._dilate(occupied, r, range(-2, 3))
                self.feed[d] = P._dilate(occupied // 2, r // 2 + 1,
                                         range(-3, 4))
                node_keys = P._dilate(P._decode(self.regions[d], r), r + 1,
                                      (0, 1))
            ring = node_keys if d <= self.full_depth else \
                P._dilate(P._decode(node_keys, r + 1), r + 1, range(-1, 2))
            if boundary == 'dirichlet':   # no hat on the domain boundary
                nodes = P._decode(node_keys, r + 1)
                node_keys = node_keys[np.all((nodes > 0) & (nodes < r), 1)]
            centroid, weight = P._gather(
                self.samples, self.count, r,
                self.regions[d] if d > self.full_depth else None)
            self.levels.append(_Level(
                P._Depth(d, node_keys), ring, self.levels[-1] if d else None,
                centroid, self.screen * r * weight))

    def stage(self, depth: int):
        """
        The samples of a stage: the samples gathered per cell of ``depth``
        (the samples themselves at the finest depth). Returns the stage of
        every sample, and the positions, weights, areas and splatting levels
        (capped at ``depth``) of the stage samples.
        """
        if depth >= self.finest:
            group = np.arange(len(self.samples))
        else:
            r = 1 << depth
            cells = np.clip(np.floor(self.samples * r).astype(np.int64),
                            0, r - 1)
            _, group = np.unique(P._encode(cells, r), return_inverse=True)
        count = np.bincount(group, self.count)
        positions = np.stack([np.bincount(group, self.count * self.samples[
            :, i]) for i in range(3)], 1) / count[:, None]
        area = np.bincount(group, self.area)
        # the deepest member: the stage sample stays inside the octree
        level = np.zeros(len(count))
        np.maximum.at(level, group, self.level)
        return group, positions, count, area, np.minimum(level, depth)

    def divergence(self, depth: int, positions: np.ndarray,
                   inward: np.ndarray, level: np.ndarray):
        """Right-hand side <grad phi, V> of every depth up to ``depth``
        (exactly as repsr.psr, with the splatting levels capped)."""
        top = np.maximum(np.ceil(level).astype(np.int64), 0)
        frac = np.where(top > 0, 1 - (top - level), 1.0)
        splats = []
        for d in range(depth + 1):
            here = np.concatenate([top == d, top == d + 1])
            weight = np.concatenate([frac, 1 - frac])[here]
            splats.append(P._splat(
                np.concatenate([positions, positions])[here],
                np.concatenate([inward, inward])[here] * weight[:, None], d))
        divergence = [None] * (depth + 1)
        for d in range(depth, -1, -1):
            part = P._divergence(*splats[d], 1 << d)
            if d < depth:
                finer = P._restrict(*divergence[d + 1], 1 << d)
                part = P._merge(np.concatenate([part[0], finer[0]]),
                                np.concatenate([part[1], finer[1]]))
            divergence[d] = part
        field = splats[0]
        for d in range(1, depth + 1):
            r = 1 << d
            if d > self.full_depth:
                keep = P._lookup(self.feed[d], field[0]) >= 0
                field = field[0][keep], field[1][keep]
            field = P._encode(P._decode(field[0], r // 2 + 1), r + 1), \
                field[1]
            part = P._divergence(*field, r, refine=True)
            keep = P._lookup(self.levels[d].dep.keys, part[0]) >= 0
            divergence[d] = P._merge(
                np.concatenate([divergence[d][0], part[0][keep]]),
                np.concatenate([divergence[d][1], part[1][keep]]))
            if d < depth:
                for axis in range(3):
                    field = P._convolve(*field, r + 1, axis, *P._REFINE, r,
                                        True)
                field = P._merge(np.concatenate([field[0], splats[d][0]]),
                                 np.concatenate([field[1], splats[d][1]]))
        return divergence

    def reconstruct(self, depth: int, positions: np.ndarray,
                    count: np.ndarray, area: np.ndarray, level: np.ndarray,
                    normals: np.ndarray, cg_tol: float, cg_iters: int):
        """Screened PSR up to ``depth`` of the stage samples with outward
        ``normals``; returns the surface (unit cube coordinates)."""
        inward = -normals * area[:, None]
        divergence = self.divergence(depth, positions, inward, level)
        for d in range(depth + 1):
            self.levels[d].solve(divergence[d], cg_tol, cg_iters)

        base = min(self.full_depth, depth)
        solved = [self.levels[base].total] + \
            [lv.dep for lv in self.levels[base + 1:depth + 1]]
        iso = np.average(P._evaluate(solved, positions), weights=count)

        # extraction on the leaves of the octree up to ``depth``
        verts, tris, offset = [], [], 0
        leaves = np.arange((1 << base) ** 3, dtype=np.int64)
        for d in range(base, depth + 1):
            r = 1 << d
            if d < depth:
                parents = P._encode(
                    P._decode(self.regions[d + 1], 2 * r) // 2, r)
                split = np.isin(leaves, parents)
                children = P._decode(leaves[split], r)[:, None, :] * 2 + \
                    P._CORNERS
                next_leaves = P._encode(children, 2 * r).ravel()
                leaves = leaves[~split]
            else:
                next_leaves = None
            finer = [lv.dep for lv in self.levels[d + 1:depth + 1]]
            total = self.levels[d].total

            def function(x, total=total, finer=finer):
                return total.evaluate(x) + P._evaluate(finer, x)

            def nodal(k, total=total, finer=finer, r=r):
                # corners 4..7 are above corners 0..3: next keys of the ring
                idx = P._lookup(total.keys, k[:, :4])
                values = total.coeffs[np.concatenate([idx, idx + 1], 1)]
                if finer:
                    nodes = P._unique(k.ravel())
                    values += P._evaluate(finer, P._decode(
                        nodes, r + 1) / r)[P._lookup(nodes, k)]
                return values

            v, t = P._marching_tetrahedra(leaves, r, nodal, function, iso)
            verts.append(v)
            tris.append(t + offset)
            offset += len(v)
            leaves = next_leaves
        self.iso = iso
        return np.concatenate(verts), np.concatenate(tris)

    def inside_out(self, depth: int) -> bool:
        """Whether the indicator function of the last reconstruction is
        above the iso-value at the corners of the domain (Neumann boundary
        conditions leave the global sign of the normals free)."""
        base = min(self.full_depth, depth)
        solved = [self.levels[base].total] + \
            [lv.dep for lv in self.levels[base + 1:depth + 1]]
        corners = P._CORNERS.astype(np.float64)
        return P._evaluate(solved, corners).mean() > self.iso

    def mesh(self, vertices: np.ndarray, triangles: np.ndarray):
        """Open3D mesh in the input coordinates."""
        return o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices * self.size + self.origin),
            o3d.utility.Vector3iVector(triangles))


def _update_normals(normals, vertices, triangles, tree, k, weighted):
    """iPSR normal update: every face adds its area-weighted normal (divided
    by the distance when ``weighted``) to its ``k`` nearest samples."""
    tri = vertices[triangles]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    k = min(k, tree.n)
    distances, neighbors = tree.query(tri.mean(1), k, workers=-1)
    distances, neighbors = distances.reshape(len(tri), k), \
        neighbors.reshape(len(tri), k)
    scale = 1.0 / (distances + 1e-6) if weighted else np.ones_like(distances)
    new = np.stack([np.bincount(
        neighbors.ravel(), (scale * face_normals[:, None, i]).ravel(),
        len(normals)) for i in range(3)], 1)
    length = np.linalg.norm(new, axis=1)
    new[length == 0] = normals[length == 0]
    return new / np.linalg.norm(new, axis=1, keepdims=True)


def fused_iterative_reconstruction(
    points: o3d.geometry.PointCloud,
    max_iter: int = 30,
    depth: int = 10,
    k_neighbors: int = 10,
    threshold: float = 0.175,
    init_method: str = 'random',
    weighted: bool = False,
    anealing: bool = False,
    point_weight: float = 10.0,
    start_depth: int | None = None,
    coarse_iters: int = 5,
    boundary: str = 'dirichlet',
    scale: float = 1.01,
    full_depth: int = 5,
    samples_per_node: float = 1.5,
    k_area: int = 16,
    cg_tol: float = 1e-3,
    cg_iters: int = 100
) -> tuple[list[o3d.geometry.TriangleMesh], list[np.ndarray]]:
    """
    Iterative Poisson surface reconstruction with the iPSR loop fused into
    the Python PSR solver, coarse to fine.

    Parameters
    ----------
    points : o3d.geometry.PointCloud
        Input point cloud (normals are not used)
    max_iter : int, optional
        Maximum number of iterations at the finest depth, by default 30
    depth : int, optional
        Maximum octree depth, by default 10 (sparse regions stop earlier)
    k_neighbors : int, optional
        Number of samples every face votes for, by default 10
    threshold : float, optional
        A stage stops when the average of the 0.1% largest normal changes
        is below it, by default 0.175 (as iPSR)
    init_method : str, optional
        Initial normals of the first stage, by default "random"
        [random, estimate, heuristic]
    weighted : bool, optional
        Divide the face normals by their distance to the samples
    anealing : bool, optional
        Decrease the number of neighbors by one every iteration
    point_weight : float, optional
        Screening weight, by default 10 (as iPSR)
    start_depth : int, optional
        Depth of the first stage, by default two depths above the finest one
    coarse_iters : int, optional
        Maximum number of iterations of the stages before the finest one,
        by default 5
    boundary : str, optional
        Boundary condition of the indicator function on the domain, by
        default "dirichlet" (zero, as iPSR: the outside is outside); with
        "neumann" (as Open3D and repsr.psr) the sign of the normals is free
        and fixed at the end
    scale, full_depth, samples_per_node, k_area, cg_tol, cg_iters : optional
        Parameters of :func:`repsr.psr.poisson_reconstruction`

    Returns
    -------
    tuple[list[o3d.geometry.TriangleMesh], list[np.ndarray]]
        The mesh of every iteration (the last one is reconstructed with the
        final normals) and the normals of the input points used for it
    """

    problem = _Problem(np.asarray(points.points, dtype=np.float64), depth,
                       full_depth, samples_per_node, point_weight, scale,
                       k_area, boundary)
    finest = problem.finest
    start = finest - 2 if start_depth is None else start_depth
    stages = list(range(max(min(start, finest), 1), finest + 1))

    meshes, normals_out = [], []
    normals = previous = None
    bar = tqdm.tqdm(total=sum(
        max_iter if d == finest else coarse_iters for d in stages))
    for d in stages:
        group, positions, count, area, level = problem.stage(d)
        if normals is None:
            cloud = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(positions))
            normals = initial_normal(cloud, method=init_method)
        else:   # inherit the normals of the coarser stage
            first = np.zeros(len(count), dtype=np.int64)
            first[group] = np.arange(len(group))
            normals = normals[previous[first]]
        previous = group
        tree = cKDTree(positions)
        iters = max_iter if d == finest else coarse_iters
        for _ in range(iters):
            bar.update()
            if anealing:
                k_neighbors = max(1, k_neighbors - 1)
            vertices, triangles = problem.reconstruct(
                d, positions, count, area, level, normals, cg_tol, cg_iters)
            meshes.append(problem.mesh(vertices, triangles))
            normals_out.append(normals[group][problem.group])
            new = _update_normals(normals, vertices, triangles, tree,
                                  k_neighbors, weighted)
            change = np.linalg.norm(new - normals, axis=1)
            top = np.sort(change)[-int(np.ceil(len(change) / 1000)):]
            normals = new
            if top.mean() < threshold:
                break
    bar.close()

    # final reconstruction with the final normals (iPSR, Algorithm 1)
    vertices, triangles = problem.reconstruct(
        finest, positions, count, area, level, normals, cg_tol, cg_iters)
    if problem.inside_out(finest):
        normals, triangles = -normals, triangles[:, ::-1]
    meshes.append(problem.mesh(vertices, triangles))
    normals_out.append(normals[group][problem.group])
    return meshes, normals_out
