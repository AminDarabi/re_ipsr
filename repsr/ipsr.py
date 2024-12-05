# This file is developed by Amin Darabi as part of the project of
# the course "Geometric Modeling and shape analysis" (IFT6113) at the
# Université de Montréal (UdeM) in Fall 2024.
"""
Iterative Poisson surface reconstruction.

This module provides a function to perform iterative Poisson surface \
reconstruction on a given point cloud. The function iteratively estimates \
the normals of the points and reconstructs the surface using the Poisson \
reconstruction algorithm. The function returns a list of meshes and normals \
for each iteration.

Example
-------
The following example demonstrates how to use the function to perform \
iterative Poisson surface reconstruction on a given point cloud.

>>> import open3d as o3d
>>> import repsr
>>> # Load the point cloud
>>> points = o3d.io.read_point_cloud("bunny.ply")
>>> # Perform iterative Poisson surface reconstruction
>>> meshes, normals = repsr.iterative_poisson_reconstruction(points)
>>> # Plot the meshes
>>> repsr.plot_meshes(meshes, normals, "Poisson Surface Reconstruction", \
    [f"Iteration {i}" for i in range(len(meshes))])
"""

import numpy as np
import scipy as sp
import open3d as o3d
import tqdm

from repsr.utility import initial_normal, calculate_triangle_centroids
from repsr.utility import calculate_surfaces_normals


def iterative_poisson_reconstruction(
    points: o3d.geometry.PointCloud,
    max_iter: int = 30,
    depth: int = 12,
    k_neighbors: int = 10,
    threshold: float = 1e-4,
    init_method: str = 'random',
    weighted: bool = False,
    anealing: bool = False
) -> tuple[list[o3d.geometry.TriangleMesh], list[np.ndarray]]:
    """
    Iterative Poisson surface reconstruction.

    Parameters
    ----------
    points : o3d.geometry.PointCloud
        Input point cloud
    max_iter : int, optional
        Maximum number of iterations, by default 30
    depth : int, optional
        Depth for trees, by default 12
    k_neighbors : int, optional
        Number of neighbors to consider, by default 10
    threshold : float, optional
        Threshold for convergence, by default 1e-5
    init_method : str, optional
        Method to estimate initial normals, by default "random"
        [random, estimate, heuristic]

    Returns
    -------
    tuple[list[o3d.geometry.TriangleMesh], list[np.ndarray]]
        List of meshes and normals for each iteration
    """

    points.normals = o3d.utility.Vector3dVector(
        initial_normal(points, method=init_method)
    )
    kd_tree = sp.spatial.KDTree(np.asarray(points.points), leafsize=depth)
    meshes = []
    normals = [np.asarray(points.normals)]

    for _ in tqdm.tqdm(range(max_iter)):

        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            points, depth=10, scale=1.01
        )
        # mesh = poisson_reconstruction(points)
        meshes.append(mesh)

        if anealing:
            k_neighbors = max(1, k_neighbors - 1)

        surfaces_normals = calculate_surfaces_normals(mesh)
        distances, neighbors = kd_tree.query(
            calculate_triangle_centroids(mesh), k_neighbors
        )

        if k_neighbors == 1:
            neighbors = neighbors[:, None]
            distances = distances[:, None]

        new_normals = np.zeros((len(points.points), 3))
        for distance, neighbor, normal in zip(
            distances, neighbors, surfaces_normals
        ):
            if weighted:
                new_normals[neighbor] += normal / (distance + 1e-6)[:, None]
            else:
                new_normals[neighbor] += normal

        new_normal_norms = np.linalg.norm(new_normals, axis=1)
        new_normals[np.isclose(new_normal_norms, 0)] = np.asarray(
            points.normals
        )[np.isclose(new_normal_norms, 0)]

        new_normals = new_normals / np.linalg.norm(
            new_normals, axis=1
        )[:, None]

        diff = np.linalg.norm(new_normals - normals[-1], axis=1).mean()
        if diff < threshold:
            break

        points.normals = o3d.utility.Vector3dVector(new_normals)
        normals.append(new_normals)

    return meshes, normals
