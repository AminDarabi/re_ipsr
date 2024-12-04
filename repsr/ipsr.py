# This file is developed by Amin Darabi as part of the project of
# the course "Geometric Modeling and shape analysis" (IFT6113) at the
# Université de Montréal (UdeM) in Fall 2024.
"""
"""

import warnings

import numpy as np
import scipy as sp
import open3d as o3d
import tqdm

from repsr.utility import initial_normal, calculate_triangle_centroids


def screened_poisson_reconstruction(
    points: o3d.geometry.PointCloud,
    depth: int = 12,
    alpha: float = 10
) -> o3d.geometry.TriangleMesh:
    """
    Screened Poisson surface reconstruction.

    Parameters
    ----------
    points : o3d.geometry.PointCloud
        Input point cloud
    depth : int, optional
        Depth for trees, by default 12
    alpha : float, optional
        Alpha value for the Poisson reconstruction, by default 10

    Returns
    -------
    o3d.geometry.TriangleMesh
        Reconstructed mesh
    """

    pass


def iterative_poisson_reconstruction(
    points: o3d.geometry.PointCloud,
    max_iter: int = 30,
    depth: int = 12,
    k_neighbors: int = 10,
    threshold: float = 1e-5,
    init_method: str = "random",
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
            points, depth=depth
        )
        meshes.append(mesh)

        mesh.compute_triangle_normals(False)
        neighbors = kd_tree.query(
            calculate_triangle_centroids(mesh), k_neighbors
        )[1]

        new_normals = np.zeros((len(points.points), 3))
        for neighbor, normal in zip(
            neighbors, np.asarray(mesh.triangle_normals)
        ):
            new_normals[neighbor] -= normal

        new_normal_norms = np.linalg.norm(new_normals, axis=1)
        new_normals[np.isclose(new_normal_norms, 0)] = np.asarray(
            points.normals
        )[np.isclose(new_normal_norms, 0)]

        new_normals = new_normals / np.linalg.norm(
            new_normals, axis=1
        )[:, None]

        if np.allclose(normals[-1], new_normals, atol=threshold):
            break
        points.normals = o3d.utility.Vector3dVector(new_normals)
        normals.append(new_normals)

    return meshes, normals
