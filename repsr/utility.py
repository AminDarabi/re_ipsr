# This file is developed by Amin Darabi as part of the project of
# the course "Geometric Modeling and shape analysis" (IFT6113) at the
# Université de Montréal (UdeM) in Fall 2024.
"""
"""

import numpy as np
import scipy as sp
import open3d as o3d


def initial_normal(
        points: o3d.geometry.PointCloud,
        method: str = "random"
):
    """
    Initial normal estimation for unoriented point cloud.

    Parameters
    ----------
    points : o3d.geometry.PointCloud
        Unoriented point cloud
    method : str, optional
        Method to estimate normals, by default "random"
        [random, estimate, heuristic]
    kd_tree : sp.spatial.KDTree, optional
        KDTree for the point cloud, by default None,
        required for the heuristicmethod only

    Returns
    -------
    np.ndarray
        Normal vectors for each point in the
    """

    if method == "random":
        normals = np.random.rand(len(points.points), 3)
        return normals / np.linalg.norm(normals, axis=1)[:, None]
    if method == "estimate":
        points.estimate_normals()
        return np.asarray(points.normals)

    points = np.asarray(points.points)
    star_points = np.random.rand(np.sqrt(len(points)).astype(int), 3)
    star_points = star_points * (points.max(0) - points.min(0)) + points.min(0)
    kd_tree = sp.spatial.KDTree(star_points)
    normals = points - star_points[kd_tree.query(points)[1]]

    return normals / np.linalg.norm(normals, axis=1)[:, None]


def calculate_triangle_centroids(
        mesh: o3d.geometry.TriangleMesh
):
    """
    Calculate the centroid of each triangle in the mesh.

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        Input triangle mesh

    Returns
    -------
    np.ndarray
        Centroids of each triangle in the mesh
    """

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    return (
        vertices[triangles[:, 0]] +
        vertices[triangles[:, 1]] +
        vertices[triangles[:, 2]]
    ) / 3
