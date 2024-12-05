# This file is developed by Amin Darabi as part of the project of
# the course "Geometric Modeling and shape analysis" (IFT6113) at the
# Université de Montréal (UdeM) in Fall 2024.
"""
Utility functions for the project.

Functions:
----------
- initial_normal(points: o3d.geometry.PointCloud, method: str = "random") -> \
np.ndarray:
    Estimates initial normals for an unoriented point cloud using various \
methods.
- calculate_triangle_centroids(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    Calculates the centroid of each triangle in the mesh.
- calculate_surfaces_normals(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    Calculates the normal of each surface in the mesh.

Dependencies:
-------------
- numpy
- scipy
- open3d
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
        theta = np.random.rand(len(points.points)) * 2 * np.pi
        phi = np.random.rand(len(points.points)) * np.pi
        return np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ]).T.copy()
    if method == "estimate":
        points.estimate_normals()
        return -np.asarray(points.normals)

    points = np.asarray(points.points)
    star_points = np.random.rand(np.sqrt(len(points)).astype(int), 3)
    star_points = star_points * (points.max(0) - points.min(0)) + points.min(0)
    kd_tree = sp.spatial.KDTree(star_points)
    normals = star_points[kd_tree.query(points)[1]] - points

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


def calculate_surfaces_normals(
    mesh: o3d.geometry.TriangleMesh
):
    """
    Calculate the normal of each surface in the mesh.

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        Input triangle mesh

    Returns
    -------
    np.ndarray
        Normal of each surface in the mesh
    """

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    return np.cross(
        vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
        vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    )
