# This python script is developed by Amin Darabi as part of the project of
# the course "Geometric Modeling and shape analysis" (IFT6113) at the
# Université de Montréal (UdeM) in Fall 2024.
import argparse
import sys
import os

import numpy as np
import open3d as o3d
from vedo import Plotter, Mesh, Points, Arrows

import repsr


def plot_meshes(
    meshes: list[o3d.geometry.TriangleMesh],
    color_map: list[np.ndarray] | np.ndarray,
    title: str,
    sub_title: list[str],
    color: str = 'coolwarm'
) -> None:
    """
    Plot the given list of meshes.

    Parameters
    ----------
    meshes : list[list[np.ndarray, np.ndarray]] | list[np.ndarray, np.ndarray]
        List of meshes to plot
    color_map : list[np.ndarray]
        List of color maps to use for the plot
    title : str
        Title of the plot
    sub_title : list[str]
        List of sub titles for each mesh
    color : str
        Color map to use for the plot
    """

    if len(color_map) != len(meshes):
        raise ValueError(
            "Number of color maps should be equal to the number of meshes"
        )

    plotter = Plotter(N=len(meshes), title=title)
    for i, mesh in enumerate(meshes):
        plotter.at(i).show(
            Mesh(
                [mesh.vertices, mesh.triangles]
            ).clone(deep=False).cmap(
                color, color_map[i], n_colors=256
            ),
            sub_title[i]
        )
    plotter.interactive().close()

    return


def plot_point_clouds(
    points: o3d.geometry.PointCloud,
    normals: list[np.ndarray],
    color_map: list[np.ndarray],
    title: str,
    sub_title: list[str],
    color: str = 'coolwarm'
) -> None:
    """
    Plot the given list of point clouds.

    Parameters
    ----------
    points : o3d.geometry.PointCloud
        List of point clouds to plot
    normals : list[np.ndarray]
        List of normal vectors for each point cloud
    title : str
        Title of the plot
    sub_title : list[str]
        List of sub titles for each point cloud
    color : str
        Color map to use for the plot
    """

    points = np.asarray(points.points)
    diff = points.max(0) - points.min(0)
    scale = np.linalg.norm(diff) * 0.01

    plotter = Plotter(N=len(normals), title=title)
    for i, normal in enumerate(normals):
        point = Points(points, r=10).clone(deep=False).cmap(
            color, color_map[i], n_colors=256
        )
        arrow = Arrows(points, points + normal * scale, s=0.5).c('black')
        plotter.at(i).show(point, arrow, sub_title[i])
    plotter.interactive().close()

    return


def main(args):
    """
    Main function for the script.
    """

    points = o3d.io.read_point_cloud(args.input)

    meshes, normals = repsr.iterative_poisson_reconstruction(
        points, max_iter=args.max_iter, depth=args.depth,
        k_neighbors=args.k_neighbors, threshold=args.threshold,
        init_method=args.init
    )

    os.makedirs(args.output, exist_ok=True)
    if args.save_iters:
        for i, mesh in enumerate(meshes):
            o3d.io.write_triangle_mesh(
                f"{args.output}/mesh_{i}.ply", mesh
            )
            points.normals = o3d.utility.Vector3dVector(normals[i])
            o3d.io.write_point_cloud(
                f"{args.output}/normals_{i}.ply", points
            )
    else:
        o3d.io.write_triangle_mesh(
            f"{args.output}/mesh.ply", meshes[-1]
        )
        points.normals = o3d.utility.Vector3dVector(normals[-1])
        o3d.io.write_point_cloud(
            f"{args.output}/normals.ply", points
        )

    if args.plot > 0:

        selected = np.linspace(
            0, len(meshes) - 1, args.plot, dtype=int
        )

        meshes = [meshes[i] for i in selected]
        normals = [normals[i] for i in selected]

        plot_meshes(
            meshes, [np.asarray(mesh.vertices)[:, 0] for mesh in meshes],
            "Iterative Poisson Surface Reconstruction",
            [f"Iteration {i + 1}" for i in selected]
        )
        plot_point_clouds(
            points, normals, [
                np.asarray(points.points)[:, 0] for i in range(args.plot)
            ],
            "Estimated Normals",
            [f"Iteration {i + 1}" for i in selected]
        )

    sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A simple python script for Surface Reconstruction"
    )
    parser.add_argument(
        'input',
        type=str,
        help="Path to unoriented point cloud file"
    )
    parser.add_argument(
        '--output', '-o',
        type=str, default="out/",
        help="Path to output directory"
    )
    parser.add_argument(
        '--init',
        type=str, default="heuristic",
        help="Method to estimate initial normals",
        choices=["random", "estimate", "heuristic"]
    )
    parser.add_argument(
        '--max-iter', '-i',
        type=int, default=30,
        help="Maximum number of iterations"
    )
    parser.add_argument(
        '--depth',
        type=int, default=12,
        help="Depth for trees"
    )
    parser.add_argument(
        '--k-neighbors', '-k',
        type=int, default=10,
        help="Number of neighbors to consider"
    )
    parser.add_argument(
        '--threshold', type=float, default=1e-5,
        help="Threshold for convergence"
    )
    parser.add_argument(
        '--plot', '-p',
        type=int, default=0,
        help="Plot the result"
    )
    parser.add_argument(
        '--alpha', type=float, default=10,
        help="Alpha value for the Screened Poisson reconstruction"
    )
    parser.add_argument(
        '--save-iters', '-s',
        action='store_true',
        help="Save intermediate iterations"
    )

    main(parser.parse_args())
