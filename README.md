![Alt text](https://github.com/AminDarabi/re_ipsr/blob/main/Sample.png)

# Iterative Poisson Surface Reconstruction

This program is a python re-implementation of iterative Poisson Surface Reconstruction(iPSR) on unoriented point clouds. It iteratively estimates point normals and reconstructs the surface using the Poisson reconstruction algorithm. The program can visualize the reconstruction process and save intermediate results.

Two extensions also implemented in this program:
- Weighted normal updating which updates the normals at each iteration as
$$n_i = \sum_{j \in m} \frac{a_jn_j}{dist(p_i, f_j)}$$ for associated
faces $$f_j$$ and normals $$n_j$$, and distance dist($$p_i$$, $$f_j$$) from point $$p_i$$ to face
$$f_j$$

- Annealing for neighbor count (K) which reduces the number of neighbors
throughout.

## Features
- Iterative normal estimation
- Multiple initialization methods for normals
- Weighted normal updating
- Annealing for neighbor count
- Interactive visualization of reconstruction steps
- Support for saving intermediate results

## Requirements
- Python 3.10 or later (3.11 tested)
- NumPy
- SciPy
- Open3D
- Vedo
- tqdm

Assuming you have Python installed, you can install the required packages using pip:
```bash
pip install numpy scipy open3d vedo tqdm
```
or,
```bash
pip install -r requirements.txt
```

## Usage
Run the program using:
```bash
python main.py input_pointcloud.ply [options]
```

### Options
- `--init`: Method to estimate initial normals (`random`, `estimate`, `heuristic`)
- `-i, --max-iter`: Maximum number of iterations (default: 30)
- `--depth`: Depth for trees (default: 10)
- `-k, --k-neighbors`: Number of neighbors to consider (default: 10)
- `--threshold`: Convergence threshold (default: 1e-4)
- `-p, --plot`: Number of iterations to plot (default: 0)
- `-o, --output`: Output directory (default: "out/")
- `-s, --save-iters`: Save intermediate iteration meshes and normals
- `-w, --weighted`: Use weighted normal updating
- `-a, --anealing`: Use annealing for number of neighbors

### Example
To reconstruct a point cloud with visualization and intermediate results:
```bash
python main.py bunny.ply -i 20 -k 15 --init heuristic -p 5 -o results/ -s -w
```

## Output
The program generates in the output directory:
- Final reconstructed mesh (`mesh.ply`)
- Final estimated normals (`normals.ply`)

## Acknowledgment
This program is developed Based on Iterative Poisson Surface Reconstruction by Fei Hou, Chiyu Wang, Wencheng Wang, Hong Qin, Chen Qian, Ying He in 2022 (https://lcs.ios.ac.cn/~houf/pages/ipsr/index.html).

This code relies on the original implementation of Poisson surface
reconstruction by Michael Kazhdan and et al. (http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version8.0/) through Open3D.
