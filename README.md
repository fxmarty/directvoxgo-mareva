## Why this repo?

This repo is an extension to [sunset1995 implementation of DirectVoxGO](https://github.com/sunset1995/DirectVoxGO/) [1], and can be used to easily convert custom datasets from colmap to the format expected by DirectVoxGO.

Additionally, you may render random trajectories around the object as a video, or use an user-control script to move around the object live (depending on your hardware).

[1] Sun et al., Direct Voxel Grid Optimization: Super-fast Convergence for Radiance Fields Reconstruction. arXiv:2111.11215 preprint, 2021.

## Conversion from colmap

DirectVoxGO expects image file names to begin by `0_` for train set, `1_` for validation set, and `2_` for test set. Use the script `rename_images.py` to decide the split and rename images.

Colmap is a software used to perform photogrammetry on a dataset of views of an objects, to recover intrinsic and extrinsic parameters. In our implementation, the camera model used in colmap is expected to be `SIMPLE_PINHOLE` and we expect a shared camera among all views. As an output, we typically have

```
model/
├── cameras.txt
├── images.txt
└── points3D.txt
```

where `cameras.txt` hold the intrinsic parameters of the camera and `images.txt` holds the extrinsic parameters for each view as a quaterion and a translation.

To format this data in the format expected by DirectVoxGO implementation, use

```
python3 colmap_to_directvoxgo.py --model path/to/model/ --output path/to/output/folder/
```
where `model/` is the repository holding the `cameras.txt` and `images.txt`. In the output folder, copy as well the images in a folder named `rgb/`.

## Create a video around the object

The center of the object needs to be known to perform rotations around the object which is not at the world origin.

After training a model, use

```
python3 run.py --config configs/path/to/config_file.py --export_coarse_only coarse_data.npz
```

to generate coarse density data used to find the center of the object.

Then, generate a random trajectory using a Bézier curve with

```
python3 generate_bezier_curve.py --output output_folder/ --init path/to/initial_pose.txt --coordinates coarse_data.npz
```

where the initial pose is used at the beginning of the trajectory, and can for example be one of the pose from the test set.

The video can be generated with

```
python3 run.py --config logs/.../config.py --render_only --render_video
```

and is saved in the log directory from the config used.

## Visualize new views interactively

Use

```
python3 interaction_3d.py --config logs/path/config.py --init path/initial_pose.txt --coordinates coarse_data.npz
```

The controls are:
* 8 <-> 5
* 4 <-> 6
* 1 <-> 2

Use `q` to exit the window gracefully.

## Authors

This work was done by Jingjing Hong and Félix Marty as part of a one week long course at Mines Paris under the supervision of Jean-Emmanuel Deschaud.
