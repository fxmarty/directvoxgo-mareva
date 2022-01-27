import argparse
import bezier
import os
import torch

import numpy as np

from lib.utils_get_center import get_center_object
from lib.utils_interaction_3D import rot_phi, rot_theta, rot_psi, translation
from lib.utils_interaction_3D import Pose

#TODO remove
import matplotlib.pyplot as plt


def dist(x, y):
    return np.sqrt(np.sum((x - y)**2))


def get_rotations(n_bezier):
    """
    Generate sample points along a continuous trajectory for (theta, phi, psi)
    """
    nodes = [np.array([0, 0, 0])]

    assert n_bezier > 4

    while len(nodes) < n_bezier + 1:
        new_point = np.random.rand(3) * 2 * np.pi

        # don't take too close points to have a nice trajectory
        if all([dist(node, new_point) > 1 for node in nodes]):
            nodes.append(new_point)

    nodes = np.array(nodes).T  # shape 3 * n_bezier

    curve = bezier.Curve(nodes, degree=n_bezier)

    t_fine = np.linspace(0, 1, 500) # Curvilinear coordinate
    points_fine = curve.evaluate_multi(t_fine)

    points_fine = np.array(points_fine).T
    kept_points = [points_fine[0]]

    for point in points_fine:
        if dist(kept_points[-1], point) > 0.075:
            kept_points.append(point.copy())

    kept_points = np.array(kept_points)

    kept_points = kept_points.T
    points_fine = points_fine.T

    kept_points_rolled = np.roll(kept_points, shift=1, axis=1)

    angles_movements = kept_points - kept_points_rolled
    angles_movements = angles_movements[:, 1:-1]  # remove bad data at beginning and end

    return angles_movements.T

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conversion parser')

    parser.add_argument('--output',
                        help='Path to an output folder (store result in test_traj.txt)',
                        type=str,
                        required=True)

    parser.add_argument("--coordinates",
                        type=str,
                        required=True,
                        help='Path to a .npz generated with the option `export_coarse_only`')

    parser.add_argument("--init",
                        type=str,
                        required=True,
                        help='Initialization pose')

    parser.add_argument("--file-by-file",
                        action='store_true',
                        help='If set true, will store poses in different .txt files')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    rotations = get_rotations(6)

    # load the initial pose
    render_pose = np.loadtxt(args.init).astype(np.float32)
    render_pose = torch.Tensor(render_pose)

    # compute the coordinates for the center of the object to handle rotations later
    coordinates = np.load(args.coordinates)
    alpha = coordinates['alpha']
    xyz_min = coordinates['xyz_min']
    xyz_max = coordinates['xyz_max']

    center = get_center_object(alpha, xyz_min, xyz_max)
    print(f'Object center at position {center}')

    pose = Pose(render_pose)

    saved_poses = []
    for rotation in rotations:
        pose.rotate(center, radians=rotation[0], angle='theta', sign=1)
        pose.rotate(center, radians=rotation[1], angle='psi', sign=1)
        pose.rotate(center, radians=rotation[2], angle='phi', sign=1)

        saved_poses.append(pose.pose.cpu().numpy().copy())

    if args.file_by_file is False:
        with open(os.path.join(args.output, 'test_traj.txt'), 'w') as f:
            for pose in saved_poses:
                for row in pose:
                    to_write = ' '.join(str(x) for x in row)
                    f.write(to_write)
                    f.write('\n')
    else:
        count = 0
        for pose in saved_poses:
            with open(os.path.join(args.output, f'{count}.txt'), 'w') as f:
                for row in pose:
                    to_write = ' '.join(str(x) for x in row)
                    f.write(to_write)
                    f.write('\n')
            count += 1
