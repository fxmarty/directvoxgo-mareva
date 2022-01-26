import numpy as np


def get_center_object(alpha, xyz_min, xyz_max, threshold=0.001):
    xyz = np.stack((alpha > threshold).nonzero(), -1)

    voxel_center = np.mean(xyz, axis=0)

    coordinates_center = voxel_center / alpha.shape * (xyz_max - xyz_min) + xyz_min

    return coordinates_center