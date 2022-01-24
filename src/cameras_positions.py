import os

import numpy as np


def quaternion_to_matrix(q_dict):
    """
    Transform a quaternion (w, x, y, z) to its matrix representation.

    See as ref: http://www.songho.ca/opengl/gl_quaternion.html
    """
    w = q_dict['qw']
    x = q_dict['qx']
    y = q_dict['qy']
    z = q_dict['qz']

    res = np.zeros((4, 4))

    res[:, 0] = np.array([1 - 2 * y**2 - 2 * z**2,
                          2 * x * y + 2 * w * z,
                          2 * x * z - 2 * w * y,
                          0])

    res[:, 1] = np.array([2 * x * y - 2 * w * z,
                          1 - 2 * x**2 - 2 * z**2,
                          2 * y * z + 2 * w * x,
                          0])

    res[:, 2] = np.array([2 * x * z + 2 * w * y,
                          2 * y * z - 2 * w * x,
                          1 - 2 * x**2 - 2 * y**2,
                          0])

    res[3][3] = 1

    return res


def build_camera_matrices(data_dict):
    """
    From a list of dict (one dict for one camera), build the 4*4 camera matrices.

    Each dict contains a quaternion and translation coordinates.

    See https://en.wikipedia.org/wiki/Camera_matrix for reference.

    See as well https://github.com/colmap/colmap/issues/797#issuecomment-583592418
    """
    cameras = []

    for i in range(len(data_dict)):
        camera_matrix = quaternion_to_matrix(data_dict[i])

        rotation = np.copy(camera_matrix[:3, :3])  # copy to be safe, is it necessary?


        translation = np.array([[data_dict[i]['tx']],
                                [data_dict[i]['ty']],
                                [data_dict[i]['tz']]])
        translation = - np.matmul(rotation.T, translation)
        translation = translation.squeeze()

        camera_matrix[:3, :3] = rotation.T
        camera_matrix[:3, 3] = translation

        cameras.append(camera_matrix)

    return cameras