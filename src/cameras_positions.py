import os

import numpy as np


def quaternion_to_matrix(q_dict):
    '''
    Transform a quaternion (w, x, y, z) to its matrix representation.

    See as ref: http://www.songho.ca/opengl/gl_quaternion.html
    '''
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
    '''
    From a list of dict (one dict for one camera), build the 4*4 camera matrices.

    Each dict contains a quaternion and translation coordinates.

    See https://en.wikipedia.org/wiki/Camera_matrix for reference.
    '''
    cameras = []

    for i in range(len(data_dict)):
        camera_matrix = quaternion_to_matrix(data_dict[i])
        camera_matrix[0, 3] = data_dict[i]['tx']
        camera_matrix[1, 3] = data_dict[i]['ty']
        camera_matrix[2, 3] = data_dict[i]['tz']

        cameras.append(camera_matrix)

    return cameras

if __name__ == '__main__':
    path = '/home/felix/Documents/Mines/3A/Option/Mini-projet/colmap_jade/simple_pinhole/images.txt'
    with open(path) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]

    data = []
    names = []
    for i in range(4, len(lines)):
        if i % 2 == 0:
            camera_data = lines[i]
            camera_data = camera_data.split(' ')
            print(camera_data)

            photo_name = camera_data[-1]

            # we don't care about the image id, camera ID is always 1
            camera_data = camera_data[1:-2]
            print(camera_data)
            camera_data = [float(x) for x in camera_data]
            data.append(camera_data)

    data_dict = []
    for i, camera_data in enumerate(data):

        camera_dict = {'qw': camera_data[0],
                       'qx': camera_data[1],
                       'qy': camera_data[2],
                       'qz': camera_data[3],
                       'tx': camera_data[4],
                       'ty': camera_data[5],
                       'tz': camera_data[6]}
        data_dict.append(camera_dict)

    cameras = build_camera_matrices(data_dict)
