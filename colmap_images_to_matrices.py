import os

import argparse

from src.cameras_positions import quaternion_to_matrix, build_camera_matrices

parser = argparse.ArgumentParser(description='Training parser')

parser.add_argument('--path',
                    help='Path to a images.txt file from colmap',
                    type=str,
                    required=True)

parser.add_argument('--save-folder',
                    help='Folder to save the camera matrices as txt files',
                    type=str,
                    required=True)

args = parser.parse_args()

path = args.path
with open(path) as f:
    lines = f.readlines()
    lines = [line.rstrip() for line in lines]

data = []
names = []
for i in range(4, len(lines)):
    if i % 2 == 0:
        camera_data = lines[i]
        camera_data = camera_data.split(' ')

        # remove the file extension
        photo_name = os.path.splitext(camera_data[-1])[0]

        # we don't care about the image id, camera ID is always 1
        camera_data = camera_data[1:-2]
        camera_data = [float(x) for x in camera_data]
        data.append(camera_data)
        names.append(photo_name)

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

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

for i, camera in enumerate(cameras):
    file_name = os.path.join(args.save_folder, names[i]) + '.txt'
    with open(file_name, 'w') as f:
        for row in cameras[i]:
            to_write = ' '.join(str(x) for x in row)
            f.write(to_write)
            f.write('\n')
