import argparse
import os

import numpy as np

from src.cameras_positions import quaternion_to_matrix, build_camera_matrices


parser = argparse.ArgumentParser(description='Conversion parser')


parser.add_argument('--model',
                    help='Path to a folder containing images.txt and cameras.txt',
                    type=str,
                    required=True)

parser.add_argument('--output',
                    help='Path to an output folder',
                    type=str,
                    required=True)

args = parser.parse_args()

if not os.path.exists(args.output):
    os.makedirs(args.output)

# cameras.txt to intrinsics.txt
with open(os.path.join(args.model, 'cameras.txt'), 'r') as f:
    mt = np.identity(4, dtype = float)
    # Group lines
    for line in f:
        if line[0] =='#': continue
        else:
            print(line)
            col1, col2, col3, col4, col5, col6, col7 = line.split()
            mt[0][0] = col5
            mt[0][2] = col6
            mt[1][1] = col5
            mt[1][2] = col7
    print('Intrisics:')
    print(mt)

with open(os.path.join(args.output, 'intrinsics.txt'), 'w') as out:
    for line in mt:
        for x in line:
            out.write(str(x) + ' ')
        out.write('\n')

# images.txt to `pose` folder
with open(os.path.join(args.model, 'images.txt')) as f:
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

pose_path = os.path.join(args.output, 'pose/')
if not os.path.exists(pose_path):
    os.makedirs(pose_path)

for i, camera in enumerate(cameras):
    print(camera)
    file_name = os.path.join(pose_path, names[i]) + '.txt'
    with open(file_name, 'w') as f:
        for row in cameras[i]:
            to_write = ' '.join(str(x) for x in row)
            f.write(to_write)
            f.write('\n')