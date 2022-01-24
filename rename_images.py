import argparse
import os
import random
import sys

import numpy as np

from shutil import copyfile


parser = argparse.ArgumentParser(description='Parser')

parser.add_argument('--folder',
                    help='Folder where .jpg images are stored',
                    type=str,
                    required=True)

parser.add_argument('--output-folder',
                    help='Folder where renamed .jpg images will be stored',
                    type=str,
                    default=None)

parser.add_argument('--split',
                    help='Split used for [train, valid, test] datasets',
                    type=float,
                    default=[0.8, 0.2, 0],
                    nargs='+')

args = parser.parse_args()

assert len(args.split) == 3

n_images = len([name for name in os.listdir(args.folder) if name.endswith('.jpg')])

print(f'Total number of images: {n_images}')

n_train = int(args.split[0] * n_images)
n_valid = int(args.split[1] * n_images)
n_test = n_images - n_train - n_valid

# just an **ugly** trick
if args.split[2] == 0:
    n_test = 0
    n_valid = n_images - n_train

# class_list hold the class of all images as 0, 1 or 2. It allows to randomly
# assign images to different classes independent of the loading order of os.listdir
class_list = [0] * n_train + [1] * n_valid + [2] * n_test
random.shuffle(class_list)

if args.output_folder is not None:
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

image_id = 0
for filename in os.listdir(args.folder):
    if filename.endswith('.jpg'):
        filename_no_extension = os.path.splitext(filename)[0]

        index = class_list[image_id]
        new_name = str(index) + '_' + filename

        if args.output_folder is None:  # keep the same folder, just rename
            os.rename(os.path.join(args.folder, filename),
                      os.path.join(args.folder, new_name))
        else:
            copyfile(os.path.join(args.folder, filename),
                     os.path.join(args.output_folder, new_name))
        print(f'{filename} --> {new_name}')

        image_id += 1
    else:
        continue
