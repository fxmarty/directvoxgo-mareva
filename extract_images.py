import argparse
import os
import sys

from shutil import copyfile

parser = argparse.ArgumentParser(description='Parser')

parser.add_argument('--folder',
                    help='Folder where .jpg images are stored',
                    type=str,
                    required=True)

args = parser.parse_args()

# normpath to remove the last '/' if written, which must not be there for split
parent_path, folder = os.path.split(os.path.normpath(args.folder))

output_folder = parent_path
output_folder = os.path.join(output_folder, folder + '_extract')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in sorted(os.listdir(args.folder)):
    if filename.endswith('.jpg'):
        parts = filename.split('-')

        n_horizontal_shot = parts[3][2:]  # e.g. hI15 so remove hI
        n_horizontal_shot = int(n_horizontal_shot)

        n_vertical_shot = parts[1][2:]  # e.g. vI2 so remove vI
        n_vertical_shot = int(n_vertical_shot)

        # don't use the very top shots where all cameras are at the same position
        if (n_horizontal_shot % 4 == 0
            and n_vertical_shot % 2 == 0
            and n_vertical_shot != 10):
            print(f'Copying {filename}...')
            copyfile(os.path.join(args.folder, filename),
                     os.path.join(output_folder, filename))