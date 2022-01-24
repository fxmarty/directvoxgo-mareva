from collections import defaultdict
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Parser')

parser.add_argument('--path',
                    help='Path to a cameras.txt file from colmap',
                    type=str,
                    required=True)

parser.add_argument('--save-file',
                    help='File to save the DirectVoxGO-friendly camera matrix as .txt',
                    type=str,
                    required=True)

args = parser.parse_args()


# open file for reading, and output file for writing to
with open(args.path) as f:
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
    print(mt)

with open(args.save_file, mode='w') as out:
    for l in mt:
        for ele in l:
            out.write(str(ele) + ' ')

        out.write('\n')