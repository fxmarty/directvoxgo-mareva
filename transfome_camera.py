# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 10:39:04 2022

@author: ASUS
"""

from collections import defaultdict
import numpy as np

# open file for reading, and output file for writing to
with open("D:\Mareva_3D_construction\data\Take1-Cube\colmap_test_cube_lowdim\pinhole_model\cameras.txt") as f, open("D:\Mareva_3D_construction\data\Take1-Cube\colmap_test_cube_lowdim\pinhole_model\cameras_tf.txt", mode="w") as out:
    d = defaultdict(dict)
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
    for l in mt:
        #np.savetxt(out, l, fmt='%.1f')
        for ele in l:
            out.write(str(ele) + ' ')
            
        out.write('\n')