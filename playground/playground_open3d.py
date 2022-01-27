import open3d as o3d
import numpy as np

import copy
import time

def create_transform_matrix_from_z(z):
    """ Return transform 4x4 transformation matrix given a Z value """
    result = np.identity(4)
    result[2,3] = z # Change the z

    return result

"""
# Create Open3d visualization window
vis = o3d.visualization.Visualizer()
vis.create_window()

# create sphere geometry
sphere1 = o3d.geometry.TriangleMesh.create_sphere(0.5)
vis.add_geometry(sphere1)

# create coordinate frame
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
vis.add_geometry(coordinate_frame)

vis.poll_events()
vis.update_renderer()
#vis.run()

print('before sleep')
time.sleep(3)

sphere2 = o3d.geometry.TriangleMesh.create_sphere(0.1)

vis.remove_geometry(sphere1)
vis.add_geometry(sphere2)
vis.poll_events()
vis.update_renderer()
#vis.run()
"""


# Create Open3d visualization window
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

# create sphere geometry
sphere1 = o3d.geometry.TriangleMesh.create_sphere(0.5)
vis.add_geometry(sphere1)

# create coordinate frame
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
vis.add_geometry(coordinate_frame)

vis.poll_events()
vis.update_renderer()
#vis.run()

print('before sleep')
time.sleep(3)

sphere2 = o3d.geometry.TriangleMesh.create_sphere(0.1)

def your_update_function():
    #Your update routine
    vis.remove_geometry(sphere1)
    vis.add_geometry(sphere2)
    vis.update_renderer()
    vis.poll_events()
    vis.run()