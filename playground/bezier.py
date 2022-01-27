import matplotlib.pylab as plt

import bezier
import numpy as np


def dist(x, y):
    return np.sqrt(np.sum((x - y)**2))

##

# Define the Bezier curve
nodes = np.array([
        [0.0, 0.2, 1.0, 0.5, 2],
        [0.0, 1.8, 0.3, 2, 3],
        [0.0, 1.8, 0.3, 2, 3] ])

"""
nodes = np.array([
        [0.0, 0.2, 1.0, 0.5, 2],
        [0.0, 1.8, 0.3, 2, 4]])
"""

curve = bezier.Curve(nodes, degree=4)

t_fine = np.linspace(0, 1, 5000) # Curvilinear coordinate
points_fine = curve.evaluate_multi(t_fine)
points_fine.shape  # (2, 60)


points_fine = np.array(points_fine).T
kept_points = [points_fine[0]]

for point in points_fine:
    if dist(kept_points[-1], point) > 0.1:
        kept_points.append(point.copy())

kept_points = np.array(kept_points)

kept_points = kept_points.T
points_fine = points_fine.T

##
# Plot
#plt.plot(*nodes, '-o', label='definition nodes')
plt.plot(*points_fine, 'ok', label='Bezier curve', markersize=1)
plt.plot(kept_points[0], kept_points[1], 'x', label='regularly spaced along x', markersize=8)
plt.xlabel('x'); plt.ylabel('y'); plt.legend()

plt.grid()
plt.show()

##

for i, point in enumerate(kept_points.T):
    print(dist(kept_points[:, i], kept_points[:, i + 1]))
##
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

ax.scatter(kept_points[0], kept_points[1], kept_points[2])

plt.show()
