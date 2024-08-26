from glob import glob
import json
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

"""
We want to find a relationship between puck location in frame and distance from kart in world coordinates
"""

base_dir = '/root/cs342/final/'
data_dir = 'random_render_data'


def distance_between_points(p1, p2):
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


distances = []
projs = []
angles = []
for file_name in glob(os.path.join(base_dir, data_dir, '*.json')):
    with open(file_name) as f:
        j = json.load(f)

    puck = j.get('puck_world_coords')
    # skip this frame if puck not in view or puck is in air
    if not j.get('puck_in_view') or puck and puck[1] > 1.0:
        continue


    kart = j.get('kart_world_coords')
    pred = j.get('puck_img_coords')
    if -0.25 <= pred[1] <= 0.0:
        distances.append(distance_between_points(puck, kart))
        projs.append(pred[-1])
        angles.append(pred[0])


def exp_func(x, a, b):
    return a*np.exp(b*x)

popt, pcov = curve_fit(exp_func, projs, distances)
a, b = popt.tolist()
print(f'popt: {popt}')
print(f'pcov: {pcov}')
print(f'a: {a} | b: {b}')
a, b = 3.0, -12.0

# 2D
plt.plot(projs, distances, 'o', color='r')
plt.plot(projs, exp_func(np.array(projs), a, b), 'o', color='g')
# plt.yscale('log')
plt.xlabel('projected y')
plt.ylabel('distance')
plt.show()

# 3D
# ax = plt.axes(projection='3d')
# ax.scatter3D(angles, projs, distances, 'o', color='r')
# ax.set_xlabel('projected x')
# ax.set_ylabel('projected y')
# ax.set_zlabel('distance')
# plt.show()
