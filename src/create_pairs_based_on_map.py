import os
import numpy as np

data_dir = 'C:/Users/Stadtpilot/Desktop/datasets/downsampled_data'
pairs_file = 'C:/Users/Stadtpilot/Desktop/datasets/pairs_based_on_map_15.txt'
map_file = 'C:/Users/Stadtpilot/Desktop/datasets/map_file_15.txt'

range_to_train = 40
file_names = [name for name in sorted(os.listdir(data_dir)) if os.path.isfile(os.path.join(data_dir, name))]
global_ego_list = [name for name in file_names if 'global_ego_data' in name]

points_list = []
for global_ego_file in global_ego_list:
    lidar_filename = str.replace(global_ego_file, 'global_ego_data', 'velodyne_scan')
    global_ego_data = np.fromfile(os.path.join(data_dir, global_ego_file), dtype=np.float64)
    easting = float(global_ego_data[16])
    northing = float(global_ego_data[17])
    points_list.append((easting, northing, lidar_filename))

map_points_list = []
with open(map_file, 'r') as f_m:
    map_points = f_m.readlines()
    for map_point in map_points:
        filename = str.replace(map_point.split(':')[0], 'global_ego_data', 'velodyne_scan')
        easting = float(map_point.split(':')[1].split('#')[0])
        northing = float(map_point.split(':')[1].split('#')[1])
        map_points_list.append((easting, northing, filename))

with open(pairs_file, 'w') as f:
    for point_anchor in points_list:
        points_around = []
        for map_point in map_points_list:
            dist = np.sqrt((point_anchor[0] - map_point[0]) ** 2 + (point_anchor[1] - map_point[1]) ** 2)
            if dist < range_to_train and dist != 0:
                points_around.append((map_point[2], dist))
        for point in points_around:
            f.write(point_anchor[2] + '#' + point[0] + '#' + str(point[1]) + '\n')
