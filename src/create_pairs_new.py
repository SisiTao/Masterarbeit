import os
import numpy as np

data_dir = 'C:/Users/Stadtpilot/Desktop/datasets/downsampled_data'
pairs_file = 'C:/Users/Stadtpilot/Desktop/datasets/pairs_file_40.txt'

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

points_idx = np.arange(len(points_list))
np.random.shuffle(points_idx)
points_list_shuffled = [points_list[i] for i in points_idx]  # avoid all selected points are before the anchor point

with open(pairs_file, 'w') as f:
    for anchor_idx in range(len(points_list_shuffled)):
        point_anchor = points_list_shuffled[anchor_idx]
        distances = np.random.uniform(0.01, range_to_train, 10)
        for dist in distances:
            for point_idx in range(len(points_list_shuffled)):
                point = points_list_shuffled[point_idx]
                if anchor_idx != point_idx:
                    dist_to_anchor = np.sqrt((point_anchor[0] - point[0]) ** 2 + (point_anchor[1] - point[1]) ** 2)
                    if np.abs(dist_to_anchor - dist) < 0.5:
                        f.write(point_anchor[2] + '#' + point[2] + '#' + str(dist_to_anchor) + '\n')
                        break
