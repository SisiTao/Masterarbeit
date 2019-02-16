import numpy as np
import os

distance_between_points = 15

data_dir = 'C:/Users/Stadtpilot/Desktop/datasets/downsampled_data'
map_file = 'C:/Users/Stadtpilot/Desktop/datasets/map_file_15.txt'
file_names = [name for name in sorted(os.listdir(data_dir)) if os.path.isfile(os.path.join(data_dir, name))]
global_ego_list = [name for name in file_names if 'global_ego_data' in name]

map_points_list = []
map_point = ()
for global_ego_file in global_ego_list:
    global_ego_data = np.fromfile(os.path.join(data_dir, global_ego_file), dtype=np.float64)
    easting = float(global_ego_data[16])
    northing = float(global_ego_data[17])
    point = (easting, northing, global_ego_file)
    if len(map_points_list) == 0:
        map_point = point
        map_points_list.append(map_point)
        print(map_point[2])
    else:
        dist = np.sqrt((map_point[0] - point[0]) ** 2 + (map_point[1] - point[1]) ** 2)
        if dist >= distance_between_points:
            map_point = point
            map_points_list.append(map_point)
            print(map_point[2], ': ', dist)

with open(map_file,'w') as f:
    for map_point in map_points_list:
        f.write(str(map_point[2])+':'+str(map_point[0])+'#'+str(map_point[1])+'\n')
