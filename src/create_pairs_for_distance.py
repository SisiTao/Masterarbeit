import os
import numpy as np
import shutil

range_to_train = 55
offset = 10

data_dir = 'C:/Users/Stadtpilot/Desktop/datasets/downsampled_data'
pairs_file = 'C:/Users/Stadtpilot/Desktop/datasets/pairs_file.txt'

file_names = [name for name in sorted(os.listdir(data_dir)) if os.path.isfile(os.path.join(data_dir, name))]
global_ego_list = [name for name in file_names if 'global_ego_data' in name]

points_list = []
for global_ego_file in global_ego_list:
    global_ego_data = np.fromfile(os.path.join(data_dir, global_ego_file), dtype=np.float64)
    easting = float(global_ego_data[16])
    northing = float(global_ego_data[17])
    num = global_ego_file[0:6]  # Index of datafile
    points_list.append((easting, northing, num))

start_point = points_list[0]
points_sublists = []
points_sublist = []

for point in points_list:
    points_sublist.append(point)
    dist = np.sqrt((start_point[0] - point[0]) ** 2 + (start_point[1] - point[1]) ** 2)
    if dist >= range_to_train:
        start_point = point
        points_sublists.append(points_sublist)
        points_sublist = []

pairs_list=[]
for points_sublist in points_sublists:
    for i in range(len(points_sublist)):
        for j in range(i+1,len(points_sublist)):
            pair=[points_sublist[i][2],points_sublist[j][2]]
            pair=[num+'_velodyne_scan.bin' for num in pair]
            dist=np.sqrt((points_sublist[i][0] - points_sublist[j][0]) ** 2 +
                         (points_sublist[i][1] - points_sublist[j][1]) ** 2)
            pair.append(dist)
            pairs_list.append(pair)

with open(pairs_file,'w') as f:
    for pair in pairs_list:
        f.write(pair[0]+'#'+pair[1]+'#'+str(pair[2])+'\n')

