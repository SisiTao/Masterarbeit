import numpy as np
import os
import shutil

in_dir = 'C:/Users/Stadtpilot/Desktop/datasets/example_1'
out_dir = 'C:/Users/Stadtpilot/Desktop/datasets/example_2'
# in_dir = 'C:/Users/Stadtpilot/Desktop/datasets/downsampled_data'
# out_dir = 'C:/Users/Stadtpilot/Desktop/datasets/downsampled_data_rotated'
if os.path.exists(out_dir) is False:
    os.mkdir(out_dir)
file_names = [name for name in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, name))]
lidar_list = [file_name for file_name in file_names if 'velodyne_scan' in file_name]

for lidar_file in lidar_list:
    lidar_file_exp = os.path.join(in_dir, lidar_file)
    lidar_data = np.fromfile(lidar_file_exp, dtype=np.float32)
    lidar_data = np.reshape(lidar_data, [2, 64, 512])
    lidar_data_cutbottom = np.zeros([2, 64, 512], dtype=np.float32)
    for i in range(2):
        for j in range(10, 64):
            for k in range(512):
                lidar_data_cutbottom[i][j][k] = lidar_data[i][j][k]

    rotate_pixel = np.random.randint(512)
    rotate_deg = rotate_pixel * 0.703125
    rotate_deg = np.array(rotate_deg, dtype=np.float64)
    rotate_deg.tofile(os.path.join(out_dir, lidar_file.replace('velodyne_scan', 'rotate_deg')))
    lidar_data_cutbottom = np.reshape(lidar_data_cutbottom, [128, 512])
    lidar_data_rotated = np.zeros([128, 512], dtype=np.float32)
    for row in range(128):
        idx = rotate_pixel
        for i in range(512):
            lidar_data_rotated[row][i] = lidar_data_cutbottom[row][idx]
            idx += 1
            if idx == 512:
                idx = 0
    lidar_data_rotated = np.reshape(lidar_data_rotated, [65536])

    lidar_data_rotated.tofile(os.path.join(out_dir, lidar_file))
    file_name_gps = lidar_file.replace('velodyne_scan', 'global_ego_data')
    file_name_odometry=lidar_file.replace('velodyne_scan','local_ego_data')
    shutil.copyfile(os.path.join(in_dir,file_name_gps),os.path.join(out_dir,file_name_gps))
    shutil.copyfile(os.path.join(in_dir, file_name_odometry), os.path.join(out_dir, file_name_odometry))
