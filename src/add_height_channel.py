import numpy as np
import os
import shutil

# in_dir = 'D:/User/Sisi_Tao/downsampled_data_4channels'
# out_dir = 'D:/User/Sisi_Tao/downsampled_data_5channels'
in_dir = 'C:/Users/Stadtpilot/Desktop/datasets/example_3'
out_dir = 'C:/Users/Stadtpilot/Desktop/datasets/example_4'
if os.path.exists(out_dir) is False:
    os.mkdir(out_dir)
file_names = [name for name in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, name))]
lidar_list = [file_name for file_name in file_names if 'velodyne_scan' in file_name]
# angles_file = 'D:/User/Sisi_Tao/leonie_sensor_angles.txt'
angles_file = 'C:/Users/Stadtpilot/Desktop/datasets/leonie_sensor_angles.txt'
with open(angles_file, 'r') as f:
    angles = f.readlines()
    angles = [float(angle) for angle in angles]
    angles = np.flip(np.array(angles), axis=0)

for lidar_file in lidar_list:
    lidar_file_exp = os.path.join(in_dir, lidar_file)
    lidar_data = np.fromfile(lidar_file_exp, dtype=np.float32)
    lidar_data = np.reshape(lidar_data, [4, 64, 512])
    distance = lidar_data[0]
    lidar_data_new = np.zeros([5, 64, 512], dtype=np.float32)
    lidar_data_new[0:4] = lidar_data

    for i in range(64):
        for j in range(512):
            phi = np.deg2rad(angles[i])
            d = distance[i][j]
            if np.tan(phi) == 0:
                lidar_data_new[4][i][j] = 0
            else:
                a = 1 + 1 / np.square(np.tan(phi))
                lidar_data_new[4][i][j] = np.float32(d / np.sqrt(a))
            if i <57 :
                lidar_data_new[4][i][j] = 0 - lidar_data_new[4][i][j]
    lidar_data_new = np.reshape(lidar_data_new, [5*64*512])
    lidar_data_new.tofile(os.path.join(out_dir, lidar_file))

    file_name_gps = lidar_file.replace('velodyne_scan', 'global_ego_data')
    file_name_odometry = lidar_file.replace('velodyne_scan', 'local_ego_data')
    shutil.copyfile(os.path.join(in_dir, file_name_gps), os.path.join(out_dir, file_name_gps))
    shutil.copyfile(os.path.join(in_dir, file_name_odometry), os.path.join(out_dir, file_name_odometry))
