import numpy as np
import os

# in_dir = 'D:/User/Sisi_Tao/rotated_downsampled_data'
# out_dir = 'D:/User/Sisi_Tao/rotated_downsampled_data_3channels'
in_dir = 'C:/Users/Stadtpilot/Desktop/datasets/bev'

file_names = [name for name in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, name))]
lidar_list = [file_name for file_name in file_names if 'velodyne_scan' in file_name]

for lidar_file in lidar_list:
    lidar_file_exp = os.path.join(in_dir, lidar_file)
    lidar_data = np.fromfile(lidar_file_exp, dtype=np.float32)
    lidar_data = np.reshape(lidar_data, [3, 400, 400])

    for i in range(400):
        for j in range(400):
            lidar_data[2][i][j] = np.sqrt((i * 0.25 - 200 * 0.25) ** 2 + (j * 0.25 - 200 * 0.25) ** 2)

    lidar_data = np.reshape(lidar_data, [400 * 400 * 3])
    lidar_data.tofile(lidar_file_exp)
    print(lidar_file)

    # file_name_gps = lidar_file.replace('velodyne_scan', 'global_ego_data')
    # file_name_odometry = lidar_file.replace('velodyne_scan', 'local_ego_data')
    # shutil.copyfile(os.path.join(in_dir, file_name_gps), os.path.join(out_dir, file_name_gps))
    # shutil.copyfile(os.path.join(in_dir, file_name_odometry), os.path.join(out_dir, file_name_odometry))
