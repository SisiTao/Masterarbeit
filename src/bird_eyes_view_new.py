import numpy as np
import os
import math

resolution = 0.25
side_length = 100
side_cells = int(side_length / resolution)
half_side_cells = side_cells / 2

# in_dir = 'D:/User/Sisi_Tao/rotated_downsampled_data'
# out_dir = 'D:/User/Sisi_Tao/rotated_downsampled_data_3channels'
in_dir = 'C:/Users/Stadtpilot/Desktop/datasets/example'
out_dir = 'C:/Users/Stadtpilot/Desktop/datasets/example_6'

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
    angles = np.deg2rad(angles)

for lidar_file in lidar_list:
    lidar_file_exp = os.path.join(in_dir, lidar_file)
    lidar_data = np.fromfile(lidar_file_exp, dtype=np.float32)
    lidar_data = np.reshape(lidar_data, [2, 64, 2000])
    distance = lidar_data[0]
    reflection = lidar_data[1]
    lidar_data_new = np.zeros([3, side_cells, side_cells], dtype=np.float32)
    rotate_deg = np.random.uniform(0., 360.)
    for i in range(64):
        for j in range(2000):

            theta = np.deg2rad(j * 360 / 2000 + rotate_deg)
            phi = angles[i]
            d = distance[i][j]
            if i < 10:
                d = 0
            a = np.square(np.tan(phi))
            b = np.square(np.tan(theta))
            if b == 0:
                x = 0
            else:
                x = np.float32(d / np.sqrt((1 + a) * (1 + 1 / b)))
            if np.sin(theta) < 0:
                x = 0 - x
            y = np.float32(d / np.sqrt((1 + a) * (1 + b)))
            if np.cos(theta) < 0:
                y = 0 - y
            if a == 0:
                height = 0
            else:
                height = np.float32(d / np.sqrt((1 + 1 / a)))
            if phi < 0:
                height = 0 - height
            height += 2
            if np.abs(x) < (side_length / 2) and np.abs(y) < (side_length / 2):
                h = int(math.ceil(x / resolution) + half_side_cells - 1)
                w = int(math.ceil(y / resolution) + half_side_cells - 1)
                if lidar_data_new[0][h][w] < height:
                    lidar_data_new[0][h][w] = height
                lidar_data_new[1][h][w] += reflection[i][j]
                lidar_data_new[2][h][w] += 1
    for h in range(side_cells):
        for w in range(side_cells):
            if lidar_data_new[2][h][w] != 0:
                lidar_data_new[1][h][w] = lidar_data_new[1][h][w] / lidar_data_new[2][h][w]
            if lidar_data_new[2][h][w] > 1000:
                lidar_data_new[2][h][w] = 0
    lidar_data_new = np.reshape(lidar_data_new, [side_cells ** 2 * 3])
    lidar_data_new.tofile(os.path.join(out_dir, lidar_file))
    print(lidar_file)

    # file_name_gps = lidar_file.replace('velodyne_scan', 'global_ego_data')
    # file_name_odometry = lidar_file.replace('velodyne_scan', 'local_ego_data')
    # shutil.copyfile(os.path.join(in_dir, file_name_gps), os.path.join(out_dir, file_name_gps))
    # shutil.copyfile(os.path.join(in_dir, file_name_odometry), os.path.join(out_dir, file_name_odometry))
