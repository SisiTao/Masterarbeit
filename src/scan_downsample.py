import os
import numpy as np

in_dir = 'E:\Dropbox\MA\example'
out_dir = 'E:\Dropbox\MA\example\outdir_downsample'

file_names = [name for name in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, name))]
lidar_list = [file_name for file_name in file_names if 'velodyne_scan' in file_name]

for lidar_file in lidar_list:
    lidar_file_exp = os.path.join(in_dir, lidar_file)
    print(lidar_file_exp)
    lidar_data = np.fromfile(lidar_file_exp, dtype=np.float32)
    lidar_data = np.reshape(lidar_data, [2, 64, 2000])
    lidar_data_new = np.zeros([2, 64, 2048], dtype=np.float32)

    for d1 in range(2):
        for d2 in range(64):
            idx_o = 0
            idx_n = 0
            for j in range(16):  # |----125----|*16 =2000
                for n in range(3):  # |---25---|+1|---50---|+1|---50---|+1|
                    if n == 0:
                        m = 25
                    else:
                        m = 50
                    for _ in range(m):
                        lidar_data_new[d1][d2][idx_n] = lidar_data[d1][d2][idx_o]
                        idx_n += 1
                        idx_o += 1
                    if idx_o == 2000:
                        lidar_data_new[d1][d2][idx_n] = lidar_data[d1][d2][1999]
                    else:
                        lidar_data_new[d1][d2][idx_n] = (lidar_data[d1][d2][idx_o - 1]
                                                         + lidar_data[d1][d2][idx_o]) / 2
                    idx_n += 1
            assert idx_o == 2000
            assert idx_n == 2048

    lidar_data_new = np.reshape(lidar_data_new, [262144])
    lidar_data_out = np.zeros([65536], dtype=np.float32)
    i = 0
    j = 0
    while i < 262144:
        lidar_data_out[j] = (lidar_data_new[i]
                             + lidar_data_new[i + 1]
                             + lidar_data_new[i + 2]
                             + lidar_data_new[i + 3]) / 4
        i += 4
        j += 1
    assert j == 65536
    print('save')
    lidar_data_out.tofile(os.path.join(out_dir, lidar_file))
