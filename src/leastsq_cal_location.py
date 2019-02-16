import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy.optimize import leastsq

MAX_RANGE = 100.0  # maximum observation range

show_animation = True


def calc_input(v, yawrate):
    # v [m/s] yawrate [rad/s]
    u = np.array([[v, yawrate]]).T
    return u


def observation(RFID, embeddings_around, embedding, xEst, xEst_0_x,xEst_0_y):
    tan = (xEst[1, 0] - xEst_0_x) / (xEst[0, 0] - xEst_0_y)
    if tan >= 0:
        p0 = np.array([xEst[0, 0] - 100, xEst[1, 0] + 100])
        p1 = np.array([xEst[0, 0] + 100, xEst[1, 0] - 100])
    else:
        p0 = np.array([xEst[0, 0] + 100, xEst[1, 0] + 100])
        p1 = np.array([xEst[0, 0] - 100, xEst[1, 0] - 100])
    # add noise to gps x-y
    z = np.zeros((0, 3))

    Di = []
    Xi = []
    Yi = []
    for i in range(len(RFID[:, 0])):
        RFID_embedding = embeddings_around[i]
        dist = np.sqrt(np.sum(np.square(np.subtract(RFID_embedding, embedding))))
        zi = np.array([[dist, float(RFID[i, 1]), float(RFID[i, 2])]])
        z = np.vstack((z, zi))

        Di.append(dist)
        Xi.append(float(RFID[i, 1]))
        Yi.append(float(RFID[i, 2]))
    Di = np.array(Di)
    Xi = np.array(Xi)
    Yi = np.array(Yi)
    p0 = leastsq(error, p0, (Xi, Yi, Di))
    p1 = leastsq(error, p1, (Xi, Yi, Di))
    x0, y0 = p0[0]
    x1, y1 = p1[0]
    x = (x0 + x1) / 2
    y = (y0 + y1) / 2
    z = np.array([[x], [y]])

    return z


def error(p, x, y, d):
    err = dist(p, x, y) - d
    return err


def dist(p, x, y):
    x0, y0 = p
    return np.sqrt((x0 - x) ** 2 + (y0 - y) ** 2)


def main(start_point_idx, end_point_idx):
    print(__file__ + " start!!")

    # inputs
    data_dir = 'C:/Users/Stadtpilot/Desktop/datasets/downsampled_data'
    map_file = 'C:/Users/Stadtpilot/Desktop/datasets/map_file_20.txt'
    sorted_file_names = [name for name in sorted(os.listdir(data_dir)) if os.path.isfile(os.path.join(data_dir, name))]

    # get embeddings for map_points and datapoints
    embeddings_file_map = 'C:/Users/Stadtpilot/Desktop/datasets/embeddings_files/embeddings_file_map_20_384.txt'
    embeddings_file_all = 'C:/Users/Stadtpilot/Desktop/datasets/embeddings_files/embeddings_file_downsampled_data_384.txt'
    with open(embeddings_file_map, 'r') as fr:
        data = fr.readline()
        embedding_dict_map = json.loads(data)
    print('embedding_dict_map loaded')
    with open(embeddings_file_all, 'r') as fr:
        data = fr.readline()
        embeddings_dict_all = json.loads(data)
    print('embeddings_dict_all loaded')

    # RFID positions [x, y]
    with open(map_file, 'r') as f:  # 以後可以直接保存成字典
        line = f.readline()
        RFID = []
        while line:
            RFID_filename = line.split(':')[0]
            RFID_easting = np.float(line.split(':')[1].split('#')[0])
            RFID_northing = np.float(line.split(':')[1].split('#')[1])
            RFID_data = [RFID_filename, RFID_easting, RFID_northing]
            RFID.append(RFID_data)
            line = f.readline()

    # Get list of file names for each type of data
    global_ego_list = [file_name for file_name in sorted_file_names if
                       'global' in file_name]
    local_ego_list = [file_name for file_name in sorted_file_names if
                      'local' in file_name]
    combined_list = list(zip(global_ego_list, local_ego_list))

    # State Vector [x y yaw v]' and initialize
    global_ego_data = np.fromfile(os.path.join(data_dir, global_ego_list[start_point_idx]), dtype=np.float64)
    xEst_x = global_ego_data[16]
    xEst_y = global_ego_data[17]
    xEst = np.array([[xEst_x], [xEst_y]])
    xTrue = xEst

    # history
    hxEst = xEst
    hxTrue = xTrue

    err_x_list = []
    err_y_list = []
    err_list = []
    print('calculate start')
    old_idx = 0
    for global_ego_file, local_ego_file in combined_list[start_point_idx:end_point_idx]:

        # embedding of this point
        embedding = embeddings_dict_all[global_ego_file.replace('global_ego_data', 'velodyne_scan')]

        # xTrue
        global_ego_data = np.fromfile(os.path.join(data_dir, global_ego_file), dtype=np.float64)
        xTrue = np.array([[global_ego_data[16]], [global_ego_data[17]]])

        # observation
        RFID_around = []

        embeddings_around = []
        dists = []
        if old_idx <= 10:
            start_idx = 0
        else:
            start_idx = old_idx - 10
        end_idx = old_idx + 10
        for r in range(start_idx, end_idx):
            RFID_data = RFID[r]
            dist = np.sqrt((RFID_data[1] - xEst[0, 0]) ** 2 + (RFID_data[2] - xEst[1, 0]) ** 2)
            if dist < MAX_RANGE:
                if len(RFID_around) == 4:
                    if dist < max(dists):
                        max_idx = dists.index(max(dists))
                        dists.remove(max(dists))
                        embeddings_around.remove(embeddings_around[max_idx])
                        RFID_around.remove(RFID_around[max_idx])

                        dists.append(dist)
                        embeddings_around.append(
                            embedding_dict_map[RFID_data[0].replace('global_ego_data', 'velodyne_scan')])
                        RFID_around.append(RFID_data)
                        old_idx = r
                else:
                    RFID_around.append(RFID_data)
                    embeddings_around.append(
                        embedding_dict_map[RFID_data[0].replace('global_ego_data', 'velodyne_scan')])
                    dists.append(dist)
                    old_idx = r
        RFID_around = np.array(RFID_around)
        if RFID_around.size == 0:
            print(global_ego_file)
            print(RFID_around[:, 0])
            print(len(RFID_around[:, 0]))
        if len(hxEst[0]) ==1:
            xEst_0_x, xEst_0_y=hxEst[0][-1]+1,hxEst[1][-1]+1
        else:
            xEst_0_x, xEst_0_y=hxEst[0][-2],hxEst[1][-2]

        xEst = observation(RFID_around, embeddings_around, embedding, xEst, xEst_0_x, xEst_0_y)

        err_x = np.abs(xEst[0, 0] - xTrue[0, 0])
        err_y = np.abs(xEst[1, 0] - xTrue[1, 0])
        err_x_list.append(err_x)
        err_y_list.append(err_y)
        err_list.append(np.sqrt(err_x ** 2 + err_y ** 2))
        print(np.sqrt(err_x ** 2 + err_y ** 2))

        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxTrue = np.hstack((hxTrue, xTrue))

        if show_animation:
            plt.cla()

            # for i in range(len(z[:, 0])):
            #     plt.plot([xEst[0, 0], z[i, 1]], [xEst[1, 0], z[i, 2]], "-k")
            # plt.plot(RFID_around[:, 1], RFID_around[:, 2], "*k")
            # plt.plot(px[0, :], px[1, :], ".r")
            plt.plot(np.array(hxTrue[0, :]).flatten(),
                     np.array(hxTrue[1, :]).flatten(), "-b")
            plt.plot(np.array(hxEst[0, :]).flatten(),
                     np.array(hxEst[1, :]).flatten(), "-r")
            # plot_covariance_ellipse(xEst, PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)

    avg_err_x = np.mean(np.array(err_x_list))
    avg_err_y = np.mean(np.array(err_y_list))
    avg_err = np.mean(np.array(err_list))
    print('err_x:\t', avg_err_x)
    print('err_y:\t', avg_err_y)
    print('err:\t', avg_err)


if __name__ == '__main__':
    main(0, 8000)
