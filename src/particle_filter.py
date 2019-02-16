import numpy as np
import math
import matplotlib.pyplot as plt
import os
import json

# Estimation parameter of PF
Q = np.diag([1.5]) ** 2  # range error
# R = np.diag([1.0, 1.0]) ** 2  # input error

#  Simulation parameter
# Qsim = np.diag([0.2]) ** 2
Rsim = np.diag([1., np.deg2rad(40.0)]) ** 2
print('Q:\t', Q)
print('R:\t', Rsim)
DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 40.  # maximum observation range

nrof_RFID_around=4
# Particle filter parameter
NP = 100  # Number of Particle
NTh = NP / 2.0  # Number of particle for re-sampling

show_animation = True


def calc_input(v, yawrate):
    # v [m/s] yawrate [rad/s]
    u = np.array([[v, yawrate]]).T
    return u


def observation(xd, u, RFID, embeddings_around, embedding):
    # add noise to gps x-y
    z = np.zeros((0, 3))

    for i in range(len(RFID[:, 0])):
        RFID_embedding = embeddings_around[i]
        dist = np.sqrt(np.sum(np.square(np.subtract(RFID_embedding, embedding))))
        zi = np.array([[dist, float(RFID[i, 1]), float(RFID[i, 2])]])
        z = np.vstack((z, zi))

    # add noise to input
    ud = u

    xd = motion_model(xd, ud)

    return z, xd, ud


def motion_model(x, u):
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])
    x = F.dot(x) + B.dot(u)

    return x


def gauss_likelihood(x, sigma):
    p = 1.0 / math.sqrt(2.0 * math.pi * sigma ** 2) * \
        math.exp(-x ** 2 / (2 * sigma ** 2))

    return p


def calc_covariance(xEst, px, pw):
    cov = np.zeros((2, 2))

    for i in range(px.shape[1]):
        dx = (px[:, i] - xEst)[0:2]
        cov += pw[0, i] * dx.dot(dx.T)

    return cov


def pf_localization(px, pw, z, u):
    """
    Localization with Particle filter
    """

    for ip in range(NP):
        x = np.array([px[:, ip]]).T
        w = pw[0, ip]
        #  Predict with random input sampling
        ud1 = u[0, 0] + np.random.randn() * Rsim[0, 0]
        ud2 = u[1, 0] + np.random.randn() * Rsim[1, 1]
        ud = np.array([[ud1, ud2]]).T
        x = motion_model(x, ud)

        #  Calc Importance Weight
        for i in range(len(z[:, 0])):
            dx = x[0, 0] - z[i, 1]
            dy = x[1, 0] - z[i, 2]
            prez = math.sqrt(dx ** 2 + dy ** 2)
            dz = prez - z[i, 0]
            # if z[i, 0] < 25:
            #     Q = np.diag([1.5]) ** 2
            # else:
            #     Q = np.diag([6.8]) ** 2
            w = w * gauss_likelihood(dz, math.sqrt(Q[0, 0]))

        px[:, ip] = x[:, 0]
        pw[0, ip] = w

    pw = pw / pw.sum()  # normalize
    if str(pw[0, 0]) == 'nan':
        print('error')
    xEst = px.dot(pw.T)
    PEst = calc_covariance(xEst, px, pw)

    px, pw = resampling(px, pw)

    return xEst, PEst, px, pw


def resampling(px, pw):
    """
    low variance re-sampling
    """

    Neff = 1.0 / (pw.dot(pw.T))[0, 0]  # Effective particle number
    if Neff < NTh:
        wcum = np.cumsum(pw)
        base = np.cumsum(pw * 0.0 + 1 / NP) - 1 / NP
        resampleid = base + np.random.rand(base.shape[0]) / NP

        inds = []
        ind = 0
        for ip in range(NP):
            while resampleid[ip] > wcum[ind]:
                ind += 1
            inds.append(ind)

        px = px[:, inds]
        pw = np.zeros((1, NP)) + 1.0 / NP  # init weight

    return px, pw


def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)

    # eigval[bigind] or eiqval[smallind] were occassionally negative numbers extremely
    # close to 0 (~10^-20), catch these cases and set the respective variable to 0
    try:
        a = math.sqrt(eigval[bigind])
    except ValueError:
        a = 0

    try:
        b = math.sqrt(eigval[smallind])
    except ValueError:
        b = 0

    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
    R = np.array([[math.cos(angle), math.sin(angle)],
                  [-math.sin(angle), math.cos(angle)]])
    fx = R.dot(np.array([[x, y]]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")


def main(start_point_idx, end_point_idx):
    print(__file__ + " start!!")

    time = 0.0

    # inputs
    data_dir = 'C:/Users/Stadtpilot/Desktop/datasets/downsampled_data'
    map_file = 'C:/Users/Stadtpilot/Desktop/datasets/map_file_15.txt'
    sorted_file_names = [name for name in sorted(os.listdir(data_dir)) if os.path.isfile(os.path.join(data_dir, name))]

    # get embeddings for map_points and datapoints
    embeddings_file_map = 'C:/Users/Stadtpilot/Desktop/datasets/embeddings_files/embeddings_file_map_15_384.txt'
    embeddings_file_all = 'C:/Users/Stadtpilot/Desktop/datasets/embeddings_files/embeddings_file_downsampled_data_384_175555.txt'
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
    xEst_yaw = global_ego_data[4]
    xEst_v = np.sqrt(global_ego_data[13] ** 2 + global_ego_data[14] ** 2)
    xEst = np.array([[xEst_x], [xEst_y], [xEst_yaw], [xEst_v]])
    xTrue = xEst

    px = np.stack((np.tile([xEst_x], NP), np.tile([xEst_y], NP), np.tile([xEst_yaw], NP),
                   np.tile([xEst_v], NP)))  # Particle store
    pw = np.zeros((1, NP)) + 1.0 / NP  # Particle weight
    xDR = xEst  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    err_x_list = []
    err_y_list = []
    err_list = []

    print('calculate start')
    old_idx = 0
    for global_ego_file, local_ego_file in combined_list[start_point_idx:end_point_idx - 1]:
        time += DT

        # embedding of this point
        embedding = embeddings_dict_all[global_ego_file.replace('global_ego_data', 'velodyne_scan')]

        # control input
        local_ego_data = np.fromfile(os.path.join(data_dir, local_ego_file), dtype=np.float64)
        v_longitudinal = local_ego_data[7]
        v_lateral = local_ego_data[8]
        yawrate = local_ego_data[10]
        v = np.sqrt(v_longitudinal ** 2 + v_lateral ** 2)
        u = calc_input(v, yawrate)

        # xTrue
        global_ego_data = np.fromfile(os.path.join(data_dir, global_ego_file), dtype=np.float64)
        xTrue = np.array([[global_ego_data[16]], [global_ego_data[17]], [0.], [0.]])

        # observation
        RFID_around = []

        embeddings_around = []
        dists = []
        if old_idx <= 10:
            start_idx = 0
        else:
            start_idx = old_idx - 10
        end_idx = old_idx + 15
        for r in range(start_idx, end_idx):
            RFID_data = RFID[r]
            dist = np.sqrt((RFID_data[1] - xEst[0, 0]) ** 2 + (RFID_data[2] - xEst[1, 0]) ** 2)
            if dist < MAX_RANGE:
                if len(RFID_around) == nrof_RFID_around:
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

        z, xDR, ud = observation(xDR, u, RFID_around, embeddings_around, embedding)

        xEst, PEst, px, pw = pf_localization(px, pw, z, ud)
        err_x = np.abs(xEst[0, 0] - xTrue[0, 0])
        err_y = np.abs(xEst[1, 0] - xTrue[1, 0])
        err_x_list.append(err_x)
        err_y_list.append(err_y)
        err_list.append(np.sqrt(err_x ** 2 + err_y ** 2))
        print(np.sqrt(err_x ** 2 + err_y ** 2))
        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))

        if show_animation:
            plt.cla()

            for i in range(len(z[:, 0])):
                plt.plot([xEst[0, 0], z[i, 1]], [xEst[1, 0], z[i, 2]], "-k")
            # plt.plot(RFID_around[:, 1], RFID_around[:, 2], "*k")
            # plt.plot(px[0, :], px[1, :], ".r")
            plt.plot(np.array(hxTrue[0, :]).flatten(),
                     np.array(hxTrue[1, :]).flatten(), "-b")
            plt.plot(np.array(hxDR[0, :]).flatten(),
                     np.array(hxDR[1, :]).flatten(), "-k")
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
    print('Q:\t', Q)
    print('R:\t', Rsim)


if __name__ == '__main__':
    main(0, 8000)
