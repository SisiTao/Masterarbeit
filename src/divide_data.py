import os
import numpy as np
import shutil

dist_between_classes = 50
class_radius = 7
min_nrof_points_per_class = 9
offsets = [0, 16, 32]

class_kern_list0 = []
class_kern_list0_idx = []
for offset in offsets:
    data_dir = 'C:/Users/Stadtpilot/Desktop/datasets/downsampled_data'
    new_dir = 'C:/Users/Stadtpilot/Desktop/datasets/divided_data_' + str(offset)
    os.mkdir(new_dir)
    file_names = [name for name in sorted(os.listdir(data_dir)) if os.path.isfile(os.path.join(data_dir, name))]
    global_ego_list = [name for name in file_names if 'global_ego_data' in name]
    global_ego_list.sort()
    easting_list = []
    northing_list = []
    points_list = []
    for global_ego_file in global_ego_list:
        global_ego_data = np.fromfile(os.path.join(data_dir, global_ego_file), dtype=np.float64)
        easting = float(global_ego_data[16])
        easting_list.append(easting)
        northing = float(global_ego_data[17])
        northing_list.append(northing)
        num = global_ego_file[0:6]  # Index of datafile
        points_list.append((easting, northing, num))
        # print(num, ': ', 'easting: ', easting, '\t', 'northing: ', northing)
    # easting_list.sort()
    # northing_list.sort()
    # print('min_easting=', easting_list[0])
    # print('max_easing=', easting_list[len(easting_list) - 1])
    # print('min_northing=', northing_list[0])
    # print('max_northing=', northing_list[len(northing_list) - 1])
    #
    # # compute the central coordinate of each class
    # east_west_dist = easting_list[len(easting_list) - 1] - easting_list[0]
    # north_south_dist = northing_list[len(northing_list) - 1] - northing_list[0]
    class_kern_list = []
    if offset == 0:
        start_kern_idx = 0

        kern_idx = start_kern_idx

        kern_x = points_list[kern_idx][0]
        kern_y = points_list[kern_idx][1]
        class_kern = [kern_x, kern_y]
        class_kern_list.append(class_kern)
        class_kern_list0_idx.append(kern_idx)
        while True:
            kern_idx += 1
            dist = np.sqrt((kern_x - points_list[kern_idx][0]) ** 2 + (kern_y - points_list[kern_idx][1]) ** 2)
            if dist >= dist_between_classes:
                kern_x = points_list[kern_idx][0]
                kern_y = points_list[kern_idx][1]
                class_kern = points_list[kern_idx]
                class_kern_list.append(class_kern)
                class_kern_list0_idx.append(kern_idx)
            if kern_idx + 2 > len(points_list):
                kern_idx = -1
            if kern_idx + 1 == start_kern_idx:
                break
        class_kern_list0 = class_kern_list
    else:
        for i in range(len(class_kern_list0)):
            kern_idx0 = class_kern_list0_idx[i]
            kern_x0 = class_kern_list0[i][0]
            kern_y0 = class_kern_list0[i][1]
            kern_idx = kern_idx0
            while True:
                kern_idx += 1
                dist = np.sqrt((kern_x0 - points_list[kern_idx][0]) ** 2 + (kern_y0 - points_list[kern_idx][1]) ** 2)
                if dist >= offset:
                    class_kern_list.append(points_list[kern_idx])
                    break
                if kern_idx + 2 > len(points_list):
                    kern_idx = -1
                if kern_idx + 1 == start_kern_idx:
                    break
    # Put the points around class_kern in each class
    nrof_sampled_points = 0
    for class_kern in class_kern_list:
        kern_x = class_kern[0]
        kern_y = class_kern[1]
        points_in_class = []
        for point in points_list:  # the structure of point:(num, easting, northing)
            if abs(kern_x - point[0]) < class_radius and abs(kern_y - point[1]) < class_radius:
                dist = np.sqrt((kern_x - point[0]) ** 2 + (kern_y - point[1]) ** 2)
                if dist < class_radius:
                    points_in_class.append(point)
        # Only save the class with at least min_nrof_points_per_class points
        if len(points_in_class) > (min_nrof_points_per_class - 1):
            class_name = str(int(kern_x)) + '_' + str(int(kern_y))
            class_dir = os.path.join(new_dir, class_name)
            os.mkdir(class_dir)
            # copy the files to the new path
            for point in points_in_class:
                src_file_global_ego = os.path.join(data_dir, point[2] + '_global_ego_data.bin')
                src_file_lidar_scan = os.path.join(data_dir, point[2] + '_velodyne_scan.bin')
                dst_file_global_ego = os.path.join(class_dir, point[2] + '_global_ego_data.bin')
                dst_file_lidar_scan = os.path.join(class_dir, point[2] + '_velodyne_scan.bin')
                shutil.copyfile(src_file_global_ego, dst_file_global_ego)
                shutil.copy(src_file_lidar_scan, dst_file_lidar_scan)
            nrof_sampled_points += len(points_in_class)
            print('points_in_class: ', len(points_in_class))
    print('nrof_sampled_points: ', nrof_sampled_points)
