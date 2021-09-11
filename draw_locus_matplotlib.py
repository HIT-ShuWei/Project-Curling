import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import os


def get_xy(xy_path):
    # 将坐标点的集合文件.csv读取，并转换成numpy数组进行返回
    xy_csv = open(xy_path, encoding='utf-8')
    xy_data = pd.read_csv(xy_csv, header=None)
    print('loaded data from CSV!')
    xy_numpy = xy_data.to_numpy()
    return xy_numpy


def plot_single_locus(img_path, xy_path, ratio=1):
    # 绘制单个轨迹在单个视角下的轨迹
    # img-背景图片  xy-坐标轨迹的numpy文件

    # 读取冰壶场景图片作为背景
    img = plt.imread(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, 510.6 * ratio, 0, 161 * ratio])

    # 读取坐标文件
    xy = get_xy(xy_path)
    print('xy is loaded!')

    # 读取单应矩阵
    H_path = 'Homography_array/Homography-1.csv'
    f = open(H_path, encoding='utf-8')
    H_pd = pd.read_csv(f, header=None)
    H = H_pd.to_numpy(dtype=float)
    print('H is loaded!')

    # 将坐标文件转换为世界坐标系
    x_world, y_world = [], []
    for i, [x,y,id,time] in enumerate(xy):
        # if id != 1:
        #     continue
        print('id=',id)
        xy_src = np.float32([[x],[y],[1]])
        xy_world = np.matmul(H, xy_src)
        x_world.append(xy_world[0]/xy_world[2])
        y_world.append(xy_world[1]/xy_world[2])

    # 将坐标轨迹动态描述出来
    x_motion, y_motion = [x_world[0], x_world[1]], [y_world[0], y_world[1]]
    plt.ion()
    for i in range(2, len(x_world)):
        x_motion[0], y_motion[0] = x_motion[1], y_motion[1]
        x_motion[1], y_motion[1] = x_world[i], y_world[i]
        plt.plot(x_motion,y_motion,'r')
        plt.pause(0.01)
    plt.pause(10)

    # 平滑曲线
    # from scipy import  interpolate
    # from scipy.interpolate import interp1d
    # x_new = np.linspace(min(x_world), max(x_world), 100)
    # f = interp1d(x_world, y_world, kind='linear')
    # ax.plot(x_new, f(x_new))


    # 原图
    # plt.plot(x_world, y_world, 'r')
    # plt.show()

    return


def plot_whole_locus(img_path, locus_path_1, locus_path_2,locus_path_3, homograthy_1, homography_2, homograthy_3, ratio=1):
    '''
    功能：根据存储好的csv坐标文件，绘制全场（3视角下）的多个冰壶球运动轨迹
    输入参数：1.场地背景图片 2.冰壶轨迹坐标 3.对应单应性矩阵存储路径 4.比例尺
    '''

    # 读取冰壶场景图片作为背景
    img = plt.imread(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, 510.6 * ratio, 0, 161 * ratio])

    # 读取坐标文件
    locus_1 = get_xy(locus_path_1)
    locus_2 = get_xy(locus_path_2)
    locus_3 = get_xy(locus_path_3)
    print('locus has been loaded!')

    # 读取单应性矩阵
    h_1 = get_xy(homograthy_1)
    h_2 = get_xy(homography_2)
    h_3 = get_xy(homograthy_3)

    # 模拟帧读取并进行绘制
    count = 0
    frame_1, frame_2, frame_3 = 0, 0, 0 # 三个指针分别指向三个坐标矩阵，初始化0为起始位置
    dict_1, dict_2, dict_3 = {}, {}, {} # 三个字典分别存储
    relenish_1, replenish_2, replenish_3 = -23, 0, 5   # 每个相机因为剪辑时间轴会错开，为了使轨迹连贯，故各自补偿帧数
    colors = ['yellow', 'red', 'blue', 'green', 'pink', 'orangered', 'cyan']

    plt.ion()
    while count <= 690:
        # 视频一共有690帧，按照这个顺序遍历

        #———————————————————--------------------—#
        #     实时更新三个视角的世界坐标到字典存储
        #————————————————————————————————————----#

        # 更新第一个视角坐标
        while frame_1 < len(locus_1) and locus_1[frame_1][3] + relenish_1 == count:
            # 如果当前frame在范围内，并且当前帧的时间戳和模拟时间相同

            x1_world, y1_world = get_world_xy(locus_1[frame_1][0],locus_1[frame_1][1], h_1)    # 转换为世界坐标
            id = str(locus_1[frame_1][2])

            if id not in dict_1.keys():
                # 如果当前id不再字典里，则新创建一个key
                dict_1[id] = ([x1_world], [y1_world])
            else:
                dict_1[id][0].append(x1_world)
                dict_1[id][1].append(y1_world)
            frame_1 += 1
            # print(dict_1)     # 检验
        # 更新第二视角坐标
        while frame_2 < len(locus_2) and locus_2[frame_2][3] + replenish_2 == count:
            # 如果当前frame在范围内，并且当前帧的时间戳和模拟时间相同

            x2_world, y2_world = get_world_xy(locus_2[frame_2][0],locus_2[frame_2][1], h_2)    # 转换为世界坐标
            id = str(locus_2[frame_2][2])

            if id not in dict_2.keys():
                # 如果当前id不再字典里，则新创建一个key
                dict_2[id] = ([x2_world], [y2_world])
            else:
                dict_2[id][0].append(x2_world)
                dict_2[id][1].append(y2_world)
            frame_2 += 1
            # print(dict_2)     # 检验
        # 更新第三视角坐标
        while frame_3 < len(locus_3) and locus_3[frame_3][3] + replenish_3 == count:
            # 如果当前frame在范围内，并且当前帧的时间戳和模拟时间相同

            x3_world, y3_world = get_world_xy(locus_3[frame_3][0],locus_3[frame_3][1], h_3)    # 转换为世界坐标
            id = str(locus_3[frame_3][2])

            if id not in dict_3.keys():
                # 如果当前id不再字典里，则新创建一个key
                dict_3[id] = ([x3_world], [y3_world])
            else:
                dict_3[id][0].append(x3_world)
                dict_3[id][1].append(y3_world)
            frame_3 += 1

        #----------------------------------#
        #           刷写坐标
        #----------------------------------#
        plt.cla()
        ax.imshow(img, extent=[0, 510.6 * ratio, 0, 161 * ratio])

        #----------------------------------#
        #       统计冰壶球个数
        #----------------------------------#
        total_nums = len(dict_1.keys())     #统计总共的冰壶数
        yellow_nums = total_nums // 2 if (total_nums % 2 == 0) else total_nums // 2 + 1       #黄色冰壶数量
        red_nums = total_nums - yellow_nums #红色冰壶数量
        plt.text(0, 165, s='yellow:{}'.format(yellow_nums), fontsize=10)
        plt.text(0, 180, s='red:{}'.format(red_nums), fontsize=10)
        # print('y=', yellow_nums)
        # print('r=', red_nums)

        #----------------------------------#
        #           实时绘制坐标
        #——————————————————————————————————#

        for i, id_1 in enumerate(dict_1):
            color = colors[i]
            plt.plot(dict_1[id_1][0], dict_1[id_1][1], color=color)
        for j, id_2 in enumerate(dict_2):
            color = colors[j]
            plt.plot(dict_2[id_2][0], dict_2[id_2][1], color=color)
        for k, id_3 in enumerate(dict_3):
            color = colors[k]
            plt.plot(dict_3[id_3][0], dict_3[id_3][1], color=color)
        plt.pause(0.01)

        count += 1

    # return locus_1, locus_2, locus_3, h_1, h_2, h_3

def get_world_xy(x,y,homography):
    '''
    功能：通过单应性矩阵将图像坐标x，y转换成世界坐标x_world,y_world
    '''
    xy_src = np.float32([[x], [y], [1]])
    xy_world = np.matmul(homography, xy_src)
    x_world = xy_world[0] / xy_world[2]
    y_world = xy_world[1] / xy_world[2]
    return x_world, y_world

if __name__ == '__main__':
    img_path = 'dihu_board.jpg'
    locus_1_path = 'center_loucs/002-1_test.csv'
    locus_2_path = 'center_loucs/002-2_test.csv'
    locus_3_path = 'center_loucs/002-3_test.csv'
    homograpthy_path_1 = 'Homography_array/Homography-1.csv'
    homograpthy_path_2 = 'Homography_array/Homography-2.csv'
    homograpthy_path_3 = 'Homography_array/Homography-3.csv'
    plot_whole_locus(img_path, locus_1_path, locus_2_path, locus_3_path,
                     homograpthy_path_1, homograpthy_path_2, homograpthy_path_3)
    # plot_single_locus(img_path, locus_1_path)
