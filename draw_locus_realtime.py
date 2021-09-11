import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import os
import socket


class Plot_Realtime():
    def __init__(self, board_path, homography_path):
        
        # load Homography array from .csv
        H_array = []
        for i, h_path in enumerate(homography_path):
            H_array.append(self.get_xy(h_path))
        print('Homography has been loaded!, Total: {}'.format(len(H_array)))


        # 读取冰壶场景图片作为背景
        img = plt.imread(board_path)
        fig, ax = plt.subplots()
        ax.imshow(img, extent=[0, 510.6 * ratio, 0, 161 * ratio])

        # make new dict to save 
        locus = [{} for i in range( len(H_array)*2 ) ]

        # UDP 建立
        #建立IPv4,UDP的socket
        sockets = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for i in range(len(H_array))]
        #绑定端口：
        for i, s in enumerate(sockets):
            s.bind(('127.0.0.3', 9990+i))
        
        # 实时接收数据，并绘制轨迹
        while True:
            #接收来自客户端的数据,使用recvfrom
            for i, s in enumerate(sockets):
                data, addr = s.recvfrom(1024)
                data=np.fromstring(data)    #,np.uint8)
                data_red, data_yellow = data[0], data[1]
                red_locus_num, yellow_locus_num = 2*i, 2*i+1    #对应locus字典的编号

                for j, cord in enumerate(data_red):
                    x_world, y_world = self.get_world_xy(cord[0], cord[1], H_array[i])

                    if cord[2] not in locus[red_locus_num].keys():
                        locus[red_locus_num][cord[2]] = ([[x_world], [y_world]])
                    else:
                        locus[red_locus_num][cord[2]][0].append(x_world)
                        locus[red_locus_num][cord[2]][1].append(y_world)
            
                for j, cord in enumerate(data_yellow):
                    x_world, y_world = self.get_world_xy(cord[0], cord[1], H_array[i])

                    if cord[2] not in locus[yellow_locus_num].keys():
                        locus[yellow_locus_num][cord[2]] = ([[x_world], [y_world]])
                    else:
                        locus[yellow_locus_num][cord[2]][0].append(x_world)
                        locus[yellow_locus_num][cord[2]][1].append(y_world)
        #----------------------------------#
        #           刷写坐标
        #----------------------------------#
        plt.cla()
        ax.imshow(img, extent=[0, 510.6 * ratio, 0, 161 * ratio])

        #----------------------------------#
        #       统计冰壶球个数
        #----------------------------------#
        yellow_nums = len(locus[-1].keys())     #黄色冰壶数量
        red_nums = len(locus[-2].keys())        #红色冰壶数量
        plt.text(0, 165, s='yellow:{}'.format(yellow_nums), fontsize=10)
        plt.text(0, 180, s='red:{}'.format(red_nums), fontsize=10)
        # print('y=', yellow_nums)
        # print('r=', red_nums)

        #----------------------------------#
        #           实时绘制坐标
        #——————————————————————————————————#
        colors = ['red', 'yellow']
        for i in range(len(locus)):
            for j, ids in enumerate(locus[i]):
                color = colors[i % 2]
                plt.plot(locus[i][ids][0], locus[i][ids][1], color=color)
        plt.pause(0.01)
        plt.ion()
    
    def get_xy(xy_path):
        # 将坐标点的集合文件.csv读取，并转换成numpy数组进行返回
        xy_csv = open(xy_path, encoding='utf-8')
        xy_data = pd.read_csv(xy_csv, header=None)
        print('loaded data from CSV!')
        xy_numpy = xy_data.to_numpy()
        return xy_numpy
    
    def get_world_xy(x,y,homography):
        '''
        功能：通过单应性矩阵将图像坐标x，y转换成世界坐标x_world,y_world
        '''
        xy_src = np.float32([[x], [y], [1]])
        xy_world = np.matmul(homography, xy_src)
        x_world = xy_world[0] / xy_world[2]
        y_world = xy_world[1] / xy_world[2]
        return x_world, y_world


if __name__ == "__main__":
    img_path = 'dihu_board.jpg'
    homograpthy_path_1 = 'Homography_array/Homography-1.csv'
    homograpthy_path_2 = 'Homography_array/Homography-2.csv'
    homograpthy_path_3 = 'Homography_array/Homography-3.csv'
    H_path = [homograpthy_path_1, homograpthy_path_2, homograpthy_path_3]

    P = Plot_Realtime(img_path, H_path)     # create a new instance
    

