#!/usr/bin/python3
# Author:Jeremy.Dong
# Introduction: 将目标检测所标注的json文件转化为txt文件

import json
import os
#文件根目录
path = 'E:/online_learning_data/turn_json_txt/'

path_list = os.listdir(path)

for dirs in path_list:
    #组合路径
    json_path = os.path.join(path, dirs)
    with open(json_path, 'r', encoding='utf-8') as fp:
        json_data = json.load(fp)
        # print(json_data)
        imagePath = json_data['imagePath']
        # print(imagePath)
        # print(json_path['shapes'][0])
        if len(json_data['shapes']) == 2:

            for i in range(len(json_data['shapes'])):

                if json_data['shapes'][i]['label'] == 'gaze':
                    gaze_point = json_data['shapes'][i]['points']

                elif json_data['shapes'][i]['label'] == 'head':
                    head_local = json_data['shapes'][i]['points']

        else:
            if json_data['shapes'][0]['label'] == 'head':
                head_local = json_data['shapes'][0]['points']
            gaze_point = [[-1, -1]]


        gaze_x = int(gaze_point[0][0])
        gaze_y = int(gaze_point[0][1])
        head_min_x = int(head_local[0][0])
        head_min_y = int(head_local[0][1])
        head_max_x = int(head_local[1][0])
        head_max_y = int(head_local[1][1])
        with open('E:/online_learning_data/head_gaze.txt','a') as f:
            f.write(imagePath+',')
            f.write(str(head_min_x)+','+str(head_min_y)+','+str(head_max_x)+','+str(head_max_y)+',')
            f.write(str(gaze_x)+','+str(gaze_y)+','+'\n')




