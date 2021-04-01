"""Script read video from local ,and clip pic rename save in local"""
import os
import cv2

class clipvideo(object):
    def __init__(self, video_path, save_path, num):
        """
        :param video_path: video path from local
        :param save_path: clip pic path from local
        :param num:  How many frames to store
        """
        video = os.listdir(video_path)  #返回指定文件夹下的视频文件
        #依次读取文件名
        for video_name in video:
            #对文件名进行拆分处理
            file_name = video_name.split('.')[0]
            folder_name = save_path + file_name #构成新目录存放视频帧
            os.makedirs(folder_name, exist_ok=True) #查看是否存在目录，否则创建
            vc = cv2.VideoCapture(video_path + video_name)
            #获取视频帧率
            fps = vc.get(cv2.CAP_PROP_FPS)
            print("filename:", fps)
            #判断视频是否可以打开
            rval = vc.isOpened()
            c = 1
            while rval:
                raval, frame = vc.read()  #第一个返回值为判断是否可以返回帧， 第二个返回的帧为返回的值
                frame_name = folder_name + "/"
                if raval:
                    if (c % num == 0):
                        cv2.imwrite(frame_name + file_name + "_" + str(c)+".jpg",frame)
                    cv2.waitKeyEx(1)
                c= c+1
            vc.release()


if __name__=='__main__':
    clip = clipvideo


