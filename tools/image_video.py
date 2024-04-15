import cv2
import numpy as np
import os
import imageio
from PIL import Image

def Image2Video(path, video_name, frame_rate = 10.0):
    # 读取图片
    files = os.listdir(path)
    files.sort()
    img = cv2.imread(path + '/' + files[0])
    height, width, channels = img.shape

    # 创建视频对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height))

    # 将图片逐帧写入视频对象
    for file in files:
        img = cv2.imread(path + '/' + file)
        video.write(img)

    # 释放视频对象
    video.release()
    return

def Image2Gif(path, gif_name, duration = 0.04):
    # 读取图片
    files = os.listdir(path)
    files.sort()

    # 创建GIF对象
    gif = imageio.get_writer(gif_name, mode = 'I', duration = duration)

    # 将图片逐帧写入GIF对象
    for file in files:
        img = cv2.imread(path + '/' + file)
        # 调色板
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gif.append_data(img)

    # 释放GIF对象
    gif.close()
    return

