'''
1、在正常的图片（铝箔）上不同位置生成异常情况（例如擦花）
2、按照时间顺序把图片合成一个视频，模拟异常从左到右移动的情况
这样做的好处是不需要考虑背景的变化了，直接对异常做文章即可。
'''

import cv2
import numpy as np
import os

original = cv2.imread('lv.jpg')
Imgrow, Imgcol, Imgchannel = original.shape

# 生成异常图片，颜色是blue-green-red
def generate_anomaly(original_image, num_points, start_point, end_point, color):
    canvas = np.copy(original_image)
    group_numrow = num_points // np.abs(end_point[0] - start_point[0])
    group_numcol = num_points // np.abs(end_point[1] - start_point[1])
    variation = num_points // 100
    for i in range(num_points):
        x = np.random.randint(i // group_numrow - variation, i // group_numrow + variation) * np.sign(end_point[0] - start_point[0]) + start_point[0]
        y = np.random.randint(i // group_numcol - variation, i // group_numcol + variation) * np.sign(end_point[1] - start_point[1]) + start_point[1]
        cv2.circle(canvas, (x, y), 1, color, -1)
    filename = f'anomaly_{end_point[0]:04d}.jpg'
    cv2.imwrite('lv_output/' + filename, canvas)


# 生成视频
def images2video(image_folder, output_video_name, frame_rate=10.0, file_extension=".jpg"):
    # 获取图像文件夹中的所有图像文件
    images = [img for img in os.listdir(image_folder) if img.endswith(file_extension)]
    print(images)

    # 读取第一张图像以获取图像尺寸
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_name, fourcc, frame_rate, (width, height))

    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        out.write(img)

    out.release()

# # 生成多帧图片
# for i in range(0, 2560, 20):
#     generate_anomaly(original, 5000, np.array([1560 - i, 500]), np.array([2560 - i, 520]), (230, 230, 230))

image_folder = 'lv_output'  # 图像文件夹路径
output_video_name = 'lv_anomaly_video.avi'  # 输出视频文件名
images2video(image_folder, output_video_name)


# 比较两行像素差距，如果过大说明有异常
# for i in range(Imgrow - 1):
#     line1 = anomaly1[i, :, :]
#     line2 = anomaly1[i+1, :, :]
#     # 两个line逐点作差，并将差的绝对值求和
#     dif = np.sum(np.abs(line1 - line2))
#     if dif > 1050000:
#         print(i)