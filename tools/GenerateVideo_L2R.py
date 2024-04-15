'''
用于生成从左到右移动的图片，图片在完全展示之后停止，从而得到一个异常缓慢出现的视频。
图片构思想法来源于木制桌面（家具），不单独考虑物体，而是直接在背景上产生瑕疵。
本文件用标准（无瑕疵图片）、异常（有瑕疵图片）生成了多组视频。
'''
import cv2
import numpy as np
import os

def generate_anomaly(original_image, mode, params):
    '''
    在图像中生成异常标记。

    Parameters:
    - original_image: 原始图像
    - mode: 异常标记的模式，可以是 'line', 'rectangle', 'circle', 或 'ellipse'
    - params: 包含标记参数的字典，根据模式不同包含不同的键值对

    Returns:
    - img_revised: 处理后的图像
    - canvas_anomaly: 包含异常标记的画布
    '''

    # 将原图变成1028*1028的正方形
    # img_revised = cv2.resize(original_image, (1028, 1028))

    # 将原图变成256*256的正方形
    img_revised = cv2.resize(original_image, (256, 256))
    height, width, channel = img_revised.shape
    # 在图上绘制异常
    canvas = np.copy(img_revised)

    # 将图像变256 *256
    # img_revised = cv2.resize(canvas, (256, 256))

    background = np.zeros((height, width, 3), dtype=np.uint8) # 创建背景（这里是黑色的）
    canvas_anomaly = np.copy(background)

    if mode == 'line':
        start_point = params.get('start_point',(130, 70))
        end_point = params.get('end_point',(135, 75))
        color = params.get('color', (0, 0, 0))
        thickness = params.get('thickness', 1)
        cv2.line(canvas, start_point, end_point, color, thickness)
        cv2.line(canvas_anomaly, start_point, end_point, (255, 255, 255), thickness)

    elif mode == 'rectangle':
        start_point = params.get('start_point',(130, 70))
        end_point = params.get('end_point',(135, 75))
        color = params.get('color', (0, 0, 0))
        thickness = params.get('thickness', 1)
        cv2.rectangle(canvas, start_point, end_point, color, thickness)
        cv2.rectangle(canvas_anomaly, start_point, end_point, (255, 255, 255), thickness)

    elif mode == 'circle':
        start_point = params.get('start_point',(130, 70))
        radius = params.get('radius', 5)
        color = params.get('color', (0, 0, 0))
        thickness = params.get('thickness', 1)
        cv2.circle(canvas, start_point, radius, color, thickness = -1)
        cv2.circle(canvas_anomaly, start_point, radius, (255, 255, 255), thickness = -1)

    elif mode == 'ellipse':
        start_point = params.get('start_point',(130, 70))
        axes = params.get('axes', (5, 3))
        angle = params.get('angle', 0)
        startAngle = params.get('startAngle', 0)
        endAngle = params.get('endAngle', 360)
        color = params.get('color', (0, 0, 0))
        thickness = params.get('thickness', 1)
        cv2.ellipse(canvas, start_point, axes, angle, startAngle, endAngle, color, thickness)
        cv2.ellipse(canvas_anomaly, start_point, axes, angle, startAngle, endAngle, (255, 255, 255), thickness)

    return canvas, canvas_anomaly


def image2frame(image, mode = None): # 这里有一个假设: framecount = width
    # 将图片转化为从左到右出现的视频
    height, width, channel = image.shape
    background = np.zeros((height, width, 3), dtype=np.uint8) # 创建背景（这里是黑色的）
    if mode == 'real':
        # rgb为29，54，24，更改对应的background
        background[:, :, 0] = 29
        background[:, :, 1] = 54
        background[:, :, 2] = 24
    frames = np.zeros((width, height, width, channel), dtype = np.uint8) # 图像形成连续帧
    for x in range(0, width):
        frame = np.copy(background)
        frame[:, : x] = image[:, width - x: width]  # 逐渐将原始图像的左侧部分添加到黑色背景上
        frames[x] = frame
    return frames

def save_video(frames, path):
    # 储存连续帧图像
    frame_count, height, width, channel = frames.shape
    fps = 30  # 指定帧率
    output_file = path # 指定输出文件名和路径
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 指定视频编码器
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))  # 创建视频写入对象
    for frame in frames:
        out.write(frame)  # 将每一帧图像写入视频
    out.release()  # 释放视频写入对象

def video2frame(video, save_path):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print('video is not opened')
        return
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 逐帧读取视频和保存图片
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 保存当前帧图片
        file_num = '%03d' % frame_count
        cv2.imwrite(os.path.join(save_path, file_num + '.jpg'), frame)
        frame_count += 1
    cap.release()
    return

# 读取原始图像
# original_white = cv2.imread('../Data/white/toilet_seat_1.jpg')
# original_wood = cv2.imread('../Data/wood/wooden_texture.jpg')
# #
# original_white = cv2.resize(original_white, (256, 256))
# original_wood = cv2.resize(original_wood, (256,256))

# video2frame('.../Anomaly_Prediction/results/white/01_generate.avi', '.../Anomaly_Prediction/results/white/background_01/')
# video2frame('.../Anomaly_Prediction/results/white/02_generate.avi', '.../Anomaly_Prediction/results/white/background_02/')
# video2frame('.../Anomaly_Prediction/results/white/03_generate.avi', '.../Anomaly_Prediction/results/white/background_03/')
# video2frame('.../Anomaly_Prediction/results/white/04_generate.avi', '.../Anomaly_Prediction/results/white/background_04/')
# video2frame('.../Anomaly_Prediction/results/white/05_generate.avi', '.../Anomaly_Prediction/results/white/background_05/')
# video2frame('.../Anomaly_Prediction/results/wood/01_generate.avi', '.../Anomaly_Prediction/results/wood/background_01/')
# video2frame('.../Anomaly_Prediction/results/wood/02_generate.avi', '.../Anomaly_Prediction/results/wood/background_02/')
# video2frame('.../Anomaly_Prediction/results/wood/03_generate.avi', '.../Anomaly_Prediction/results/wood/background_03/')
# video2frame('.../Anomaly_Prediction/results/wood/04_generate.avi', '.../Anomaly_Prediction/results/wood/background_04/')
# video2frame('.../Anomaly_Prediction/results/wood/05_generate.avi', '.../Anomaly_Prediction/results/wood/background_05/')
# video2frame('.../Anomaly_Prediction/results/wood/06_generate.avi', '.../Anomaly_Prediction/results/wood/background_06/')

# original_gt folder, in Data/{dataset}/original_gt
# save_video(image2frame(original_white), '../Data/white_o/toilet_seat.avi')
# video2frame('../Data/white_o/toilet_seat.avi', '../Data/white_o1/')
# save_video(image2frame(original_white, 'real'), '../Data/white_o/toilet_seat.avi')
# video2frame('../Data/white_o/toilet_seat.avi', '../Data/white_o/')
# save_video(image2frame(original_wood), '../Data/wood_o/wood.avi')
# video2frame('../Data/wood_o/wood.avi', '../Data/wood_o/')

#
# params_line = {'start_point': (130, 70),
#                'end_point': (135, 75),
#                'color': (0, 0, 0),
#                'thickness': 1}
#
# params_line1 = {'start_point': (20, 100),
#                 'end_point': (120, 50),
#                 'color': (255, 255, 255),
#                 'thickness': 1}
#
# params_line2 = {'start_point': (150, 120),
#                 'end_point': (190, 160),
#                 'color': (255, 255, 255),
#                 'thickness': 2}
#
# params_line3 = {'start_point': (190, 135),
#                 'end_point': (265, 160),
#                 'color': (255, 255, 255),
#                 'thickness': 3}
#
# anomaly_wood1, anomaly_wood1_pixel = generate_anomaly(original_wood, 'line', params_line1)
# save_video(image2frame(anomaly_wood1), '../Data/wood/testing/Anomaly1_wood.avi')
# video2frame('../Data/wood/testing/Anomaly1_wood.avi', '../Data/wood/testing/01/')
# save_video(image2frame(anomaly_wood1_pixel), '../Data/wood/testing_gt/Anomaly1_wood_pixel.avi')
# video2frame('../Data/wood/testing_gt/Anomaly1_wood_pixel.avi', '../Data/wood/testing_gt/01/')
#
# anomaly_wood2, anomaly_wood2_pixel = generate_anomaly(original_wood, 'line', params_line2)
# save_video(image2frame(anomaly_wood2), '../Data/wood/testing/Anomaly2_wood.avi')
# video2frame('../Data/wood/testing/Anomaly2_wood.avi', '../Data/wood/testing/02/')
# save_video(image2frame(anomaly_wood2_pixel), '../Data/wood/testing_gt/Anomaly2_wood_pixel.avi')
# video2frame('../Data/wood/testing_gt/Anomaly2_wood_pixel.avi', '../Data/wood/testing_gt/02/')
#
# anomaly_wood3, anomaly_wood3_pixel = generate_anomaly(original_wood, 'line', params_line3)
# save_video(image2frame(anomaly_wood3), '../Data/wood/testing/Anomaly3_wood.avi')
# video2frame('../Data/wood/testing/Anomaly3_wood.avi', '../Data/wood/testing/03/')
# save_video(image2frame(anomaly_wood3_pixel), '../Data/wood/testing_gt/Anomaly3_wood_pixel.avi')
# video2frame('../Data/wood/testing_gt/Anomaly3_wood_pixel.avi', '../Data/wood/testing_gt/03/')
#
#
# params_rectangle = {'start_point': (145, 110),
#                     'end_point': (147, 103),
#                     'color': (0, 0, 0),
#                     'thickness': 2}
#
# params_rectangle2 = {'start_point': (160, 135),
#                     'end_point': (165, 120),
#                     'color': (255, 255, 255),
#                     'thickness': 2}
#
# anomaly3, anomaly3_pixel = generate_anomaly(original_white, 'rectangle', params_rectangle)
# save_video(image2frame(anomaly3, 'real'), '../Data/white/testing/Anomaly3_toilet_seat.avi')
# video2frame('../Data/white/testing/Anomaly3_toilet_seat.avi', '../Data/white/testing/03/')
# save_video(image2frame(anomaly3_pixel), '../Data/white/testing_gt/Anomaly3_toilet_seat_pixel.avi')
# video2frame('../Data/white/testing_gt/Anomaly3_toilet_seat_pixel.avi', '../Data/white/testing_gt/03/')
#
# anomaly_wood4, anomaly_wood4_pixel = generate_anomaly(original_wood, 'rectangle', params_rectangle2)
# save_video(image2frame(anomaly_wood4), '../Data/wood/testing/Anomaly4_wood.avi')
# video2frame('../Data/wood/testing/Anomaly4_wood.avi', '../Data/wood/testing/04/')
# save_video(image2frame(anomaly_wood4_pixel), '../Data/wood/testing_gt/Anomaly4_wood_pixel.avi')
# video2frame('../Data/wood/testing_gt/Anomaly4_wood_pixel.avi', '../Data/wood/testing_gt/04/')
#
# params_circle = {'start_point': (150, 75),
#                  'radius': 4,
#                  'color': (0, 0, 0),
#                  'thickness': -1}
#
# params_circle2 = {'start_point': (160, 120),
#                   'radius': 4,
#                   'color': (255, 255, 255),
#                   'thickness': -1}
#
# anomaly4, anomaly4_pixel = generate_anomaly(original_white, 'circle', params_circle)
# save_video(image2frame(anomaly4, 'real'), '../Data/white/testing/Anomaly4_toilet_seat.avi')
# video2frame('../Data/white/testing/Anomaly4_toilet_seat.avi', '../Data/white/testing/04/')
# save_video(image2frame(anomaly4_pixel), '../Data/white/testing_gt/Anomaly4_toilet_seat_pixel.avi')
# video2frame('../Data/white/testing_gt/Anomaly4_toilet_seat_pixel.avi', '../Data/white/testing_gt/04/')
#
# anomaly_wood5, anomaly_wood5_pixel = generate_anomaly(original_wood, 'circle', params_circle2)
# save_video(image2frame(anomaly_wood5), '../Data/wood/testing/Anomaly5_wood.avi')
# video2frame('../Data/wood/testing/Anomaly5_wood.avi', '../Data/wood/testing/05/')
# save_video(image2frame(anomaly_wood5_pixel), '../Data/wood/testing_gt/Anomaly5_wood_pixel.avi')
# video2frame('../Data/wood/testing_gt/Anomaly5_wood_pixel.avi', '../Data/wood/testing_gt/05/')
#
#
# params_ellipse = {'start_point': (155, 70),
#                   'axes': (2, 1),
#                   'angle': 0,
#                   'startAngle': 0,
#                   'endAngle': 360,
#                   'color': (0, 0, 0),
#                   'thickness': -1}
#
# params_ellipse2 = {'start_point': (220, 120),
#                    'axes': (2, 1),
#                    'angle': 0,
#                    'startAngle': 0,
#                    'endAngle': 360,
#                    'color': (255, 255, 255),
#                    'thickness': -1}
#
# anomaly5, anomaly5_pixel = generate_anomaly(original_white, 'ellipse', params_ellipse)
# save_video(image2frame(anomaly5, 'real'), '../Data/white/testing/Anomaly5_toilet_seat.avi')
# video2frame('../Data/white/testing/Anomaly5_toilet_seat.avi', '../Data/white/testing/05/')
# save_video(image2frame(anomaly5_pixel), '../Data/white/testing_gt/Anomaly5_toilet_seat_pixel.avi')
# video2frame('../Data/white/testing_gt/Anomaly5_toilet_seat_pixel.avi', '../Data/white/testing_gt/05/')
#
# anomaly_wood6, anomaly_wood6_pixel = generate_anomaly(original_wood, 'ellipse', params_ellipse2)
# save_video(image2frame(anomaly_wood6), '../Data/wood/testing/Anomaly6_wood.avi')
# video2frame('../Data/wood/testing/Anomaly6_wood.avi', '../Data/wood/testing/06/')
# save_video(image2frame(anomaly_wood6_pixel), '../Data/wood/testing_gt/Anomaly6_wood_pixel.avi')
# video2frame('../Data/wood/testing_gt/Anomaly6_wood_pixel.avi', '../Data/wood/testing_gt/06/')


# original = cv2.imread('../Data/wood/wooden_texture.jpg')
#
#
# anomaly1, anomaly1_pixel = generate_anomaly_line(original)
# # save_video(image2frame(anomaly1), '../Data/wood/testing/Anomaly1_wooden_texture.avi')
# # video2frame('../Data/wood/testing/Anomaly1_wooden_texture.avi', '../Data/wood/testing/01')
# save_video(image2frame(anomaly1_pixel), '../Data/wood/testing_gt/Anomaly1_wooden_texture_pixel.avi')
# video2frame('../Data/wood/testing_gt/Anomaly1_wooden_texture_pixel.avi', '../Data/wood/testing_gt/01')
#
# anomaly2, anomaly2_pixel = generate_anomaly_line(original, start_point = (150, 120), end_point = (190, 160), color = (255, 255, 255), thickness = 2)
# # save_video(image2frame(anomaly2), '../Data/wood/testing/Anomaly2_wooden_texture.avi')
# # video2frame('../Data/wood/testing/Anomaly2_wooden_texture.avi', '../Data/wood/testing/02')
# save_video(image2frame(anomaly2_pixel), '../Data/wood/testing_gt/Anomaly2_wooden_texture_pixel.avi')
# video2frame('../Data/wood/testing_gt/Anomaly2_wooden_texture_pixel.avi', '../Data/wood/testing_gt/02')
#
# anomaly3, anomaly3_pixel = generate_anomaly_line(original, start_point = (190, 135), end_point = (265, 160), color = (255, 255, 255), thickness = 3)
# # save_video(image2frame(anomaly3), '../Data/wood/testing/Anomaly3_wooden_texture.avi')
# # video2frame('../Data/wood/testing/Anomaly3_wooden_texture.avi', '../Data/wood/testing/03')
# save_video(image2frame(anomaly3_pixel), '../Data/wood/testing_gt/Anomaly3_wooden_texture_pixel.avi')
# video2frame('../Data/wood/testing_gt/Anomaly3_wooden_texture_pixel.avi', '../Data/wood/testing_gt/03')
