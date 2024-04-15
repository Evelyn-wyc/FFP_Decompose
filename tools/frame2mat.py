'''
Compare with STSSD, transform frames to mat (gray).
'''
import os
import numpy as np
import scipy.io as scio
import cv2

def np_load_frame(filename, resize_h, resize_w):
    img = cv2.imread(filename)
    image_resized = cv2.resize(img, (resize_w, resize_h)).astype('float32')
    img_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    # image_resized = (image_resized / 127.5) - 1.0  # to -1 ~ 1
    # image_resized = np.transpose(image_resized, [2, 0, 1])  # to (C, W, H)
    return img_gray

def frame2mat(video_folder, resize_h, resize_w):
    all_imgs = os.listdir(video_folder)
    all_imgs.sort()
    video_clip = []
    for i in range(len(all_imgs)):
        video_clip.append(np_load_frame(os.path.join(video_folder, all_imgs[i]), resize_h, resize_w))
    video_clip = np.array(video_clip).reshape((-1, resize_h, resize_w))
    video_clip = np.transpose(video_clip, [1, 2, 0])
    video_clip = video_clip.astype(np.float32)
    return video_clip

def GenerateMat(dataset, resize_h, resize_w):
    read_path = '../Data/' + dataset + '/' + 'testing/'
    save_path = '../comparison/STSSD_' + dataset + '/'
    for i in os.listdir(read_path):
        current_path = read_path + f'{i}/'
        video_clip = frame2mat(current_path, resize_h, resize_w)
        scio.savemat(save_path + f'{dataset}_{i}.mat', {'data': video_clip})

# GenerateMat('wood', 256, 256)
GenerateMat('white', 256, 256)
# GenerateMat('solar', 256, 256)
