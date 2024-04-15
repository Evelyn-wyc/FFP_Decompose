'''
Use the solar data from STSSD, transform mat to frames (gray).
'''
import os
import numpy as np
import scipy.io as scio
import cv2
import matplotlib.pyplot as plt


def mat2frame(mat_path, save_path):
    mat = scio.loadmat(mat_path)['data']
    mat = np.transpose(mat, [2, 0, 1])
    for i in range(mat.shape[0]):
        plt.imshow(mat[i], cmap='hot')
        plt.axis('off')
        plt.savefig(save_path + f'{i:03d}' + '.jpg', bbox_inches='tight', pad_inches=0)
    return

# resize into 256*256
def resize_frame(frame_path, save_path):
    frame_list = os.listdir(frame_path)
    for frame in frame_list:
        img = cv2.imread(frame_path + frame)
        img = cv2.resize(img, (256, 256))
        cv2.imwrite(save_path + f'{frame}', img)
        # plt.imshow(img, cmap='hot')
        # plt.axis('off')
        # plt.savefig(save_path + frame, bbox_inches='tight', pad_inches=0)
    return

mat2frame('D:/thu/research/detection/VideoPredictAD/Simulation/solar.mat', '../Data/solar/testing/01/')
resize_frame('../Data/solar/testing/01/', '../Data/solar/testing/02/')