'''
RGB image to gray image
'''
import cv2
import os
import numpy as np

def RGB2Gray(dataset):
    read_path = f'../Data/{dataset}/testing/'
    save_path = f'../Data/{dataset}/testing_gray/'
    for folder in os.listdir(read_path):
        print('Processing folder: ', folder)
        folder_path = os.path.join(read_path, folder)
        save_folder_path = os.path.join(save_path, folder)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        for frame in os.listdir(folder_path):
            frame_path = os.path.join(folder_path, frame)
            save_frame_path = os.path.join(save_folder_path, frame)
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(save_frame_path, frame)
    return

if __name__ == '__main__':
    RGB2Gray('white')
    RGB2Gray('wood')
    print('Done!')