'''
To revise the anomaly images.
'''
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import PIL.Image as Image
import io
from function import *
# from skimage.measure import compare_ssim as ssim

threshold = 80
def read_image_FFPD(dataset, number, scenario):
    read_path = f'results/{dataset}/anomaly'
    save_path = f'results/{dataset}/anomaly_revised/'
    folder_path = f'{read_path}/{number}/{scenario}/'
    # 遍历所有文件
    for frame in os.listdir(folder_path):
        revised_save_path = f'{save_path}/{number}/{scenario}/{frame}'
        frame_path = os.path.join(folder_path, frame)
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if dataset == 'white':
            if scenario == '4':
                frame[frame <= threshold] = 0
                frame[frame > threshold] = 255
        elif dataset == 'wood':
            # if number != '03':
            #     frame[frame <= 55] = 0
            #     frame[frame >= 55] = 255
            # else:
                frame[frame <= threshold] = 0
                frame[frame > threshold] = 255
        cv2.imwrite(revised_save_path, frame)
    return

def read_image_FFP(dataset, number):
    read_path = f'../Anomaly_Prediction/results/{dataset}'
    save_path = f'../Anomaly_Prediction/results/{dataset}_anomaly_revised/'
    folder_path = f'{read_path}/{number}/'
    # 遍历所有文件
    for frame in os.listdir(folder_path):
        revised_save_path = f'{save_path}/{number}/{frame}'
        frame_path = os.path.join(folder_path, frame)
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if dataset == 'white':
            frame[frame <= threshold] = 0
            frame[frame > threshold] = 255
        elif dataset == 'wood':
            frame[frame <= threshold] = 0
            frame[frame >= threshold] = 255
        cv2.imwrite(revised_save_path, frame)
    return


def read_image_STSSD(dataset, number):
    read_path = f'comparison/STSSD_frame/anomaly_{dataset}_{number}'
    save_path = f'comparison/STSSD_revised/{dataset}_{number}'
    # 遍历所有文件
    for frame in os.listdir(read_path):
        revised_save_path = f'{save_path}/{frame}'
        frame_path = os.path.join(read_path, frame)
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if dataset == 'white':
            frame[frame <= threshold] = 0
            frame[frame > threshold] = 255
        elif dataset == 'wood':
            frame[frame <= threshold] = 0
            frame[frame > threshold] = 255
        cv2.imwrite(revised_save_path, frame)
    return

def read_image_RTD(dataset, number):
    read_path = f'comparison/RTD_frame/anomaly_{dataset}_{number}'
    save_path = f'comparison/RTD_revised/{dataset}_{number}'
    # 遍历所有文件
    for frame in os.listdir(read_path):
        revised_save_path = f'{save_path}/{frame}'
        frame_path = os.path.join(read_path, frame)
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if dataset == 'white':
            frame[frame <= threshold] = 0
            frame[frame > threshold] = 255
        elif dataset == 'wood':
            frame[frame <= threshold] = 0
            frame[frame > threshold] = 255
        cv2.imwrite(revised_save_path, frame)
    return



if __name__ == '__main__':
    dataset = 'wood'
    scenario = '4'
    read_image_FFPD(dataset, '01', scenario)
    read_image_FFPD(dataset, '02', scenario)
    read_image_FFPD(dataset, '03', scenario)
    read_image_FFPD(dataset, '04', scenario)
    read_image_FFPD(dataset, '05', scenario)
    read_image_FFPD(dataset, '06', scenario)

    read_image_FFP(dataset, '01')
    read_image_FFP(dataset, '02')
    read_image_FFP(dataset, '03')
    read_image_FFP(dataset, '04')
    read_image_FFP(dataset, '05')
    read_image_FFP(dataset, '06')

    read_image_STSSD(dataset, '01')
    read_image_STSSD(dataset, '02')
    read_image_STSSD(dataset, '03')
    read_image_STSSD(dataset, '04')
    read_image_STSSD(dataset, '05')
    read_image_STSSD(dataset, '06')

    read_image_RTD(dataset, '01')
    read_image_RTD(dataset, '02')
    read_image_RTD(dataset, '03')
    read_image_RTD(dataset, '04')
    read_image_RTD(dataset, '05')
    read_image_RTD(dataset, '06')

    # img = cv2.imread('comparison/RTD_frame/anomaly_white_01/230.jpg')
    # print(img)
    # dice_groups = calculation(dataset, scenario)
    # print("dice_groups: ", dice_groups)
    # print(dice_groups.mean(axis=1))
    # with open(f'results/{dataset}/iou_curve/dice_groups_FFP_{dataset}_{scenario}.txt', 'w') as f:
    #     for item in dice_groups:
    #         f.write("%s\n" % item)
    #     # 输入平均值
    #     f.write("%s\n" % dice_groups.mean(axis=1))

    # psnr_groups = calculation_PSNR(dataset, scenario)
    # print("psnr_groups: ", psnr_groups)
    # print(psnr_groups.mean(axis=1))
    # with open(f'results/{dataset}/iou_curve/psnr_groups_FFP_{dataset}_{scenario}.txt', 'w') as f:
    #     for item in psnr_groups:
    #         f.write("%s\n" % item)
    #     # 输入平均值
    #     f.write("%s\n" % psnr_groups.mean(axis=1))

    # ssim_groups = calculation_SSIM(dataset, scenario)
    # print("ssim_groups: ", ssim_groups)
    # print(ssim_groups.mean(axis=1))
    # with open(f'results/{dataset}/iou_curve/ssim_groups_FFP_{dataset}_{scenario}.txt', 'w') as f:
    #     for item in ssim_groups:
    #         f.write("%s\n" % item)
    #     # 输入平均值
    #     f.write("%s\n" % ssim_groups.mean(axis=1))