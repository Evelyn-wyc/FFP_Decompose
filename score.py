import torch
import cv2
import numpy as np
import os
from function import *
import matplotlib.pyplot as plt
# anomaly_revised.py before score.py

# def calculation_score(dataset, method, scenario = '4'):
#     gt_path = f'Data/{dataset}/testing_gt/'
#     scores = []
#     if method == 'STSSD':
#         anomaly_path = f'comparison/STSSD_frame/'
#     elif method == 'RTD':
#         anomaly_path = f'comparison/RTD_frame/'
#
#     for i, folder in enumerate(os.listdir(gt_path)):
#         print("Calculating folder: ", folder)
#         gt_folder = os.path.join(gt_path, folder)
#         psnrs = []
#         folder_scores = []
#
#         for j, frame in enumerate(os.listdir(gt_folder)):
#             if j < 4:
#                 continue
#             gt_frame = cv2.imread(os.path.join(gt_folder, frame))
#             gt_frame_gray = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
#             if method == 'FFPD':
#                 background_frame = cv2.imread(f'results/{dataset}/background_noise/{folder}/{scenario}/background_noise_0{frame}')
#                 psnr = cv2.PSNR(gt_frame, background_frame)
#             elif method == 'FFP':
#                 background_frame = cv2.imread(f'../Anomaly_Prediction/results/{dataset}/background_{folder}/background_{frame}')
#                 psnr = cv2.PSNR(gt_frame, background_frame)
#             elif method == 'STSSD' or method == 'RTD':
#                 anomaly_frame = cv2.imread(os.path.join(anomaly_path, 'anomaly_'+f'{dataset}_'+folder, frame), cv2.IMREAD_GRAYSCALE)
#                 background_frame = (gt_frame_gray - anomaly_frame).astype('uint8')
#                 psnr = cv2.PSNR(gt_frame_gray, background_frame)
#             psnrs.append(psnr)
#         psnrs = np.array(psnrs)
#         score = (psnrs - psnrs.min()) / (psnrs.max() - psnrs.min())
#         # 根据psnrs画图，title为method-dataset-folder
#         plt.clf()
#         plt.title(f'{method}-{dataset}-{folder}')
#         plt.plot(psnrs)
#         plt.savefig(f'results/{dataset}/iou_curve/score_{method}_{dataset}_{folder}.png')
#         avg_score = np.mean(score)
#         scores.append(avg_score)
#     scores = np.array(scores)
#     return scores

# print('wood, FFPD: ', calculation_score('wood', 'FFPD', '4'))
# print('wood, FFP: ', calculation_score('wood', 'FFP'))
# print('wood, STSSD: ', calculation_score('wood', 'STSSD'))
# print('wood, RTD: ', calculation_score('wood', 'RTD'))
# print('white, FFPD: ', calculation_score('white', 'FFPD', '4'))
# print('white, FFP: ', calculation_score('white', 'FFP'))
# print('white, STSSD: ', calculation_score('white', 'STSSD'))
# print('white, RTD: ', calculation_score('white', 'RTD'))

def calculation_PSNR_background(dataset, method, scenario = '4'):
    gt_path = f'Data/{dataset}/testing_gt/'
    psnr_groups = []
    if method == 'STSSD':
        anomaly_path = f'comparison/STSSD_frame/'
    elif method == 'RTD':
        anomaly_path = f'comparison/RTD_frame/'

    for i, folder in enumerate(os.listdir(gt_path)):
        print("Calculating folder: ", folder)
        psnrs = []
        if dataset == 'white' and folder == '01':
            gt_path1 = f'Data/{dataset}/original_gt1/'
        else:
            gt_path1 = f'Data/{dataset}/original_gt/'

        for j, frame in enumerate(os.listdir(gt_path1)):
            if j < 4:
                continue
            gt_frame = cv2.imread(os.path.join(gt_path1, frame))
            gt_frame_gray = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)

            if method == 'FFPD':
                background_frame = cv2.imread(f'results/{dataset}/background_noise/{folder}/{scenario}/background_noise_0{frame}')
            elif method == 'FFP':
                background_frame = cv2.imread(f'../Anomaly_Prediction/results/{dataset}/background_{folder}/background_{frame}')
            elif method == 'STSSD' or method == 'RTD':
                anomaly_frame = cv2.imread(os.path.join(anomaly_path, 'anomaly_'+f'{dataset}_'+folder, frame), cv2.IMREAD_GRAYSCALE)
                background_frame = (gt_frame_gray - anomaly_frame).astype('uint8')

            if method == 'FFPD' or method == 'FFP':
                psnr = cv2.PSNR(gt_frame, background_frame)
            elif method == 'STSSD':
                cv2.imwrite(f'comparison/STSSD_background/{dataset}_{folder}/{frame}', background_frame)
                psnr = cv2.PSNR(gt_frame_gray, background_frame)
            else:
                cv2.imwrite(f'comparison/RTD_background/{dataset}_{folder}/{frame}', background_frame)
                psnr = cv2.PSNR(gt_frame_gray, background_frame)

            psnrs.append(psnr)

        # 根据psnrs画图，title为method-dataset-folder
        plt.clf()
        plt.title(f'{method}-{dataset}-{folder}')
        plt.plot(psnrs[10:])
        plt.savefig(f'results/{dataset}/iou_curve/{method}_{dataset}_{folder}.png')
        avg_psnr = np.mean(psnrs)
        psnr_groups.append(avg_psnr)
        # print(f'method: {method} folder: {folder} average-psnr: {avg_psnr}')
    psnr_groups = np.array(psnr_groups)
    return psnr_groups

# save psnr_groups for every method
# with open(f'results/wood/iou_curve/wood_psnr_groups.txt', 'w') as f:
#     f.write("\nFFPD: ")
#     for item in calculation_PSNR_background('wood', 'FFPD', '4'):
#         f.write("%s, " % item)
#     f.write("\nFFP: ")
#     for item in calculation_PSNR_background('wood', 'FFP'):
#         f.write("%s, " % item)
#     f.write("\nSTSSD: ")
#     for item in calculation_PSNR_background('wood', 'STSSD'):
#         f.write("%s, " % item)
#     f.write("\nRTD: ")
#     for item in calculation_PSNR_background('wood', 'RTD'):
#         f.write("%s, " % item)
#
# with open(f'results/white/iou_curve/white_psnr_groups.txt', 'w') as f:
#     f.write("\nFFPD: ")
#     for item in calculation_PSNR_background('white', 'FFPD', '4'):
#         f.write("%s, " % item)
#     f.write("\nFFP: ")
#     for item in calculation_PSNR_background('white', 'FFP'):
#         f.write("%s, " % item)
#     f.write("\nSTSSD: ")
#     for item in calculation_PSNR_background('white', 'STSSD'):
#         f.write("%s, " % item)
#     f.write("\nRTD: ")
#     for item in calculation_PSNR_background('white', 'RTD'):
#         f.write("%s, " % item)

# print('wood, FFPD: ', calculation_PSNR_background('wood', 'FFPD', '4'))
# print('wood, FFP: ', calculation_PSNR_background('wood', 'FFP'))
# print('wood, STSSD: ', calculation_PSNR_background('wood', 'STSSD'))
# print('wood, RTD: ', calculation_PSNR_background('wood', 'RTD'))
# print('white, FFPD: ', calculation_PSNR_background('white', 'FFPD', '4'))
# print('white, FFP: ', calculation_PSNR_background('white', 'FFP'))
# print('white, STSSD: ', calculation_PSNR_background('white', 'STSSD'))
# print('white, RTD: ', calculation_PSNR_background('white', 'RTD'))

def calculation_PSNR(dataset, method, scenario = '4'):
    gt_path = f'Data/{dataset}/testing_gt/'
    psnr_groups = []
    if method == 'FFPD':
        anomaly_path = f'results/{dataset}/anomaly_revised/'
    elif method == 'FFP':
        anomaly_path = f'../Anomaly_Prediction/results/{dataset}_anomaly_revised/'
    elif method == 'STSSD':
        anomaly_path = f'comparison/STSSD_frame/'
    elif method == 'RTD':
        anomaly_path = f'comparison/RTD_frame/'

    for i, folder in enumerate(os.listdir(gt_path)):
        print("Calculating folder: ", folder)
        gt_folder = os.path.join(gt_path, folder)
        psnrs = []

        for j, frame in enumerate(os.listdir(gt_folder)):
            if j < 4:
                continue
            gt_frame = cv2.imread(os.path.join(gt_folder, frame), cv2.IMREAD_GRAYSCALE)
            if method == 'FFPD':
                anomaly_frame = cv2.imread(os.path.join(anomaly_path, folder, scenario, 'anomaly_0'+frame), cv2.IMREAD_GRAYSCALE)
            elif method == 'FFP':
                anomaly_frame = cv2.imread(os.path.join(anomaly_path, folder, frame), cv2.IMREAD_GRAYSCALE)
            elif method == 'STSSD' or method == 'RTD':
                anomaly_frame = cv2.imread(os.path.join(anomaly_path, 'anomaly_'+f'{dataset}_'+folder, frame), cv2.IMREAD_GRAYSCALE)
            psnr = cv2.PSNR(gt_frame, anomaly_frame)
            psnrs.append(psnr)
        # 根据psnrs画图，title为method-dataset-folder
        plt.clf()
        plt.title(f'{method}-{dataset}-{folder}')
        plt.plot(psnrs[10:])
        plt.savefig(f'results/{dataset}/iou_curve/anomaly_{method}_{dataset}_{folder}.png')
        # 每个folder的平均值
        avg_psnr = np.mean(psnrs)
        psnr_groups.append(avg_psnr)
    psnr_groups = np.array(psnr_groups)
    return psnr_groups

# print(calculation_PSNR('wood', 'FFPD', '4'))
# print(calculation_PSNR('wood', 'FFP'))
# print(calculation_PSNR('wood', 'STSSD'))
# print(calculation_PSNR('wood', 'RTD'))
# print(calculation_PSNR('white', 'FFPD', '4'))
# print(calculation_PSNR('white', 'FFP'))
# print(calculation_PSNR('white', 'STSSD'))
# print(calculation_PSNR('white', 'RTD'))

def SSIM(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()



def calculation_SSIM(dataset, method, scenario = '4'):
    gt_path = f'Data/{dataset}/testing_gt/'
    ssim_groups = []
    if method == 'FFPD':
        anomaly_path = f'results/{dataset}/anomaly_revised/'
    elif method == 'FFP':
        anomaly_path = f'../Anomaly_Prediction/results/{dataset}_anomaly_revised/'
    elif method == 'STSSD':
        anomaly_path = f'comparison/STSSD_frame/'
    elif method == 'RTD':
        anomaly_path = f'comparison/RTD_frame/'

    for i, folder in enumerate(os.listdir(gt_path)):
        print("Calculating folder: ", folder)
        gt_folder = os.path.join(gt_path, folder)
        ssims = []

        for j, frame in enumerate(os.listdir(gt_folder)):
            if j < 4:
                continue
            gt_frame = cv2.imread(os.path.join(gt_folder, frame), cv2.IMREAD_GRAYSCALE)
            if method == 'FFPD':
                anomaly_frame = cv2.imread(os.path.join(anomaly_path, folder, scenario, 'anomaly_0' + frame),
                                           cv2.IMREAD_GRAYSCALE)
            elif method == 'FFP':
                anomaly_frame = cv2.imread(os.path.join(anomaly_path, folder, frame), cv2.IMREAD_GRAYSCALE)
            elif method == 'STSSD' or method == 'RTD':
                anomaly_frame = cv2.imread(os.path.join(anomaly_path, 'anomaly_' + f'{dataset}_' + folder, frame),
                                           cv2.IMREAD_GRAYSCALE)
            ssim = SSIM(gt_frame, anomaly_frame)
            ssims.append(ssim)
        # 画图
        plt.clf()
        plt.title(f'{method}-{dataset}-{folder}')
        plt.plot(ssims[10:])
        plt.savefig(f'results/{dataset}/iou_curve/ssim_{method}_{dataset}_{folder}.png')
        avg_ssim = np.mean(ssims)
        ssim_groups.append(avg_ssim)
    ssim_groups = np.array(ssim_groups)
    return ssim_groups

# print(calculation_SSIM('wood', 'FFPD', '4'))
# print(calculation_SSIM('wood', 'FFP'))
# print(calculation_SSIM('wood', 'STSSD'))
# print(calculation_SSIM('wood', 'RTD'))
# print(calculation_SSIM('white', 'FFPD', '4'))
# print(calculation_SSIM('white', 'FFP'))
# print(calculation_SSIM('white', 'STSSD'))
# print(calculation_SSIM('white', 'RTD'))

def calculation_IOU(dataset, method, scenario = '4'):
    gt_path = f'Data/{dataset}/testing_gt/'
    iou_groups = []
    if method == 'FFPD':
        anomaly_path = f'results/{dataset}/anomaly_revised/'
    elif method == 'FFP':
        anomaly_path = f'../Anomaly_Prediction/results/{dataset}_anomaly_revised/'
    elif method == 'STSSD':
        anomaly_path = f'comparison/STSSD_revised/'
    elif method == 'RTD':
        anomaly_path = f'comparison/RTD_revised/'

    for i, folder in enumerate(os.listdir(gt_path)):
        print("Calculating folder: ", folder)
        gt_folder = os.path.join(gt_path, folder)
        ious = []
        ious_1 = []

        for j, frame in enumerate(os.listdir(gt_folder)):
            if j < 4:
                continue
            gt_frame = cv2.imread(os.path.join(gt_folder, frame), cv2.IMREAD_GRAYSCALE)
            gt_frame = np.array(gt_frame / 255)
            if method == 'FFPD':
                anomaly_frame = cv2.imread(os.path.join(anomaly_path, folder, scenario, 'anomaly_0' + frame),
                                           cv2.IMREAD_GRAYSCALE)
                # scale anomaly_frame to 0-1
                anomaly_frame = np.array(anomaly_frame / 255)
            elif method == 'FFP':
                anomaly_frame = cv2.imread(os.path.join(anomaly_path, folder, frame), cv2.IMREAD_GRAYSCALE)
                # scale anomaly_frame to 0-1
                anomaly_frame = np.array(anomaly_frame / 255)
            elif method == 'STSSD':
                anomaly_frame = cv2.imread(os.path.join(anomaly_path, f'{dataset}_' + folder, frame),
                                           cv2.IMREAD_GRAYSCALE)
                # scale anomaly_frame to 0-1
                anomaly_frame = np.array(anomaly_frame / 255)
            elif method == 'RTD':
                anomaly_frame = cv2.imread(os.path.join(anomaly_path, f'{dataset}_' + folder, frame),
                                           cv2.IMREAD_GRAYSCALE)
                # scale anomaly_frame to 0-1
                anomaly_frame = np.array(anomaly_frame / 255)
            iou = dice_coefficient(gt_frame, anomaly_frame)
            ious_1.append(iou)
            if iou != 1 and iou != 0:
                # print(f'folder: {folder} frame: {frame} iou: {iou}')
                ious.append(iou)
        # 画图
        plt.clf()
        plt.title(f'{method}-{dataset}{i+1}')
        # x轴取值范围是0-256，y轴取值范围是0-1
        plt.xlabel('threshold')
        plt.ylabel('iou')
        plt.xlim(0, 256)
        plt.ylim(0, 1)
        plt.plot(ious_1)
        plt.savefig(f'results/{dataset}/iou_curve/ious_{method}_{dataset}_{folder}.png')
        avg_iou = np.mean(ious)
        iou_groups.append(avg_iou)
    iou_groups = np.array(iou_groups)
    return iou_groups

threshold = 80
with open(f'results/wood/iou_curve/wood_iou_groups_{threshold}.txt', 'w') as f:
    f.write("\nFFPD: ")
    for item in calculation_IOU('wood', 'FFPD', '4'):
        f.write("%s, " % item)
    f.write("\nFFP: ")
    for item in calculation_IOU('wood', 'FFP'):
        f.write("%s, " % item)
    f.write("\nSTSSD: ")
    for item in calculation_IOU('wood', 'STSSD'):
        f.write("%s, " % item)
    f.write("\nRTD: ")
    for item in calculation_IOU('wood', 'RTD'):
        f.write("%s, " % item)


with open(f'results/white/iou_curve/white_iou_groups_{threshold}.txt', 'w') as f:
    f.write("\nFFPD: ")
    for item in calculation_IOU('white', 'FFPD', '4'):
        f.write("%s, " % item)
    f.write("\nFFP: ")
    for item in calculation_IOU('white', 'FFP'):
        f.write("%s, " % item)
    f.write("\nSTSSD: ")
    for item in calculation_IOU('white', 'STSSD'):
        f.write("%s, " % item)
    f.write("\nRTD: ")
    for item in calculation_IOU('white', 'RTD'):
        f.write("%s, " % item)

# print(calculation_IOU('wood', 'FFPD', '4'))
# print(calculation_IOU('wood', 'FFP'))
# print(calculation_IOU('wood', 'STSSD'))
# print(calculation_IOU('wood', 'RTD'))
# print(calculation_IOU('white', 'FFPD', '4'))
# print(calculation_IOU('white', 'FFP'))
# print(calculation_IOU('white', 'STSSD'))
# print(calculation_IOU('white', 'RTD'))