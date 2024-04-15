import scipy.io as scio
import numpy as np
import cv2
import os
import function
import matplotlib.pyplot as plt

# abnormal_events = scio.loadmat('ped2_original.mat', squeeze_me=True)['gt']
# print(abnormal_events)
# print(abnormal_events.shape)

# avenue = scio.loadmat('avenue/avenue.mat', squeeze_me=True)['gt']
# print(avenue)

# label_ped2 = [np.array([[61, 180]], dtype=np.uint8),
#               np.array([[95, 180]], dtype=np.uint8),
#               np.array([[1, 146]], dtype=np.uint8),
#               np.array([[31, 180]], dtype=np.uint8),
#               np.array([[1, 129]], dtype=np.uint8),
#               np.array([[1, 159]], dtype=np.uint8),
#               np.array([[46, 180]], dtype=np.uint8),
#               np.array([[1, 180]], dtype=np.uint8),
#               np.array([[1, 120]], dtype=np.uint8),
#               np.array([[1, 150]], dtype=np.uint8),
#               np.array([[1, 180]], dtype=np.uint8),
#               np.array([[88, 180]], dtype=np.uint8)]
# scio.savemat('ped2/ped2.mat', {'gt': label_ped2})
# abnormal_events = scio.loadmat('ped2/ped2.mat')['gt']


# label1 = [np.array([[146,265]], dtype=np.uint16)]
# scio.savemat('wood_ano/wood_ano.mat', {'gt': label1})
#
# abnormal_events = scio.loadmat('wood_ano/wood_ano.mat')['gt']


# label_wood_ano2 = [np.array([[76, 265]], dtype=np.uint16)]
# scio.savemat('wood_ano2/wood_ano2.mat', {'gt': label_wood_ano2})

# abnormal_events = scio.loadmat('wood_ano2/wood_ano2.mat')['gt']

# label_wood_ano3 = [np.array([[1, 265]], dtype=np.uint16)]
# scio.savemat('wood_ano3/wood_ano3.mat', {'gt': label_wood_ano3})
#
# abnormal_events = scio.loadmat('wood_ano3/wood_ano3.mat')['gt']
# print(abnormal_events)
# print(abnormal_events.shape)
# for i in range(abnormal_events.shape[0]):
#     one= abnormal_events[i]
#     for j in range(one.shape[0]):
#         print(one[j,0]-1)
#         print(one[j,1])


# different methods threshold
def threshold_method(gt_frame, anomaly_frame, threshold):
    temp_copy = anomaly_frame.copy()
    temp_copy[anomaly_frame > threshold] = 255
    temp_copy[anomaly_frame <= threshold] = 0
    temp_copy = np.array(temp_copy / 255)
    value = function.dice_coefficient(gt_frame, temp_copy)
    return value

def cal_threshold(dataset, number, scenario):
    gt_path = f'Data/{dataset}/testing_gt/0{number}/'
    FFPD = []
    FFP = []
    RTD = []
    STSSD = []

    num = 0
    for test_image in os.listdir(gt_path):
        num += 1
        print(dataset, number, num)
        if num <= 4:
            continue
        gt_frame = cv2.imread(os.path.join(gt_path, test_image), cv2.IMREAD_GRAYSCALE)
        gt_frame = np.array(gt_frame / 255)

        # 读取不同方法生成的异常帧
        anomaly_frame_RTD = f'comparison/RTD_frame/anomaly_{dataset}_0{number}/{test_image}'
        temp_revised_RTD = cv2.imread(anomaly_frame_RTD, cv2.IMREAD_GRAYSCALE)
        dice_RTD = []

        anomaly_frame_STSSD = f'comparison/STSSD_frame/anomaly_{dataset}_0{number}/{test_image}'
        temp_revised_STSSD = cv2.imread(anomaly_frame_STSSD, cv2.IMREAD_GRAYSCALE)
        dice_STSSD = []

        anomaly_frame_FFP = f'../Anomaly_Prediction/results/{dataset}/0{number}/{test_image}'
        temp_revised_FFP = cv2.imread(anomaly_frame_FFP, cv2.IMREAD_GRAYSCALE)
        dice_FFP = []

        anomaly_frame_FFPD = f'results/{dataset}/anomaly/0{number}/{scenario}/anomaly_0{test_image}'
        temp_revised_FFPD = cv2.imread(anomaly_frame_FFPD, cv2.IMREAD_GRAYSCALE)
        dice_FFPD = []

        for threshold in range(0, 255):
            dice_RTD.append(float('%.4f' % (threshold_method(gt_frame, temp_revised_RTD, threshold))))
            dice_STSSD.append(float('%.4f' % (threshold_method(gt_frame, temp_revised_STSSD, threshold))))
            dice_FFP.append(float('%.4f' % (threshold_method(gt_frame, temp_revised_FFP, threshold))))
            dice_FFPD.append(float('%.4f' % (threshold_method(gt_frame, temp_revised_FFPD, threshold))))

        FFPD.append(dice_FFPD)
        FFP.append(dice_FFP)
        RTD.append(dice_RTD)
        STSSD.append(dice_STSSD)

    FFPD = np.array(FFPD)
    FFP = np.array(FFP)
    RTD = np.array(RTD)
    STSSD = np.array(STSSD)

    # 对于每种方法，计算平均值
    FFPD_mean = np.mean(FFPD, axis=0)
    FFP_mean = np.mean(FFP, axis=0)
    RTD_mean = np.mean(RTD, axis=0)
    STSSD_mean = np.mean(STSSD, axis=0)

    # 绘制dice，横轴是threshold，纵轴是dice
    plt.clf()
    plt.xlabel('threshold')
    plt.ylabel('dice coefficient')
    plt.title(f'dice-threshold curve for {dataset}{number}')
    plt.xlim(0, 255)
    plt.ylim(0, 1)
    plt.plot(range(0, 255), FFPD_mean, label='FFPD')
    plt.plot(range(0, 255), FFP_mean, label='FFP')
    plt.plot(range(0, 255), RTD_mean, label='RTD')
    plt.plot(range(0, 255), STSSD_mean, label='STSSD')
    plt.legend()
    plt.savefig(f'results/{dataset}/iou_curve/0{number}.png')
    return



scenario = '4'
cal_threshold('wood', '1', scenario)
# cal_threshold('wood', '2', scenario)
# cal_threshold('wood', '3', scenario)
# cal_threshold('wood', '4', scenario)
# cal_threshold('wood', '5', scenario)
# cal_threshold('wood', '6', scenario)
# cal_threshold('white', '1', scenario)
# cal_threshold('white', '2', scenario)
# cal_threshold('white', '3', scenario)
# cal_threshold('white', '4', scenario)
# cal_threshold('white', '5', scenario)
