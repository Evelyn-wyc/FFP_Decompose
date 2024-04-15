'''
This label: Compute AUC and compare with FFP
Compare with STSSD, see frame2mat.py
'''
import scipy.io as scio
import numpy as np

save_path = 'Data/'


# label_wood = [np.array([[146, 256]], dtype=np.uint16),
#               np.array([[76, 256]], dtype=np.uint16),
#               np.array([[1, 256]], dtype=np.uint16),
#               np.array([[91, 256]], dtype=np.uint16),
#               np.array([[91, 256]], dtype=np.uint16),
#               np.array([[34, 256]], dtype=np.uint16)]
# scio.savemat(save_path + 'wood/wood_label.mat', {'gt': label_wood})
# abnormal_events = scio.loadmat('Data/wood/wood_label.mat')['gt']
#
# label_solar = [np.array([[217, 241]], dtype=np.uint16),
#                np.array([[217, 241]], dtype=np.uint16)]
# scio.savemat(save_path + 'solar/solar_label.mat', {'gt': label_solar})
# abnormal_events = scio.loadmat('Data/solar/solar_label.mat')['gt']
#
label_white = [np.array([[121, 256]], dtype=np.uint16),
               np.array([[121, 256]], dtype=np.uint16),
               np.array([[108, 256]], dtype=np.uint16),
               np.array([[104, 256]], dtype=np.uint16),
                np.array([[99, 256]], dtype=np.uint16)]
scio.savemat(save_path + 'white/white_label.mat', {'gt': label_white})
abnormal_events = scio.loadmat('Data/white/white_label.mat')['gt']

# label_fly = [np.array([[0, 30]], dtype=np.uint16)]
# scio.savemat(save_path + 'fly/fly.mat', {'gt': label_fly})
# abnormal_events = scio.loadmat('Data/fly/fly.mat')['gt']

# label_ped2 = [np.array([[61, 180]], dtype=np.uint8)]
# scio.savemat(save_path + 'ped2/ped2_label.mat', {'gt': label_ped2})
# abnormal_events = scio.loadmat('Data/ped2/ped2_label.mat')['gt']

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


# print(abnormal_events)
# print(abnormal_events.shape)
# for i in range(abnormal_events.shape[0]):
#     one = abnormal_events[i]
#     for j in range(one.shape[0]):
#         print(one[j,0]-1)
#         print(one[j,1])