import cv2
import torch
import argparse
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import io
from PIL import Image

from config import update_config
import Dataset
import anomaly
from unet import UNet
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description = 'Future Frame Prediction with Decomposition')
parser.add_argument('--dataset', default = 'white', type = str, help = 'The name of the dataset to train.')
parser.add_argument('--trained_model', default = 'avenue_15000.pth', type = str, help = 'The pre-trained model to evaluate.')
parser.add_argument('--scenario', default = '4', type = str, help = 'The scenario to evaluate.')
parser.add_argument('--show_anomaly', default = False, action = 'store_true',
                    help = 'Show and save the anomaly frame real-timely.')

torch.cuda.memory_allocated()

# 循环更新传入预测网络的帧，利用Bi, Bi+1, ..., Bi+3预测\hatBi+4
def val(cfg):

    # 获取预训练的网络
    generator = UNet(input_channels=12, output_channel=3).cuda().eval()
    generator.load_state_dict(torch.load('weights/' + cfg.trained_model)['net_g'])
    print(f'The pre-trained generator has been loaded from \'weights/{cfg.trained_model}\'.\n')

    # 获取测试集
    test_folders = os.listdir(cfg.test_data)
    test_folders.sort()
    test_folders = [os.path.join(cfg.test_data, aa) for aa in test_folders] #列表化

    # 获取背景集，前四张图片需要手动复制到对应的文件夹中
    background_folders = os.listdir(cfg.background_save_path)
    background_folders.sort()
    background_folders = [os.path.join(cfg.background_save_path, aa) for aa in background_folders]

    # 获取背景+噪声集合
    background_noise_folders = os.listdir(cfg.background_noise_save_path)
    background_noise_folders.sort()
    background_noise_folders = [os.path.join(cfg.background_noise_save_path, aa) for aa in background_noise_folders]

    # 获取异常集
    anomaly_folders = os.listdir(cfg.anomaly_save_path)
    anomaly_folders.sort()
    anomaly_folders = [os.path.join(cfg.anomaly_save_path, aa) for aa in anomaly_folders]

    # writer = SummaryWriter('./tensorboard_log')
    fps = 0

    if cfg.show_anomaly:
        cv2.namedWindow('anomaly frames', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('anomaly frames', 256, 256)
        cv2.moveWindow("anomaly frames", 100, 100)

    for i, folder in enumerate(test_folders):
        # if i == 0 or i == 1 or i == 2 or i == 3 or i == 4 or i == 5:
        if i == 3:
            b_folder = background_folders[i]
            bn_folder = background_noise_folders[i]
            a_folder = anomaly_folders[i]
            test_dataset = Dataset.test_dataset(cfg, folder)

            # 获取视频名称
            name = folder.split('/')[-1]
            fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

            if cfg.show_anomaly:
                anomaly_writer = cv2.VideoWriter(f'results/{cfg.dataset}/anomaly/{name}_scenario{cfg.scenario}_anomaly.avi', fourcc, 30, cfg.img_size)
                background_writer = cv2.VideoWriter(f'results/{cfg.dataset}/gen_background/{name}_scenario{cfg.scenario}_background.avi', fourcc, 30, cfg.img_size)

            for j, clip in enumerate(test_dataset):
                # predict_clips取b_folder里的第j，j+1，j+2，j+3帧，然后预测第j+4帧
                predict_clips = []
                for frame_id in range(j, j + 4):
                    # 加入b_folder里的图片
                    frame = os.path.join(b_folder + '/' + cfg.scenario + '/background_' + f'{frame_id:04d}' + '.jpg')
                    predict_clips.append(Dataset.np_load_frame(frame, cfg.img_size[0], cfg.img_size[1]))
                predict_clips = np.array(predict_clips).reshape((-1, cfg.img_size[0], cfg.img_size[1]))
                clip = np.array(clip).reshape((-1, cfg.img_size[0], cfg.img_size[1]))
                input_np = predict_clips[0:12, :, :]
                target_np = clip[12:15, :, :]
                input_frames = torch.from_numpy(input_np).unsqueeze(0).cuda()
                target_frame = torch.from_numpy(target_np).unsqueeze(0).cuda()
                with torch.no_grad():
                    G_frame = generator(input_frames) # 这是\hatBi+4


                # 对应不同scenario，选择不同的分解方法
                if cfg.scenario == '1':
                    # 这是情况一中的Ai+4
                    A_frame = anomaly.directly_minus(target_frame, G_frame)
                    B_frame = G_frame
                elif cfg.scenario == '2':
                    # 这是情况二中的Ai+4
                    A_frame = anomaly.directly_minus_noise(target_frame - G_frame, lambda_val = 1.5)
                    B_frame = G_frame
                    BN_frame = target_frame - A_frame
                elif cfg.scenario == '3':
                    # 这是情况三中的Ai+4
                    # white
                    # B_frame = anomaly.renew_background(target_frame, G_frame, mu_val = 0.4)
                    # wood_01\02\03
                    B_frame = anomaly.renew_background(target_frame, G_frame, mu_val = 2)
                    A_frame = anomaly.directly_minus(target_frame, B_frame)
                elif cfg.scenario == '4':
                    # lambda和mu都比较小时，意味着噪音项的约束比较大，噪音项更可能趋于0.
                    # 噪音项趋于0，意味着“噪音”信息都包含在了背景和异常当中。
                    # 此时更有可能得到边缘清晰的背景和异常，但同时也会有画布信息保留在异常中。

                    # 这是情况四中的Ai+4
                    # solar的参数，noise不归类到B_frame
                    # A_frame, B_frame = anomaly.renew_background_noise(target_frame, G_frame, lambda_val = 0.7, mu_val = 0.6)
                    # white_01\02\03\04\05的参数
                    # A_frame, B_frame, BN_frame = anomaly.renew_background_noise(target_frame, G_frame, lambda_val = 1.5, mu_val = 8)
                    # white_05的参数
                    A_frame, B_frame, BN_frame = anomaly.renew_background_noise(target_frame, G_frame, lambda_val=1.5, mu_val=10)
                    # wood_01\02\04\05\06的参数
                    # A_frame, B_frame, BN_frame = anomaly.renew_background_noise(target_frame, G_frame, lambda_val=1.0, mu_val=5)
                    # wood_03的参数
                    # A_frame, B_frame, BN_frame = anomaly.renew_background_noise(target_frame, G_frame, lambda_val=0.5, mu_val=3)
                    # fly-4的参数，需要单独设计。fly是灰度图。
                    # A_frame, B_frame = anomaly.renew_background_noise(target_frame, G_frame, lambda_val=0.4, mu_val=100)

                # 将预测结果Bi+4与真实帧Fi+4对比，进行分解计算
                # 将生成结果储存
                B_frame = B_frame.cpu().detach().numpy()
                B_frame = B_frame.squeeze(0)
                B_frame = ((B_frame + 1) * 127.5).transpose(1, 2, 0).astype('uint8')
                cv2.imwrite(os.path.join(b_folder + '/' + cfg.scenario + '/background_' + f'{j+4:04d}' + '.jpg'), B_frame)

                BN_frame = BN_frame.cpu().detach().numpy()
                BN_frame = BN_frame.squeeze(0)
                BN_frame = ((BN_frame + 1) * 127.5).transpose(1, 2, 0).astype('uint8')
                cv2.imwrite(os.path.join(bn_folder + '/' + cfg.scenario + '/background_noise_' + f'{j+4:04d}' + '.jpg'), BN_frame)

                # 将分解结果储存
                # # gray：
                # # A_frame = torch.sum(A_frame.squeeze(0), 0)
                A_frame = A_frame.cpu().detach().numpy()
                A_frame = A_frame.squeeze(0)
                # black_anomaly (except sxenario01\02):
                A_frame = -A_frame
                A_frame = (((A_frame - A_frame.min()) * 255) / (1 - A_frame.min())).transpose(1, 2, 0).astype('uint8')
                # A_frame = ((A_frame + 1) * 127.5).transpose(1, 2, 0).astype('uint8')
                A_frame = cv2.cvtColor(A_frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(a_folder + '/' + cfg.scenario + '/anomaly_' + f'{j+4:04d}' + '.jpg'), A_frame)

                if cfg.show_anomaly:
                    cv2.imshow('anomaly frames', A_frame)
                    cv2.waitKey(1)
                    anomaly_writer.write(A_frame)  # Write anomaly frames.
                    cv2.imshow('background frames', B_frame)
                    cv2.waitKey(1)
                    background_writer.write(B_frame)


                torch.cuda.synchronize()
                end = time.time()
                if j > 1:  # Compute fps by calculating the time used in one completed iteration, this is more accurate.
                    fps = 1 / (end - temp)
                temp = end
                print(f'\rDetecting: [{i + 1:02d}] {j + 1}/{len(test_dataset)}, {fps:.2f} fps.', end='')

        if cfg.show_anomaly:
            anomaly_writer.release()
            background_writer.release()

        # writer.close()


if __name__ == '__main__':

    args = parser.parse_args()
    decompose_cfg = update_config(args, mode='decompose')
    decompose_cfg.print_cfg()
    val(decompose_cfg)

