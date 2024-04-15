from tools.image_video import Image2Video, Image2Gif
import os


def GenerateVideoResult(dataset, mode):
    path = './results/' + dataset + '/' +  mode +'/'
    for i in os.listdir(path): # i folders
        frame_path = path + f'{i}/'
        for j in range(1, 5): # 4 scenarios
            frame_path_scenario = frame_path + f'{j}'
            print(frame_path_scenario)
            video_name = frame_path + f'anomaly_video_scenario{j}.avi'
            Image2Video(frame_path_scenario, video_name)
    return

def GenerateGifResult(dataset, mode):
    path = './results/' + dataset + '/' +  mode +'/'
    for i in os.listdir(path): # i folders
        frame_path = path + f'{i}/'
        for j in range(1, 5): # 4 scenarios
            frame_path_scenario = frame_path + f'{j}'
            print(frame_path_scenario)
            gif_name = frame_path + f'anomaly_gif_scenario{j}.gif'
            Image2Gif(frame_path_scenario, gif_name)
    return

GenerateVideoResult('wood', 'anomaly')
GenerateVideoResult('wood', 'gen_background')
GenerateGifResult('wood', 'anomaly')
GenerateGifResult('wood', 'gen_background')