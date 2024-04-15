import os
import cv2

# 图片路径
bmp_path = '/Data/fly/testing/Set_11/'
jpg_path = 'D:/thu/research/detection/VideoPredictAD/FFP_based/FFP_Decompose/Data/fly/testing/Set_11/Set_11_jpg/'
filelists = os.listdir(bmp_path)

for i, file in enumerate(filelists):
    file_path = os.path.join(bmp_path, file)
    img = cv2.imread(file_path, -1)
    newName = file.replace('.bmp', '.jpg')
    newImg = os.path.join(jpg_path, newName)
    cv2.imwrite(newImg, img)
    print('converting %s to jpg' % file)