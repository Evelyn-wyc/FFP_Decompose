# IPAD: Iterative Prediction and Decomposition for Sparse Anomaly Detection in Video With Non-Low-Rank Background

This repository of Pytorch implementation of paper ["IPAD: Iterative Prediction and Decomposition for Sparse Anomaly Detection in Video With Non-Low-Rank Background"](https://onlinelibrary.wiley.com/doi/abs/10.1002/qre.3723). Parts of the code are inspired by [this link](https://github.com/feiyuhuahuo/Anomaly_Prediction).

## Network Pipeline

![IPAD](https://github.com/user-attachments/assets/7517383e-a1ce-4ae5-a494-9e240e946dc8)

## Data & Requirments
Data can be downloaded in [Google Drive](https://drive.google.com/file/d/1TDnWhqEdHu9N7oOgr9EK7IgGPsOqAQFo/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1B2fLc3-iq2uCDQ9UBK15Yg?pwd=e344).

Require PyTorch, opencv and other common packages.

## File Description

### Method (Stage 1): Prediction
`config.py`, `Dataset.py`

`unet.py`: The U-Net architecture.

`train.py`: The training file.

`loss.py`: Calculate the loss when training NN.

### Method (Stage 2): Single-frame Decomposition

`anomaly.py`: 4 different scenarios.

`decompose.py`: Core code of proposed method.

### Experiment

`function.py`: Define PSNR, dice coefficient.

`anomaly_revised.py`: Set threshold to calculate dice coefficient.

`score.py`: Calculate PSNR, dice coefficient for different methods.

`test_threshold.py`: Draw comparison figure for different methods.

`video_result.py`: Show video form of consecutive frames.

### Tools

`bmp2jpg.py`, `RGB2Gray.py`

`frame2mat.py`: This file makes all the frames together into a mat, which is a datatype in Matlab. Use the mat transform by frame2mat.py, we can compare with STSSD by the new image sequences.

`mat2frame.py`: This file change the solar dataset from mat to jpg. It is a gray dataset.

`GenerateVideo_DiffPos.py`:	Generate anomaly feature with fixed background (not used).

`GenerateVideo_L2R.py`: Generate the anomaly with moving background from left to right. Tools like image2frame (a single image transform into an image sequence, moving from left to right), save_video, video2frame (save the image sequences into testing path) are also in this file.

## Evaluation
```
# Validate with a pretrained model.

python decompose.py --dataset=wood --trained_model=avenue_15000.pth --scenario=4 --show_anomaly=False
```
