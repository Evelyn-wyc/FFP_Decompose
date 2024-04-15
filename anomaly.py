import torch
import cv2
import numpy as np

def directly_minus(gt_frames, gen_frames):
    return gt_frames - gen_frames


def directly_minus_noise(diff_frame, lambda_val):
    # min||Fi+4 - \hatBi+4 - Ai+4||^2 + lambda * ||Ai+4||_1, that is, min||diff_frame - Ai+4||^2 + lambda * ||Ai+4||_1
    A_frame = torch.nn.Parameter(torch.zeros_like(diff_frame), requires_grad=True)
    optimizer = torch.optim.Adam([A_frame], lr=0.1)
    criterion1 = torch.nn.MSELoss()
    criterion2 = torch.nn.L1Loss()
    for epoch in range(1000):
        optimizer.zero_grad()
        loss1 = criterion1(diff_frame, A_frame)
        loss2 = lambda_val * criterion2(A_frame, torch.zeros_like(A_frame))
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        # 打印每轮迭代的loss
        if epoch % 100 == 0:
            print(f'EPOCH: {epoch}, Loss1: {loss1.item()}, Loss2: {loss2.item()}')
    return A_frame

def renew_background(target_frame, G_frame, mu_val):
    # min||Fi+4 - Bi+4||_1 + mu * ||Bi+4 - \hatB_i+4||^2
    B_frame = torch.nn.Parameter(torch.zeros_like(target_frame), requires_grad=True)
    optimizer = torch.optim.Adam([B_frame], lr=0.001)
    criterion1 = torch.nn.L1Loss()
    criterion2 = torch.nn.MSELoss()
    for epoch in range(1000):
        optimizer.zero_grad()
        loss1 = criterion1(target_frame, B_frame)
        loss2 = mu_val * criterion2(G_frame, B_frame)
        loss = loss1 + loss2
        # loss.requires_grad = True
        loss.backward()
        optimizer.step()
        # 打印每轮迭代的loss
        if epoch % 100 == 0:
            print(f'EPOCH: {epoch}, Loss1: {loss1.item()}, Loss2: {loss2.item()}')
    return B_frame

def renew_background_noise(target_frame, G_frame, lambda_val, mu_val):
    # min||Fi+4 - Bi+4 - Ai+4||^2 + lambda * ||Ai+4||_1 + mu * ||Bi+4 - \hatB_i+4||^2
    A_frame = torch.nn.Parameter(torch.zeros_like(target_frame - G_frame), requires_grad=True)
    B_frame = torch.nn.Parameter(torch.zeros_like(target_frame), requires_grad=True)
    optimizer = torch.optim.Adam([A_frame, B_frame], lr=0.001)
    criterion1 = torch.nn.MSELoss()
    criterion2 = torch.nn.L1Loss()
    criterion3 = torch.nn.MSELoss()
    for epoch in range(1000):
        optimizer.zero_grad()
        loss1 = criterion1(target_frame, B_frame + A_frame)
        loss2 = lambda_val * criterion2(A_frame, torch.zeros_like(A_frame))
        loss3 = mu_val * criterion3(B_frame, G_frame)
        loss = loss1 + loss2 + loss3
        loss.backward()
        optimizer.step()
        # 输出损失值
        if epoch % 100 == 0:
            print(f"Iteration {epoch}, Loss1: {loss1.item()}, Loss2: {loss2.item()}, Loss3: {loss3.item()}")
    # 将噪音项加回原图
    BN_frame = target_frame - A_frame
    # normalize A_frame
    return A_frame, B_frame, BN_frame
