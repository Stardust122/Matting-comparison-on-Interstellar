import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# 计算两个视频帧之间的平均绝对误差（MAD）
def calculate_mad_for_frames(folder1, folder2):
    # 获取帧文件列表，并排序确保帧对齐
    frames1 = sorted([os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith('.png')])
    frames2 = sorted([os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith('.png')])
    
    mad_values = []
    
    for frame1_path, frame2_path in zip(frames1[1:], frames2[1:]):
        frame1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
        frame2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)
        # 归一化
        frame1 = frame1 / 255.0
        frame2 = frame2 / 255.0
        
        # 确保两个帧的尺寸一致
        if frame1.shape != frame2.shape:
            frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]))
        
        # 计算每个像素的绝对误差
        abs_diff = np.abs(frame1.astype(np.float32) - frame2.astype(np.float32))
        
        # 计算这一帧的平均绝对误差
        mad = np.mean(abs_diff)
        mad_values.append(mad)
    # 计算整个视频的平均绝对误差
    overall_mad = np.mean(mad_values)
    
    return overall_mad * 1e3

# 计算两个视频帧之间的均方误差（MSE）
def calculate_mse_for_frames(folder1, folder2):
    # 获取帧文件列表，并排序确保帧对齐
    frames1 = sorted([os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith('.png')])
    frames2 = sorted([os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith('.png')])
    
    mse_values = []
    
    for frame1_path, frame2_path in zip(frames1[1:], frames2[1:]):
        frame1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
        frame2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)
        # 归一化
        frame1 = frame1 / 255.0
        frame2 = frame2 / 255.0
        # plt.subplot(1, 2, 1)
        # plt.imshow(frame1)
        # plt.subplot(1, 2, 2)
        # plt.imshow(frame2)
        # plt.show()
        
        # 确保两个帧的尺寸一致
        if frame1.shape != frame2.shape:
            frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]))
        
        # 计算每个像素的误差
        diff = frame1.astype(np.float32) - frame2.astype(np.float32)
        # 计算这一帧的均方误差
        mse = np.mean(diff ** 2)
        mse_values.append(mse)
    # 计算整个视频的均方误差
    overall_mse = np.mean(mse_values)
    
    return overall_mse * 1e3

# 计算两个视频帧之间的 Grad(Spatial Gradient)
def calculate_grad_for_frames(folder1, folder2):
    # 高斯滤波器的标准差
    sigma = 1.4
    # 乘方次数
    q = 2
    # 获取帧文件列表，并排序确保帧对齐
    frames1 = sorted([os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith('.png')])
    frames2 = sorted([os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith('.png')])
    
    grad_values = []
    
    for frame1_path, frame2_path in zip(frames1[1:], frames2[1:]):
        frame1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
        frame2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)
        # 归一化
        frame1 = frame1 / 255.0
        frame2 = frame2 / 255.0
        
        # 确保两个帧的尺寸一致
        if frame1.shape != frame2.shape:
            frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]))
        
        # 计算每个像素的绝对梯度误差
        pred_grad = gaussian_filter(frame1.astype(np.float32), sigma, order=1)
        ref_grad = gaussian_filter(frame2.astype(np.float32), sigma, order=1)
        diff = np.abs(pred_grad - ref_grad)

        # 计算这一帧的绝对误差
        grad = np.sum(diff ** q)
        grad_values.append(grad)

    # 计算整个视频的平均绝对误差
    overall_grad = np.mean(grad_values)
    
    return overall_grad

# 计算两个视频帧之间的 Connectivity
def calculate_connectivity_for_frames(folder1, folder2):
    # 设置阈值步长
    step = 0.1
    # 设置阈值
    theta = 0.15
    # 乘方次数
    p = 1
    # 获取帧文件列表，并排序确保帧对齐
    frames1 = sorted([os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith('.png')])
    frames2 = sorted([os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith('.png')])
    
    connectivity_values = []
    
    for frame1_path, frame2_path in zip(frames1[1:], frames2[1:]):
        frame1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
        frame2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)
        # 归一化
        frame1 = frame1 / 255.0
        frame2 = frame2 / 255.0
        
        # 确保两个帧的尺寸一致
        if frame1.shape != frame2.shape:
            frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]))
        
        # 生成阈值数组和初始阈值映射
        thresh_steps = np.arange(0, 1 + step, step)
        round_down_map = -np.ones_like(frame1)
        
        # 逐步调整阈值，并计算在当前阈值下预测和真实掩码的交集
        for i in range(1, len(thresh_steps)):
            true_thresh = frame1 >= thresh_steps[i]
            pred_thresh = frame2 >= thresh_steps[i]
            intersection = (true_thresh & pred_thresh).astype(np.uint8)

            # 对交集进行连通区域分析，获取每个连通区域的大小
            _, output, stats, _ = cv2.connectedComponentsWithStats(intersection, connectivity=4)
            size = stats[1:, -1]

            # 选择最大的连通区域作为前景
            omega = np.zeros_like(frame1)
            if len(size) != 0:
                max_id = np.argmax(size)
                omega[output == max_id + 1] = 1

            # 更新阈值映射
            mask = (round_down_map == -1) & (omega == 0)
            round_down_map[mask] = thresh_steps[i - 1]

        # 处理剩余的像素
        round_down_map[round_down_map == -1] = 1

        # 计算真实和预测的差异
        true_diff = frame1 - round_down_map
        pred_diff = frame2 - round_down_map

        # 计算真实和预测的连通度
        true_phi = 1 - true_diff * (true_diff >= theta)
        pred_phi = 1 - pred_diff * (pred_diff >= theta)

        connectivity_error = np.abs(true_phi - pred_phi) ** p
        connectivity_values.append(connectivity_error)
    # 计算整个视频的平均绝对误差
    overall_connectivity = np.mean(connectivity_values)
    
    return overall_connectivity * 1e3

# 计算两个视频帧之间的 dtSSD
def calculate_dtSSD_for_frames(folder1, folder2):
    # 获取帧文件列表，并排序确保帧对齐
    frames1 = sorted([os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith('.png')])
    frames2 = sorted([os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith('.png')])
    
    dtSSD_values = []
    
    for pre_frame1_path, pre_frame2_path, frame1_path, frame2_path in zip(frames1[:-1], frames2[:-1], frames1[1:], frames2[1:]):
        pre_frame1 = cv2.imread(pre_frame1_path, cv2.IMREAD_GRAYSCALE)
        pre_frame2 = cv2.imread(pre_frame2_path, cv2.IMREAD_GRAYSCALE)
        frame1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
        frame2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)
        # 归一化
        pre_frame1 = pre_frame1 / 255.0
        pre_frame2 = pre_frame2 / 255.0
        frame1 = frame1 / 255.0
        frame2 = frame2 / 255.0
        
        # 确保两个帧的尺寸一致
        if pre_frame1.shape != pre_frame2.shape:
            pre_frame1 = cv2.resize(pre_frame1, (pre_frame2.shape[1], pre_frame2.shape[0]))
        if frame1.shape != frame2.shape:
            frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]))
        
        # 计算每一帧与前一帧的差异（差分）
        dtSSD = ((frame1 - pre_frame1) - (frame2 - pre_frame2)) ** 2
        dtSSD = np.sqrt(np.mean(dtSSD))

        dtSSD_values.append(dtSSD)
    # 计算整个视频的均方误差
    overall_dtSSD = np.mean(dtSSD_values)
    
    return overall_dtSSD * 1e2

ref_folder = 'Interstellar/alpha'
pred_folder = 'robust_video_matting_results/alpha'

mad = calculate_mad_for_frames(ref_folder, pred_folder)
print(f'两个 alpha matte 视频之间的平均绝对误差（MAD）为：{mad}')
mse = calculate_mse_for_frames(ref_folder, pred_folder)
print(f'两个 alpha matte 视频之间的均方误差（MSE）为：{mse}')
grad = calculate_grad_for_frames(ref_folder, pred_folder)
print(f'两个 alpha matte 视频之间的 Grad 为：{grad}')
connectivity = calculate_connectivity_for_frames(ref_folder, pred_folder)
print(f'两个 alpha matte 视频之间的 Connectivity 为：{connectivity}')
dtSSD = calculate_dtSSD_for_frames(ref_folder, pred_folder)
print(f'两个 alpha matte 视频之间的 dtSSD 为：{dtSSD}')
