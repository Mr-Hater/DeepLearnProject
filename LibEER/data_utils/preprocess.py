import torch.nn.functional as F

import numpy as np
import scipy.signal
from scipy import signal
from scipy.signal import filtfilt, stft

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from functools import partial


# 对eeg信号进行各项数据预处理操作（去除眼动干扰，带通滤波，提取频段，分段样本，提取特征, 归一化等）
# 并且同时
# 最终希望处理成为能够经过划分后就能输入模型的数据


def preprocess(data, baseline, sample_rate, pass_band, extract_bands, time_window, overlap, car=False, whiten=False,
               sample_length=1, stride=1, only_seg=False, feature_type='DE', eog_clean=True, normalization=False):
    """
    提供预处理操作的完整流程
    input shape -> data:  (session, subject, trail, channel, original_data)
                   label: (session, subject, trail, label)
    output shape -> data :  (session, subject, trail, sample, time, channel, feature)
                    label : (session, subject, trail, sample, label)
    """
    if baseline is not None:
        data = baseline_removal(data, baseline)  # 基线校正
    if not only_seg:  # 如果不是仅进行分段
        if pass_band != [-1, -1]:  # 如果需要带通滤波
            data = bandpass_filter(data, sample_rate, pass_band)  # 带通滤波
        if eog_clean:  # 如果需要去除眼动伪影
            data = eog_remove(data)  # 去除眼动伪影
        # data, label = frequency_band_extraction(data, label, sample_rate, extract_bands, time_window)
        data = feature_extraction(data, sample_rate, extract_bands, time_window, overlap, feature_type)  # 特征提取
    data, feature_dim = segment_data(data, sample_length, stride)  # 数据分段
    return data, feature_dim


def noise_label(train_label, num_classes=3, level=0.1):
    """
    为训练标签添加噪声，用于标签平滑或噪声标签学习

    参数:
        train_label: 原始标签
        num_classes: 类别数量
        level: 噪声水平

    返回:
        噪声标签（概率分布形式）
    """
    if type(train_label[0]) is np.ndarray:
        train_label = [np.where(tl == 1)[0] for tl in train_label]  # 将one-hot编码转换为类别索引

    noised_label = [[] for _ in train_label]  # 初始化噪声标签列表
    if num_classes == 4:  # 四分类情况
        for i, label in enumerate(train_label):
            if label == 0:
                noised_label[i] = [1 - 3 / 4 * level, 1 / 4 * level, 1 / 4 * level, 1 / 4 * level]
            elif label == 1:
                noised_label[i] = [1 / 3 * level, 1 - 2 / 3 * level, 1 / 3 * level, 0]
            elif label == 2:
                noised_label[i] = [1 / 4 * level, 1 / 4 * level, 1 - 3 / 4 * level, 1 / 4 * level]
            else:
                noised_label[i] = [1 / 3 * level, 0, 1 / 3 * level, 1 - 2 / 3 * level]
    elif num_classes == 3:  # 三分类情况
        for i, label in enumerate(train_label):
            if label == 0:
                noised_label[i] = [1 - 2 / 3 * level, 2 / 3 * level, 0]
            elif label == 1:
                noised_label[i] = [1 / 3 * level, 1 - 2 / 3 * level, 1 / 3 * level]
            else:
                noised_label[i] = [0, 2 / 3 * level, 1 - 2 / 3 * level]
    elif num_classes == 2:  # 二分类情况
        for i, label in enumerate(train_label):
            if label == 0:
                noised_label[i] = [1, 0]
            elif label == 1:
                noised_label[i] = [0, 1]
    return noised_label


def baseline_removal(data, base):
    """
    基线校正：从EEG数据中减去基线信号

    参数:
        data: EEG数据，shape: (session, subject, trail, channel, original_data)
        base: 基线数据，shape: (session, subject, trail, channel, base_data)

    返回:
        基线校正后的EEG数据
    """
    for ses_i, ses_data in enumerate(data):
        for sub_i, sub_data in enumerate(ses_data):
            for trail_i, trail_data in enumerate(sub_data):
                trail_time = trail_data.shape[1]  # 当前试次的时间点数
                base_time = base[ses_i][sub_i][trail_i].shape[1]  # 基线数据的时间点数
                base_data = base[ses_i][sub_i][trail_i]  # 基线数据
                # 分段减去基线
                for i in range(int(trail_time / base_time)):
                    trail_data[:, i * base_time:(i + 1) * base_time] = trail_data[:,
                                                                       i * base_time:(i + 1) * base_time] - base_data
                last = trail_time % base_time  # 剩余部分
                if last != 0:
                    trail_data[:, -last:] = trail_data[:, -last:] - base_data[:, :last]  # 处理剩余部分
                data[ses_i][sub_i][trail_i] = trail_data
    return data


def bandpass_filter(data, frequency, pass_band):
    """
    对EEG信号进行带通滤波操作

    参数:
        data: EEG信号，shape: (session, subject, trail, channel, original_data)
        frequency: 采样率
        pass_band: 通带频率范围 [low, high]

    返回:
        带通滤波后的EEG信号，shape: (session, subject, trail, channel, filter_data)
    """
    # 定义奈奎斯特频率（防止信号混叠的最小采样率）
    nyq = 0.5 * frequency
    # 获取Butterworth滤波器系数
    b, a = signal.butter(N=5, Wn=[pass_band[0] / nyq, pass_band[1] / nyq], btype='bandpass')
    # 对所有通道进行线性滤波
    for ses_i, ses_data in enumerate(data):
        for sub_i, sub_data in enumerate(ses_data):
            for trail_i, trail_data in enumerate(sub_data):
                data[ses_i][sub_i][trail_i] = \
                    filtfilt(b, a, trail_data)  # 使用零相位滤波

    return data


def whiten(data):
    """
    数据白化操作（去相关和标准化方差）

    参数:
        data: 输入数据

    返回:
        白化后的数据
    """
    # 中心化操作
    new_data = []
    for session in range(len(data)):
        new_session = []
        for subject in range(len(data[0])):
            new_subject = []
            for trail in range(len(data[0][0])):
                trail = np.array(trail)
                # 中心化操作
                trail_mean = trail.mean(axis=0)
                trail_center = trail - trail_mean
                # 计算协方差矩阵
                cov = np.dot(trail_center.T, trail_center) / (trail_center.shape[0])
                # 特征值计算
                eig_vals, eig_vecs = np.linalg.eigh(cov)
                D = np.diag(1.0 / np.sqrt(eig_vals))  # 特征值对角矩阵的逆平方根
                W = np.dot(eig_vecs, D).dot(eig_vecs.T)  # 白化矩阵

                trail_whitened = np.dot(trail_center, W)  # 白化变换
                new_subject.append(trail_whitened)
            new_session.append(new_subject)
        new_data.append(new_session)
    return new_data


# def ica_eog_remove(data):

def eog_remove(data):
    """
    通过伪影子空间重构去除眼动干扰
    输入: 原始EEG数据
    输出: 去除眼动伪影的EEG数据
    input shape : (session, subject, trail, channel, filter_data)
    output shape : (session, subject, trail, channel, filter_data)
    """
    pca = PCA()  # 初始化PCA（目前未实现完整功能）
    return data  # 返回原始数据（占位符）


def feature_extraction(data, sample_rate, extract_bands, time_window, overlap, feature_type):
    """
    特征提取函数

    参数:
        data: 带通滤波后的信息
        sample_rate: 采样率
        extract_bands: 要提取的频段列表
        time_window: 时间窗口长度
        overlap: 重叠率
        feature_type: 特征类型（psd, de, de_reduced）

    input shape -> data:  (session, subject, trail, channel, band, filter_data)
    output shape -> data:  (session, subject, trail, sample, channel, band, band_feature)
    """
    isLds = False
    if feature_type.endswith("_lds"):  # 检查是否需要进行LDS处理
        isLds = True
        feature_type = feature_type[:-4]  # 移除后缀
    # 根据特征类型选择提取函数
    fe = {
        'psd': psd_extraction,  # 功率谱密度
        'de': de_extraction,  # 微分熵
        'de_reduced': de_reduced_extraction  # 加速的微分熵提取
    }[feature_type]

    feature_data = []
    for ses_i, ses_data in enumerate(data):
        ses_fe = []
        for sub_i, sub_data in enumerate(ses_data):
            sub_fe = []
            for trail_i, trail_data in enumerate(sub_data):
                sub_fe_data = fe(trail_data, sample_rate, extract_bands, time_window, overlap)  # 特征提取
                if isLds:  # 如果需要LDS处理
                    sub_fe_data = lds(sub_fe_data)  # 线性动态系统处理
                # if trail_i == 0 and sub_i == 0:
                #     print(sub_fe_data)

                sub_fe.append(sub_fe_data)
            ses_fe.append(sub_fe)
        feature_data.append(ses_fe)
    return feature_data


def psd_extraction(data, sample_rate, extract_bands, time_window, overlap):
    """
    功率谱密度特征提取

    参数:
        data: EEG数据，shape: (channel, filter_data)
        sample_rate: 采样率
        extract_bands: 要提取的频段
        time_window: 时间窗口长度（秒）
        overlap: 重叠率

    返回:
        PSD特征，shape: (sample, channel, band_psd_feature)
    """
    if extract_bands is None:  # 默认频段
        extract_bands = [[1, 4], [4, 8], [8, 14], [14, 31], [31, 50]]  # delta, theta, alpha, beta, gamma
    noverlap = int(overlap * sample_rate)  # 重叠样本数
    window_size = int(time_window * sample_rate)  # 窗口样本数
    if noverlap != 0:  # 如果有重叠
        sample_num = (data.shape[1] - window_size) // (window_size - noverlap)  # 计算样本数
    else:  # 无重叠
        sample_num = (data.shape[1]) // window_size
    psd_data = np.zeros((sample_num, data.shape[0], len(extract_bands)))  # 初始化PSD数据
    t = 0  # 时间指针
    for i in range(sample_num):
        # 使用Welch方法计算功率谱密度
        f, psd = scipy.signal.welch(data[:, t:t + window_size],
                                    fs=sample_rate, nperseg=window_size, window='hamming')
        for b_i, bands in enumerate(extract_bands):  # 对每个频段
            # 计算频段内的平均功率（dB单位）
            psd_data[i, :, b_i] = np.mean(10 * np.log10(psd[:, bands[0]:bands[1] + 1]), axis=1)
        t += window_size - noverlap  # 移动时间指针
    return psd_data


def de_reduced_extraction(data, sample_rate, extract_bands, time_window, overlap):
    """
    使用简化方法加速DE特征提取（基于STFT）

    参数:
        data: 原始EEG数据，shape: (channel, filter_data)
        sample_rate: EEG信号的采样率
        extract_bands: 需要提取的频段
        time_window: 一次提取的时间窗口
        overlap: 重叠率

    返回:
        需要计算的DE特征
    """
    if extract_bands is None:  # 默认频段
        extract_bands = [[1, 4], [4, 8], [8, 14], [14, 31], [31, 50]]
    noverlap = int(overlap * sample_rate)  # 重叠样本数
    window_size = int(time_window * sample_rate)  # 窗口样本数
    if noverlap != 0:  # 如果有重叠
        sample_num = (data.shape[1] - window_size) // (window_size - noverlap)  # 计算样本数
    else:  # 无重叠
        sample_num = (data.shape[1]) // window_size
    de_data = np.zeros((sample_num, data.shape[0], len(extract_bands)))  # 初始化DE数据

    # 使用STFT计算短时傅里叶变换
    fs, ts, Zxx = stft(data, fs=sample_rate, window='hamming', nperseg=window_size,
                       noverlap=noverlap, boundary=None)

    for b_idx, band in enumerate(extract_bands):  # 对每个频段
        fb_indices = np.where((fs >= band[0]) & (fs <= band[1]))[0]  # 获取频段索引
        fourier_coe = np.real(Zxx[:, fb_indices, :])  # 取实部
        parseval_energy = np.mean(np.square(fourier_coe), axis=1)  # 计算Parseval能量
        # 计算微分熵：log2(100 * 能量)
        de_data[:, :, b_idx] = np.transpose(np.log2(100 * parseval_energy))[:sample_num]
    return de_data


def de_extraction(data, sample_rate, extract_bands, time_window, overlap):
    """
    DE（微分熵）特征提取

    参数:
        data: 原始EEG数据，shape: (channel, filter_data)
        sample_rate: EEG信号的采样率
        extract_bands: 需要提取的频段
        time_window: 一次提取的时间窗口
        overlap: 重叠率

    返回:
        需要计算的DE特征
    """
    if extract_bands is None:  # 默认频段
        extract_bands = [[0.5, 4], [4, 8], [8, 14], [14, 30], [30, 50]]
    nyq = 0.5 * sample_rate  # 奈奎斯特频率
    noverlap = int(overlap * sample_rate)  # 重叠样本数
    window_size = int(time_window * sample_rate)  # 窗口样本数
    if noverlap != 0:  # 如果有重叠
        sample_num = (data.shape[1] - window_size) // (window_size - noverlap)  # 计算样本数
    else:  # 无重叠
        sample_num = (data.shape[1]) // window_size
    de_data = np.zeros((sample_num, data.shape[0], len(extract_bands)))  # 初始化DE数据

    for b_idx, band in enumerate(extract_bands):  # 对每个频段
        # 设计带通滤波器
        b, a = signal.butter(3, [band[0] / nyq, band[1] / nyq], 'bandpass')
        # 应用滤波器
        band_data = signal.filtfilt(b, a, data)
        t = 0  # 时间指针
        for i in range(sample_num):
            # 计算微分熵：1/2 * log2(2πe * 方差)
            de_data[i, :, b_idx] = 1 / 2 * np.log2(2 * np.pi * np.e *
                                                   np.var(band_data[:, t:t + window_size], axis=1, ddof=1))
            t += window_size - noverlap  # 移动时间指针
    return de_data


def lds(data):
    """
    使用线性动态系统方法处理数据（卡尔曼滤波）

    参数:
        data: 输入数据，shape: (time, channel, feature)

    返回:
        处理后的数据，shape: (time, channel, feature)
    """
    [num_t, num_channel, num_feature] = data.shape
    # 展平通道和特征维度
    data = data.reshape((data.shape[0], -1))

    # 初始化参数
    prior_correlation = 0.01  # 先验协方差
    transition_matrix = 1  # 状态转移矩阵
    noise_correlation = 0.0001  # 过程噪声协方差
    observation_matrix = 1  # 观测矩阵
    observation_correlation = 1  # 观测噪声协方差

    # 计算均值用于初始化
    mean = np.mean(data, axis=0)
    data = data.T  # 转置以便更容易处理时间维度

    num_features, num_samples = data.shape
    P = np.zeros(data.shape)  # 预测协方差
    U = np.zeros(data.shape)  # 状态估计
    K = np.zeros(data.shape)  # 卡尔曼增益
    V = np.zeros(data.shape)  # 估计协方差

    # 初始卡尔曼滤波设置
    K[:, 0] = prior_correlation * observation_matrix / (
            observation_matrix * prior_correlation * observation_matrix + observation_correlation) * np.ones(
        (num_features,))
    U[:, 0] = mean + K[:, 0] * (data[:, 0] - observation_matrix * prior_correlation)
    V[:, 0] = (np.ones((num_features,)) - K[:, 0] * observation_matrix) * prior_correlation

    # 随时间应用卡尔曼滤波
    for i in range(1, num_samples):
        P[:, i - 1] = transition_matrix * V[:, i - 1] * transition_matrix + noise_correlation
        K[:, i] = P[:, i - 1] * observation_matrix / (
                observation_matrix * P[:, i - 1] * observation_matrix + observation_correlation)
        U[:, i] = transition_matrix * U[:, i - 1] + K[:, i] * (
                data[:, i] - observation_matrix * transition_matrix * U[:, i - 1])
        V[:, i] = (1 - K[:, i] * observation_matrix) * P[:, i - 1]

    # 返回处理后的数据，重塑以匹配原始输入形状
    return U.T.reshape((num_t, num_channel, num_feature))


def segment_data(data, sample_length, stride):
    """
    数据分段函数

    特征数据:
    input shape -> data:  (session, subject, trail, sample1, channel, band)
                    label: (session, subject, trail, sample1, label)
    output shape -> data:  (session, subject, trail, sample2, sample2_length, channel, band)
                    label: (session, subject, trail, sample2, label)

    原始数据:
    input shape -> data: (session, subject, trail, channel, data_points)
                   label: (session, subject, trail)
    output shape -> data: (session, subject, trail, sample, channel, seg_data_points)
                    label: (session, subject, trail)
    """
    if sample_length == 1:  # 如果不需要分段
        print(len(data[0][0]))
        print(len(data[0][0][0]))
        print(len(data[0][0][0][0]))
        return data, len(data[0][0][0][0][0])  # 返回原始数据和特征维度
    else:  # 需要分段
        seg_data = []
        for ses_i, session in enumerate(data):
            seg_session = []
            for sub_i, subject in enumerate(data[ses_i]):
                seg_sub = []
                seg_sub_label = []
                for t_i, trail in enumerate(data[ses_i][sub_i]):
                    seg_trail = None
                    trail = np.array(trail)
                    if len(trail.shape) == 3:  # 特征数据情况
                        # trail shape -> (sample, channel, band)
                        trail = np.asarray(trail)
                        num_sample = (len(trail) - sample_length) // stride + 1  # 计算分段后的样本数
                        seg_trail = np.zeros((num_sample, sample_length, len(trail[0]), len(trail[0][0])))
                        # 通过滑动窗口切割一维数组形成二维数组
                        for i in range(num_sample):
                            seg_trail[i] = trail[i * stride:i * stride + sample_length]  # 滑动窗口分段
                    elif len(trail.shape) == 2:  # 原始数据情况
                        # trail shape -> (channel, data_points)
                        num_sample = (len(trail[0]) - sample_length) // stride + 1
                        seg_trail = np.zeros((num_sample, len(trail), sample_length))
                        for i in range(num_sample):
                            seg_trail[i] = trail[:, i * stride:i * stride + sample_length]
                    seg_sub.append(seg_trail)
                seg_session.append(seg_sub)
            seg_data.append(seg_session)
        # 根据数据类型返回特征维度
        if len(seg_data[0][0][0].shape) == 4:  # 特征数据
            return seg_data, len(seg_data[0][0][0][0][0][0])  # 返回特征维度
        elif len(seg_data[0][0][0].shape) == 3:  # 原始数据
            return seg_data, len(seg_data[0][0][0][0][0])


def label_process(data, label, bounds=None, onehot=False, label_used=None):
    """
    标签处理函数

    input shape -> data: (session, subject, trail, sample)
                   label: (session, subject, trail)
    output shape -> data: (session, subject, trail, sample)
                    label: (session, subject, trail, sample)

    bounds shape -> 2, 高情绪状态 > bounds[1], 低情绪状态 < bounds[0]
    如果数据集是hci, deap, dreamer, 则标签按效价、唤醒度、支配度、喜好度排序
    """
    available_label = ['valence', 'arousal', 'dominance', 'liking']  # 可用的标签类型
    if label_used is None:  # 如果未指定使用的标签
        label_used = ['valence']  # 默认使用效价
    used_id = [available_label.index(item) for item in label_used]  # 获取使用的标签索引

    if type(label[0][0][0]) is np.ndarray:  # 如果标签是多维的（如DEAP数据集）
        num_classes = np.power(2, len(used_id))  # 类别数为2的标签数次方
    else:  # 单标签
        num_classes = len(np.unique(label))  # 唯一标签数

    new_label = []
    new_data = []
    for ses_i, ses_label in enumerate(label):
        new_ses_label = []
        new_ses_data = []
        for sub_i, sub_label in enumerate(ses_label):
            new_sub_label = []
            new_sub_data = []
            for trail_i, trail_label in enumerate(sub_label):
                new_trail_label = []
                new_trail_data = data[ses_i][sub_i][trail_i]
                num_sample = len(new_trail_data)
                if type(trail_label) is np.ndarray:  # 处理多维标签
                    pro_label = []
                    for value_id in used_id:  # 对每个使用的标签维度
                        value = trail_label[value_id]
                        if value <= bounds[0]:  # 低于下限为低情绪
                            pro_label.append(0)
                        elif value >= bounds[1]:  # 高于上限为高情绪
                            pro_label.append(1)
                    # pro_label shape -> (num_used_label, 2)
                    # 处理成普通标签
                    if len(pro_label) == len(used_id):
                        # 将二进制列表转换为整数标签
                        trail_label = int("".join(str(i) for i in pro_label), 2)
                    else:
                        # 丢弃数据和标签
                        continue
                if onehot:  # 如果需要one-hot编码
                    oh_code = np.zeros((1, num_classes), dtype='int32')
                    oh_code[0][trail_label] = 1
                    trail_label = oh_code
                    new_trail_label = np.tile(trail_label, (num_sample, 1))  # 复制到每个样本
                else:  # 普通标签
                    trail_label = np.ones(1, dtype='int32') * trail_label
                    new_trail_label = np.tile(trail_label, num_sample)  # 复制到每个样本
                new_sub_data.append(new_trail_data)
                new_sub_label.append(new_trail_label)
            new_ses_label.append(new_sub_label)
            new_ses_data.append(new_sub_data)
        new_label.append(new_ses_label)
        new_data.append(new_ses_data)

    return new_data, new_label, num_classes


def normalize(train_data, val_data, test_data=None, dim="sample", method="z-score"):
    """
    数据归一化函数

    参数:
        train_data: 训练数据
        val_data: 验证数据
        test_data: 测试数据
        dim: 归一化维度 ("sample"或"electrode")
        method: 归一化方法 ("z-score"或"minmax")

    返回:
        归一化后的训练、验证、测试数据
    """

    all_data = np.concatenate((train_data, val_data), axis=0)  # 合并训练和验证数据
    data_shape = all_data.shape
    scaler = None
    scaled_test_data = None

    if dim == "sample":  # 按样本维度归一化
        if len(data_shape) == 3:  # 三维数据
            all_data = all_data.reshape(data_shape[0], data_shape[1] * data_shape[2])
        elif len(data_shape) == 4:  # 四维数据
            all_data = all_data.reshape(data_shape[0], data_shape[1] * data_shape[2] * data_shape[3])
        scaled_data = None

        if method == "z-score":  # Z-score标准化
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(all_data)
        if method == "minmax":  # Min-Max归一化
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(all_data)

        # 恢复原始形状
        if len(data_shape) == 3:
            scaled_data = scaled_data.reshape(data_shape[0], data_shape[1], data_shape[2])
        elif len(data_shape) == 4:
            scaled_data = scaled_data.reshape(data_shape[0], data_shape[1], data_shape[2], data_shape[3])

        if test_data is not None:  # 如果有测试数据
            # 重塑测试数据
            if len(test_data.shape) == 3:
                test_data_reshaped = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])
            elif len(test_data.shape) == 4:
                test_data_reshaped = test_data.reshape(test_data.shape[0],
                                                       test_data.shape[1] * test_data.shape[2] * test_data.shape[3])

            scaled_test_data = scaler.transform(test_data_reshaped)  # 使用训练数据的scaler变换测试数据

            # 恢复测试数据形状
            if len(test_data.shape) == 3:
                scaled_test_data = scaled_test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2])
            elif len(test_data.shape) == 4:
                scaled_test_data = scaled_test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2],
                                                            test_data.shape[3])
        return scaled_data[:len(train_data)], scaled_data[len(train_data):], scaled_test_data  # 分割回训练和验证数据

    if dim == "electrode":  # 按电极维度归一化
        # data shape -> (sample, channel, band)
        all_data_t = all_data.reshape(len(all_data), len(all_data[0]) * len(all_data[0][0])).T  # 转置
        test_data_t = test_data.reshape(len(test_data), len(test_data[0]) * len(test_data[0][0])).T

        for i in range(all_data_t.shape[0]):  # 对每个电极
            _range = np.max(all_data_t[i]) - np.min(all_data_t[i])  # 计算范围
            all_data_t[i] = (all_data_t[i] - np.min(all_data_t[i])) / _range  # Min-Max归一化
            test_data_t[i] = (test_data_t[i] - np.min(all_data_t[i])) / _range  # 使用相同的范围

        # 恢复原始形状
        norm_data = all_data_t.T.reshape(len(all_data), len(all_data[0]), len(all_data[0][0]))
        norm_test_data = test_data_t.T.reshape(len(test_data), len(test_data[0]), len(test_data[0][0]))
        return norm_data[:len(train_data)], norm_data[len(train_data):], norm_test_data


def ele_normalize(all_data):
    """
    电极维度归一化（简化的按电极归一化）

    参数:
        all_data: 输入数据，shape: (sample, channel, band)

    返回:
        归一化后的数据
    """
    # data shape -> (sample, channel, band)
    all_data_t = all_data.reshape(len(all_data), len(all_data[0]) * len(all_data[0][0])).T  # 转置
    for i in range(all_data_t.shape[0]):  # 对每个电极
        _range = np.max(all_data_t[i]) - np.min(all_data_t[i])  # 计算范围
        all_data_t[i] = (all_data_t[i] - np.min(all_data_t[i])) / _range  # Min-Max归一化
    norm_data = all_data_t.T.reshape(len(all_data), len(all_data[0]), len(all_data[0][0]))  # 恢复原始形状
    return norm_data


def baseline_normalisation(data, baseline):
    """
    基线标准化（除以基线）

    input shape : data (session, subject, trail)
                  baseline (session, subject, trail)
    """
    norm_data = []
    for ses_i in range(len(data)):
        session = data[ses_i]
        ses_base = baseline[ses_i]
        normal_session = []
        for sub_i in range(len(session)):
            subject = session[sub_i]
            sub_base = ses_base[sub_i]
            norm_subject = []
            for trail_i in range(len(subject)):
                trail = subject[trail_i]
                trail_base = sub_base[trail_i]
                norm_trail = []
                base_len = len(trail_base)
                for i in range(len(trail) // base_len):
                    norm_trail[i * base_len:(i + 1) * base_len] = trail[
                                                                  i * base_len:(i + 1) * base_len] / trail_base  # 除以基线
                # if trail_i == 0 and sub_i == 0:
                #     print(norm_trail)
                norm_subject.append(norm_trail)
            normal_session.append(norm_subject)
        norm_data.append(normal_session)
    return norm_data


def generate_adjacency_matrix(channel_names, channel_adjacent):
    """
    生成邻接矩阵（基于相邻关系）

    参数:
        channel_names: 通道名称列表
        channel_adjacent: 通道相邻关系的字典

    返回:
        邻接矩阵
    """
    channel_names = np.array(channel_names)
    channel_num = len(channel_names)
    adjacency_matrix = np.zeros((channel_num, channel_num))  # 初始化邻接矩阵
    for key, value in channel_adjacent.items():  # 遍历相邻关系
        idx1 = np.where(channel_names == key)[0][0]  # 获取通道索引
        for chan in value:  # 遍历相邻通道
            idx2 = np.where(channel_names == chan)[0][0]
            adjacency_matrix[idx1][idx2] = 1  # 设置相邻关系
    return adjacency_matrix


def generate_rgnn_adjacency_matrix(channel_names, channel_loc, global_channel_pair):
    """
    生成RGNN（区域图神经网络）邻接矩阵（基于空间距离）

    参数:
        channel_names: 通道名称列表
        channel_loc: 通道位置字典
        global_channel_pair: 全局通道对列表

    返回:
        RGNN邻接矩阵
    """
    channel_names = np.array(channel_names)
    channel_num = len(channel_names)
    adjacency_matrix = np.zeros((channel_num, channel_num))

    for chan1 in channel_names:  # 遍历所有通道对
        idx1 = np.where(channel_names == chan1)[0][0]
        for chan2 in channel_names:
            idx2 = np.where(channel_names == chan2)[0][0]
            if chan1 == chan2:  # 对角线设置为1
                adjacency_matrix[idx1][idx2] = 1
            else:  # 计算空间距离并转换为权重
                cor1 = np.array(channel_loc[chan1]) / 10  # 归一化坐标
                cor2 = np.array(channel_loc[chan2]) / 10
                dis_sq = 0
                for i in range(3):  # 计算欧氏距离平方
                    dis_sq += np.square(cor1[i] - cor2[i])
                # 权重 = min(5/距离平方, 1)
                adjacency_matrix[idx1][idx2] = min(5 / dis_sq, 1)
                adjacency_matrix[idx2][idx1] = min(5 / dis_sq, 1)
    # print((np.where(adjacency_matrix > 0.1)[0].shape[0])/62/62)
    # 应用差分不对称性杠杆
    adjacency_matrix = differential_asymmetry_leverage(channel_names, adjacency_matrix, global_channel_pair)
    return adjacency_matrix


def differential_asymmetry_leverage(channel_names, adjacency_matrix, global_channel_pair):
    """
    应用差分不对称性杠杆（对特定通道对调整权重）

    参数:
        channel_names: 通道名称列表
        adjacency_matrix: 当前邻接矩阵
        global_channel_pair: 全局通道对列表

    返回:
        调整后的邻接矩阵
    """
    for pair in global_channel_pair:  # 遍历全局通道对
        idx1 = np.where(channel_names == pair[0])[0][0]
        idx2 = np.where(channel_names == pair[1])[0][0]
        adjacency_matrix[idx1][idx2] -= 1  # 减少权重
        adjacency_matrix[idx2][idx1] -= 1
    return adjacency_matrix