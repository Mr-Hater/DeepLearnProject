# 导入必要的数值计算库
import numpy as np
# 导入PyTorch深度学习框架
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import torch.optim as optim
# 导入数据集常量定义
from LibEER.data_utils.constants.deap import DEAP_CHANNEL_NAME
from LibEER.data_utils.constants.seed import SEED_CHANNEL_NAME

# 导入工具库
from tqdm import tqdm  # 进度条显示
import yaml  # YAML配置文件解析

# 导入自定义工具模块
from LibEER.utils.store import save_state  # 模型状态保存
from LibEER.utils.metric import Metric  # 评估指标计算

class TSception(nn.Module):
    """
    核心方法介绍：TSception模型
    基于Inception架构的时空卷积神经网络，专门用于脑电信号分类
    特点：同时捕获时间维度和空间维度的多尺度特征

    参数说明：
    - num_electrodes: 电极数量（空间维度）
    - num_datapoints: 数据点数量（时间维度）
    - num_classes: 分类类别数
    - inception_window: 多尺度时间窗口比例列表，默认为[0.5, 0.25, 0.125]
    - num_T: 时间卷积核数量，控制时间特征提取能力
    - num_S: 空间卷积核数量，控制空间特征提取能力
    - hidden: 全连接层隐藏单元数
    - dropout_rate: Dropout比率，防止过拟合
    """

    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        """
        构建卷积块：卷积层 + 激活函数 + 池化层

        参数说明：
        - in_chan: 输入通道数
        - out_chan: 输出通道数
        - kernel: 卷积核大小 (height, width)
        - step: 卷积步长
        - pool: 池化窗口大小

        返回：顺序连接的卷积模块
        """
        return nn.Sequential(
            # 二维卷积层
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step),  # 卷积核大小和步长
            nn.LeakyReLU(),  # LeakyReLU激活函数，解决梯度消失问题
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))  # 平均池化层，只在时间维度池化

    def __init__(self, num_electrodes, num_datapoints, num_classes, inception_window=None, num_T=15, num_S=15, hidden=32, dropout_rate=0.5):
        # input_size: 1 x EEG channel x datapoint
        # 输入大小: 1 x EEG通道数 x 数据点数
        super(TSception, self).__init__()  # 调用父类构造函数

        # 根据数据集调整电极数量（移除边缘电极）
        if num_electrodes == 62:
            # SEED数据集：从62个电极中选取54个核心电极
            num_electrodes = 54
        elif num_electrodes == 32:
            # DEAP数据集：从32个电极中选取28个核心电极
            num_electrodes = 28

        # 设置多尺度时间窗口比例
        if inception_window is not None:
            self.inception_window = inception_window  # 使用传入的时间窗口配置
        else:
            self.inception_window = [0.5, 0.25, 0.125]  # 默认时间窗口：50%、25%、12.5%的数据点长度

        self.pool = 8  # 池化系数，控制特征图下采样比例

        # 通过设置卷积核为(1, length)和步长为1，可以使用conv2d实现1d卷积操作

        # 构建多尺度时间卷积分支（Tception）
        # 三个不同尺度的时间卷积，捕获不同时间范围的特征
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * num_datapoints)), 1, self.pool)
        self.Tception2 = self.conv_block(1, num_T, (1, int(self.inception_window[1] * num_datapoints)), 1, self.pool)
        self.Tception3 = self.conv_block(1, num_T, (1, int(self.inception_window[2] * num_datapoints)), 1, self.pool)

        # 构建多尺度空间卷积分支（Sception）
        # 两个不同尺度的空间卷积，捕获不同空间范围的特征
        self.Sception1 = self.conv_block(num_T, num_S, (int(num_electrodes), 1), 1, int(self.pool*0.25))
        self.Sception2 = self.conv_block(num_T, num_S, (int(num_electrodes * 0.5), 1), (int(num_electrodes * 0.5), 1),
                                         int(self.pool*0.25))

        # 特征融合层
        self.fusion_layer = self.conv_block(num_S, num_S, (3, 1), 1, 4)

        # 批归一化层，加速训练并提高稳定性
        self.BN_t = nn.BatchNorm2d(num_T)  # 时间特征批归一化
        self.BN_s = nn.BatchNorm2d(num_S)  # 空间特征批归一化
        self.BN_fusion = nn.BatchNorm2d(num_S)  # 融合特征批归一化

        # 全连接分类层
        self.fc = nn.Sequential(
            nn.Linear(num_S, hidden),  # 第一个全连接层
            nn.ReLU(),  # ReLU激活函数
            nn.Dropout(dropout_rate),  # Dropout层，防止过拟合
            nn.Linear(hidden, num_classes)  # 输出层，映射到类别数
        )

    def forward(self, x):
        """
        前向传播过程

        参数说明：
        - x: 输入张量，形状为 (batch_size, num_electrodes, num_datapoints)

        返回：
        - 分类结果，形状为 (batch_size, num_classes)
        """
        # 增加通道维度：从 (batch, electrodes, datapoints) 到 (batch, 1, electrodes, datapoints)
        x = x.unsqueeze(1)

        # 多尺度时间特征提取
        y = self.Tception1(x)  # 使用最大时间窗口提取特征
        out = y  # 初始化输出
        y = self.Tception2(x)  # 使用中等时间窗口提取特征
        out = torch.cat((out, y), dim=-1)  # 在时间维度拼接特征
        y = self.Tception3(x)  # 使用最小时间窗口提取特征
        out = torch.cat((out, y), dim=-1)  # 在时间维度拼接所有时间特征
        out = self.BN_t(out)  # 时间特征批归一化

        # 多尺度空间特征提取
        z = self.Sception1(out)  # 全空间范围卷积
        out_ = z  # 初始化空间输出
        z = self.Sception2(out)  # 半空间范围卷积
        out_ = torch.cat((out_, z), dim=2)  # 在空间维度拼接特征
        out = self.BN_s(out_)  # 空间特征批归一化

        # 特征融合和最终处理
        out = self.fusion_layer(out)  # 融合多尺度特征
        out = self.BN_fusion(out)  # 融合特征批归一化
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)  # 全局平均池化并压缩维度
        out = self.fc(out)  # 全连接分类层

        return out


def generate_TS_channel_order(original_order: list):
    """
        This function will generate the channel order for TSception
        Parameters
        ----------
        original_order: list of the channel names

        Returns
        -------
        TS: list of channel names which is for TSception
        """
    """
    核心方法介绍：生成TSception模型专用的通道重排序索引
    根据电极名称将电极按奇偶分组，优化空间卷积效果

    参数说明：
    - original_order: 原始通道名称列表

    返回：
    - TS: 为TSception模型优化的通道顺序索引数组
    """
    chan_name, chan_num, chan_final = [], [], []  # 初始化存储列表
    # 解析每个通道名称，分离字母部分和数字部分
    for channel in original_order:
        chan_name_len = len(channel)  # 通道名称总长度
        k = 0  # 数字字符计数器
        # 统计通道名称中的数字字符数量
        for s in [*channel[:]]:
            if s.isdigit():  # 如果是数字字符
                k += 1
        # 分离通道名称的字母部分和数字部分
        if k != 0:  # 如果包含数字
            chan_name.append(channel[:chan_name_len - k])  # 字母部分（如'Fp', 'C'等）
            chan_num.append(int(channel[chan_name_len - k:]))  # 数字部分（如1, 2, 3等）
            chan_final.append(channel)  # 完整通道名称

    chan_pair = []  # 存储配对通道
    # 为每个通道找到对应的奇偶配对通道
    for ch, id in enumerate(chan_num):
        if id % 2 == 0:  # 如果是偶数编号
            chan_pair.append(chan_name[ch] + str(id - 1))  # 配对的奇数通道
        else:  # 如果是奇数编号
            chan_pair.append(chan_name[ch] + str(id + 1))  # 配对的偶数通道

    chan_no_duplicate = []  # 存储去重后的通道对
    # 构建无重复的通道对列表
    [chan_no_duplicate.extend([f, chan_pair[i]]) for i, f in enumerate(chan_final) if f not in chan_no_duplicate]

    # 重新排列通道顺序：所有奇数编号在前，偶数编号在后
    chans = chan_no_duplicate[0::2] + chan_no_duplicate[1::2]  # 奇数索引 + 偶数索引

    # 生成新顺序在原始顺序中的索引位置
    indexes = [original_order.index(c) for c in chans]
    return np.array(indexes)  # 返回索引数组