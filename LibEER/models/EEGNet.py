import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
import yaml

from LibEER.utils.store import save_state
from LibEER.utils.metric import Metric

from LibEER.data_utils.preprocess import normalize

# EEGNet模型参数配置文件路径
param_path = 'config/model_param/EEGNet.yaml'

class EEGNet(nn.Module):
    """EEGNet模型 - 用于脑电信号分类的轻量级卷积神经网络"""
    def __init__(self, num_electrodes=62, datapoints=128, num_classes=3, F1=8, D=2, dropout=0.5):
        """
        初始化EEGNet模型

        Args:
            num_electrodes: EEG电极数量，默认62个
            datapoints: 每个电极的数据点数，默认128
            num_classes: 分类类别数，默认3类
            F1: 第一个卷积层的滤波器数量，默认8
            D: 深度乘数，用于控制深度卷积的通道扩展，默认2
            dropout: Dropout比率，默认0.5
        """
        super().__init__()
        # 初始化模型参数
        self.F1 = F1
        self.D = D
        self.dropout = dropout

        # 第一层：时间维度卷积 - 提取时间特征
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(1, datapoints//2), padding='same', bias=False)
        # 批归一化
        self.BN1 = nn.BatchNorm2d(self.F1)

        # self.depth_conv = nn.Conv2d(in_channels=self.F1, out_channels=self.F1 * self.D, kernel_size=(num_electrodes, 1), bias=False,
        #                             groups=self.F1)
        # 第二层：深度可分离卷积 - 空间滤波（电极维度）
        # 使用带约束的卷积层，限制权重范围防止过拟合
        self.depth_conv = Conv2dWithConstraint(in_channels=self.F1, out_channels=self.F1 * self.D, kernel_size=(num_electrodes, 1), bias=False,
                                    groups=self.F1)
        self.BN2 = nn.BatchNorm2d(self.D * self.F1) # 批归一化
        self.act1 = nn.ELU(inplace=True)    # 激活函数

        # 第一层池化和dropout
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=4) # 平均池化
        self.dropout1 = nn.Dropout(dropout) # Dropout正则化

        # 可分离卷积模块：深度卷积 + 逐点卷积
        self.sep_conv = nn.ModuleList()
        # 深度卷积：每个通道独立卷积
        self.sep_conv.append(
            nn.Conv2d(in_channels=self.D * self.F1, out_channels=self.D * self.F1, kernel_size=(1, 16), padding='same', bias=False,
                      groups=self.D * self.F1))
        F2 = self.D * self.F1   # 第二层滤波器数量

        # 逐点卷积：1x1卷积，融合通道信息
        self.sep_conv.append(nn.Conv2d(in_channels=self.D * self.F1, out_channels=F2, kernel_size=1, bias=False))
        self.BN3 = nn.BatchNorm2d(F2)   # 批归一化
        self.act2 = nn.ELU(inplace=True)    # 激活函数

        # 第二层池化和dropout
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=8) # 平均池化
        self.dropout2 = nn.Dropout(dropout) # Dropout正则化

        # 全连接层：输出分类结果
        # 计算展平后的特征维度：F2 * (datapoints // 32)
        self.fc = nn.Linear(F2 * (datapoints // 32), num_classes)

    def get_param(self):
        """获取模型参数（待实现）"""
        return

    def init_weight(self):
        """初始化模型权重"""
        # 使用Kaiming正态分布初始化卷积层权重（适合ReLU/ELU激活函数）
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.depth_conv.weight)
        nn.init.kaiming_normal_(self.sep_conv[0].weight)
        nn.init.kaiming_normal_(self.sep_conv[1].weight)

        # 它的核心目标是：在训练开始时，防止网络层中的输入信号梯度在向前或向后传播过程中过快地消失或爆炸
        # 使用Xavier正态分布初始化全连接层权重
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)    # 偏置初始化为0


    def weight_constraint(self, parameters, min_value, max_value):
        """
        权重约束函数：将权重限制在指定范围内

        Args:
            parameters: 需要约束的权重参数
            min_value: 最小值
            max_value: 最大值
        """
        for param in parameters:
            param.data.clamp_(min_value, max_value)

    def forward(self, x):
        """
        前向传播过程
        Args:
            x: 输入张量，形状为(batch_size, channels, datapoints)

        Returns:
            分类结果，形状为(batch_size, num_classes)
        """
        # 重塑输入维度：增加通道维度 -> (batch_size, 1, channels, datapoints)
        # x shape -> (batch_size, channels, datapoints)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        # 第一卷积块：时间特征提取
        x = self.conv1(x)           # 时间卷积
        x = self.BN1(x)             # 批归一化

        # 第二卷积块：空间特征提取
        x = self.depth_conv(x)      # 深度卷积
        x = self.BN2(x)             # 批归一化
        x = self.act1(x)            # 激活函数
        x = self.pool1(x)           # 池化
        x = self.dropout1(x)        # Dropout

        # 可分离卷积块：进一步特征提取
        x = self.sep_conv[0](x)     # 深度卷积
        x = self.sep_conv[1](x)     # 逐点卷积
        x = self.BN3(x)             # 批归一化
        x = self.act2(x)            # 激活函数
        x = self.pool2(x)           # 池化
        x = self.dropout2(x)        # Dropout

        # 分类层
        x = torch.flatten(x, 1)     # 展平特征
        x = self.fc(x)              # 全连接分类
        return x

class Conv2dWithConstraint(nn.Module):
    """带约束的二维卷积层：限制输出值的最大范围"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, max_value=1.0, bias=False, groups=1):
        """
        初始化带约束的卷积层
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            max_value: 输出值的最大限制
            bias: 是否使用偏置
            groups: 分组卷积的组数
        """
        super(Conv2dWithConstraint, self).__init__()
        # 创建标准的二维卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.max_value = max_value  # 输出值的上限

    def forward(self, x):
        """前向传播：执行卷积并限制输出范围"""
        output = self.conv(x)   # 标准卷积操作
        output = torch.clamp(output, max=self.max_value)    # 限制最大值防止梯度爆炸
        return output