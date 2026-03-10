# 导入必要的PyTorch库
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import torch.optim as optim

# 导入其他工具库
from tqdm import tqdm  # 进度条显示
import yaml  # YAML配置文件解析

# 导入自定义工具模块
from LibEER.utils.store import save_state  # 模型状态保存
from LibEER.utils.metric import Metric  # 评估指标计算

# 定义模型参数配置文件路径
param_path = 'config/model_param/STRNN.yaml'


class STRNN(nn.Module):
    """
    核心方法介绍：STRNN (Spatio-Temporal Recurrent Neural Network)
    时空递归神经网络，结合空间和时间维度的RNN处理
    适用于脑电信号等时空序列数据的分类任务

    参数说明：
    - sample_length: 时间序列长度，表示每个样本包含的时间步数
    - num_electrodes: 电极数量，表示空间维度的大小
    - in_channels: 输入通道数，每个电极点的特征维度
    - num_classes: 分类类别数，输出层的维度
    - sp_hidden: 空间RNN的隐藏层大小
    - tp_hidden: 时间RNN的隐藏层大小
    - sp_projection: 空间投影维度，空间特征降维后的尺寸
    - tp_projection: 时间投影维度，时间特征降维后的尺寸
    """
    def __init__(self, sample_length=9, num_electrodes=62, in_channels=5, num_classes=3, sp_hidden=30, tp_hidden=30
                 , sp_projection=10, tp_projection=5):
        # 调用父类构造函数初始化神经网络模块
        super(STRNN, self).__init__()
        # 初始化模型参数
        self.num_electrodes = num_electrodes    # 电极数量（空间节点数）
        self.in_channels = in_channels          # 输入通道数（每个节点的特征维度）
        self.num_classes = num_classes          # 分类类别数（输出维度）
        self.sp_hidden = sp_hidden              # 空间RNN隐藏层维度
        self.tp_hidden = tp_hidden              # 时间RNN隐藏层维度
        self.sp_projection = sp_projection      # 空间特征投影维度
        self.tp_projection = tp_projection      # 时间特征投影维度
        self.sample_length = sample_length      # 时间序列长度（时间步数）
        self.get_param()                        # 获取额外参数配置
        self.directions = []                    # 空间传播方向列表
        self.pos = []                           # 电极位置坐标列表

        # 如果是62通道配置，使用预定义的62通道方向和位置
        if self.num_electrodes == 62:
            self.directions = Sixtytwo_channel_directions  # 62通道的4种空间传播方向
            self.pos = Sixtytwo_channel_coor  # 62通道的二维坐标位置

        # 初始化空间RNN模块，处理电极间的空间关系
        self.sp_rnn = SRNN(num_electrodes=self.num_electrodes, in_channels=self.in_channels,
                           num_hidden=self.sp_hidden,
                           num_projection=self.sp_projection, directions=self.directions, pos=self.pos)

        # 初始化前向时间RNN模块，处理正向时间序列
        self.f_tp_rnn = TRNN(sample_length=self.sample_length, num_hidden=self.tp_hidden,
                             num_projection=self.tp_projection)

        # 初始化反向时间RNN模块，处理反向时间序列
        self.b_tp_rnn = TRNN(sample_length=self.sample_length, num_hidden=self.tp_hidden,
                             num_projection=self.tp_projection)

        # 前向时间RNN的输出层第一部分：将时间隐藏特征映射到标量
        self.fp_1 = nn.Linear(self.tp_hidden, 1)
        # 前向时间RNN的输出层第二部分：将投影特征映射到分类结果
        self.fp_2 = nn.Linear(self.tp_projection, self.num_classes)

        # 反向时间RNN的输出层第一部分
        self.bp_1 = nn.Linear(self.tp_hidden, 1)
        # 反向时间RNN的输出层第二部分
        self.bp_2 = nn.Linear(self.tp_projection, self.num_classes)

    def get_param(self):
        """获取模型参数（当前为空实现，可用于从配置文件加载参数）"""
        return

    def init_weight(self):
        """初始化模型权重，使用Xavier正态分布初始化线性层"""
        # 初始化前向输出层权重
        nn.init.xavier_normal_(self.fp_1.weight)  # Xavier正态分布初始化权重
        nn.init.zeros_(self.fp_1.bias)  # 偏置初始化为0
        nn.init.xavier_normal_(self.fp_2.weight)
        nn.init.zeros_(self.fp_2.bias)
        # 初始化反向输出层权重
        nn.init.xavier_normal_(self.bp_1.weight)
        nn.init.zeros_(self.bp_1.bias)
        nn.init.xavier_normal_(self.bp_2.weight)
        nn.init.zeros_(self.bp_2.bias)

    def forward(self, x):
        """
       前向传播过程
       参数:
           x: 输入张量，形状为 (batch_size, sample_length, num_electrodes, num_features)
               - batch_size: 批次大小
               - sample_length: 时间序列长度
               - num_electrodes: 电极数量
               - num_features: 特征维度
       返回:
           output: 输出分类结果，形状为 (batch_size, num_classes)
       """
        # 调整输入张量维度:
        # shape of x -> (batch_size, sample_length, num_electrodes, num_features)
        # reshape to -> (sample_length, batch_size, num_electrodes, num_features)

        # 这样便于按时间步迭代处理
        x = x.permute(1, 0, 2, 3)
        ms = None  # 初始化空间特征序列容器
        # 遍历每个时间步，处理每个时间点的空间特征
        for idx, x_i in enumerate(x):
            # x_i形状: (batch_size, num_electrodes, num_features)
            if idx == 0:
                # 第一个时间步：初始化空间特征序列
                # self.sp_rnn(x_i)返回形状: (batch_size, sp_hidden)
                # unsqueeze(dim=0)增加时间维度 -> (1, batch_size, sp_hidden)
                ms = self.sp_rnn(x_i).unsqueeze(dim=0)
            else:
                # 后续时间步：将新的空间特征拼接到序列中
                ms = torch.concatenate((ms, self.sp_rnn(x_i).unsqueeze(dim=0)), dim=0)

        # ms shape: (t, batch_size, sp_hidden) - 空间特征时间序列
        # 前向时间RNN处理：从第一个时间步到最后一个时间步
        q_f = self.f_tp_rnn(ms)  # 形状: (batch_size, num_projection, tp_hidden)
        # 反向时间RNN处理：将序列反转，从最后一个时间步到第一个时间步
        q_b = self.b_tp_rnn(torch.flip(ms, dims=[0]))  # 形状: (batch_size, num_projection, tp_hidden)

        # concatenated the forward and backward temporal output
        # 合并前向和反向时间RNN的输出
        # self.fp_1(q_f)形状: (batch_size, num_projection, 1)
        # torch.squeeze(..., dim=-1)形状: (batch_size, num_projection)
        # 最终输出是两个方向输出的加和
        output = self.fp_2(torch.squeeze(self.fp_1(q_f), dim=-1)) + self.bp_2(torch.squeeze(self.bp_1(q_b), dim=-1))
        return output # 形状: (batch_size, num_classes)


class SRNN(nn.Module):
    """
    核心方法介绍：SRNN (Spatial Recurrent Neural Network)
    空间递归神经网络，处理电极间的空间关系
    基于电极的物理位置和预定义方向进行空间信息传播

    参数说明：
    - num_electrodes: 电极数量
    - in_channels: 输入特征维度
    - num_hidden: 隐藏层维度
    - num_projection: 投影输出维度
    - directions: 空间传播方向列表
    - pos: 电极位置坐标列表
    """
    def __init__(self, num_electrodes, in_channels, num_hidden, num_projection, directions, pos):
        super(SRNN, self).__init__()
        # 初始化参数
        self.num_electrodes = num_electrodes  # 电极总数
        self.in_channels = in_channels  # 输入特征维度
        self.num_hidden = num_hidden  # 隐藏状态维度
        self.num_projection = num_projection  # 投影输出维度
        self.pos = pos  # 电极位置坐标列表，每个元素是[x, y]坐标
        self.directions = directions  # 空间传播方向列表，每个方向是一个电极序号列表

        # self.hiddens = nn.ModuleList()
        # # There are hidden layers for each channel in each direction
        # for _ in range(len(directions)):
        #     hidden = nn.ModuleList()
        #     # for _ in range(len(num_electrodes)):
        #     #     hidden.append(nn.Parameter(torch.tensor(self.num_hidden), requires_grad=True))
        #     self.hiddens.append(hidden)

        # 初始化邻接矩阵Ns，表示电极间的连接关系
        self.Ns = nn.Parameter(torch.zeros((len(self.directions), self.num_electrodes, self.num_electrodes)),
                               requires_grad=False) # 不参与梯度更新

        # 根据方向和位置信息构建邻接矩阵
        for di, direction in enumerate(self.directions):  # 遍历每个方向
            for pi, point in enumerate(direction):  # 遍历当前方向中的每个电极点
                # 获取当前点的邻居索引
                for ni in n_set(pi, di, self.pos, direction):
                    self.Ns[di][pi][ni] = 1  # 标记连接关系，1表示连接

        # Process input x using matrix U
        # 初始化输入处理矩阵U（每个方向一个线性层）
        self.sp_Ums = nn.ModuleList()   # 模块列表，存储每个方向的输入变换层
        for _ in directions:
            # 将输入特征映射到隐藏空间: in_channels -> num_hidden
            self.sp_Ums.append(nn.Linear(self.in_channels, self.num_hidden, bias=True))

        # Update the hidden layer with matrix W
        # 初始化隐藏层更新矩阵W（每个方向一个线性层）
        self.sp_Wms = nn.ModuleList()  # 模块列表，存储每个方向的隐藏状态更新层
        for _ in directions:
            # 将邻居隐藏状态聚合映射: num_hidden -> num_hidden
            self.sp_Wms.append(nn.Linear(self.num_hidden, self.num_hidden, bias=True))

        # the bias in SRNN
        # 空间RNN的偏置参数，每个方向一个偏置向量
        self.sp_bs = nn.Parameter(torch.Tensor(len(directions), self.num_hidden), requires_grad=True)
        self.relu = nn.ReLU()      # ReLU激活函数

        # the projection matrx to downsample
        # 空间投影矩阵，用于降维（每个方向一个投影层）
        self.sp_projections = nn.ModuleList()
        for _ in directions:
            # 将电极维度投影到更低维度: num_electrodes -> num_projection
            self.sp_projections.append(nn.Linear(self.num_electrodes, self.num_projection))

        # gather each direction feature
        # 聚合各方向特征的线性层
        # 输入: num_projection * num_directions
        # 输出: 1（通过后续操作得到最终特征）
        self.sp_ps = nn.Linear(self.num_projection * len(directions), 1)
        self.init_weight()      # 初始化权重

    def init_weight(self):
        """初始化SRNN模块的所有权重参数"""
        for i in range(len(self.directions)):
            # 初始化输入处理层权重
            nn.init.xavier_normal_(self.sp_Ums[i].weight)  # Xavier正态分布初始化
            nn.init.zeros_(self.sp_Ums[i].bias)  # 偏置初始化为0
            # 初始化隐藏层更新权重
            nn.init.xavier_normal_(self.sp_Wms[i].weight)
            nn.init.zeros_(self.sp_Wms[i].bias)
            # 初始化偏置
            nn.init.zeros_(self.sp_bs)  # 偏置初始化为0
            # 初始化投影层权重
            nn.init.xavier_normal_(self.sp_projections[i].weight)
            nn.init.zeros_(self.sp_projections[i].bias)
        # 初始化特征聚合层权重
        nn.init.xavier_normal_(self.sp_ps.weight)
        nn.init.zeros_(self.sp_ps.bias)

    def forward(self, x):
        """
        空间RNN前向传播
        参数:
            x: 输入张量，形状为 (batch_size, num_ele, num_feature)
                - batch_size: 批次大小
                - num_ele: 电极数量
                - num_feature: 特征维度
        返回:
            m: 空间特征聚合结果，形状为 (batch_size, num_hidden)
        """
        # x shape->(batch size, num_ele, num_feature)
        s = None    # 初始化各方向特征聚合容器
        # # hiddens shape -> (num_directions, num_ele, batch_size, num_hidden)
        # hiddens = torch.zeros((len(self.directions), x.shape[1], x.shape[0], self.num_hidden), device=x.device)

        # 遍历每个空间方向
        for ri, direction in enumerate(self.directions):
            # 初始化当前方向的隐藏状态
            # 形状: (batch_size, 1, num_hidden)
            # update all the hidden layers in one direction
            hidden_di = torch.zeros((x.shape[0], 1, self.num_hidden), device=x.device)
            # 遍历当前方向上的每个电极点（按预定顺序）
            for pi, point in enumerate(direction):
                # Iterate through the hidden layer in a specific direction

                # find the neighbors near
                # neighbor_indexes = n_set(pi, ri, self.pos, direction)
                # hidden layer shape of one electrode -> (batch, num_hidden)
                # print(self.Ns[ri][pi][0:pi+1].unsqueeze(0).repeat(x.shape[0], 1, pi+1).shape)

                # 计算邻居隐藏状态的聚合
                if pi == 0:
                    # 第一个点没有前驱邻居，聚合项为0
                    h_aggregation = torch.zeros((x.shape[0], self.num_hidden), device=x.device)
                else:
                    # 聚合前驱邻居的隐藏状态
                    # self.Ns[ri][pi][0:pi] 形状: (pi,) - 当前点与前pi个点的连接关系
                    # unsqueeze(0).repeat(x.shape[0], 1, 1) 形状: (batch_size, 1, pi)
                    # hidden_di 形状: (batch_size, pi, num_hidden)
                    # matmul结果形状: (batch_size, 1, num_hidden)
                    # squeeze(1) 形状: (batch_size, num_hidden)
                    h_aggregation = torch.matmul(self.Ns[ri][pi][0:pi].unsqueeze(0).repeat(x.shape[0], 1, 1),
                                                 hidden_di).squeeze(1)
                # Adjust the hidden layer based on the input

                # 更新当前点的隐藏状态
                # 输入变换 + 邻居聚合 + 偏置
                hidden_pi = self.sp_Ums[ri](x[:, point - 1]) + self.sp_Wms[ri](h_aggregation) + self.sp_bs[ri]
                hidden_pi = self.relu(hidden_pi)     # 应用ReLU激活函数
                # # hiddens[ri][pi] += self.sp_Ums[ri](x[:, point - 1])
                # for ni in neighbor_indexes:
                #     # Adjust hidden layers based on past ones
                #     hidden_pi += self.sp_Wms[ri](hidden_di[ni])
                # the bias add on the hidden layers
                # hidden_pi += self.sp_bs[ri]

                # 将当前点隐藏状态添加到序列中
                if pi == 0:
                    # 第一个点，初始化隐藏状态序列
                    hidden_di = hidden_pi.unsqueeze(dim=1)      # 形状: (batch_size, 1, num_hidden)
                else:
                    # 后续点，拼接隐藏状态
                    hidden_di = torch.concatenate((hidden_di, hidden_pi.unsqueeze(dim=1)), dim=1)       # 形状: (batch_size, pi+1, num_hidden)
            # hidden_di shape -> (batch_size, num_ele, num_hidden)
            # reshape hidden_di to -> (batch_size, num_hidden, num_ele)
            # project the all electrodes to the output feature vector
            # 对当前方向的隐藏状态进行投影和聚合
            # hidden_di形状: (batch_size, num_ele, num_hidden)
            # permute(0, 2, 1) -> (batch_size, num_hidden, num_ele)
            # 投影到: (batch_size, num_hidden, num_projection)
            if ri == 0:
                # 第一个方向，初始化特征容器
                s = self.sp_projections[ri](hidden_di.permute(0, 2, 1))
            else:
                # 后续方向，拼接特征
                # 形状: (batch_size, num_hidden, num_projection * num_directions)
                s = torch.concatenate((s, self.sp_projections[ri](hidden_di.permute(0, 2, 1))), dim=-1)
        # s shape -> (batch_size, num_hidden, num_pro * len(direction))
        # s 形状: (batch_size, num_hidden, num_pro * len(direction))
        # 最终特征聚合: 将多方向特征聚合为单个特征向量
        # self.sp_ps(s)形状: (batch_size, num_hidden, 1)
        # squeeze(dim=2)形状: (batch_size, num_hidden)
        m = torch.squeeze(self.sp_ps(s), dim=2)
        # m shape -> (batch_size, num_feature)  - 最终的空间特征表示
        return m


class TRNN(nn.Module):
    """
    核心方法介绍：TRNN (Temporal Recurrent Neural Network)
    时间递归神经网络，处理时间序列的时序依赖关系

    参数说明：
    - sample_length: 时间序列长度
    - num_hidden: 隐藏层维度
    - num_projection: 投影输出维度
    """
    def __init__(self, sample_length=9, num_hidden=30, num_projection=5):
        super(TRNN, self).__init__()
        # 初始化参数

        self.sample_length = sample_length  # 时间序列长度
        self.num_hidden = num_hidden  # 隐藏状态维度
        self.num_projection = num_projection  # 时间投影维度

        self.hidden = nn.ModuleList()  # 隐藏层列表（当前未使用）
        self.tp_rs = nn.Linear(self.num_hidden, self.num_hidden)  # 当前输入变换矩阵
        self.tp_vs = nn.Linear(self.num_hidden, self.num_hidden)  # 前一时刻隐藏状态变换矩阵
        self.tp_bs = nn.Parameter(torch.Tensor(self.num_hidden), requires_grad=True)  # 偏置参数
        self.activation = nn.ReLU()  # 激活函数
        self.tp_projection = nn.Linear(self.sample_length, self.num_projection)  # 时间投影层
        self.init_weight()  # 初始化权重

    def init_weight(self):
        """初始化TRNN模块的所有权重参数"""
        nn.init.xavier_normal_(self.tp_rs.weight)  # 输入变换权重初始化
        nn.init.zeros_(self.tp_rs.bias)  # 输入变换偏置初始化
        nn.init.xavier_normal_(self.tp_vs.weight)  # 隐藏状态变换权重初始化
        nn.init.zeros_(self.tp_vs.bias)  # 隐藏状态变换偏置初始化
        nn.init.zeros_(self.tp_bs)  # 偏置参数初始化为0
        nn.init.xavier_normal_(self.tp_projection.weight)  # 投影层权重初始化
        nn.init.zeros_(self.tp_projection.bias)  # 投影层偏置初始化

    def forward(self, ms):
        """
        时间RNN前向传播
        参数:
            ms: 输入序列，形状为 (t, batch_size, num_features)
                - t: 时间步数
                - batch_size: 批次大小
                - num_features: 特征维度
        返回:
            q: 时间特征投影结果，形状为 (batch_size, num_projection, num_features)
        """
        # shape of ms -> (t, batch_size, num_features)
        # zero initial hidden
        # 初始化隐藏状态（全零），形状: (1, batch_size, num_hidden)
        hiddens = torch.zeros((1, ms.shape[1], self.num_hidden), device=ms.device)
        # 遍历时间步，从1开始计数
        for t, m_t in enumerate(ms, 1):
            # 更新隐藏状态: r(当前输入) + v(前一隐藏状态) + b(偏置)
            # self.tp_rs(m_t): 当前输入变换，形状: (batch_size, num_hidden)
            # self.tp_vs(hiddens[t-1]): 前一隐藏状态变换，形状: (batch_size, num_hidden)
            # 相加后应用ReLU激活
            hidden_t = self.activation(self.tp_rs(m_t) + self.tp_vs(hiddens[t - 1]) + self.tp_bs)
            # 将新隐藏状态添加到序列中
            hiddens = torch.concatenate((hiddens, hidden_t.unsqueeze(dim=0)), dim=0)
            # hiddens形状变为: (t+1, batch_size, num_hidden)

        # shape of hiddens -> (t, batch_size, num_feature)
        # reshape hiddens to (batch_size, num_feature, t)
        # hiddens 形状: (t+1, batch_size, num_hidden)

        # 去掉初始的零隐藏状态，从第1个时间步开始
        # hiddens[1:]形状: (t, batch_size, num_hidden)
        # permute(1, 2, 0) -> (batch_size, num_hidden, t)
        # 时间投影: (batch_size, num_hidden, t) -> (batch_size, num_hidden, num_projection)
        # 再次permute(0, 2, 1) -> (batch_size, num_projection, num_hidden)
        q = self.tp_projection(hiddens[1:].permute(1, 2, 0)).permute(0, 2, 1)
        # shape of q -> (batch_size, num_projection, num_feature)
        return q


# 62通道的4种空间传播方向定义
# 每个方向是一个电极编号列表，表示在该方向上的传播顺序
Sixtytwo_channel_directions = [
    # 方向1: 顺序传播，按电极编号从小到大
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
     32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 58, 52, 53, 54, 55, 56, 62, 60, 61, 51,
     57, 59],
    # 方向2: 逆序传播，从中心向四周扩散
    [59, 57, 51, 61, 60, 62, 56, 55, 54, 53, 52, 58, 50, 41, 32, 23, 14, 49, 40, 31, 22, 13, 48, 39, 30, 21, 12, 47, 38,
     29, 20, 11, 46, 37, 28, 19, 10, 45, 36, 27, 18, 9, 44, 35, 26, 17, 8, 43, 34, 25, 16, 7, 42, 33, 24, 15, 6, 5, 4, 3
        , 2, 1],
    # 方向3: 完全逆序传播，按电极编号从大到小
    [59, 57, 51, 61, 60, 62, 56, 55, 54, 53, 52, 58, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34,
     33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3
        , 2, 1],
    # 方向4: 按行传播，模拟在电极网格上的行优先遍历
    [1, 2, 3, 4, 5, 6, 15, 24, 33, 42, 7, 16, 25, 34, 43, 8, 17, 26, 35, 44, 9, 18, 27, 36, 45, 10, 19, 28, 37, 46, 11,
     20, 29, 38, 47, 12, 21, 30, 39, 48, 13, 22, 31, 40, 49, 14, 23, 32, 41, 50, 58, 52, 53, 54, 55, 56, 62, 60, 61, 51,
     57, 59]]


# Gets the adjacent hidden layer based on the direction
def n_set(pi, ri, pos, direction):
    """
    核心方法介绍：获取指定方向和位置的邻居电极索引
    基于电极的二维坐标位置计算空间邻居关系

    参数:
        pi: 当前点在方向序列中的索引
        ri: 方向索引（0-3对应4种邻居定义）
        pos: 所有电极的位置坐标列表
        direction: 当前方向序列

    返回:
        available_neighbor_indexes: 可用的邻居在方向序列中的索引列表
    """
    point = direction[pi]  # 当前电极点编号
    i = pos[point - 1][0]  # 当前点的行坐标（电极布局中的行）
    j = pos[point - 1][1]  # 当前点的列坐标（电极布局中的列）

    # 定义4种方向的邻居坐标偏移
    # 每个方向对应3个邻居位置偏移
    neighbors = {
        # 'top-left': [[i, j-1], [i-1, j-1], [i-1, j]],
        # 'top-right': [[i, j+1], [i-1, j], [i-1, j+1]],
        # 'bottom-left': [[i, j-1], [i+1, j-1], [i+1, j]],
        # 'bottom-right': [[i, j-1], [i-1, j-1], [i-1, j]]
        0: [[i, j - 1], [i - 1, j - 1], [i - 1, j]],  # 左上方向邻居
        1: [[i, j + 1], [i - 1, j], [i - 1, j + 1]],  # 右上方向邻居
        2: [[i, j - 1], [i + 1, j - 1], [i + 1, j]],  # 左下方向邻居
        3: [[i, j - 1], [i - 1, j - 1], [i - 1, j]]  # 另一种左上方向
    }

    # 获取当前方向的邻居坐标定义
    neighbor_set = neighbors[ri]
    neighbor_indexes = []  # 存储存在的邻居电极全局编号

    # 查找实际存在的邻居电极
    for neighbor in neighbor_set:
        if neighbor in pos: # 检查该坐标位置是否有电极
            # location of electrodes
            # 计算邻居电极的全局索引（位置索引+1）
            neighbor_indexes.append(pos.index(neighbor)+1)

    # 筛选在当前方向序列中且在当前点之前的邻居
    available_neighbor_indexes = []
    for n_idx in neighbor_indexes:
        if n_idx in direction[0:pi]:    # 只考虑当前点之前的邻居（在传播顺序上先被处理）
            # Returns its index in the specified direction
            # 返回邻居在方向序列中的局部索引
            available_neighbor_indexes.append(direction.index(n_idx))

    return available_neighbor_indexes


# 62个电极的二维坐标定义（基于10x9的网格布局）
# Two-dimensional coordinates for channels 1 through 62
# https://bcmi.sjtu.edu.cn/home/seed/img/seed-FRA/montage.png
# 每个电极用[row, col]坐标表示在电极帽上的物理位置
Sixtytwo_channel_coor = [[1, 4], [1, 5], [1, 6], [2, 4], [2, 6], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7],
                     [3, 8], [3, 9], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [5, 1],
                     [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [6, 1], [6, 2], [6, 3], [6, 4],
                     [6, 5], [6, 6], [6, 7], [6, 8], [6, 9], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7],
                     [7, 8], [7, 9], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [9, 3], [9, 4], [9, 5],
                     [9, 6], [9, 7]]
