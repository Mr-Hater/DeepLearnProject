import numpy as np
from sklearn.preprocessing import StandardScaler
from LibEER.utils.store import save_data
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, train_test_split
import random

# def train_test_split(data, label, setting):
#     """
#     Provides division of training set and test set under various experimental settings No matter how the experimental
#     settings are, they are all based on trail division, so trail is a basic division unit. For the three typical
#     experimental settings on a dataset, subject-dependent, subject-independent, cross-session,they can be operated
#     based on each subject’s trail, each subject, each session
#           input : all the eeg data and label which can directly be taken as an input
#           output : data and label that make up the training set or test set
#           input shape -> data :   (session, subject, trail, sample, sample_length, time_window, channel, band_feature)
#                          label :  (session, subject, trail, sample, label)
#           output shape -> data :  (sample, sample_length, time_window, channel, band_feature)
#                           label : (sample, label)
#     """
#     train_data = []
#     train_label = []
#     test_data = []
#     test_label = []
#     if setting.experiment_mode == "subject-dependent":
#         # reshape to (sample, sample_length, time_window, channel, band_feature)
#         train_data = [sample for session in data for subject in session for i in setting.train_part for sample in
#                       subject[i - 1]]
#         train_label = [sample for session in label for subject in session for i in setting.train_part for sample in
#                        subject[i - 1]]
#
#         test_part = list(set(range(1, len(data[0][0]) + 1)) - set(setting.train_part))
#
#         test_data = [sample for session in data for subject in session for i in test_part for sample in
#                      subject[i - 1]]
#         test_label = [sample for session in label for subject in session for i in test_part for sample in
#                       subject[i - 1]]
#
#     elif setting.experiment_mode == "subject-independent":
#
#         # reshape to (sample, sample_length, time_window, channel, band_feature)
#         train_data = [sample for session in data for i in setting.train_part for trail in session[i-1]
#                       for sample in trail]
#         train_label = [sample for session in label for i in setting.train_part for trail in session[i-1]
#                        for sample in trail]
#
#         test_part = list(set(range(1, len(data[0]) + 1)) - set(setting.train_part))
#
#         test_data = [sample for session in data for i in test_part for trail in session[i-1] for sample in trail]
#         test_label = [sample for session in label for i in test_part for trail in session[i-1] for sample in trail]
#
#     elif setting.experiment_mode == "cross-session":
#
#         # reshape to (sample, sample_length, time_window, channel, band_feature)
#         train_data = [sample for i in setting.train_part for session in data[i-1] for subject in session for trail in
#                       subject for sample in trail]
#         train_label = [sample for i in setting.train_part for session in label[i-1] for subject in session for trail in
#                        subject for sample in trail]
#
#         test_part = list(set(range(1, len(data) + 1)) - set(setting.train_part))
#
#         test_data = [sample for i in test_part for subject in data[i-1] for trail in subject for sample in trail]
#         test_label = [sample for i in test_part for subject in label[i-1] for trail in subject for sample in trail]
#
#     train_data = np.asarray(train_data)
#     train_label = np.asarray(train_label)
#     test_data = np.asarray(test_data)
#     test_label = np.asarray(test_label)
#     if setting.normalize:
#         for i in range(len(train_data[0][0])):
#             scaler = StandardScaler()
#             train_data[:, :, i] = scaler.fit_transform(train_data[:, :, i])
#             test_data[:, :, i] = scaler.transform(test_data[:, :, i])
#     # if setting.save_data:
#     #     save_data(train_data, train_label, test_data, test_label)
#     return train_data, train_label, test_data, test_label

# 原train_test_split函数已注释，提供了更灵活的index-based分割方式

def index_to_data(data, label, train_indexes, test_indexes, val_indexes, keep_dim=False):
    """
    根据索引将原始数据分割为训练集、验证集和测试集

    参数:
        data: 原始数据数组
        label: 原始标签数组
        train_indexes: 训练集索引列表
        test_indexes: 测试集索引列表
        val_indexes: 验证集索引列表，[-1]表示无验证集
        keep_dim: 是否保持原始维度（True：保持分组结构，False：展平为一维数组）

    返回:
        (train_data, train_label, val_data, val_label, test_data, test_label)
    """
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    test_data = []
    test_label = []

    if keep_dim:
        # 保持维度模式：保留原始的分组结构
        for train_index in train_indexes:
            train_data.append(data[train_index])
            train_label.append(label[train_index])
        for test_index in test_indexes:
            test_data.append(data[test_index])
            test_label.append(label[test_index])
        if val_indexes[0] != -1:  # 有验证集
            for val_index in val_indexes:
                val_data.append(data[val_index])
                val_label.append(label[val_index])
    else:
        # 展平模式：将所有样本展平为一维数组
        for train_index in train_indexes:
            train_data.extend(data[train_index])  # 扩展，而不是追加
            train_label.extend(label[train_index])
        for test_index in test_indexes:
            test_data.extend(data[test_index])
            test_label.extend(label[test_index])
        if val_indexes[0] != -1:  # 有验证集
            for val_index in val_indexes:
                val_data.extend(data[val_index])
                val_label.extend(label[val_index])

        # 转换为numpy数组
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        train_label = np.array(train_label)
        test_label = np.array(test_label)
        val_data = np.array(val_data)
        val_label = np.array(val_label)

    return train_data, train_label, val_data, val_label, test_data, test_label


def get_split_index(data, label, setting=None):
    """
    根据实验设置获取数据分割的索引

    参数:
        data: 原始数据
        label: 原始标签
        setting: 实验设置对象

    返回:
        tts: 字典，包含'train'、'test'、'val'的索引列表
    """
    tts = {}  # train-test-split字典

    # ==================== K折交叉验证 ====================
    if setting.split_type == "kfold":
        # 创建K折交叉验证对象
        kf = KFold(
            setting.fold_num,
            shuffle=True if setting.fold_shuffle == 'true' or setting.fold_shuffle == 'True' else False,
            random_state=setting.seed if setting.fold_shuffle == 'true' else None
        )
        # 获取每折的训练和测试索引
        tts['train'] = [list(train_index) for train_index, _ in kf.split(label)]
        tts['test'] = [list(test_index) for _, test_index in kf.split(label)]

    # ==================== 留一法交叉验证 ====================
    elif setting.split_type == "leave-one-out":
        loo = LeaveOneOut()  # 创建留一法对象
        tts['train'] = [list(train_index) for train_index, _ in loo.split(label)]
        tts['test'] = [list(test_index) for _, test_index in loo.split(label)]

    # ==================== 前部-后部分割 ====================
    elif setting.split_type == "front-back":
        # 验证前部大小是否合理
        if setting.front >= len(label):
            print(f"使用前部-后部分割类型和 {setting.experiment_mode} 实验模式")
            print(f"前部大小 {setting.front} > 分割部分数 {len(label)}")
            print("请检查您的实验模式或分割类型")
            exit(1)

        # 前部作为训练集，后部作为测试集
        tts['train'] = [[i for i in range(setting.front)]]
        tts['test'] = [[setting.front + i for i in range(len(label) - setting.front)]]

    # ==================== 早停分割（训练-验证-测试） ====================
    elif setting.split_type == "early-stop":
        if setting.experiment_mode == "subject-dependent":
            # 被试依赖模式：需要平衡分割，保持标签分布
            tts['test'] = [[]]
            tts['train'] = [[]]
            tts['val'] = [[]]

            # ==================== 按标签分组 ====================
            # 将相同标签的样本索引分组，确保每个标签在训练/验证/测试集中都有代表性
            groups = {}
            for index, value in enumerate(label):
                # 处理numpy数组和普通值的键转换
                if isinstance(value[0], np.ndarray):
                    value_key = tuple(value[0])  # 数组转元组作为字典键
                else:
                    value_key = value[0]  # 普通值直接作为键

                if value_key in groups:
                    groups[value_key].append(index)
                else:
                    groups[value_key] = [index]

            # ==================== 平衡分割 ====================
            # 对每个标签组进行随机分割
            others = []  # 存储因取整问题剩余的样本
            for indexes in groups.values():
                random.shuffle(indexes)  # 随机打乱
                total_length = len(indexes)

                # 计算每个集合的理论样本数
                test_num = int(setting.test_size * total_length)
                val_num = int(setting.val_size * total_length)
                train_num = int((1 - setting.test_size - setting.val_size) * total_length)

                # 分割样本
                tts['test'][0].extend(indexes[:test_num])
                tts['val'][0].extend(indexes[test_num:test_num + val_num])
                tts['train'][0].extend(indexes[test_num + val_num:test_num + val_num + train_num])

                # 收集因取整问题剩余的样本
                others.extend(indexes[test_num + val_num + train_num:])

            # ==================== 处理剩余样本 ====================
            # 将剩余样本按比例分配到各集合，确保总比例接近设定值
            if len(others) != 0:
                random.shuffle(others)
                expect_test_num = int(len(label) * setting.test_size)
                expect_val_num = int(len(label) * setting.val_size)

                # 计算当前与期望数量的差值
                test_num = expect_test_num - len(tts['test'][0])
                val_num = expect_val_num - len(tts['val'][0])

                # 分配剩余样本
                tts['test'][0].extend(others[:test_num])
                tts['val'][0].extend(others[test_num:test_num + val_num])
                tts['train'][0].extend(others[test_num + val_num:])

        else:
            # 非被试依赖模式：简单随机分割
            tts['test'] = [[]]
            tts['train'] = [[]]
            tts['val'] = [[]]

            indexes = [i for i in range(len(label))]
            random.shuffle(indexes)  # 随机打乱
            total_length = len(indexes)

            # 计算每个集合的样本数
            test_num = int(setting.test_size * total_length)
            val_num = int(setting.val_size * total_length)
            train_num = total_length - test_num - val_num

            # 分割样本
            tts['test'][0].extend(indexes[:test_num])
            tts['val'][0].extend(indexes[test_num:test_num + val_num])
            tts['train'][0].extend(indexes[test_num + val_num:])

    else:
        print("错误的分割类型，请检查")
        exit(1)

    # ==================== 次级轮次处理 ====================
    # 支持选择特定的轮次/折叠进行实验（如只运行第1、3、5折）
    assert setting.sr is None or (max(setting.sr) <= len(label) and min(setting.sr) > 0), \
        "次级轮次超出限制或次级轮次设置小于0"

    if setting.sr is not None:
        # 选择特定的轮次/折叠
        tts['train'] = [tts['train'][i - 1] for i in setting.sr]
        tts['test'] = [tts['test'][i - 1] for i in setting.sr]
        if 'val' in tts:
            tts['val'] = [tts['val'][i - 1] for i in setting.sr]

    # ==================== 验证集默认设置 ====================
    # 如果分割方法不产生验证集，设置为[-1]表示无验证集
    if 'val' not in tts:
        tts['val'] = [[-1] for _ in tts['train']]

    return tts


def merge_to_part(data, label, setting=None):
    """
    根据实验模式将多维度数据合并为合适的分组结构

    将 (session, subject, trail, sample) 合并为 (corresponding_part, sample)
    根据不同的实验模式采用不同的合并策略

    参数:
        data: 原始数据，形状 -> (session, subject, trail, sample, ...)
        label: 原始标签，形状 -> (session, subject, trail, sample, ...)
        setting: 实验设置对象
            - experiment_mode: ["subject-dependent", "subject-independent", "cross-session"]
            - sessions: 使用的会话列表（索引从1开始，默认全部）
            - cross_trail: 是否跨试次（仅对subject-dependent模式有效）

    返回:
        根据实验模式返回不同结构的数据和标签：
        1. 非subject-dependent模式:
            data: -> (corresponding_part, sample)
            label: -> (corresponding_part, sample)
        2. subject-dependent且cross_trail='true':
            data: -> (subject, trail, sample)
            label: -> (subject, trail, sample)
        3. subject-dependent且cross_trail='false':
            data: -> (subject, sample)
            label: -> (subject, sample)
    """

    # ==================== 会话选择验证 ====================
    # 验证会话选择是否有效
    assert setting.sessions is None or (max(setting.sessions) <= len(label) and min(setting.sessions) >= 0), \
        "会话设置错误，数据集中不存在该会话"

    # 确定使用的会话索引
    if setting.sessions is None:
        sessions = range(len(data))  # 使用所有会话
    else:
        sessions = [i - 1 for i in setting.sessions]  # 转换为0-based索引

    m_data = []  # 合并后的数据
    m_label = []  # 合并后的标签

    # ==================== 被试依赖模式 - 跨试次 ====================
    # 保持试次结构，用于试次级别的分析或模型
    if setting.experiment_mode == "subject-dependent" and setting.cross_trail == 'true':
        # 创建数据结构：每个被试一个列表，每个列表包含其所有试次
        m_data = [[] for _ in range(len(data[0]) * len(sessions))]
        m_label = [[] for _ in range(len(data[0]) * len(sessions))]

        for i in sessions:
            for idx1, subject in enumerate(data[i]):
                for idx2, trail in enumerate(subject):
                    # 按试次分组
                    m_data[i * len(data[i]) + idx1].append(trail)

        for i in sessions:
            for idx1, subject in enumerate(label[i]):
                for idx2, trail in enumerate(subject):
                    m_label[i * len(data[i]) + idx1].append(trail)

    # ==================== 被试依赖模式 - 不跨试次 ====================
    # 展平试次结构，将所有试次的样本合并
    elif setting.experiment_mode == "subject-dependent" and setting.cross_trail == 'false':
        m_data = [[] for _ in range(len(data[0]))]
        m_label = [[] for _ in range(len(data[0]))]

        for i in sessions:
            for idx1, subject in enumerate(data[i]):
                for idx2, trail in enumerate(subject):
                    for sample in trail:
                        # 展平试次，按样本存储
                        m_data[i * len(data[0]) + idx1].append([sample])

        for i in sessions:
            for idx1, subject in enumerate(label[i]):
                for idx2, trail in enumerate(subject):
                    for sample in trail:
                        m_label[i * len(data[0]) + idx1].append([sample])

    # ==================== 被试独立模式 ====================
    # 将被试作为分组单位，用于跨被试泛化评估
    elif setting.experiment_mode == "subject-independent":
        # 数据结构：[所有被试][每个被试的所有样本]
        m_data = [[[] for _ in range(len(data[0]))]]
        m_label = [[[] for _ in range(len(data[0]))]]

        for i in sessions:
            for idx, subject in enumerate(data[i]):
                for trail in subject:
                    # 将所有试次的样本合并到被试级别
                    m_data[0][idx].extend(trail)

        for i in sessions:
            for idx, subject in enumerate(label[i]):
                for trail in subject:
                    m_label[0][idx].extend(trail)

    # ==================== 跨会话模式 ====================
    # 将会话作为分组单位，用于跨时间泛化评估
    elif setting.experiment_mode == "cross-session":
        # 数据结构：[所有会话][每个会话的所有样本]
        m_data = [[[] for _ in range(len(sessions))]]
        m_label = [[[] for _ in range(len(sessions))]]

        for i in sessions:
            for subject in data[i]:
                for trail in subject:
                    # 将所有被试和试次的样本合并到会话级别
                    m_data[0][i].extend(trail)

        for i in sessions:
            for subject in label[i]:
                for trail in subject:
                    m_label[0][i].extend(trail)

    # ==================== 主轮次处理 ====================
    # 支持选择特定的主轮次进行实验
    assert setting.pr is None or (max(setting.pr) <= len(m_label) and min(setting.pr) > 0), \
        "主轮次超出限制或主轮次设置小于0"

    if setting.pr is not None:
        # 选择特定的主轮次
        m_data = [m_data[i - 1] for i in setting.pr]
        m_label = [m_label[i - 1] for i in setting.pr]

    return m_data, m_label