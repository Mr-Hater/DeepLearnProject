# 导入必要的库和模块
import numpy as np  # 数值计算库

# 从自定义模块导入相关类和方法
from models.Models import Model  # 模型定义
from config.setting import seed_sub_dependent_front_back_setting, preset_setting, set_setting_by_args  # 配置设置
from data_utils.load_data import get_data  # 数据加载
from data_utils.split import merge_to_part, index_to_data, get_split_index  # 数据分割处理
from utils.args import get_args_parser  # 参数解析
from utils.metric import Metric  # 评估指标
from utils.store import make_output_dir  # 输出目录管理
from utils.utils import state_log, result_log, setup_seed, sub_result_log  # 工具函数
from Trainer.training import train  # 训练器
import torch  # PyTorch深度学习框架
import torch.optim as optim  # 优化器
import torch.nn as nn  # 神经网络模块


def main(args):
    """
    核心方法介绍：
    主函数，负责整个机器学习流程的执行，包括：
    1. 实验配置设置
    2. 数据加载和预处理
    3. 模型训练和评估
    4. 结果记录和输出
    """

    # 根据参数设置实验配置
    if args.setting is not None:  # 如果指定了预设配置
        setting = preset_setting[args.setting](args)  # 使用预设配置
    else:  # 如果没有指定预设配置
        setting = set_setting_by_args(args)  # 根据命令行参数设置配置

    setup_seed(args.seed)  # 设置随机种子以确保实验可重复性

    # 加载数据，返回数据、标签、通道数、特征维度和类别数
    data, label, channels, feature_dim, num_classes = get_data(setting)

    # 将数据合并到指定部分（根据实验设置）
    data, label = merge_to_part(data, label, setting)

    best_metrics = []  # 存储每轮实验的最佳指标
    subjects_metrics = [[] for _ in range(len(data))]  # 存储每个受试者的指标列表

    # 遍历每个受试者的数据（rridx从1开始计数）
    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        # 获取数据分割索引（训练集、测试集、验证集）
        tts = get_split_index(data_i, label_i, setting)

        # 遍历每种数据分割方式（ridx从1开始计数）
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts['train'], tts['test'], tts['val']),
                                                                          1):
            setup_seed(args.seed)  # 为每轮实验设置随机种子

            # 打印数据分割信息
            if val_indexes[0] == -1:  # 如果没有验证集
                print(f"train indexes:{train_indexes}, test indexes:{test_indexes}")
            else:  # 如果有验证集
                print(f"train indexes:{train_indexes}, val indexes:{val_indexes}, test indexes:{test_indexes}")

            # 根据指定的实验模式分割训练和测试数据
            train_data, train_label, val_data, val_label, test_data, test_label = \
                index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, args.keep_dim)

            # 将标签从one-hot编码转换为类别索引
            train_label = np.argmax(train_label, axis=1)  # 训练集标签
            test_label = np.argmax(test_label, axis=1)  # 测试集标签
            val_label = np.argmax(val_label, axis=1)  # 验证集标签

            # 如果验证集为空，使用测试集作为验证集
            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label

            # 创建SVM模型
            model = Model['svm'](channels, feature_dim, num_classes)

            # 重塑数据形状以适应SVM输入要求
            train_data = train_data.reshape(train_data.shape[0], -1)  # 将训练数据展平
            test_data = test_data.reshape(test_data.shape[0], -1)  # 将测试数据展平

            # 训练SVM模型
            model.svc.fit(train_data, train_label)  # 使用训练数据拟合SVM

            # 在测试集上进行预测
            pred = model.svc.predict(test_data)  # 预测测试集标签

            # 计算评估指标
            metric = Metric(args.metrics)  # 初始化评估指标计算器
            metric.update(pred, test_label)  # 更新指标计算
            metric.value()  # 计算指标值
            round_metric = metric.values  # 获取本轮实验的指标值

            # 打印每个指标的测试结果
            for m in args.metrics:
                print(f"best_test_{m}: {round_metric[m]:.2f}")

            best_metrics.append(round_metric)  # 记录本轮指标

            # 如果是受试者相关实验模式，记录受试者特定指标
            if setting.experiment_mode == "subject-dependent":
                subjects_metrics[rridx - 1].append(round_metric)

    # 根据实验模式记录最终结果
    if setting.experiment_mode == "subject-dependent":  # 受试者相关模式
        sub_result_log(args, subjects_metrics)  # 记录受试者级别的结果
    else:  # 其他实验模式
        result_log(args, best_metrics)  # 记录总体结果


if __name__ == '__main__':
    """
    程序入口点：
    1. 解析命令行参数
    2. 记录训练状态
    3. 执行主函数
    """
    args = get_args_parser()  # 获取参数解析器
    args = args.parse_args()  # 解析命令行参数
    state_log(args)  # 记录训练状态和参数
    main(args)  # 执行主函数