# 导入操作系统模块，用于设置环境变量
import os

# 导入 numpy 数值计算库
import numpy as np

# 导入 PyTorch 深度学习框架
import torch

# 导入 random 随机数生成模块
import random

# 从自定义模块导入保存结果的函数
from LibEER.utils.store import save_res


def state_log(args):
    """
    打印实验状态信息的表格（美观的格式化输出）

    参数:
        args: 命令行参数对象，包含实验配置
    """
    # 创建要显示的参数字典，选择关键参数
    log_dict = {
        "dataset": args.dataset,  # 数据集名称
        "feature type": args.feature_type,  # 特征类型
        "model": args.model,  # 模型名称
        "batch size": args.batch_size,  # 批大小
        "epochs": args.epochs,  # 训练轮数
        "learning rate": args.lr,  # 学习率
        "experiment mode": args.experiment_mode,  # 实验模式
        "split_type": args.split_type,  # 数据分割类型
        "log dir": args.log_dir,  # 日志目录
        "output dir": args.output_dir,  # 输出目录
    }

    # 打印分隔线，43个下划线（与表格宽度匹配）
    print('_' * 43)

    # 遍历字典中的每个键值对，格式化打印为表格行
    for key, value in zip(log_dict.keys(), log_dict.values()):
        # 使用字符串格式化，居中对齐，每列宽度20
        print("|{:^20}|{:^20}|".format(key, value))

    # 打印底部边框线
    print('-' * 43)


def result_log(args, best_metrics):
    """
    打印并保存多轮实验的统计结果

    参数:
        args: 命令行参数对象
        best_metrics: 列表，包含每轮实验的最佳指标字典
                      例如 [{'acc': 0.85, 'f1': 0.83}, {'acc': 0.86, 'f1': 0.84}]
    """
    # 初始化输出字典，用于存储各指标的列表
    output = {}

    # 构建表头字符串
    s = "|{:^10}|".format("Result")  # 第一列标题："Result"，宽度10，居中
    for metric_name in args.metrics:
        # 为每个指标创建空列表
        output[metric_name] = []
        # 在表头中添加指标列
        s += "{:^10}|".format(metric_name)

    # 打印表头
    print(s)

    # 遍历每一轮实验的结果
    for idx, metric in enumerate(best_metrics):
        # 构建当前轮次的行字符串
        s_i = "|{:^10}|".format(idx + 1)  # 轮次编号（从1开始）

        # 遍历每个指标
        for n in args.metrics:
            # 将当前轮次的指标值添加到对应列表中
            output[n].append(metric[n])
            # 格式化当前指标值（3位小数），添加到行字符串
            s_i += "{:^10.3f}|".format(metric[n])

        # 打印当前轮次的结果行
        print(s_i)

    # 计算并打印所有轮次的统计结果
    for metric in args.metrics:
        # 计算该指标的均值和标准差
        mean_val = np.mean(output[metric])
        std_val = np.std(output[metric])

        # 打印统计信息（4位小数）
        print("ALLRound Mean and Std of {} : {:.4f}/{:.4f}".format(
            metric, mean_val, std_val))

        # 保存统计结果到日志文件
        save_res(args, "ALLRound Mean and Std of {} : {:.4f}/{:.4f}".format(
            metric, mean_val, std_val))


def sub_result_log(args, subjects_metrics):
    """
    处理并打印基于被试（subject）的实验结果

    参数:
        args: 命令行参数对象
        subjects_metrics: 列表的列表，格式为 subjects_metrics[subject][round][metric]
                          例如：subjects_metrics[0][0]['acc'] = 0.85
    """
    # 初始化字典，存储每个被试的统计结果
    sub_outputs = {}

    # 遍历每个被试
    for i, sub_metric in enumerate(subjects_metrics):
        # 初始化当前被试的指标字典
        sub_output = {}

        # 遍历每个指标
        for metric in args.metrics:
            # 初始化当前被试当前指标的累积值
            sub_output[metric] = 0

            # 遍历该被试的所有轮次
            for r_metric in sub_metric:
                # 累加该轮次的指标值
                sub_output[metric] += r_metric[metric]

            # 计算该被试该指标的平均值
            sub_output[metric] /= len(subjects_metrics[i])

        # 将当前被试的结果存入字典，键为"sub i"
        sub_outputs[f"sub {i}"] = sub_output

    # sub_outputs 结构: (subject, metric_dict)
    # 保存被试级别的结果
    save_res(args, sub_outputs)

    # 初始化字典，存储所有被试的均值和标准差
    sub_mean_std = {}

    # 计算每个指标的跨被试统计
    for metric in args.metrics:
        # 收集所有被试在该指标上的值
        sub_metrics = []
        for sub_metric in sub_outputs.values():
            sub_metrics.append(sub_metric[metric])

        # 计算均值和标准差
        mean_val = np.mean(sub_metrics)
        std_val = np.std(sub_metrics)

        # 存储到字典中
        sub_mean_std[metric] = {"mean": mean_val, "std": std_val}

        # 打印统计信息
        print("ALLRound Mean and Std of {} : {:.4f}/{:.4f}".format(
            metric, mean_val, std_val))

        # 保存到日志文件
        save_res(args, f"ALL Subjects {metric}: Mean: {mean_val}, Std: {std_val}")


def setup_seed(seed):
    """
    设置随机种子以确保实验的可重复性

    参数:
        seed: 随机种子值（整数）
    """
    # 设置 PyTorch CPU 随机种子
    torch.manual_seed(seed)

    # 设置 PyTorch 所有 GPU 的随机种子
    torch.cuda.manual_seed_all(seed)

    # 设置 numpy 随机种子
    np.random.seed(seed)

    # 设置 Python 内置 random 模块随机种子
    random.seed(seed)

    # 设置 Python 哈希种子环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 设置 CuDNN 确定性模式（确保GPU计算可重复）
    torch.backends.cudnn.deterministic = True

    # 禁用 CuDNN 基准测试（如果启用基准测试，确定性可能为False）
    torch.backends.cudnn.benchmark = False

    # 禁用 CuDNN（更彻底的可重复性设置）
    torch.backends.cudnn.enabled = False