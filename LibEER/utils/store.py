# 导入命令行参数解析模块
import argparse

# 导入操作系统相关功能模块
import os.path

# 导入时间模块
import time

# 导入路径处理模块（Python 3.4+ 推荐使用）
from pathlib import Path

# 导入 PyTorch 深度学习框架
import torch


def make_output_dir(args, model):
    """
    根据参数创建输出目录的层级结构

    参数:
        args: 命令行参数对象
        model: 模型名称

    返回:
        Path: 构建好的输出目录路径对象
    """
    # 将输出目录字符串转换为 Path 对象
    output_dir = Path(args.output_dir)

    # 第一级：模型名称目录
    output_dir = output_dir / model

    # 如果有预设设置，使用预设设置作为第二级目录
    if args.setting is not None:
        output_dir = output_dir / args.setting
    else:
        # 否则使用实验模式和分割类型作为第二、三级目录
        output_dir = output_dir / args.experiment_mode
        output_dir = output_dir / args.split_type

    # 如果指定了使用的标签，添加标签相关目录
    if args.label_used is not None:
        if len(args.label_used) == 1:
            # 如果只有一个标签，直接使用标签名作为目录
            output_dir = output_dir / args.label_used[0]
        else:
            # 如果有多个标签，用"both"连接所有标签名
            # 例如：如果 label_used = ['valence', 'arousal']，目录名为"bothvalencearousal"
            output_dir = output_dir / "both".join(label for label in args.label_used)

    return output_dir


def save_state(output_dir, model, optimizer, epoch, r_idx='last', rr_idx='last', metric=None, state='best'):
    """
    保存模型状态（检查点）

    参数:
        output_dir: 输出目录路径或参数对象
        model: 要保存的模型
        optimizer: 优化器
        epoch: 当前训练轮数
        r_idx: 主轮次索引（如交叉验证的fold编号），默认'last'
        rr_idx: 次轮次索引，默认'last'
        metric: 指标值（用于命名），默认None
        state: 状态标识（如'best', 'last'），默认'best'
    """
    # 兼容性处理：如果output_dir是参数对象，而不是路径
    if type(output_dir) is argparse.Namespace:
        # 使用参数对象创建输出目录
        output_dir = make_output_dir(output_dir, output_dir.model)
    else:
        # 如果是路径字符串，转换为Path对象
        output_dir = Path(output_dir)

    # 如果不是最后的主次轮次，添加轮次目录
    if not (r_idx == 'last' and rr_idx == 'last'):
        output_dir = output_dir / str(r_idx)  # 主轮次目录
        output_dir = output_dir / str(rr_idx)  # 次轮次目录

    # 创建目录（如果不存在）
    try:
        # exist_ok=True 表示目录已存在时不报错
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        # 如果创建目录出错，打印错误信息
        print(f"An error occurred: {e.strerror}")

    # 构建检查点文件路径
    if metric is None:
        # 如果没有指定metric，使用epoch命名
        checkpoint_path = output_dir / f'checkpoint-{str(epoch)}'
    else:
        # 如果有metric，使用状态和metric命名
        checkpoint_path = output_dir / f'checkpoint-{state}{metric}'

    # 构建要保存的字典数据
    save = {
        'model': model.state_dict(),  # 模型参数
        'optimizer': optimizer.state_dict(),  # 优化器状态
        'epoch': epoch,  # 当前训练轮数
    }

    # 使用PyTorch保存检查点
    torch.save(save, checkpoint_path)

    # 打印保存信息
    print(f"save model to {checkpoint_path}")


def save_data(args, data, label):
    """
    保存处理后的数据

    参数:
        args: 命令行参数对象
        data: 要保存的数据
        label: 数据标签
    """
    # 创建数据保存目录
    save_dir = Path(args.data_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 定义实验模式的缩写映射
    mode = {
        'subject-dependent': 'sub-dep',  # 被试内
        'subject-independent': 'sub-In',  # 被试间
        'cross-session': 'cro-sess'  # 跨会话
    }

    # 构建数据保存路径
    # 第一级：数据集名称
    save_path = save_dir / f'{args.dataset}'

    # 第二级：特征类型-时间窗口-重叠参数
    save_path = save_path / f'{args.feature_type}-tw-{args.time_window}ol-{args.overlap}'

    # 创建目录（如果不存在）
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 打印保存信息
    print(f"Saving Processed Data To {save_path}")

    # 注意：这个函数实际没有保存data和label参数！
    # 这里应该添加数据保存逻辑，例如：
    # torch.save({'data': data, 'label': label}, save_path / 'data.pt')


def save_res(args, metric):
    """
    保存实验结果到日志文件

    参数:
        args: 命令行参数对象
        metric: 评估指标结果
    """
    # 创建日志目录
    log_dir = Path(args.log_dir)
    add_dir(log_dir)  # 确保目录存在

    # 构建日志文件名：使用参数中的时间格式化
    log_file = log_dir / time.strftime("%Y-%m-%d %H-%M-%S", args.time)

    # 如果日志文件不存在，创建并写入参数信息
    if not os.path.exists(log_file):
        f = open(log_file, 'w')
        f.write(str(args))  # 写入所有参数
        f.close()

    # 追加写入指标结果
    f = open(log_file, 'a')  # 'a' 表示追加模式
    f.write('\n' + str(metric))  # 换行后写入指标
    f.close()


def add_dir(path):
    """
    辅助函数：确保目录存在，如果不存在则创建

    参数:
        path: 目录路径
    """
    if not os.path.exists(path):
        os.makedirs(path)  # 创建目录（包括所有父目录）