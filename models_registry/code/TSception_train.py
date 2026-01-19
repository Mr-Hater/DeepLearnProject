from data_utils.constants.deap import DEAP_CHANNEL_NAME
from data_utils.constants.seed import SEED_CHANNEL_NAME
from models.Models import Model
from models.TSception import generate_TS_channel_order
from config.setting import seed_sub_dependent_front_back_setting, preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from Trainer.training import train
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


# run this file with
# deap batch 64 hci batch 32


#    seed dep
#    CUDA_VISIBLE_DEVICES=0 nohup python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1' -setting seed_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED/ -dataset seed_raw -batch_size 16 -epochs 200 -only_seg -sample_length 200 -stride 200 -seed 2024 >TSception/b16e200.log
#    0.6401/0.1644	0.6053/0.1851
#    seed iv dep
#    CUDA_VISIBLE_DEVICES=1 nohup python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1' -setting seediv_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED_IV -dataset seediv_raw -batch_size 16 -epochs 300 -only_seg -sample_length 200 -stride 200 -seed 2024 >TSception/s4_b16e300.log
#    0.3606/0.1512	0.3277/0.1508

#    seed indep
#    CUDA_VISIBLE_DEVICES=1 nohup python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1' -setting seed_sub_independent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED/ -dataset seed_raw -batch_size 32 -epochs 200 -only_seg -sample_length 200 -stride 200 -seed 2024 >TSception_indep/b32e200.log
#    0.456	0.4354
#    seed iv indep
#    CUDA_VISIBLE_DEVICES=3 nohup python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1' -setting seediv_sub_independent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED_IV -dataset seediv_raw -batch_size 16 -epochs 300 -only_seg -sample_length 200 -stride 200 -seed 2024 >TSception_indep/s4_b16e300.log
#    0.3419	0.2683

#    deap indep
#    valence
#    python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 64 -epochs 300 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence -seed 2024 >TSception_indep/deap_valence_b64e300lr0.001.log
#    0.5444	0.4894
#    arousal
#    python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >TSception/hci_arousal_b32e300lr0.001.log
#    0.459	0.4556
#    both
#    python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.002 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >TSception/hci_arousal_b32e300lr0.002.log
#    0.2464	0.2324

#   hci indep
#   valence
#   python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence -seed 2024 >TSception/hci_valence_b32e300lr0.001.log
#   0.5736	0.5476
#   arousal
#   python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >TSception_indep/hci_arousal_b32e300lr0.001.log
#   0.523	0.5025
#   python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence arousal -seed 2024 >TSception_indep/hci_both_b32e300lr0.001.log
#   0.2699	0.2195

#   hci dep
#   arousal
#   python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.002 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >TSception/hci_arousal_b32e300lr0.002.log
#   0.6826/0.2310	0.5629/0.2379
#   valence
#   python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.002 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence -seed 2024 >TSception/hci_valence_b32e300lr0.002.log
#   0.6112/0.1552	0.5051/0.1669
#   both
#   python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.002 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence arousal -seed 2024 >TSception/hci_both_b32e300lr0.002.log
#   0.4000/0.2060	0.2719/0.1365

def main(args):
    """
    核心方法介绍：主训练流程控制函数
    负责完整的模型训练流程，包括：
    1. 实验配置设置
    2. 数据加载和预处理
    3. 模型训练和评估
    4. 结果记录和保存

    参数说明：
    - args: 命令行参数对象，包含所有训练配置参数
    """
    # 根据参数设置实验配置
    if args.setting is not None:
        # 如果指定了预设配置名称，从预设配置字典中获取对应的配置函数并执行
        setting = preset_setting[args.setting](args)
    else:
        # 如果没有指定预设配置，根据命令行参数动态设置配置
        setting = set_setting_by_args(args)

    # 设置随机种子以确保实验可重复性
    setup_seed(args.seed)

    # 加载数据，返回数据、标签、通道数、特征维度和类别数
    # data: 数据数组，label: 对应标签，channels: 通道数量
    # feature_dim: 特征维度，num_classes: 分类类别数
    data, label, channels, feature_dim, num_classes = get_data(setting)

    # 根据实验设置将数据合并到指定部分
    data, label = merge_to_part(data, label, setting)

    # 设置训练设备（CPU或GPU）
    device = torch.device(args.device)

    # 初始化存储最佳指标的列表
    best_metrics = []
    # 为每个受试者初始化指标列表（用于subject-dependent模式）
    subjects_metrics = [[]for _ in range(len(data))]

    # 遍历每个受试者的数据（rridx从1开始计数）
    # rridx: 受试者索引，data_i: 当前受试者数据，label_i: 当前受试者标签
    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        # 获取数据分割索引（训练集、测试集、验证集）
        # tts包含'train', 'test', 'val'三个键，每个键对应索引列表
        tts = get_split_index(data_i, label_i, setting)

        # 遍历每种数据分割方式（ridx从1开始计数）
        # ridx: 分割轮次索引，train_indexes: 训练集索引
        # test_indexes: 测试集索引，val_indexes: 验证集索引
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts['train'], tts['test'], tts['val']), 1):
            # 为每轮实验设置随机种子以确保可重复性
            setup_seed(args.seed)

            # 打印数据分割信息
            if val_indexes[0] == -1:
                # 如果没有验证集，只打印训练集和测试集索引
                print(f"train indexes:{train_indexes}, test indexes:{test_indexes}")
            else:
                # 如果有验证集，打印所有索引信息
                print(f"train indexes:{train_indexes}, val indexes:{val_indexes}, test indexes:{test_indexes}")

            # split train and test data by specified experiment mode
            # 根据指定的实验模式分割训练和测试数据
            # train_data, train_label: 训练数据和标签
            # val_data, val_label: 验证数据和标签
            # test_data, test_label: 测试数据和标签
            # args.keep_dim: 是否保持数据维度
            train_data, train_label, val_data, val_label, test_data, test_label = \
                index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, args.keep_dim)

            # print(len(train_data))
            # 如果验证集为空，使用测试集作为验证集
            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label

            # model to train
            # 根据数据集选择模型配置
            if args.dataset.startswith('hci'):
                # 对于HCI数据集，使用特定的inception窗口配置
                # inception_window: TSception模型的时间窗口大小配置
                model = Model['TSception'](channels, feature_dim, num_classes, inception_window=[0.25, 0.125, 0.0625])
            else:
                # 对于其他数据集，使用默认配置
                model = Model['TSception'](channels, feature_dim, num_classes)

            # 生成通道重排序索引
            indexes = np.array([])
            if args.dataset == "deap" or args.dataset == "hci":
                # DEAP和HCI数据集使用DEAP通道名称顺序
                indexes = generate_TS_channel_order(DEAP_CHANNEL_NAME)
            elif args.dataset.startswith("seed"):
                # SEED数据集使用SEED通道名称顺序
                indexes = generate_TS_channel_order(SEED_CHANNEL_NAME)

            # 按照指定顺序重排数据通道维度
            # 确保所有数据集使用统一的通道顺序
            train_data = train_data[:, indexes, :]
            val_data = val_data[:, indexes, :]
            test_data = test_data[:, indexes, :]

            # 创建PyTorch数据集对象
            # Train one round using the train one round function defined in the model
            # TensorDataset: 将数据和标签封装成数据集
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))

            # 定义优化器
            # Adam优化器参数：
            # - lr: 学习率
            # - weight_decay: L2正则化系数，防止过拟合
            # - eps: 数值稳定性参数
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-8)

            # 定义损失函数（交叉熵损失，适用于多分类问题）
            criterion = nn.CrossEntropyLoss()

            # 创建输出目录用于保存模型和结果
            output_dir = make_output_dir(args, "TSception")

            # 执行一轮训练
            # 参数说明：
            # - model: 要训练的模型
            # - dataset_train/val/test: 训练/验证/测试数据集
            # - device: 训练设备
            # - output_dir: 输出目录
            # - metrics: 评估指标列表
            # - metric_choose: 选择最佳模型的指标
            # - optimizer: 优化器
            # - batch_size: 批次大小
            # - epochs: 训练轮数
            # - criterion: 损失函数
            round_metric = train(model=model,
                                 dataset_train=dataset_train,
                                 dataset_val=dataset_val,
                                 dataset_test=dataset_test,
                                 device=device,
                                 output_dir=output_dir,
                                 metrics=args.metrics,
                                 metric_choose=args.metric_choose,
                                 optimizer=optimizer,
                                 batch_size=args.batch_size,
                                 epochs=args.epochs,
                                 criterion=criterion
                                 )
            # 记录本轮的最佳指标
            best_metrics.append(round_metric)

            # 如果是受试者相关实验模式，记录受试者特定指标
            if setting.experiment_mode == "subject-dependent":
                subjects_metrics[rridx-1].append(round_metric)

    # best metrics: every round metrics dict
    # subjects metrics: (subject, sub_round_metric)
    # 根据实验模式记录最终结果
    # best metrics: 每轮实验的指标字典列表
    # subjects metrics: (受试者, 子轮次指标) 的嵌套列表
    if setting.experiment_mode == "subject-dependent":
        # 受试者相关模式：记录每个受试者的详细结果
        sub_result_log(args, subjects_metrics)
    else:
        # 其他模式：记录总体结果
        result_log(args, best_metrics)

if __name__ == '__main__':
    """
    程序入口点
    执行流程：
    1. 解析命令行参数
    2. 记录训练状态
    3. 执行主训练函数
    """
    # 获取参数解析器
    args = get_args_parser()
    # 解析命令行参数
    args = args.parse_args()
    # 记录训练状态和参数配置
    state_log(args)
    # 执行主训练函数
    main(args)
