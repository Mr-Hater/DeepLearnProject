from LibEER.models.Models import Model
from LibEER.config.setting import seed_sub_dependent_front_back_setting, preset_setting, set_setting_by_args
from LibEER.data_utils.load_data import get_data
from LibEER.data_utils.split import merge_to_part, index_to_data, get_split_index
from LibEER.utils.args import get_args_parser
from LibEER.utils.store import make_output_dir
from LibEER.utils.utils import state_log, result_log, setup_seed, sub_result_log
from LibEER.Trainer.training import train
import torch
import torch.optim as optim
import torch.nn as nn

# run this file with
# CUDA_VISIBLE_DEVICES=0  nohup python EEGNet_train.py -setting seed_sub_dependent_front_back_setting  -dataset seed_raw -onehot -batch_size 512 -sample_length 200 -stride 200 -only_seg -lr 0.001 -sessions 1 2 -epochs 200 > EEGNet/repro512_0001.log &

#    python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1' -setting seed_sub_dependent_train_val_test_setting -dataset_path D:/Study/EEG/LibEER/data_utils/Dataset/SEED -dataset seed_raw -sample_length 200 -stride 200 -only_seg -batch_size 256 -epochs 100 -seed 2024 -lr 0.001 -onehot > D:/Study/EEG/LibEER/result/EEGNet/b256e100l001.log
#    seed dep
#    python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1' -setting seed_sub_dependent_train_val_test_setting -dataset_path D:/Study/EEG/LibEER/data_utils/Dataset/SEED -dataset seed_raw -sample_length 200 -stride 200 -only_seg -batch_size 256 -epochs 100 -seed 2024 -lr 0.001 -onehot >EEGNet/b256e100l001.log
#    0.5881/0.1622	0.5441/0.1759
#    seed iv dep
#    python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1' -setting seediv_sub_dependent_train_val_test_setting -dataset_path D:/Study/EEG/LibEER/data_utils/Dataset/SEED_IV -dataset seediv_raw -batch_size 512 -epochs 150 -seed 2024 -sample_length 200 -stride 200 -only_seg -onehot >EEGNet/s4_b512e150l001.log
#    0.2989/0.1353	0.2659/0.1358


#    hci dep
#    arousal
#    CUDA_VISIBLE_DEVICES=2 nohup python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 256 -epochs 300 -lr 0.04 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >EEGNet/hci_arousal_b256e300lr0.04.log
#    0.6742/0.2171	0.5450/0.2005
#    valence
#    CUDA_VISIBLE_DEVICES=2 nohup python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 128 -epochs 300 -lr 0.04 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence -seed 2024 >EEGNet/hci_valence_b128e300lr0.04.log
#    0.6115/0.1676	0.5035/0.1728
#    both
#    CUDA_VISIBLE_DEVICES=2 nohup python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 128 -epochs 300 -lr 0.04 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence arousal -seed 2024 >EEGNet/hci_both_b128e300lr0.04.log
#    0.3832/0.1951	0.2456/0.1371

#    deap dep
#    arousal
#    python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path D:\Study\EEG\LibEER\data_utils\Dataset\DEAP\data_preprocessed_python -dataset deap -batch_size 256 -epochs 300 -lr 0.02 -only_seg -sample_length 512 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 -onehot >EEGNet/deap_arousal_b512e300lr0.02.log
#    0.6130/0.1588	0.5326/0.1305
#    valence
#    python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path D:\Study\EEG\LibEER\data_utils\Dataset\DEAP\data_preprocessed_python -dataset deap -batch_size 256 -epochs 300 -lr 0.02 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence -seed 2024 -onehot >EEGNet/deap_valence_b256e300lr0.02.log
#    0.5150/0.1157	0.4785/0.1170
#    both
#    python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path D:\Study\EEG\LibEER\data_utils\Dataset\DEAP\data_preprocessed_python -dataset deap -batch_size 256 -epochs 300 -lr 0.04 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence arousal -seed 2024 -onehot >EEGNet/deap_both_b256e300lr0.04.log
#    0.3941/0.1153	0.2919/0.1021

#    seed iv indep
#    python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1' -setting seediv_sub_independent_train_val_test_setting -dataset_path D:/Study/EEG/LibEER/data_utils/Dataset/SEED_IV -dataset seediv_raw -batch_size 128 -epochs 150 -seed 2024 -sample_length 200 -stride 200 -only_seg >EEGNet_indep/s4_b128e150.log &
#    28.19%	0.2835


#    hci indep
#    valence
#    CUDA_VISIBLE_DEVICES=1 nohup python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 256 -epochs 300 -lr 0.04 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence -seed 2024 >EEGNet_indep/hci_valence_b256e300lr0.04.log
#    0.5706	0.5383
#    arousal
#    CUDA_VISIBLE_DEVICES=1 nohup python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 512 -epochs 300 -lr 0.02 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >EEGNet_indep/hci_arousal_b512e300lr0.02.log
#    0.547	0.5402
#    both
#    CUDA_VISIBLE_DEVICES=1 nohup python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 256 -epochs 300 -lr 0.02 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence arousal -seed 2024 >EEGNet_indep/hci_both_b256e300lr0.02.log
#    0.3484	0.2796


#    deap indep
#    valence
#    CUDA_VISIBLE_DEVICES=3 nohup python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path D:\Study\EEG\LibEER\data_utils\Dataset\DEAP\data_preprocessed_python -dataset deap -batch_size 128 -epochs 300 -lr 0.02 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence -seed 2024 >EEGNet_indep/deap_valence_b128e300lr0.02.log
#    0.5236	0.4974
#    arousal
#    CUDA_VISIBLE_DEVICES=3 nohup python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path /D:\Study\EEG\LibEER\data_utils\Dataset\DEAP\data_preprocessed_python -dataset deap -batch_size 512 -epochs 300 -lr 0.02 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >EEGNet_indep/deap_arousal_b512e300lr0.02.log
#    0.4894	0.4894
#    both
#    CUDA_VISIBLE_DEVICES=3 nohup python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path D:\Study\EEG\LibEER\data_utils\Dataset\DEAP\data_preprocessed_python -dataset deap -batch_size 256 -epochs 300 -lr 0.02 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence arousal -seed 2024 >EEGNet_indep/deap_both_b256e300lr0.02.log
#    0.2541	0.2444

def main(args):
    """
    主训练函数：负责整个训练流程的协调和执行

    参数:
        args: 命令行参数对象，包含所有训练配置
    """

    # ==================== 1. 实验设置初始化 ====================
    # 根据预设设置或命令行参数创建实验设置对象
    if args.setting is not None:
        # 使用预设的实验设置（如"seed_sub_dependent_train_val_test_setting"）
        setting = preset_setting[args.setting](args)
    else:
        # 使用命令行参数自定义设置
        setting = set_setting_by_args(args)

    # 设置随机种子以确保实验可重复性
    setup_seed(args.seed)

    # ==================== 2. 数据加载与预处理 ====================
    # 加载原始数据并获取相关元信息
    # data: 原始EEG数据，label: 对应标签
    # channels: 脑电通道数，feature_dim: 特征维度，num_classes: 情感类别数
    data, label, channels, feature_dim, num_classes = get_data(setting)

    # 根据实验模式合并数据到合适的分组结构
    # 例如：将被试依赖模式的数据合并为(subject, sample)格式
    data, label = merge_to_part(data, label, setting)

    # 设置训练设备（CPU/GPU）
    device = torch.device(args.device)

    # ==================== 3. 实验循环初始化 ====================
    # best_metrics: 存储所有轮次的最佳指标
    best_metrics = []

    # subjects_metrics: 用于被试依赖模式下，存储每个被试的指标
    # 结构：[被试1的指标列表, 被试2的指标列表, ...]
    subjects_metrics = [[] for _ in range(len(data))]

    # ==================== 4. 主实验循环 ====================
    # 外层循环：遍历不同轮次/实验重复（如交叉验证的不同折）
    # rridx: round repeat index（轮次重复索引），从1开始计数
    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        """
        循环说明:
        - data_i: 当前轮次的数据（如某个特定实验设置下的数据）
        - label_i: 当前轮次的标签
        - rridx: 当前轮次索引，用于跟踪和记录
        """

        # 获取当前轮次的数据分割索引
        # 根据设置（如K折、留一法等）生成训练、验证、测试集的索引
        tts = get_split_index(data_i, label_i, setting)

        # 内层循环：遍历当前轮次的不同数据分割（如K折交叉验证的各折）
        # ridx: round index（轮次内索引），从1开始计数
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts['train'], tts['test'], tts['val']),
                                                                          1):
            """
            循环说明:
            - train_indexes: 训练集索引列表
            - test_indexes: 测试集索引列表
            - val_indexes: 验证集索引列表（[-1]表示无验证集）
            - ridx: 当前分割的索引
            """

            # 重置随机种子，确保每轮实验的可重复性
            setup_seed(args.seed)

            # 打印当前分割的索引信息（调试用）
            if val_indexes[0] == -1:
                # 无验证集的情况
                print(f"训练集索引: {train_indexes}, 测试集索引: {test_indexes}")
            else:
                # 有验证集的情况
                print(f"训练集索引: {train_indexes}, 验证集索引: {val_indexes}, 测试集索引: {test_indexes}")

            # ==================== 5. 数据分割 ====================
            # 根据索引将数据分割为训练集、验证集、测试集
            # args.keep_dim: 是否保持原始维度（如保持试次结构）
            train_data, train_label, val_data, val_label, test_data, test_label = \
                index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, args.keep_dim)



            # 如果没有验证集，使用测试集作为验证集（简化实验设置）
            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label

            # ==================== 6. 模型初始化 ====================
            # 创建EEGNet模型实例
            # Model['EEGNet']: 从模型字典中获取EEGNet模型类
            model = Model['EEGNet'](channels, feature_dim, num_classes)

            # ==================== 7. 数据封装 ====================
            # 将numpy数组转换为PyTorch的TensorDataset对象
            dataset_train = torch.utils.data.TensorDataset(
                torch.Tensor(train_data),
                torch.Tensor(train_label)
            )
            dataset_val = torch.utils.data.TensorDataset(
                torch.Tensor(val_data),
                torch.Tensor(val_label)
            )
            dataset_test = torch.utils.data.TensorDataset(
                torch.Tensor(test_data),
                torch.Tensor(test_label)
            )

            # ==================== 8. 优化器和损失函数 ====================
            # 使用Adam优化器，包含L2正则化（weight_decay）
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.lr,  # 学习率
                weight_decay=1e-4,  # 权重衰减（L2正则化）
                eps=1e-8  # 数值稳定性参数
            )

            # 使用交叉熵损失函数（适用于分类任务）
            criterion = nn.CrossEntropyLoss()



            # ==================== 9. 输出目录创建 ====================
            # 创建保存模型和结果的目录
            output_dir = make_output_dir(args, "EEGNet")

            # ==================== 10. 模型训练 ====================
            # 调用训练函数进行单轮训练
            round_metric = train(
                model=model,
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                dataset_test=dataset_test,
                device=device,
                output_dir=output_dir,
                metrics=args.metrics,  # 评估指标列表
                metric_choose=args.metric_choose,  # 选择最佳模型的主要指标
                optimizer=optimizer,
                batch_size=args.batch_size,  # 批大小
                epochs=args.epochs,  # 训练轮数
                criterion=criterion  # 损失函数
            )

            # ==================== 11. 结果记录 ====================
            # 将当前轮次的指标添加到总结果中
            best_metrics.append(round_metric)

            # 如果是被试依赖模式，按被试记录结果
            if setting.experiment_mode == "subject-dependent":
                subjects_metrics[rridx - 1].append(round_metric)

    # ==================== 12. 最终结果汇总 ====================
    # 根据实验模式选择不同的结果记录方式
    if setting.experiment_mode == "subject-dependent":
        # 被试依赖模式：按被试报告结果
        sub_result_log(args, subjects_metrics)
    else:
        # 其他模式：报告总体结果
        result_log(args, best_metrics)


if __name__ == '__main__':
    """
    程序入口点
    """

    # 获取命令行参数
    args = get_args_parser()
    args = args.parse_args()

    # 记录训练状态（如开始时间、参数配置等）
    state_log(args)

    # 执行主训练函数
    main(args)