import numpy as np

from config.setting import Setting, preset_setting, set_setting_by_args
from models.Models import Model

from utils.utils import state_log, result_log, setup_seed, sub_result_log
from utils.args import get_args_parser
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from utils.store import save_res


# LibEER/main.py（只需要添加这个函数）


def main(args):
    # 根据参数args设置数据集和数据预处理参数
    if args.setting is not None:
        # 如果指定了预设配置，使用预设配置
        setting = preset_setting[args.setting](args)
    else:
        # 否则根据参数动态设置配置
        setting = set_setting_by_args(args)
    # 设置随机种子以确保结果可重现
    setup_seed(args.seed)

    # 根据配置获取数据，包括数据、标签、通道数、特征维度和类别数
    data, label, channels, feature_dim, num_classes = get_data(setting)

    # 如果设置中启用了保存数据功能
    # if setting.save_data:

    # 根据实验模式合并数据到相应部分
    data, label = merge_to_part(data, label, setting)
    # 获取训练和测试索引以及分割类型

    # 初始化存储最佳指标和受试者指标的列表
    best_metrics = []
    subjects_metrics = []

    # 如果是受试者依赖实验模式，为每个受试者初始化指标列表
    if setting.experiment_mode == "subject-dependent":
        subjects_metrics = [[] for _ in range(len(data))]

    # 遍历每个受试者的数据和标签（rridx从1开始计数）
    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        # 获取当前受试者的数据分割索引
        tts = get_split_index(data_i, label_i, setting)

        # 遍历每个分割轮次（ridx从1开始计数）
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts['train'], tts['test'], tts['val']),
                                                                          1):
            # 每轮开始前设置随机种子
            setup_seed(args.seed)

            # 根据验证索引是否存在，打印不同的分割信息
            if val_indexes[0] == -1:
                print(f"训练索引:{train_indexes}, 测试索引:{test_indexes}")
            else:
                print(f"训练索引:{train_indexes}, 验证索引:{val_indexes}, 测试索引:{test_indexes}")

            # 根据索引分割训练、验证和测试数据
            train_data, train_label, val_data, val_label, test_data, test_label = \
                    index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, args.keep_dim)
            # print(len(train_data))
            # model to train
            # 根据参数选择不同的模型初始化方式
            if args.sample_length == 1 or args.only_seg:
                # 使用基本参数初始化模型
                model = Model[args.model](channels, feature_dim, num_classes)
            else:
                # 使用包含样本长度的参数初始化模型
                model = Model[args.model](args.sample_length, channels, feature_dim, num_classes)

            # 训练模型一轮并获取评估指标
            round_metric = model.train_one_round(args, ridx, rridx, train_data, train_label, val_data, val_label,
                                                 test_data, test_label)

            # 保存当前轮次的最佳指标
            best_metrics.append(round_metric)
            # 保存结果到文件
            save_res(args, round_metric)

            # 如果是受试者依赖模式，将指标添加到对应受试者的列表中
            if setting.experiment_mode == "subject-dependent":
                subjects_metrics[rridx - 1].append(round_metric)

    # 根据实验模式输出最终结果
    # best_metrics: 每轮的最佳指标字典
    # subjects_metrics: (受试者, 子轮次指标)
    if setting.experiment_mode == "subject-dependent":
        # 受试者依赖模式：按受试者记录结果
        sub_result_log(args, subjects_metrics)
    else:
        # 其他模式：直接记录所有轮次结果
        result_log(args, best_metrics)


if __name__ == '__main__':
    # 获取参数解析器
    args = get_args_parser()
    # 解析命令行参数
    args = args.parse_args()
    # 记录训练状态日志
    state_log(args)
    # 调用主函数执行训练流程
    main(args)
