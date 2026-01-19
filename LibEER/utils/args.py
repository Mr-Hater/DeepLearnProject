# 导入 argparse 模块，用于解析命令行参数
import argparse

# 导入 time 模块，用于获取时间信息
import time

# 从自定义模块导入预设设置
from LibEER.config.setting import preset_setting

# 从自定义模块导入可用数据集列表
from LibEER.data_utils.load_data import available_dataset


# 定义获取命令行参数的函数
def get_args_parser():
    """
    创建并配置命令行参数解析器
    返回: 配置好的 argparse.ArgumentParser 对象
    """

    # 创建 ArgumentParser 对象
    # - "EEG Lib for emotion recognition based on EEG": 程序描述
    # - add_help=False: 禁用默认的 -h/--help 选项（可以自定义）
    parser = argparse.ArgumentParser("EEG Lib for emotion recognition based on EEG", add_help=False)

    # --- 训练参数部分 ---
    # 批大小参数，默认128，整型
    parser.add_argument('-batch_size', default=128, type=int, help='batch size per GPU')
    # 训练轮数，默认40，整型
    parser.add_argument('-epochs', default=40, type=int)
    # 训练设备，默认cuda，可选cuda或cpu
    parser.add_argument('-device', default='cuda', type=str, choices=['cuda', 'cpu'], help='which devices to train')
    # 评估模式标志，默认为False，如果设置为True则仅进行评估
    parser.add_argument('-eval', default=False, action='store_true', help='if eval, perform evaluation only')
    # 随机种子，默认1，整型，用于结果可复现性
    parser.add_argument('-seed', default=1, type=int, help='random seed')
    # 数据加载的工作进程数，默认4，整型
    parser.add_argument('-num_workers', default=4, type=int)
    # 损失函数，默认交叉熵损失
    parser.add_argument('-loss_func', default='crossEntropyLoss', type=str, help="the loss function")
    # 评估指标，默认使用准确率，nargs='+'表示可接受多个值
    parser.add_argument('-metrics', default=['acc'], type=str, nargs='+', help='which metrics used to evaluate')
    # 选择最佳模型的指标，默认使用准确率
    parser.add_argument('-metric_choose', default='acc', type=str, help='which best metric choose to test')
    # 学习率，默认0.001，浮点型
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    # 处理后的数据保存路径
    parser.add_argument('-data_dir', default='./data_processed', type=str, help='the location to save processed data')

    # --- 恢复训练参数 ---
    # 从检查点恢复训练标志
    parser.add_argument('-resume', action='store_true', help='resume from checkpoint')
    # 恢复的起始轮数，默认0
    parser.add_argument('-resume_epoch', default=0, type=int, help='resume epoch')
    # 检查点文件路径
    parser.add_argument('-checkpoint', default=None, type=str, help='checkpoint')

    # --- 模型参数 ---
    # 模型名称，默认DGCNN（动态图卷积神经网络）
    parser.add_argument('-model', default='DGCNN', type=str, help='Name of model to train')

    # --- 日志参数 ---
    # 日志目录，默认'./log/'
    parser.add_argument('-log_dir', default='./log/', help='location of log dir')
    # 输出目录，默认'./result/'
    parser.add_argument('-output_dir', default='./result/', help='location of output dir')
    # 当前时间，使用time.localtime()获取本地时间
    parser.add_argument('-time', default=time.localtime(), help='the time now')

    # --- 预设参数 ---
    # 预设设置选择，从preset_setting中选择
    parser.add_argument('-setting', default='seed_sub_dependent_front_back_setting', choices=preset_setting,
                        help='using preset setting')

    # --- 数据集参数 ---
    # 数据集选择，从available_dataset中选择
    parser.add_argument('-dataset', default='seed_de_lds', type=str, choices=available_dataset,
                        help=f'available dataset are {available_dataset}')
    # 数据集原始文件路径
    parser.add_argument('-dataset_path', default='/LibEER/data_utils/Dataset/SEED', type=str,
                        help='the location of dataset')
    # 带通滤波器低频截止频率，默认0.3Hz
    parser.add_argument('-low_pass', default=0.3, type=float, help='the minimum frequency of bandpass filter')
    # 带通滤波器高频截止频率，默认50Hz
    parser.add_argument('-high_pass', default=50, type=float, help='the maximum frequency of bandpass filter')
    # 预处理时间窗口长度，单位秒，默认1秒
    parser.add_argument('-time_window', default=1, type=float,
                        help='the num of sample points of preprocessing time window/s')
    # 时间窗口重叠长度，单位秒，默认0秒（无重叠）
    parser.add_argument('-overlap', default=0, type=float,
                        help='the length of overlap for each pretreatment/s')
    # 每个样本的序列长度，默认1
    parser.add_argument('-sample_length', default=1, type=int,
                        help='sequence length of each sample')
    # 滑动窗口的步长，用于数据提取
    parser.add_argument('-stride', default=1, type=int,
                        help='the stride of a sliding window for data extraction')
    # 特征类型，默认'de_lds'（微分熵+局部动态系统？）
    parser.add_argument('-feature_type', default='de_lds', type=str,
                        help='the feature type need to compute')
    # 是否清除EOG（眼电）信号标志
    parser.add_argument('-eog_clean', default=False, action='store_true',
                        help='whether clean eog')
    # 是否仅进行数据分割标志
    parser.add_argument('-only_seg', default=False, action='store_true',
                        help='whether only segment data')
    # 是否保存处理后的数据标志
    parser.add_argument('-save_data', default=False, action='store_true',
                        help='if save processed data')
    # 是否进行数据归一化，默认True（注意：这里没有指定类型，可能是布尔值）
    parser.add_argument('-normalize', default=True, )

    # --- 训练测试分割参数 ---
    # 是否使用跨试验设置，默认'true'（字符串类型）
    parser.add_argument('-cross_trail', default='true', type=str,
                        help="whether use cross-trail setting")
    # 实验模式，默认'subject-dependent'（被试内设计）
    parser.add_argument('-experiment_mode', default='subject-dependent', type=str,
                        help='which experiment mode be selected')
    # 数据分割类型，三选一：kfold, leave-one-out, front-back
    parser.add_argument('-split_type', default='front-back', type=str,
                        choices=['kfold', 'leave-one-out', 'front-back'],
                        help="choose which method to split dataset")
    # K折交叉验证的折数，默认5
    parser.add_argument('-fold_num', default=5, type=int, help='the number of folds')
    # 使用K折分割时是否打乱数据，默认'true'
    parser.add_argument('-fold_shuffle', default='true', type=str,
                        help='whether shuffle when using k-fold split')
    # 前向分割中前几个数据集作为训练集，默认9
    parser.add_argument('-front', default=9, type=int,
                        help='convert the first few data sets into training sets')
    # 使用哪些会话（session）进行训练，nargs='+'表示可接受多个整数值
    parser.add_argument('-sessions', default=None, type=int, nargs='+',
                        help="which sessions used to train")
    # 测试集比例，默认0.2
    parser.add_argument('-test_size', default=0.2, type=float,
                        help="the ratio of the test dataset")
    # 验证集比例，默认0.2
    parser.add_argument('-val_size', default=0.2, type=float,
                        help="the ratio of the val dataset")
    # 主要轮次（primary rounds）选择，用于某些特定的实验设计
    parser.add_argument('-pr', default=None, type=int, nargs='+',
                        help="which primary rounds to train")
    # 次要轮次（secondary rounds）选择
    parser.add_argument('-sr', default=None, type=int, nargs='+',
                        help="which secondary rounds to train")
    # 情感得分边界：[低分界，高分界]，用于将连续情感得分转为分类标签
    parser.add_argument('-bounds', default=None, type=float, nargs='+',
                        help="emotion score bounds:[low, high]")
    # 是否使用one-hot编码标签，默认True
    parser.add_argument('-onehot', default=True, action='store_true',
                        help="if use onehot code")
    # 使用哪些情感维度作为标签，可选：valence, arousal, dominance, liking
    parser.add_argument('-label_used', default=None, type=str, nargs='+',
                        help="valence, arousal, dominance, liking")
    # 是否保持维度，默认False
    parser.add_argument('-keep_dim', default=False, action='store_true')

    # 返回配置好的参数解析器
    return parser