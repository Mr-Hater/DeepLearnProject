class Setting:
    """
    实验设置配置类，用于管理EEG情感识别实验的所有参数

    这个类集中管理数据预处理、实验设计、训练测试分割等所有配置参数，
    确保实验的可重复性和配置的统一性。
    """

    def __init__(self, dataset, dataset_path, pass_band, extract_bands, time_window, overlap, sample_length, stride,
                 seed,
                 feature_type, only_seg=False, cross_trail='true', experiment_mode="subject-dependent", train_part=None,
                 eog_clean=True,
                 metrics=None, normalize=False, save_data=True, split_type="kfold", fold_num=5, fold_shuffle=True,
                 front=9, test_size=0.2, val_size=0.2, sessions=None, pr=None, sr=None, bounds=None,
                 onehot=False, label_used=None):
        """
        初始化实验设置

        参数:
            dataset: 数据集名称 (如 'seed', 'seediv', 'deap', 'hci', 'dreamer')
            dataset_path: 数据集文件路径
            pass_band: 带通滤波器范围 [低频, 高频]
            extract_bands: 要提取的频段列表，每个元素是[低频, 高频]
            time_window: 时间窗口大小（数据点数）
            overlap: 时间窗口重叠长度（数据点数）
            sample_length: 输入模型的样本序列长度
            stride: 滑动窗口的步长
            seed: 随机种子
            feature_type: EEG信号特征类型
            only_seg: 是否只进行数据分段
            cross_trail: 是否跨试次实验
            experiment_mode: 实验模式，可选：
                - "subject-dependent": 被试依赖（同一被试数据分割）
                - "subject-independent": 被试独立（不同被试数据分割）
                - "cross-session": 跨会话实验
            train_part: 训练部分指定
            eog_clean: 是否去除眼电伪影
            metrics: 评估指标列表
            normalize: 是否进行数据归一化
            save_data: 是否保存预处理数据
            split_type: 数据分割类型，可选：
                - "kfold": K折交叉验证
                - "front-back": 前部作为训练，后部作为测试
                - "early-stop": 训练-验证-测试分割
                - "leave-one-out": 留一法
            fold_num: K折交叉验证的折数
            fold_shuffle: 是否打乱数据
            front: 前部样本数（用于front-back分割）
            test_size: 测试集比例
            val_size: 验证集比例
            sessions: 会话列表
            pr: 预处理相关参数
            sr: 采样率
            bounds: 数据边界限制
            onehot: 是否使用one-hot编码标签
            label_used: 使用的标签类型
        """

        # ==================== 随机种子设置 ====================
        # 设置随机种子以确保实验可重复性
        self.seed = seed

        # ==================== 数据集设置 ====================
        # 数据集名称和路径
        self.dataset = dataset  # 数据集标识符
        self.dataset_path = dataset_path  # 数据文件存储路径

        # ==================== 数据预处理设置 ====================
        # 带通滤波器设置：索引0和1分别代表滤波器的下限和上限频率
        self.pass_band = pass_band

        # 频段提取设置：二维数组，每个元素代表一个要提取的频段范围
        # 例如：[[4, 7], [8, 13]] 表示提取θ波和α波
        self.extract_bands = extract_bands if extract_bands is None else extract_bands

        # 时间窗口设置：预处理时的时间窗口大小（数据点数量）
        self.time_window = time_window

        # 重叠设置：相邻时间窗口之间的重叠长度（数据点数量）
        self.overlap = overlap

        # 样本长度：一次输入模型的样本序列长度
        self.sample_length = sample_length

        # 滑动窗口步长：用于数据提取的滑动窗口移动步长
        self.stride = stride

        # 特征类型：EEG信号的特征提取方法
        self.feature_type = feature_type

        # 眼电伪影去除：是否移除眼动干扰
        self.eog_clean = eog_clean

        # 数据归一化：是否对数据进行归一化处理
        self.normalize = normalize

        # 数据保存：是否保存预处理后的数据
        self.save_data = save_data

        # 仅分段：是否只进行数据分段而不进行其他处理
        self.only_seg = only_seg

        # ==================== 训练测试设置 ====================
        # 跨试次设置：是否使用跨试次实验设计
        self.cross_trail = cross_trail

        # 实验模式：决定如何分割数据
        #   - "subject-dependent": 被试内分割，同一被试的数据分为训练测试
        #   - "subject-independent": 被试间分割，不同被试的数据分为训练测试
        #   - "cross-session": 跨会话分割
        self.experiment_mode = experiment_mode

        # 分割类型：具体的数据分割方法
        self.split_type = split_type

        # 根据分割类型选择训练集或测试集
        self.fold_num = fold_num  # K折交叉验证的折数
        self.fold_shuffle = fold_shuffle  # 是否打乱数据
        self.front = front  # 前部样本数（用于front-back分割）
        self.test_size = test_size  # 测试集比例
        self.val_size = val_size  # 验证集比例
        self.sessions = sessions  # 会话列表
        self.pr = pr  # 预处理参数
        self.sr = sr  # 采样率

        # ==================== 其他设置 ====================
        self.bounds = bounds  # 数据边界限制（如裁剪范围）
        self.onehot = onehot  # 是否使用one-hot编码标签
        self.label_used = label_used  # 使用的标签类型

def set_setting_by_config(config):
    """
    根据配置对象创建Setting对象
    修改为接收 config 对象而不是 args 对象
    """
    # 参数验证
    if config.dataset_path is None:
        print("请设置数据集路径")
    if config.dataset is None:
        print("请选择要训练的数据集")

    # 创建Setting对象
    return Setting(
        dataset=config.dataset,
        dataset_path=config.dataset_path,
        pass_band=[config.low_pass, config.high_pass],
        extract_bands=None,
        time_window=config.time_window,
        overlap=config.overlap,
        sample_length=config.sample_length,
        stride=config.stride,
        seed=config.seed,
        feature_type=config.feature_type,
        only_seg=config.only_seg,
        cross_trail=config.cross_trail,
        experiment_mode=config.experiment_mode,
        metrics=config.metrics,
        normalize=config.normalize,
        split_type=config.split_type,
        fold_num=config.fold_num,
        fold_shuffle=config.fold_shuffle,
        front=config.front,
        sessions=config.sessions,
        pr=config.pr,
        sr=config.sr,
        bounds=config.bounds,
        onehot=config.onehot,
        label_used=config.label_used
    )


def set_setting_by_args(args):
    """
    根据命令行参数创建Setting对象

    参数:
        args: 命令行参数对象

    返回:
        Setting: 配置好的Setting对象
    """
    # 参数验证
    if args.dataset_path is None:
        print("请设置数据集路径")
    if args.dataset is None:
        print("请选择要训练的数据集")

    # 创建Setting对象
    return Setting(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        pass_band=[args.low_pass, args.high_pass],
        extract_bands=None,
        time_window=args.time_window,
        overlap=args.overlap,
        sample_length=args.sample_length,
        stride=args.stride,
        seed=args.seed,
        feature_type=args.feature_type,
        only_seg=args.only_seg,
        cross_trail=args.cross_trail,
        experiment_mode=args.experiment_mode,
        metrics=args.metrics,
        normalize=args.normalize,
        split_type=args.split_type,
        fold_num=args.fold_num,
        fold_shuffle=args.fold_shuffle,
        front=args.front,
        sessions=args.sessions,
        pr=args.pr,
        sr=args.sr,
        bounds=args.bounds,
        onehot=args.onehot,
        label_used=args.label_used
    )


def seed_sub_dependent_front_back_setting(args):
    """
    SEED数据集被试依赖实验模式 - 前部训练后部测试

    说明: 对每个被试，前9个试次作为训练集，后6个试次作为测试集
    """
    if not args.dataset.startswith('seed'):
        print('未使用SEED数据集，请检查设置')
        exit(1)

    print("使用默认SEED被试依赖实验模式，\n"
          "每个被试的前9个试次作为训练集，后6个试次作为测试集")

    return Setting(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        pass_band=[args.low_pass, args.high_pass],
        extract_bands=None,
        time_window=args.time_window,
        overlap=args.overlap,
        sample_length=args.sample_length,
        stride=args.stride,
        seed=args.seed,
        feature_type=args.feature_type,
        only_seg=args.only_seg,
        experiment_mode="subject-dependent",
        normalize=args.normalize,
        split_type='front-back',
        front=9,
        sessions=args.sessions,
        pr=args.pr,
        sr=args.sr,
        onehot=args.onehot,
        label_used=args.label_used
    )


def seed_sub_dependent_train_val_test_setting(args):
    """
    SEED数据集被试依赖实验模式 - 训练验证测试分割

    说明: 对每个被试，随机选择9个试次作为训练集，3个试次作为验证集，
          后3个试次作为测试集，在验证集上选择最佳结果，在测试集上测试
    """
    if not args.dataset.startswith('seed'):
        print('未使用SEED数据集，请检查设置')
        exit(1)

    print("使用SEED被试依赖训练验证测试实验模式，\n"
          "对每个被试，随机9个试次作为训练集，随机3个试次作为验证集，\n"
          "后3个试次作为测试集，在验证集上选择最佳结果，在测试集上测试")

    return Setting(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        pass_band=[args.low_pass, args.high_pass],
        extract_bands=None,
        time_window=args.time_window,
        overlap=args.overlap,
        sample_length=args.sample_length,
        stride=args.stride,
        seed=args.seed,
        feature_type=args.feature_type,
        only_seg=args.only_seg,
        experiment_mode="subject-dependent",
        normalize=args.normalize,
        split_type='early-stop',
        test_size=0.2,
        val_size=0.2,
        sessions=args.sessions,
        pr=args.pr,
        sr=args.sr,
        onehot=args.onehot,
        label_used=args.label_used
    )


def seediv_sub_dependent_train_val_test_setting(args):
    """
    SEED-IV数据集被试依赖实验模式 - 训练验证测试分割

    说明: 对每个被试，随机选择16个试次作为训练集，4个试次作为验证集，
          后4个试次作为测试集
    """
    if not args.dataset.startswith('seediv'):
        print('未使用SEED IV数据集，请检查设置')
        exit(1)

    print("使用SEED IV被试依赖早停实验模式，\n"
          "对每个被试，随机16个试次作为训练集，4个试次作为验证集，\n"
          "后4个试次作为测试集，在验证集上选择最佳结果，在测试集上测试")

    return Setting(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        pass_band=[args.low_pass, args.high_pass],
        extract_bands=None,
        time_window=args.time_window,
        overlap=args.overlap,
        sample_length=args.sample_length,
        stride=args.stride,
        seed=args.seed,
        feature_type=args.feature_type,
        only_seg=args.only_seg,
        experiment_mode="subject-dependent",
        normalize=args.normalize,
        split_type='early-stop',
        test_size=0.2,
        val_size=0.2,
        sessions=args.sessions,
        pr=args.pr,
        sr=args.sr,
        onehot=args.onehot,
        label_used=args.label_used
    )


def seed_sub_dependent_5fold_setting(args):
    """
    SEED数据集被试依赖实验模式 - 5折交叉验证

    说明: 使用5折交叉验证，按试次顺序分组测试集
    """
    if not args.dataset.startswith('seed'):
        print('未使用SEED数据集，请检查设置')
        exit(1)

    print("使用默认SEED被试依赖实验模式，\n"
          "使用5折交叉验证，按试次顺序分组测试集")

    return Setting(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        pass_band=[args.low_pass, args.high_pass],
        extract_bands=None,
        time_window=args.time_window,
        overlap=args.overlap,
        sample_length=args.sample_length,
        stride=args.stride,
        seed=args.seed,
        feature_type=args.feature_type,
        only_seg=args.only_seg,
        cross_trail=args.cross_trail,
        experiment_mode="subject-dependent",
        normalize=args.normalize,
        split_type='kfold',
        fold_num=5,
        fold_shuffle=False,
        sessions=args.sessions,
        pr=args.pr,
        sr=args.sr,
        onehot=args.onehot,
        label_used=args.label_used
    )


def seed_sub_independent_leave_one_out_setting(args):
    """
    SEED数据集被试独立实验模式 - 留一法

    说明: 使用留一法，1个被试的所有试次作为测试集，
          其他14个被试的所有试次作为训练集，循环15次报告平均结果
    """
    if not args.dataset.startswith('seed'):
        print('未使用SEED数据集，请检查设置')
        exit(1)

    print("使用默认SEED被试独立早停实验模式，\n"
          "使用留一法，1个被试的所有试次作为测试集，\n"
          "其他14个被试的所有试次作为训练集，循环15次报告平均结果")

    return Setting(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        pass_band=[args.low_pass, args.high_pass],
        extract_bands=None,
        time_window=args.time_window,
        overlap=args.overlap,
        sample_length=args.sample_length,
        stride=args.stride,
        seed=args.seed,
        feature_type=args.feature_type,
        only_seg=args.only_seg,
        experiment_mode="subject-independent",
        normalize=args.normalize,
        split_type='leave-one-out',
        sessions=[1] if args.sessions is None else args.sessions,
        pr=args.pr,
        sr=args.sr,
        onehot=args.onehot,
        label_used=args.label_used
    )


def seed_sub_independent_train_val_test_setting(args):
    """
    SEED数据集被试独立实验模式 - 训练验证测试分割

    说明: 随机选择9个被试的数据作为训练集，3个被试作为验证集，
          3个被试作为测试集，在验证集上选择最佳模型
    """
    if not args.dataset.startswith('seed'):
        print('未使用SEED数据集，请检查设置')
        exit(1)

    print("使用早停SEED被试独立实验模式，\n"
          "随机9个被试的数据作为训练集，随机3个被试的数据作为验证集，\n"
          "随机3个被试的数据作为测试集，在验证集上选择最佳结果，在测试集上测试")

    return Setting(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        pass_band=[args.low_pass, args.high_pass],
        extract_bands=None,
        time_window=args.time_window,
        overlap=args.overlap,
        sample_length=args.sample_length,
        stride=args.stride,
        seed=args.seed,
        feature_type=args.feature_type,
        only_seg=args.only_seg,
        experiment_mode="subject-independent",
        normalize=args.normalize,
        split_type='early-stop',
        test_size=0.2,
        val_size=0.2,
        sessions=[1] if args.sessions is None else args.sessions,
        pr=args.pr,
        sr=args.sr,
        onehot=args.onehot,
        label_used=args.label_used
    )


def hci_sub_dependent_train_val_test_setting(args):
    """
    HCI数据集被试依赖实验模式 - 训练验证测试分割
    """
    if not args.dataset.startswith('hci'):
        print('未使用HCI数据集，请检查设置')
        exit(1)

    print("使用HCI被试依赖早停实验模式")

    return Setting(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        pass_band=[args.low_pass, args.high_pass],
        extract_bands=None,
        time_window=args.time_window,
        overlap=args.overlap,
        sample_length=args.sample_length,
        stride=args.stride,
        seed=args.seed,
        feature_type=args.feature_type,
        only_seg=args.only_seg,
        experiment_mode="subject-dependent",
        normalize=args.normalize,
        split_type='early-stop',
        test_size=0.2,
        val_size=0.2,
        sessions=args.sessions,
        pr=args.pr,
        sr=args.sr,
        onehot=args.onehot,
        bounds=args.bounds,
        label_used=args.label_used
    )


def seediv_sub_independent_train_val_test_setting(args):
    """
    SEED-IV数据集被试独立实验模式 - 训练验证测试分割
    """
    if not args.dataset.startswith('seediv'):
        print('未使用SEED IV数据集，请检查设置')
        exit(1)

    print("使用SEED IV被试独立早停实验模式")

    return Setting(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        pass_band=[args.low_pass, args.high_pass],
        extract_bands=None,
        time_window=args.time_window,
        overlap=args.overlap,
        sample_length=args.sample_length,
        stride=args.stride,
        seed=args.seed,
        feature_type=args.feature_type,
        only_seg=args.only_seg,
        experiment_mode="subject-independent",
        normalize=args.normalize,
        split_type='early-stop',
        test_size=0.2,
        val_size=0.2,
        sessions=[1] if args.sessions is None else args.sessions,
        pr=args.pr,
        sr=args.sr,
        onehot=args.onehot,
        label_used=args.label_used
    )


def deap_sub_independent_train_val_test_setting(args):
    """
    DEAP数据集被试独立实验模式 - 训练验证测试分割
    """
    if not args.dataset.startswith('deap'):
        print('未使用DEAP数据集，请检查设置')
        exit(1)

    return Setting(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        pass_band=[args.low_pass, args.high_pass],
        extract_bands=None,
        time_window=args.time_window,
        overlap=args.overlap,
        sample_length=args.sample_length,
        stride=args.stride,
        seed=args.seed,
        feature_type=args.feature_type,
        only_seg=args.only_seg,
        experiment_mode="subject-independent",
        normalize=args.normalize,
        split_type='early-stop',
        test_size=0.2,
        val_size=0.2,
        sessions=args.sessions,
        pr=args.pr,
        sr=args.sr,
        onehot=args.onehot,
        bounds=args.bounds,
        label_used=args.label_used
    )


def hci_sub_independent_train_val_test_setting(args):
    """
    HCI数据集被试独立实验模式 - 训练验证测试分割
    """
    if not args.dataset.startswith('hci'):
        print('未使用HCI数据集，请检查设置')
        exit(1)

    print("使用HCI被试独立早停实验模式")

    return Setting(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        pass_band=[args.low_pass, args.high_pass],
        extract_bands=None,
        time_window=args.time_window,
        overlap=args.overlap,
        sample_length=args.sample_length,
        stride=args.stride,
        seed=args.seed,
        feature_type=args.feature_type,
        only_seg=args.only_seg,
        experiment_mode="subject-independent",
        normalize=args.normalize,
        split_type='early-stop',
        test_size=0.2,
        val_size=0.2,
        sessions=args.sessions,
        pr=args.pr,
        sr=args.sr,
        onehot=args.onehot,
        bounds=args.bounds,
        label_used=args.label_used
    )


def deap_sub_dependent_train_val_test_setting(args):
    """
    DEAP数据集被试依赖实验模式 - 训练验证测试分割
    """
    if not args.dataset.startswith('deap'):
        print('未使用DEAP数据集，请检查设置')
        exit(1)

    print("使用DEAP被试依赖早停实验模式")

    return Setting(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        pass_band=[args.low_pass, args.high_pass],
        extract_bands=None,
        time_window=args.time_window,
        overlap=args.overlap,
        sample_length=args.sample_length,
        stride=args.stride,
        seed=args.seed,
        feature_type=args.feature_type,
        only_seg=args.only_seg,
        experiment_mode="subject-dependent",
        normalize=args.normalize,
        split_type='early-stop',
        test_size=0.2,
        val_size=0.2,
        sessions=args.sessions,
        pr=args.pr,
        sr=args.sr,
        onehot=args.onehot,
        bounds=args.bounds,
        label_used=args.label_used
    )


def seed_cross_session_setting(args):
    """
    SEED数据集跨会话实验模式

    说明: 三个会话的数据，一个作为测试数据集
    """
    if not args.dataset.startswith('seed'):
        print('未使用SEED数据集，请检查设置')
        exit(1)

    print("使用默认SEED跨会话实验模式，\n"
          "三个会话的数据，一个作为测试数据集")

    return Setting(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        pass_band=[args.low_pass, args.high_pass],
        extract_bands=None,
        time_window=args.time_window,
        overlap=args.overlap,
        sample_length=args.sample_length,
        stride=args.stride,
        seed=args.seed,
        feature_type=args.feature_type,
        only_seg=args.only_seg,
        experiment_mode="cross-session",
        normalize=args.normalize,
        split_type='leave-one-out',
        sessions=args.sessions,
        pr=args.pr,
        sr=args.sr,
        onehot=args.onehot,
        label_used=args.label_used
    )


def deap_sub_independent_leave_one_out_setting(args):
    """
    DEAP数据集被试独立实验模式 - 留一法
    """
    if not args.dataset.startswith('deap'):
        print('未使用DEAP数据集，请检查设置')
        exit(1)

    print("使用默认DEAP被试独立实验模式")

    return Setting(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        pass_band=[args.low_pass, args.high_pass],
        extract_bands=[[4, 7], [8, 10], [8, 12], [13, 30], [30, 47]],
        time_window=args.time_window,
        overlap=args.overlap,
        sample_length=args.sample_length,
        stride=args.stride,
        seed=args.seed,
        feature_type=args.feature_type,
        only_seg=args.only_seg,
        experiment_mode="subject-independent",
        normalize=args.normalize,
        split_type='leave-one-out',
        pr=args.pr,
        sr=args.sr,
        bounds=args.bounds,
        onehot=args.onehot,
        label_used=args.label_used
    )


def deap_sub_dependent_10fold_setting(args):
    """
    DEAP数据集被试依赖实验模式 - 10折交叉验证
    """
    if not args.dataset.startswith('deap'):
        print('未使用DEAP数据集，请检查设置')
        exit(1)

    print("使用默认DEAP被试依赖实验模式")

    return Setting(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        pass_band=[args.low_pass, args.high_pass],
        extract_bands=[[4, 7], [8, 10], [8, 12], [13, 30], [30, 47]],
        time_window=args.time_window,
        overlap=args.overlap,
        sample_length=args.sample_length,
        stride=args.stride,
        seed=args.seed,
        feature_type=args.feature_type,
        only_seg=args.only_seg,
        experiment_mode="subject-dependent",
        normalize=args.normalize,
        cross_trail=args.cross_trail,
        split_type='kfold',
        fold_num=10,
        pr=args.pr,
        sr=args.sr,
        bounds=args.bounds,
        onehot=args.onehot,
        label_used=args.label_used
    )


def dreamer_sub_independent_setting(args):
    """
    Dreamer数据集被试独立实验模式
    """
    if not args.dataset.startswith('dreamer'):
        print('未使用Dreamer数据集，请检查设置')
        exit(1)

    print("使用默认Dreamer被试独立实验模式")

    return Setting(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        pass_band=[args.low_pass, args.high_pass],
        extract_bands=[[4, 7], [8, 13], [14, 30]],
        time_window=args.time_window,
        overlap=args.overlap,
        sample_length=args.sample_length,
        stride=args.stride,
        seed=args.seed,
        feature_type=args.feature_type,
        only_seg=args.only_seg,
        experiment_mode="subject-independent",
        normalize=args.normalize,
        split_type='leave-one-out',
        pr=args.pr,
        sr=args.sr,
        bounds=args.bounds,
        onehot=args.onehot,
        label_used=args.label_used
    )


def dreamer_sub_dependent_setting(args):
    """
    Dreamer数据集被试依赖实验模式
    """
    if not args.dataset.startswith('dreamer'):
        print('未使用Dreamer数据集，请检查设置')
        exit(1)

    print("使用默认Dreamer被试依赖实验模式")

    return Setting(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        pass_band=[args.low_pass, args.high_pass],
        extract_bands=[[4, 7], [8, 13], [14, 30]],
        time_window=args.time_window,
        overlap=args.overlap,
        sample_length=args.sample_length,
        stride=args.stride,
        seed=args.seed,
        feature_type=args.feature_type,
        only_seg=args.only_seg,
        experiment_mode="subject-dependent",
        normalize=args.normalize,
        cross_trail=args.cross_trail,
        split_type='leave-one-out',
        pr=args.pr,
        sr=args.sr,
        bounds=args.bounds,
        onehot=args.onehot,
        label_used=args.label_used
    )


# ==================== 预设设置字典 ====================
# 将设置函数名映射到对应的函数，便于通过名称快速调用
preset_setting = {
    # 训练-验证-测试分割设置（主流设置）
    "seed_sub_dependent_train_val_test_setting": seed_sub_dependent_train_val_test_setting,
    "seediv_sub_dependent_train_val_test_setting": seediv_sub_dependent_train_val_test_setting,
    "seed_sub_independent_train_val_test_setting": seed_sub_independent_train_val_test_setting,
    "seediv_sub_independent_train_val_test_setting": seediv_sub_independent_train_val_test_setting,
    "deap_sub_dependent_train_val_test_setting": deap_sub_dependent_train_val_test_setting,
    "hci_sub_dependent_train_val_test_setting": hci_sub_dependent_train_val_test_setting,
    "deap_sub_independent_train_val_test_setting": deap_sub_independent_train_val_test_setting,
    "hci_sub_independent_train_val_test_setting": hci_sub_independent_train_val_test_setting,

    # ***********************************************************************
    # 其他实验设置
    "seed_sub_dependent_5fold_setting": seed_sub_dependent_5fold_setting,
    "seed_sub_dependent_front_back_setting": seed_sub_dependent_front_back_setting,
    "seed_sub_independent_leave_one_out_setting": seed_sub_independent_leave_one_out_setting,
    "seed_cross_session_setting": seed_cross_session_setting,
    "deap_sub_independent_leave_one_out_setting": deap_sub_independent_leave_one_out_setting,
    "deap_sub_dependent_10fold_setting": deap_sub_dependent_10fold_setting,
    "dreamer_sub_independent_setting": dreamer_sub_independent_setting,
    "dreamer_sub_dependent_setting": dreamer_sub_dependent_setting,

    # 默认设置：根据命令行参数设置
    None: set_setting_by_args
}