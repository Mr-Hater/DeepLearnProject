class ModelConfig:
    """模型训练配置类，包含args.py中的所有参数"""

    def __init__(self):
        # ========== 训练参数部分 (第19-39行) ==========
        self.batch_size = 128
        self.epochs = 40
        self.device = 'cuda'
        self.eval = False
        self.seed = 1
        self.num_workers = 4
        self.loss_func = 'crossEntropyLoss'
        self.metrics = ['acc']
        self.metric_choose = 'acc'
        self.lr = 0.001
        self.data_dir = './data_processed'

        # ========== 恢复训练参数 (第42-47行) ==========
        self.resume = False
        self.resume_epoch = 0
        self.checkpoint = None

        # ========== 模型参数 (第50-53行) ==========
        self.model = 'DGCNN'

        # ========== 日志参数 (第56-61行) ==========
        self.log_dir = './log/'
        self.output_dir = './result/'

        # ========== 预设参数 (第64-67行) ==========
        self.setting = 'seed_sub_dependent_front_back_setting'

        # ========== 数据集参数 (第70-109行) ==========
        self.dataset = 'seed_de_lds'
        self.dataset_path = 'D:/Work/Studio/PyCharm/PythonProject/DeepLearn/LibEER/data_utils/Dataset/SEED'
        self.low_pass = 0.3
        self.high_pass = 50
        self.time_window = 1
        self.overlap = 0
        self.sample_length = 1
        self.stride = 1
        self.feature_type = 'de_lds'
        self.eog_clean = False
        self.only_seg = False
        self.save_data = False
        self.normalize = True  # 注意：args.py中这里没有指定类型，只有default=True

        # ========== 训练测试分割参数 (第112-148行) ==========
        self.cross_trail = 'true'
        self.experiment_mode = 'subject-dependent'
        self.split_type = 'front-back'
        self.fold_num = 5
        self.fold_shuffle = 'true'
        self.front = 9
        self.sessions = None
        self.test_size = 0.2
        self.val_size = 0.2
        self.pr = None
        self.sr = None
        self.bounds = None
        self.onehot = False  # 注意：args.py中这里是action='store_true'，默认为False
        self.label_used = None
        self.keep_dim = False

        import time
        # 添加可能遗漏的：args.py第27行
        self.time = time.localtime()

    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"ModelConfig没有属性 '{key}'。有效的属性有：{', '.join(self.__dict__.keys())}")

    def __repr__(self):
        """打印所有配置"""
        attrs = []
        for key, value in sorted(self.__dict__.items()):
            attrs.append(f"  {key}: {value}")
        return "ModelConfig:\n" + "\n".join(attrs)