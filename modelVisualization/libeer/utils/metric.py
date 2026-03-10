# 导入 numpy 库，用于数值计算
import numpy as np

# 导入 torch 库，PyTorch 深度学习框架
import torch

# 从 sklearn.metrics 导入常用评估指标
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


class Metric:
    """
    使用 value 类计算各种评估指标
    用于收集模型输出和真实标签，批量计算多个评估指标
    """

    def __init__(self, metrics):
        """
        初始化 Metric 类

        参数:
            metrics: list，包含需要计算的指标名称，如 ['acc', 'macro-f1']
        """
        # 用于存储计算得到的指标值，字典形式 {指标名: 值}
        self.values = {}

        # 存储所有批次的模型预测结果（累积）
        self.outputs = []

        # 存储所有批次的真实标签（累积）
        self.targets = []

        # 存储所有批次的损失值（可选）
        self.losses = []

        # 需要计算的指标列表
        self.metrics = metrics

    def accuracy(self):
        """
        计算准确率 (accuracy)

        返回:
            float: 准确率值
        """
        # 使用 sklearn 的 accuracy_score 计算准确率
        self.values['acc'] = accuracy_score(self.targets, self.outputs)
        # 计算准确率
        return self.values['acc']

    def update(self, outputs, targets, loss=None):
        """
        更新当前批次的预测结果和真实标签

        参数:
            outputs: 模型预测输出，可以是 tensor 或 numpy 数组
            targets: 真实标签，可以是 tensor 或 numpy 数组
            loss: 当前批次的损失值（可选）
        """
        # 将当前批次的预测结果和真实标签添加到累积列表中
        if torch.is_tensor(outputs):
            # 如果是 PyTorch tensor，转换为 CPU，分离计算图，再转为列表
            self.outputs += outputs.cpu().detach().tolist()
            self.targets += targets.cpu().detach().tolist()
        else:
            # 如果是 numpy 数组或其他，直接转为列表
            self.outputs += outputs.tolist()
            self.targets += targets.tolist()

        # 如果有损失值，添加到损失列表中
        if loss is not None:
            self.losses.append(loss)

    def macro_f1_score(self):
        """
        计算宏平均 F1 分数 (macro F1-score)
        对所有类别的 F1 分数取算术平均值

        返回:
            float: 宏平均 F1 分数
        """
        # 使用 sklearn 的 f1_score，设置 average='macro' 计算宏平均
        self.values['macro-f1'] = f1_score(self.targets, self.outputs, average='macro')
        # 计算宏平均 F1 分数
        return self.values['macro-f1']

    def micro_f1_score(self):
        """
        计算微平均 F1 分数 (micro F1-score)
        先汇总所有类别的 TP/FP/FN，再计算 F1

        返回:
            float: 微平均 F1 分数
        """
        # 使用 sklearn 的 f1_score，设置 average='micro' 计算微平均
        self.values['micro-f1'] = f1_score(self.targets, self.outputs, average='micro')
        # 计算微平均 F1 分数
        return self.values['micro-f1']

    def weighted_f1_score(self):
        """
        计算加权平均 F1 分数 (weighted F1-score)
        根据每个类别的样本数量加权平均

        返回:
            float: 加权平均 F1 分数
        """
        # 使用 sklearn 的 f1_score，设置 average='weighted' 计算加权平均
        self.values['weighted-f1'] = f1_score(self.targets, self.outputs, average='weighted')
        return self.values['weighted-f1']

    def ck_coe(self):
        """
        计算 Cohen's Kappa 系数
        用于评估分类器的一致性，考虑随机猜测的影响

        返回:
            float: Cohen's Kappa 系数
        """
        # 使用 sklearn 的 cohen_kappa_score 计算 Kappa 系数
        self.values['ck'] = cohen_kappa_score(self.targets, self.outputs)
        # 计算 Kappa 系数
        return self.values['ck']

    def value(self):
        """
        计算所有指定的指标并格式化为字符串输出

        返回:
            str: 包含所有指标值和平均损失的格式化字符串
        """
        # 如果标签是 one-hot 编码格式，转换为普通标签格式
        # 检查 targets 的第一个元素是否是列表（one-hot 编码）
        if type(self.targets[0]) is list:
            try:
                # 将 one-hot 编码转换为类别索引
                # 例如 [0, 1, 0] -> 1（找到值为 1 的位置索引）
                self.targets = [self.targets[i].index(1) for i in range(len(self.targets))]
            except ValueError:
                # 如果转换失败（比如没有找到值为1的元素），返回不可用信息
                return "unavailable"

        # 定义指标名称到计算函数的映射字典
        func = {
            'acc': self.accuracy,  # 准确率
            'macro-f1': self.macro_f1_score,  # 宏平均 F1
            'micro-f1': self.micro_f1_score,  # 微平均 F1
            'ck': self.ck_coe,  # Cohen's Kappa
            'weighted-f1': self.weighted_f1_score,  # 加权平均 F1
        }

        # 构建输出字符串
        out = ""
        # 遍历所有需要计算的指标
        for m in self.metrics:
            # 调用对应的计算函数，并格式化为3位小数
            out += f"{m}: {func[m]():.3f}   "

        # 如果有损失值，计算平均损失并添加到输出字符串
        if len(self.losses) != 0:
            # 计算所有批次的平均损失，格式化为4位小数
            return out + f"loss: {sum(self.losses) / len(self.losses):.4f}"
        else:
            # 如果没有损失值，只返回指标值
            return out