import numpy as np  # 数值计算库
from sklearn import svm  # scikit-learn中的支持向量机模块
from LibEER.data_utils.preprocess import normalize  # 数据预处理模块中的归一化函数
from LibEER.utils.metric import Metric  # 评估指标计算模块


class SVM:
    """
    SVM分类器类
    核心方法介绍：
    这个类封装了scikit-learn的SVC（支持向量分类器），
    使用RBF（径向基函数）核函数，适用于多分类问题
    """

    def __init__(self, num_electrodes, num_datapoints, num_classes):
        """
        初始化SVM分类器

        参数:
        num_electrodes: int - 电极数量（输入特征的第一维度）
        num_datapoints: int - 数据点数量（输入特征的第二维度）
        num_classes: int - 分类类别数量
        """
        # 创建RBF核的SVC分类器
        # kernel='rbf': 使用径向基函数核
        # C=1.0: 正则化参数，控制分类器的复杂度与训练误差的平衡
        # gamma='scale': 核函数系数，'scale'表示使用1/(n_features * X.var())作为gamma值
        self.svc = svm.SVC(kernel='rbf', C=1.0, gamma='scale')