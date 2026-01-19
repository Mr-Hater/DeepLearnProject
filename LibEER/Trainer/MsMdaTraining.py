import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm  # 进度条显示
import torch.nn.functional as F  # 神经网络函数
import torch.nn as nn  # 神经网络模块
import math  # 数学函数

from LibEER.utils.metric import Metric  # 自定义的评估指标模块
from LibEER.utils.store import save_state  # 模型保存模块


def train(model, datasets_train, dataset_val, dataset_test, samples_source, device,
          output_dir=None, metrics=['acc'], metric_choose=None, optimizer=None,
          scheduler=None, batch_size=16, epochs=40, criterion=None, loss_func=None,
          loss_param=None):
    """
    多源多领域自适应(MsMDA)训练函数
    适用于从多个源域向目标域迁移学习的场景

    参数:
        model: 多领域自适应模型
        datasets_train: 多个源域训练数据集的列表
        dataset_val: 目标域验证数据集（用于领域自适应）
        dataset_test: 目标域测试数据集
        samples_source: 源域样本总数（用于计算迭代次数）
        device: 训练设备（CPU/GPU）
        output_dir: 模型保存路径
        metrics: 评估指标列表，默认['acc']
        metric_choose: 用于选择最佳模型的主要指标
        optimizer: 优化器
        scheduler: 学习率调度器
        batch_size: 批大小，默认16
        epochs: 训练轮数，默认40
        criterion: 损失函数
        loss_func: 额外的损失函数
        loss_param: 额外损失函数的参数

    返回:
        metric_value: 测试集上的评估结果
    """

    # 设置默认的评估指标
    if metrics is None:
        metrics = ['acc']
    if metric_choose is None:
        metric_choose = metrics[0]  # 默认使用第一个指标

    # ==================== 数据加载器准备 ====================
    # 为每个源域创建数据加载器
    source_loaders = []
    for j, dataset_train in enumerate(datasets_train):
        sampler_train = RandomSampler(dataset_train)
        source_loaders.append(DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=batch_size, num_workers=4, drop_last=True
        ))

    # 为目标域验证集和测试集创建数据加载器
    sampler_test = SequentialSampler(dataset_test)
    sampler_val = SequentialSampler(dataset_val)

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=batch_size, num_workers=4, drop_last=True
    )
    data_loader_test = DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=batch_size, num_workers=4, drop_last=True
    )

    # 将模型移动到指定设备
    model = model.to(device)

    # 初始化最佳指标记录
    best_metric = {s: 0. for s in metrics}

    # ==================== 迭代次数计算 ====================
    # 计算每个epoch的迭代次数（基于源域样本数）
    iteration = math.ceil(samples_source / batch_size)
    iterations = epochs * iteration  # 总迭代次数
    log_interval = 10  # 日志记录间隔

    # ==================== 迭代器初始化 ====================
    # 创建目标域数据迭代器
    target_iter = iter(data_loader_val)

    # 为每个源域创建数据迭代器
    source_iters = []
    for i in range(len(source_loaders)):
        source_iters.append(iter(source_loaders[i]))

    # ==================== 主训练循环 ====================
    for epoch in range(epochs):
        # 创建进度条，显示当前训练轮次
        _tqdm = tqdm(range(iteration), desc=f"训练轮次 {epoch + 1}/{epochs}", leave=False)

        # 每个epoch中的迭代循环
        for idx in _tqdm:
            model.train()  # 设置模型为训练模式

            # ==================== 遍历所有源域 ====================
            for j in range(len(source_iters)):
                try:
                    # 尝试从当前源域获取下一批数据
                    source_data, source_label = next(source_iters[j])
                except Exception as err:
                    # 如果当前源域数据已遍历完，重新创建迭代器
                    source_iters[j] = iter(source_loaders[j])
                    source_data, source_label = next(source_iters[j])

                try:
                    # 尝试从目标域获取下一批数据（仅数据，无标签）
                    target_data, _ = next(target_iter)
                except Exception as err:
                    # 如果目标域数据已遍历完，重新创建迭代器
                    target_iter = iter(data_loader_val)
                    target_data, _ = next(target_iter)

                # 将数据移动到指定设备
                source_data, source_label = source_data.to(device), source_label.to(device)
                target_data = target_data.to(device)

                optimizer.zero_grad()  # 清空梯度

                # ==================== 前向传播与多任务损失计算 ====================
                # 模型返回三个损失项：
                # 1. cls_loss: 分类损失（源域监督学习）
                # 2. mmd_loss: 最大均值差异损失（领域对齐）
                # 3. l1_loss: L1正则化损失（稀疏性约束）
                cls_loss, mmd_loss, l1_loss = model(
                    source_data,
                    number_of_source=len(source_loaders),
                    data_tgt=target_data,
                    label_src=source_label,
                    mark=j  # 标记当前是第几个源域
                )

                # ==================== 动态权重调整 ====================
                # gamma: MMD损失的动态权重，随着训练进行逐渐增加
                # 使用Sigmoid曲线变化：从0逐渐增加到接近1
                gamma = 2 / (1 + math.exp(-10 * (epoch * iteration + idx) / iterations)) - 1

                # beta: L1损失的权重，设置为gamma的1%
                beta = gamma / 100

                # ==================== 总损失计算 ====================
                # 总损失 = 分类损失 + gamma * MMD损失 + beta * L1损失
                loss = cls_loss + gamma * mmd_loss + beta * l1_loss

                # ==================== 反向传播与优化 ====================
                loss.backward()
                optimizer.step()

                # 更新进度条显示
                _tqdm.set_postfix_str(f"loss: {loss.item():.2f}")

            # ==================== 验证集评估 ====================
            # 在验证集上评估模型性能
            metric_value = evaluate(
                model, data_loader_val, device, metrics,
                nn.NLLLoss(), source_num=len(source_loaders)
            )

            # ==================== 模型保存 ====================
            # 检查并保存最佳模型
            for m in metrics:
                if metric_value[m] > best_metric[m]:
                    best_metric[m] = metric_value[m]
                    save_state(output_dir, model, optimizer, epoch + 1, metric=m)

    # ==================== 最终评估 ====================
    # 加载基于主要指标的最佳模型
    model.load_state_dict(torch.load(f"{output_dir}/checkpoint-best{metric_choose}")['model'])

    # 在测试集上评估最佳模型
    metric_value = evaluate(
        model, data_loader_test, device, metrics,
        criterion, source_num=len(source_loaders)
    )

    # 打印最佳指标结果
    for m in metrics:
        print(f"最佳验证集_{m}: {best_metric[m]:.2f}")
        print(f"最佳测试集_{m}: {metric_value[m]:.2f}")

    return metric_value


@torch.no_grad()  # 禁用梯度计算，节省内存
def evaluate(model, data_loader_test, device, metrics, criterion, source_num,
             loss_func=None, loss_param=None):
    """
    MsMDA模型评估函数，用于验证和测试阶段

    参数:
        model: 要评估的多领域自适应模型
        data_loader_test: 测试数据加载器
        device: 评估设备
        metrics: 评估指标列表
        criterion: 损失函数
        source_num: 源域数量
        loss_func: 额外的损失函数
        loss_param: 额外损失函数的参数

    返回:
        metric.values: 评估指标结果字典
    """

    model.eval()  # 设置模型为评估模式

    # 创建评估指标对象
    metric = Metric(metrics)

    # 遍历数据批次进行评估
    for idx, (data, target) in tqdm(
            enumerate(data_loader_test),
            total=len(data_loader_test),
            desc=f"评估中 : ",
            leave=False
    ):
        # 将数据移动到指定设备
        data = data.to(device)
        target = target.to(device)

        # 前向传播：获取所有源域的预测结果
        preds = model(data, source_num)

        # ==================== 预测结果融合 ====================
        # 对每个源域的预测结果进行Softmax归一化
        for i in range(len(preds)):
            preds[i] = F.softmax(preds[i], dim=1)

        # 融合策略：取所有源域预测结果的平均值
        pred = sum(preds) / len(preds)

        # ==================== 损失计算 ====================
        # 使用负对数似然损失（需要先进行log_softmax）
        test_loss = criterion(
            F.log_softmax(pred, dim=1),
            target.long().squeeze()
        )

        # 更新评估指标
        metric.update(
            torch.argmax(pred, dim=1),
            target.data.squeeze(),
            test_loss.item()
        )

    # 打印评估结果（蓝色显示）
    print("\033[34m eval state: " + metric.value())

    return metric.values