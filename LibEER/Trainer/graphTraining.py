import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch_geometric.data import Data  # 用于图神经网络的数据结构
from tqdm import tqdm  # 进度条显示

from LibEER.utils.metric import Metric  # 自定义的评估指标模块
from LibEER.utils.store import save_state  # 模型保存模块


def train(model, dataset_train, dataset_val, dataset_test, edge_adj, device,
          output_dir, metrics=None, metric_choose=None, optimizer=None,
          scheduler=None, batch_size=16, epochs=40, criterion=None,
          loss_func=None, loss_param=None):
    """
    图神经网络的训练函数，支持训练、验证和测试

    参数:
        model: 要训练的模型
        dataset_train: 训练数据集
        dataset_val: 验证数据集
        dataset_test: 测试数据集
        edge_adj: 图的邻接矩阵（用于构建图结构）
        device: 训练设备（CPU/GPU）
        output_dir: 模型保存路径
        metrics: 评估指标列表，默认['acc']
        metric_choose: 用于选择最佳模型的主要指标
        optimizer: 优化器
        scheduler: 学习率调度器
        batch_size: 批大小，默认16
        epochs: 训练轮数，默认40
        criterion: 主损失函数
        loss_func: 额外的损失函数（如正则化项）
        loss_param: 额外损失函数的参数

    返回:
        metric_value: 测试集上的评估结果
    """

    # 设置默认的评估指标
    if metrics is None:
        metrics = ['acc']  # 默认使用准确率
    if metric_choose is None:
        metric_choose = metrics[0]  # 默认使用第一个指标选择最佳模型

    # 创建数据采样器
    # 训练集使用随机采样，验证和测试集使用顺序采样
    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)
    sampler_test = SequentialSampler(dataset_test)

    # 创建数据加载器
    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train, batch_size=batch_size, num_workers=4
    )
    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val, batch_size=batch_size, num_workers=4
    )
    data_loader_test = DataLoader(
        dataset_test, sampler=sampler_test, batch_size=batch_size, num_workers=4
    )

    # 将邻接矩阵转换为图神经网络需要的边索引格式
    # torch_geometric 使用 edge_index 表示图的连接关系
    edge_index = edge_adj.to_sparse()._indices()

    # 将模型移动到指定设备（CPU/GPU）
    model = model.to(device)

    # 初始化最佳指标记录
    best_metric = {s: 0. for s in metrics}

    # 开始训练循环
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        optimizer.zero_grad()  # 清空梯度

        # 创建评估指标对象
        metric = Metric(metrics)

        # 创建训练进度条
        train_bar = tqdm(enumerate(data_loader_train),
                         total=len(data_loader_train),
                         desc=f"Train Epoch {epoch}/{epochs}: lr:{optimizer.param_groups[0]['lr']}")

        # 遍历训练批次
        for idx, (samples, targets) in train_bar:
            # 将数据移动到指定设备
            samples = samples.to(device)
            edge_index = edge_index.to(device)
            targets = targets.to(device)

            # 创建图数据结构
            # x: 节点特征, edge_index: 边连接, y: 图标签（这里用节点数占位）
            data = Data(x=samples, edge_index=edge_index, y=samples.shape[0])

            optimizer.zero_grad()  # 清空梯度

            # 前向传播：执行情感识别
            outputs = model(data)

            # 计算损失：主损失 + 额外损失（如正则化）
            loss = criterion(outputs, targets) + (0 if loss_func is None else loss_func(loss_param))

            # 更新评估指标
            metric.update(torch.argmax(outputs, dim=1), targets, loss.item())

            # 更新进度条显示
            train_bar.set_postfix_str(f"loss: {loss.item():.2f}")

            # 反向传播和优化
            loss.backward()
            optimizer.step()

        # 更新学习率（如果有调度器）
        if scheduler is not None:
            scheduler.step()

        # 打印训练结果（绿色显示）
        print("\033[32m train state: " + metric.value())

        # 在验证集上评估模型
        metric_value = evaluate(model, data_loader_val, edge_adj, device,
                                metrics, criterion, loss_func, loss_param)

        # 检查并保存最佳模型
        for m in metrics:
            if metric_value[m] > best_metric[m]:
                best_metric[m] = metric_value[m]
                save_state(output_dir, model, optimizer, epoch + 1, metric=m)

    # 加载最佳模型（基于主要指标）
    model.load_state_dict(torch.load(f"{output_dir}/checkpoint-best{metric_choose}")['model'])

    # 在测试集上评估最佳模型
    metric_value = evaluate(model, data_loader_test, edge_adj, device,
                            metrics, criterion, loss_func, loss_param)

    # 打印最佳指标结果
    for m in metrics:
        print(f"best_val_{m}: {best_metric[m]:.2f}")
        print(f"best_test_{m}: {metric_value[m]:.2f}")

    return metric_value


@torch.no_grad()  # 禁用梯度计算，节省内存
def evaluate(model, data_loader, edge_adj, device, metrics, criterion,
             loss_func, loss_param):
    """
    模型评估函数，用于验证和测试阶段

    参数:
        model: 要评估的模型
        data_loader: 数据加载器
        edge_adj: 图的邻接矩阵
        device: 评估设备
        metrics: 评估指标列表
        criterion: 主损失函数
        loss_func: 额外的损失函数
        loss_param: 额外损失函数的参数

    返回:
        metric.values: 评估指标结果字典
    """

    model.eval()  # 设置模型为评估模式

    # 创建评估指标对象
    metric = Metric(metrics)

    # 将邻接矩阵转换为边索引
    edge_index = edge_adj.to_sparse()._indices()

    # 遍历数据批次进行评估
    for idx, (samples, targets) in tqdm(enumerate(data_loader),
                                        total=len(data_loader),
                                        desc=f"Evaluating : "):
        # 将数据移动到指定设备
        samples = samples.to(device)
        edge_index = edge_index.to(device)
        targets = targets.to(device)

        # 创建图数据结构
        data = Data(x=samples, edge_index=edge_index, y=samples.shape[0])

        # 前向传播：执行情感识别
        outputs = model(data)

        # 计算损失
        loss = criterion(outputs, targets) + (0 if loss_func is None else loss_func(loss_param))

        # 更新评估指标
        metric.update(torch.argmax(outputs, dim=1), targets, loss.item())

    # 打印评估结果（蓝色显示）
    print("\033[34m eval state: " + metric.value())

    return metric.values