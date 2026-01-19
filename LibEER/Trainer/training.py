import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm  # 进度条显示

from LibEER.utils.metric import Metric  # 自定义的评估指标模块
from LibEER.utils.store import save_state  # 模型保存模块


def train(model, dataset_train, dataset_val, dataset_test, device,
          output_dir="result/", metrics=None, metric_choose=None,
          optimizer=None, scheduler=None, batch_size=16, epochs=40,
          criterion=None, loss_func=None, loss_param=None):
    """
    通用深度学习训练函数，适用于标准神经网络训练流程

    参数:
        model: 要训练的模型
        dataset_train: 训练数据集
        dataset_val: 验证数据集
        dataset_test: 测试数据集
        device: 训练设备（CPU/GPU）
        output_dir: 模型保存路径，默认"result/"
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

    # ==================== 数据加载器准备 ====================
    # 创建数据采样器
    # 训练集使用随机采样，增强数据随机性
    sampler_train = RandomSampler(dataset_train)
    # 验证集和测试集使用顺序采样，确保结果一致性
    sampler_val = SequentialSampler(dataset_val)
    sampler_test = SequentialSampler(dataset_test)

    # 创建数据加载器
    # 训练数据加载器
    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train, batch_size=batch_size, num_workers=4
    )
    # 验证数据加载器
    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val, batch_size=batch_size, num_workers=4
    )
    # 测试数据加载器
    data_loader_test = DataLoader(
        dataset_test, sampler=sampler_test, batch_size=batch_size, num_workers=4
    )

    # 将模型移动到指定设备（CPU/GPU）
    model = model.to(device)

    # 初始化最佳指标记录字典
    best_metric = {s: 0. for s in metrics}

    # ==================== 主训练循环 ====================
    for epoch in range(epochs):
        # 设置模型为训练模式
        model.train()
        # 清空优化器的梯度缓存
        optimizer.zero_grad()

        # 创建评估指标对象，用于记录本epoch的训练指标
        metric = Metric(metrics)

        # 创建训练进度条，显示当前训练状态
        train_bar = tqdm(enumerate(data_loader_train),
                         total=len(data_loader_train),
                         desc=f"训练轮次 {epoch}/{epochs}: lr:{optimizer.param_groups[0]['lr']}")

        # ==================== 批次训练循环 ====================
        for idx, (samples, targets) in train_bar:
            # 将数据移动到指定设备
            samples = samples.to(device)
            targets = targets.to(device)

            # 清空梯度（每个批次前都需要清空）
            optimizer.zero_grad()

            # 前向传播：执行情感识别
            outputs = model(samples)

            # 计算损失：主损失 + 额外损失（如正则化项）
            # 如果loss_func为None，则只使用主损失
            loss = criterion(outputs, targets) + (0 if loss_func is None else loss_func(loss_param))

            # 更新训练指标
            # torch.argmax(outputs, dim=1): 获取预测类别
            metric.update(torch.argmax(outputs, dim=1), targets, loss.item())

            # 更新进度条显示当前损失
            train_bar.set_postfix_str(f"损失: {loss.item():.2f}")

            # 反向传播：计算梯度
            loss.backward()
            # 优化器更新：更新模型参数
            optimizer.step()

        # ==================== 学习率调整 ====================
        # 如果有学习率调度器，更新学习率
        if scheduler is not None:
            scheduler.step()

        # 打印训练结果（绿色显示）
        print("\033[32m 训练状态: " + metric.value())

        # ==================== 验证集评估 ====================
        # 在验证集上评估模型性能
        metric_value = evaluate(model, data_loader_val, device, metrics,
                                criterion, loss_func, loss_param)

        # ==================== 模型保存 ====================
        # 检查并保存最佳模型
        for m in metrics:
            # 如果当前验证集指标比历史最佳更好，保存模型
            if metric_value[m] > best_metric[m]:
                best_metric[m] = metric_value[m]
                # 保存模型状态
                save_state(output_dir, model, optimizer, epoch + 1, metric=m)

    # ==================== 最终测试评估 ====================
    # 加载基于主要指标的最佳模型
    # 注意：这里假设模型保存在指定格式的文件中
    model.load_state_dict(torch.load(f"{output_dir}/checkpoint-best{metric_choose}")['model'])

    # 在测试集上评估最佳模型
    metric_value = evaluate(model, data_loader_test, device, metrics,
                            criterion, loss_func, loss_param)

    # 打印最佳指标结果
    for m in metrics:
        print(f"最佳验证集_{m}: {best_metric[m]:.2f}")
        print(f"最佳测试集_{m}: {metric_value[m]:.2f}")

    return metric_value


@torch.no_grad()  # 禁用梯度计算，节省内存和计算资源
def evaluate(model, data_loader, device, metrics, criterion, loss_func, loss_param):
    """
    通用模型评估函数，用于验证和测试阶段

    参数:
        model: 要评估的模型
        data_loader: 数据加载器
        device: 评估设备
        metrics: 评估指标列表
        criterion: 主损失函数
        loss_func: 额外的损失函数
        loss_param: 额外损失函数的参数

    返回:
        metric.values: 评估指标结果字典
    """

    # 设置模型为评估模式
    # 注意：这会禁用dropout、batchnorm等训练特有的层
    model.eval()

    # 创建评估指标对象
    metric = Metric(metrics)

    # 遍历数据批次进行评估
    for idx, (samples, targets) in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc=f"评估中 : "
    ):
        # 将数据移动到指定设备
        samples = samples.to(device)
        targets = targets.to(device)

        # 前向传播：执行情感识别
        outputs = model(samples)

        # 计算损失：主损失 + 额外损失
        loss = criterion(outputs, targets) + (0 if loss_func is None else loss_func(loss_param))

        # 更新评估指标
        # 使用argmax获取预测类别，与真实标签比较
        metric.update(torch.argmax(outputs, dim=1), targets, loss.item())

    # 打印评估结果（蓝色显示）
    print("\033[34m 评估状态: " + metric.value())

    return metric.values