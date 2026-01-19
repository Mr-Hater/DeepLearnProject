import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm  # 进度条显示

from LibEER.utils.metric import Metric  # 自定义的评估指标模块
from LibEER.utils.store import save_state  # 模型保存模块


def train(model, dataset_train, dataset_val, dataset_test, device,
          output_dir="result/", metrics=None, metric_choose=None,
          batch_size=16, epochs=40):
    """
    深度信念网络(DBN)的训练函数，包含预训练、无监督微调、监督微调三阶段

    参数:
        model: 要训练的DBN模型（包含多个RBM层）
        dataset_train: 训练数据集
        dataset_val: 验证数据集
        dataset_test: 测试数据集
        device: 训练设备（CPU/GPU）
        output_dir: 模型保存路径，默认"result/"
        metrics: 评估指标列表，默认['acc']
        metric_choose: 用于选择最佳模型的主要指标
        batch_size: 批大小，默认16
        epochs: 监督微调阶段的训练轮数，默认40

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

    # 将模型移动到指定设备
    model = model.to(device)
    model.train()  # 设置模型为训练模式

    # ==================== 第一阶段：逐层预训练 ====================
    print("=" * 60)
    print("开始RBM层的逐层贪心预训练...")
    print("=" * 60)

    for epoch in range(10):  # 预训练10轮
        # 创建训练进度条
        train_bar = tqdm(enumerate(data_loader_train),
                         total=len(data_loader_train),
                         desc=f"预训练 Epoch {epoch + 1}: ")

        for idx, (samples, targets) in train_bar:
            # 将数据移动到指定设备
            samples = samples.to(device)
            # 重塑输入数据：[batch, feature_dim1, feature_dim2] -> [batch, feature_dim1*feature_dim2]
            # 假设原始特征维度为31x10=310维
            samples = samples.reshape(samples.shape[0], samples.shape[1] * samples.shape[2])

            # 第一层RBM的对比散度训练（贪婪逐层训练）
            model.rbm1.constrastive_divergence(samples, batch_size=batch_size, device=device)

            # 获取第一层RBM的隐藏层输出，作为第二层RBM的输入
            _, onelayerout = model.rbm1.sample_h(samples)

            # 第二层RBM的对比散度训练
            model.rbm2.constrastive_divergence(onelayerout, batch_size=batch_size, device=device)

    print(f"预训练完成 ✓")

    # ==================== 第二阶段：无监督微调（编码器-解码器） ====================
    print("=" * 60)
    print("开始无监督微调（自编码器重建任务）...")
    print("=" * 60)

    # 使用均方误差作为重建损失
    criterion = torch.nn.MSELoss()
    # 使用随机梯度下降优化器
    optimizer = optim.SGD(model.parameters(), lr=0.5)

    for epoch in range(5):  # 无监督微调5轮
        model.train()

        # 创建训练进度条
        train_bar = tqdm(enumerate(data_loader_train),
                         total=len(data_loader_train),
                         desc=f"无监督微调 Epoch {epoch}: lr:{optimizer.param_groups[0]['lr']}")

        for idx, (samples, targets) in train_bar:
            # 将数据移动到指定设备并重塑
            samples = samples.to(device)
            samples = samples.reshape(samples.shape[0], samples.shape[1] * samples.shape[2])

            optimizer.zero_grad()  # 清空梯度

            # 执行重建：编码→解码
            recon = model.reconstruct(samples, device=device)

            # 计算重建损失
            loss = criterion(samples, recon)

            # 更新进度条显示
            train_bar.set_postfix_str(f"loss: {loss.item():.2f}")

            # 反向传播和优化
            loss.backward()
            optimizer.step()

    # ==================== 第三阶段：监督微调（分类任务） ====================
    print("=" * 60)
    print("开始监督微调（分类任务）...")
    print("=" * 60)

    # 切换到分类任务的损失函数
    criterion = nn.CrossEntropyLoss()
    # 使用随机梯度下降优化器（学习率调整）
    optimizer = optim.SGD(model.parameters(), lr=0.2)
    # 可以添加学习率调度器（当前被注释）
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    # 初始化最佳指标记录
    best_metric = {s: 0. for s in metrics}

    for epoch in range(epochs):
        model.train()

        # 创建评估指标对象
        metric = Metric(metrics)

        # 创建训练进度条
        train_bar = tqdm(enumerate(data_loader_train),
                         total=len(data_loader_train),
                         desc=f"监督微调 Epoch {epoch}: lr:{optimizer.param_groups[0]['lr']}")

        for idx, (samples, targets) in train_bar:
            # 将数据移动到指定设备
            samples = samples.to(device)
            samples = samples.reshape(samples.shape[0], samples.shape[1] * samples.shape[2])
            targets = targets.to(device)

            optimizer.zero_grad()  # 清空梯度

            # 前向传播：执行情感识别分类
            outputs = model(samples)

            # 计算交叉熵损失
            loss = criterion(outputs, targets)

            # 更新评估指标
            metric.update(torch.argmax(outputs, dim=1), targets, loss.item())

            # 更新进度条显示
            train_bar.set_postfix_str(f"loss: {loss.item():.2f}")

            # 反向传播和优化
            loss.backward()
            optimizer.step()

        # 更新学习率（如果使用调度器）
        # scheduler.step()

        # 打印训练结果（绿色显示）
        print("\033[32m train state: " + metric.value())

        # 在验证集上评估模型
        metric_value = evaluate(model, data_loader_val, device, metrics, criterion)

        # 检查并保存最佳模型
        for m in metrics:
            if metric_value[m] > best_metric[m]:
                best_metric[m] = metric_value[m]
                save_state(output_dir, model, optimizer, epoch + 1, metric=m)

    # ==================== 测试阶段 ====================
    print("=" * 60)
    print("加载最佳模型并在测试集上评估...")
    print("=" * 60)

    # 加载基于主要指标的最佳模型
    model.load_state_dict(torch.load(f"{output_dir}/checkpoint-best{metric_choose}")['model'])

    # 在测试集上评估最佳模型
    metric_value = evaluate(model, data_loader_test, device, metrics, criterion)

    # 打印最佳指标结果
    for m in metrics:
        print(f"最佳验证集_{m}: {best_metric[m]:.2f}")
        print(f"最佳测试集_{m}: {metric_value[m]:.2f}")

    return metric_value


@torch.no_grad()  # 禁用梯度计算，节省内存
def evaluate(model, data_loader, device, metrics, criterion):
    """
    DBN模型评估函数，用于验证和测试阶段

    参数:
        model: 要评估的DBN模型
        data_loader: 数据加载器
        device: 评估设备
        metrics: 评估指标列表
        criterion: 损失函数（监督微调阶段使用交叉熵）

    返回:
        metric.values: 评估指标结果字典
    """

    model.eval()  # 设置模型为评估模式

    # 创建评估指标对象
    metric = Metric(metrics)

    # 遍历数据批次进行评估
    for idx, (samples, targets) in tqdm(enumerate(data_loader),
                                        total=len(data_loader),
                                        desc=f"评估中 : "):
        # 将数据移动到指定设备并重塑
        samples = samples.to(device)
        samples = samples.reshape(samples.shape[0], samples.shape[1] * samples.shape[2])
        targets = targets.to(device)

        # 前向传播：执行情感识别分类
        outputs = model(samples)

        # 计算损失
        loss = criterion(outputs, targets)

        # 更新评估指标
        metric.update(torch.argmax(outputs, dim=1), targets, loss.item())

    # 打印评估结果（蓝色显示）
    print("\033[34m eval state: " + metric.value())

    return metric.values