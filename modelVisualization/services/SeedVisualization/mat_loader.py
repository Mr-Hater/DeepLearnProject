# backend/services/mat_loader.py
"""
MAT文件读取服务
用于加载SEED数据集的MAT文件
"""

import scipy.io as sio
import numpy as np
import os
from pathlib import Path


class MATLoader:
    """MAT文件加载器类"""

    def __init__(self, data_root=None):
        """
        初始化MAT文件加载器

        参数:
            data_root: SEED数据集根目录，默认使用当前目录下的data/SEED
        """
        if data_root is None:
            # 自动查找data目录
            current_dir = Path(__file__).resolve().parent.parent.parent.parent
            self.data_root = current_dir / 'LibEER/data_utils/Dataset' / 'SEED'
        else:
            self.data_root = Path(data_root)

        self.datasets = {
            'SEED': {
                'path': self.data_root,
                'preprocessed_dir': 'Preprocessed_EEG',
                'extracted_dir': 'ExtractedFeatures'
            }
        }

        # SEED数据集的标准电极名称
        self.ch_names = [
            'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
            'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
            'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
            'C6', 'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
            'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4',
            'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1',
            'O1', 'OZ', 'O2', 'CB2'
        ]

    def get_available_datasets(self):
        """
        获取可用的数据集列表

        返回:
            datasets: 数据集信息列表
        """
        datasets = []
        for name, info in self.datasets.items():
            if info['path'].exists():
                datasets.append({
                    'id': name,
                    'name': name,
                    'path': str(info['path'])
                })
        return datasets

    def get_mat_files(self, dataset_name='SEED', file_type='preprocessed'):
        """
        获取指定数据集下的所有MAT文件

        参数:
            dataset_name: 数据集名称
            file_type: 文件类型 ('preprocessed' 或 'extracted')

        返回:
            files: MAT文件列表
        """
        if dataset_name not in self.datasets:
            return []

        dataset_info = self.datasets[dataset_name]
        print(self.datasets)
        if file_type == 'preprocessed':
            target_dir = dataset_info['path'] / dataset_info['preprocessed_dir']
            print("preprocessed")
        else:
            target_dir = dataset_info['path'] / dataset_info['extracted_dir']
            print("extracted")
        print("读取文件地址")
        print(target_dir)
        mat_files = []
        if target_dir.exists():
            print("找到文件了")
            for file_path in target_dir.glob('*.mat'):
                # 解析文件名获取session和subject信息
                filename = file_path.stem
                session_info = self._parse_filename(filename)

                mat_files.append({
                    'id': filename,
                    'name': filename,
                    'path': str(file_path),
                    'type': file_type,
                    'session': session_info['session'],
                    'subject': session_info['subject'],
                    'is_label': filename == 'label'
                })

        # 按文件名排序
        mat_files.sort(key=lambda x: x['name'])
        return mat_files

    def _parse_filename(self, filename):
        """
        解析文件名，提取session和subject信息

        例如: '1_20131027' -> session=1, subject=20131027
        """
        if filename == 'label':
            return {'session': None, 'subject': 'label'}

        parts = filename.split('_')
        if len(parts) >= 2:
            return {
                'session': parts[0],
                'subject': parts[1]
            }
        return {'session': None, 'subject': filename}

    def load_mat_file(self, file_path):
        """
        加载MAT文件

        参数:
            file_path: MAT文件路径

        返回:
            data: 加载的数据
        """
        try:
            # 使用scipy加载MAT文件
            mat_data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)

            # 移除MATLAB的元数据
            data = {}
            for key in mat_data:
                if not key.startswith('__'):
                    data[key] = mat_data[key]

            return {
                'success': True,
                'data': data,
                'variables': list(data.keys())
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def load_preprocessed_eeg(self, file_path, trial_index=None):
        """
        加载预处理后的EEG数据

        SEED预处理数据的结构:
        - 通常是三维数组: [trials, channels, time_points]
        - 或者字典结构包含 'data' 字段

        参数:
            file_path: MAT文件路径
            trial_index: 指定要加载的trial索引，None表示加载所有

        返回:
            eeg_data: EEG数据
            metadata: 元数据
        """
        result = self.load_mat_file(file_path)

        if not result['success']:
            raise Exception(f"加载文件失败: {result['error']}")

        data = result['data']

        # 处理不同的数据格式
        eeg_data = None
        metadata = {}

        # SEED数据常见的变量名
        possible_names = ['data', 'eeg', 'EEG', 'eeg_data', 'feat', 'x']

        for var_name in possible_names:
            if var_name in data:
                eeg_data = data[var_name]
                break

        # 如果没找到，取第一个非元数据的变量
        if eeg_data is None:
            for key in result['variables']:
                if key not in ['__header__', '__version__', '__globals__']:
                    eeg_data = data[key]
                    break

        if eeg_data is None:
            # 生成模拟数据用于测试
            print(f"警告: 无法识别EEG数据格式，使用模拟数据")
            eeg_data = np.random.randn(62, 1000) * 10

        # 转换为numpy数组
        if not isinstance(eeg_data, np.ndarray):
            eeg_data = np.array(eeg_data)

        # 获取数据维度
        if eeg_data.ndim == 2:
            # 2D数据: [channels, time_points]
            n_channels, n_times = eeg_data.shape
            metadata['format'] = '2d'
            metadata['n_channels'] = n_channels
            metadata['n_times'] = n_times
            metadata['n_trials'] = 1

            if trial_index is not None and trial_index > 0:
                raise Exception(f"trial_index={trial_index} 超出范围，数据只有1个trial")

        elif eeg_data.ndim == 3:
            # 3D数据: [trials, channels, time_points]
            n_trials, n_channels, n_times = eeg_data.shape
            metadata['format'] = '3d'
            metadata['n_trials'] = n_trials
            metadata['n_channels'] = n_channels
            metadata['n_times'] = n_times

            if trial_index is not None:
                if trial_index >= n_trials:
                    raise Exception(f"trial_index={trial_index} 超出范围，共有{n_trials}个trials")
                eeg_data = eeg_data[trial_index]
        else:
            raise Exception(f"不支持的EEG数据维度: {eeg_data.ndim}")

        return {
            'eeg_data': eeg_data,
            'metadata': metadata,
            'channels': self.ch_names[:n_channels] if n_channels <= len(self.ch_names) else [f'CH{i + 1}' for i in
                                                                                             range(n_channels)],
            'sampling_rate': 200  # SEED数据集固定采样率
        }

    def load_extracted_features(self, file_path, trial_index=None):
        """
        加载提取的特征数据

        SEED特征数据的结构:
        - 通常是三维数组: [trials, channels, features]
        - 或者字典结构

        参数:
            file_path: MAT文件路径
            trial_index: 指定要加载的trial索引

        返回:
            features: 特征数据
            metadata: 元数据
        """
        result = self.load_mat_file(file_path)

        if not result['success']:
            raise Exception(f"加载文件失败: {result['error']}")

        data = result['data']

        # 查找特征数据
        feature_data = None
        feature_names = ['feat', 'features', 'feature', 'x', 'data']

        for name in feature_names:
            if name in data:
                feature_data = data[name]
                break

        if feature_data is None:
            for key in result['variables']:
                if key not in ['__header__', '__version__', '__globals__']:
                    feature_data = data[key]
                    break

        if feature_data is None:
            # 生成模拟数据
            print(f"警告: 无法识别特征数据格式，使用模拟数据")
            feature_data = np.random.randn(1, 62, 10) * 5

        if not isinstance(feature_data, np.ndarray):
            feature_data = np.array(feature_data)

        # 特征名称（根据SEED数据集的特征定义）
        feature_type_names = {
            'time': ['mean', 'std', 'variance', 'rms', 'peak_to_peak', 'skewness', 'kurtosis'],
            'frequency': ['delta', 'theta', 'alpha', 'beta', 'gamma', 'total_power', 'peak_freq'],
            'nonlinear': ['approx_entropy', 'sample_entropy', 'hurst_exponent']
        }

        metadata = {
            'feature_names': feature_type_names
        }

        # 处理不同维度的数据
        if feature_data.ndim == 2:
            # [channels, features]
            n_channels, n_features = feature_data.shape
            metadata['n_trials'] = 1
            metadata['n_channels'] = n_channels
            metadata['n_features'] = n_features

            if trial_index is not None and trial_index > 0:
                raise Exception(f"trial_index={trial_index} 超出范围，数据只有1个trial")

        elif feature_data.ndim == 3:
            # [trials, channels, features]
            n_trials, n_channels, n_features = feature_data.shape
            metadata['n_trials'] = n_trials
            metadata['n_channels'] = n_channels
            metadata['n_features'] = n_features

            if trial_index is not None:
                if trial_index >= n_trials:
                    raise Exception(f"trial_index={trial_index} 超出范围，共有{n_trials}个trials")
                feature_data = feature_data[trial_index]
        else:
            # 其他维度，尝试reshape
            feature_data = feature_data.flatten()
            n_channels = 1
            n_features = len(feature_data)
            metadata['n_trials'] = 1
            metadata['n_channels'] = 1
            metadata['n_features'] = n_features

        return {
            'features': feature_data,
            'metadata': metadata,
            'channels': self.ch_names[:n_channels] if n_channels <= len(self.ch_names) else [f'CH{i + 1}' for i in
                                                                                             range(n_channels)]
        }

    def load_label_file(self, file_path):
        """
        加载情绪标签文件 (label.mat)

        参数:
            file_path: label.mat文件路径

        返回:
            labels: 情绪标签数组
            metadata: 元数据
        """
        result = self.load_mat_file(file_path)

        if not result['success']:
            raise Exception(f"加载标签文件失败: {result['error']}")

        data = result['data']

        # SEED标签文件中常见的变量名
        label_vars = ['label', 'labels', 'gt', 'ground_truth', 'emotion']

        labels = None
        for var in label_vars:
            if var in data:
                labels = data[var]
                break

        if labels is None:
            for key in result['variables']:
                if key not in ['__header__', '__version__', '__globals__']:
                    labels = data[key]
                    break

        if labels is None:
            # 生成模拟标签
            print(f"警告: 无法识别标签数据格式，使用模拟数据")
            labels = np.random.choice([-1, 0, 1], size=15)

        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        # 确保是一维数组
        if labels.ndim > 1:
            labels = labels.flatten()

        # 情绪标签映射
        emotion_map = {
            1: '积极',
            0: '中立',
            -1: '消极'
        }

        # 转换为整数类型
        labels = labels.astype(int)

        # 统计信息
        unique, counts = np.unique(labels, return_counts=True)
        distribution = []
        for label, count in zip(unique, counts):
            distribution.append({
                'label': int(label),
                'emotion': emotion_map.get(int(label), f'未知({label})'),
                'count': int(count),
                'percentage': float(count / len(labels) * 100) if len(labels) > 0 else 0
            })

        return {
            'labels': labels.tolist(),
            'distribution': distribution,
            'total': len(labels),
            'metadata': {
                'unique_labels': unique.tolist()
            }
        }

    def get_trial_list(self, file_path):
        """
        获取文件中的trial列表

        参数:
            file_path: MAT文件路径

        返回:
            trials: trial信息列表
        """
        try:
            # 快速检查文件而不加载全部数据
            mat_data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)

            # 查找数据变量
            data = None
            for key in mat_data:
                if not key.startswith('__'):
                    if isinstance(mat_data[key], np.ndarray):
                        data = mat_data[key]
                        break

            if data is None:
                return [{'index': 0, 'name': 'Trial 1'}]

            if data.ndim == 3:
                n_trials = data.shape[0]
                return [{'index': i, 'name': f'Trial {i + 1}'} for i in range(n_trials)]
            else:
                return [{'index': 0, 'name': 'Trial 1'}]

        except Exception as e:
            print(f"获取trial列表失败: {e}")
            return [{'index': 0, 'name': 'Trial 1'}]  # 默认返回一个trial