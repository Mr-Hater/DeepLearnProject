# backend/services/signal_processor.py
"""
脑电信号处理核心服务
使用 SciPy/NumPy/MNE 进行信号处理
"""

import numpy as np
import scipy.signal as signal
from scipy import stats
import mne
import base64
import io
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from mne.channels import make_standard_montage


class SignalProcessor:
    """信号处理器类"""

    def __init__(self):
        # SEED数据集的标准电极名称（62通道）
        self.ch_names = [
            'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
            'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
            'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
            'C6', 'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
            'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4',
            'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1',
            'O1', 'OZ', 'O2', 'CB2'
        ]
        self.sampling_rate = 200  # SEED数据集采样率 200Hz

        # 创建标准电极位置（使用标准1020系统）
        try:
            self.montage = make_standard_montage('standard_1020')
        except:
            print("警告: 无法加载标准电极位置，将使用模拟位置")
            self.montage = None

    def _get_channel_positions(self, n_channels):
        """
        获取电极位置

        参数:
            n_channels: 通道数量

        返回:
            positions: 电极位置数组
        """
        if self.montage is not None:
            # 使用标准电极位置
            ch_pos = self.montage.get_positions()['ch_pos']
            positions = []
            valid_ch_names = []

            for ch_name in self.ch_names[:n_channels]:
                if ch_name in ch_pos:
                    positions.append(ch_pos[ch_name])
                    valid_ch_names.append(ch_name)
                else:
                    # 如果没有标准位置，使用模拟位置
                    theta = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
                    phi = np.linspace(0, np.pi / 2, n_channels)
                    positions.append([
                        np.sin(phi[len(positions)]) * np.cos(theta[len(positions)]),
                        np.sin(phi[len(positions)]) * np.sin(theta[len(positions)]),
                        np.cos(phi[len(positions)])
                    ])
                    valid_ch_names.append(ch_name)

            return np.array(positions), valid_ch_names
        else:
            # 使用模拟的电极位置（球面上的点）
            theta = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
            phi = np.linspace(0, np.pi / 2, n_channels)
            positions = []
            for i in range(n_channels):
                positions.append([
                    np.sin(phi[i]) * np.cos(theta[i]),
                    np.sin(phi[i]) * np.sin(theta[i]),
                    np.cos(phi[i])
                ])
            return np.array(positions), self.ch_names[:n_channels]

    def calculate_psd(self, eeg_data, trial_index=None):
        """
        计算PSD功率谱密度

        参数:
            eeg_data: numpy数组，形状为 (channels, time_points) 或 (trials, channels, time_points)
            trial_index: 指定要计算的trial索引

        返回:
            freqs: 频率数组
            psd: 功率谱密度数组
        """
        # 处理三维数据 (trials, channels, time_points)
        if eeg_data.ndim == 3:
            if trial_index is not None:
                data = eeg_data[trial_index]
            else:
                # 如果没有指定trial，取第一个trial
                data = eeg_data[0]
        else:
            data = eeg_data

        # 计算每个通道的PSD
        freqs, psd = signal.welch(
            data,
            fs=self.sampling_rate,
            nperseg=min(256, data.shape[1] // 4),
            noverlap=128,
            axis=1
        )

        return {
            'frequencies': freqs.tolist(),
            'psd': psd.tolist(),
            'channels': self.ch_names[:len(data)]
        }

    def detect_stable_region(self, eeg_data, trial_index=None, threshold=0.1, window_size=50):
        """
        检测EEG信号中的稳定区域

        参数:
            eeg_data: numpy数组
            threshold: 稳定阈值
            window_size: 滑动窗口大小

        返回:
            stable_regions: 稳定区域列表，每个区域为[start, end]索引
            stability_scores: 每个时间点的稳定得分
        """
        if eeg_data.ndim == 3:
            if trial_index is not None:
                data = eeg_data[trial_index]
            else:
                data = eeg_data[0]
        else:
            data = eeg_data

        # 计算所有通道的平均信号
        mean_signal = np.mean(data, axis=0)

        # 计算滑动窗口内的标准差
        stability_scores = []
        half_window = window_size // 2

        for i in range(len(mean_signal)):
            start = max(0, i - half_window)
            end = min(len(mean_signal), i + half_window)
            window_std = np.std(mean_signal[start:end])
            stability_scores.append(1.0 / (1.0 + window_std))  # 稳定得分

        stability_scores = np.array(stability_scores)

        # 检测稳定区域（得分高于阈值）
        stable_regions = []
        in_stable = False
        start_idx = 0

        for i, score in enumerate(stability_scores):
            if score > threshold and not in_stable:
                in_stable = True
                start_idx = i
            elif score <= threshold and in_stable:
                in_stable = False
                if i - start_idx > window_size:  # 只保留足够长的稳定区域
                    stable_regions.append([int(start_idx), int(i)])

        # 处理最后一个稳定区域
        if in_stable and len(mean_signal) - start_idx > window_size:
            stable_regions.append([int(start_idx), int(len(mean_signal))])

        return {
            'stable_regions': stable_regions,
            'stability_scores': stability_scores.tolist(),
            'time_points': list(range(len(mean_signal)))
        }

    def generate_topomap(self, eeg_data, trial_index=None, time_point=None, time_window=None):
        """
        生成Topomap脑电拓扑图

        参数:
            eeg_data: numpy数组
            trial_index: trial索引
            time_point: 时间点索引
            time_window: 时间窗口 [start, end]

        返回:
            base64编码的图片
        """
        try:
            if eeg_data.ndim == 3:
                if trial_index is not None:
                    data = eeg_data[trial_index]
                else:
                    data = eeg_data[0]
            else:
                data = eeg_data

            n_channels = len(data)

            # 选择要绘制的数据
            if time_window:
                start, end = time_window
                plot_data = np.mean(data[:, start:end], axis=1)
            elif time_point:
                plot_data = data[:, time_point]
            else:
                # 默认取中间时间点
                time_point = data.shape[1] // 2
                plot_data = data[:, time_point]

            # 创建图形
            fig, ax = plt.subplots(figsize=(8, 6))

            # 获取电极位置
            positions, valid_ch_names = self._get_channel_positions(n_channels)

            if len(positions) == len(plot_data):
                # 使用 imshow 绘制拓扑图（简化版本）
                # 创建网格
                x = positions[:, 0]
                y = positions[:, 1]

                # 创建插值网格
                grid_x, grid_y = np.mgrid[-1:1:100j, -1:1:100j]

                from scipy.interpolate import griddata
                grid_z = griddata(
                    np.column_stack([x, y]),
                    plot_data,
                    (grid_x, grid_y),
                    method='cubic',
                    fill_value=0
                )

                # 绘制
                im = ax.imshow(
                    grid_z.T,
                    extent=(-1, 1, -1, 1),
                    origin='lower',
                    cmap='RdBu_r',
                    interpolation='bilinear'
                )

                # 绘制电极点
                ax.scatter(x, y, c='black', s=30, zorder=5)

                # 添加电极名称
                for i, (xi, yi, name) in enumerate(zip(x, y, valid_ch_names)):
                    if i % 4 == 0:  # 每隔4个显示一个标签，避免太密集
                        ax.annotate(name, (xi, yi), xytext=(5, 5),
                                    textcoords='offset points', fontsize=8)

                plt.colorbar(im, ax=ax, label='幅值 (μV)')
            else:
                # 如果无法获取位置，使用简化的圆形图
                theta = np.linspace(0, 2 * np.pi, len(plot_data), endpoint=False)
                radii = np.ones_like(plot_data) * 0.8

                ax = plt.subplot(111, projection='polar')
                bars = ax.bar(theta, radii, width=2 * np.pi / len(plot_data),
                              bottom=0.0, alpha=0.7)

                # 根据数据值着色
                norm_data = (plot_data - plot_data.min()) / (plot_data.max() - plot_data.min() + 1e-10)
                for j, bar in enumerate(bars):
                    bar.set_facecolor(plt.cm.RdBu_r(norm_data[j]))

                ax.set_xticks(theta[::4])
                ax.set_xticklabels(self.ch_names[:len(plot_data)][::4], fontsize=8)

            ax.set_title('EEG Topomap')

            # 将图片转换为base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            return f'data:image/png;base64,{img_base64}'

        except Exception as e:
            print(f"Topomap生成详细错误: {str(e)}")
            # 如果失败，返回一个简单的错误提示图片
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'Topomap生成失败\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            plt.close(fig)
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            return f'data:image/png;base64,{img_base64}'

    def extract_features(self, eeg_data, trial_index=None):
        """
        提取EEG特征

        返回:
            特征字典，包含多个域的特征
        """
        if eeg_data.ndim == 3:
            if trial_index is not None:
                data = eeg_data[trial_index]
            else:
                data = eeg_data[0]
        else:
            data = eeg_data

        features = {
            'time_domain': [],  # 时域特征
            'frequency_domain': [],  # 频域特征
            'nonlinear': []  # 非线性特征
        }

        for channel_data in data:
            # 时域特征
            time_features = {
                'mean': float(np.mean(channel_data)),
                'std': float(np.std(channel_data)),
                'variance': float(np.var(channel_data)),
                'rms': float(np.sqrt(np.mean(channel_data ** 2))),  # 均方根
                'peak_to_peak': float(np.ptp(channel_data)),  # 峰峰值
                'skewness': float(stats.skew(channel_data)),  # 偏度
                'kurtosis': float(stats.kurtosis(channel_data))  # 峰度
            }
            features['time_domain'].append(time_features)

            # 频域特征
            freqs, psd = signal.welch(channel_data, fs=self.sampling_rate)

            # 计算频带功率
            delta_power = np.mean(psd[(freqs >= 0.5) & (freqs < 4)]) if np.any((freqs >= 0.5) & (freqs < 4)) else 0
            theta_power = np.mean(psd[(freqs >= 4) & (freqs < 8)]) if np.any((freqs >= 4) & (freqs < 8)) else 0
            alpha_power = np.mean(psd[(freqs >= 8) & (freqs < 13)]) if np.any((freqs >= 8) & (freqs < 13)) else 0
            beta_power = np.mean(psd[(freqs >= 13) & (freqs < 30)]) if np.any((freqs >= 13) & (freqs < 30)) else 0
            gamma_power = np.mean(psd[(freqs >= 30) & (freqs < 50)]) if np.any((freqs >= 30) & (freqs < 50)) else 0

            freq_features = {
                'delta_power': float(delta_power),
                'theta_power': float(theta_power),
                'alpha_power': float(alpha_power),
                'beta_power': float(beta_power),
                'gamma_power': float(gamma_power),
                'total_power': float(np.sum(psd)),
                'peak_frequency': float(freqs[np.argmax(psd)]) if len(psd) > 0 else 0
            }
            features['frequency_domain'].append(freq_features)

            # 非线性特征
            approx_entropy = self._approximate_entropy(channel_data)
            sample_entropy = self._sample_entropy(channel_data)

            nonlinear_features = {
                'approximate_entropy': float(approx_entropy),
                'sample_entropy': float(sample_entropy),
                'hurst_exponent': float(self._hurst_exponent(channel_data))
            }
            features['nonlinear'].append(nonlinear_features)

        return features

    def _approximate_entropy(self, data, m=2, r=0.2):
        """计算近似熵"""
        N = len(data)
        if N < m + 1:
            return 0

        def _phi(m):
            patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            for i in range(N - m + 1):
                # 计算与所有模式的距离
                dist = np.max(np.abs(patterns - patterns[i]), axis=1)
                C[i] = np.sum(dist <= r * np.std(data)) / (N - m + 1)
            return np.mean(np.log(C + 1e-10))  # 避免log(0)

        return abs(_phi(m + 1) - _phi(m))

    def _sample_entropy(self, data, m=2, r=0.2):
        """计算样本熵"""
        N = len(data)
        if N < m + 2:
            return 0

        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])

        def _phi(m):
            if N - m + 1 <= 0:
                return 0
            patterns = [data[i:i + m] for i in range(N - m + 1)]
            B = 0
            for i in range(len(patterns)):
                for j in range(len(patterns)):
                    if j != i and _maxdist(patterns[i], patterns[j]) <= r * np.std(data):
                        B += 1
            return B / (len(patterns) * (len(patterns) - 1)) if len(patterns) > 1 else 0

        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        if phi_m == 0 or phi_m1 == 0:
            return 0
        return -np.log(phi_m1 / phi_m)

    def _hurst_exponent(self, data):
        """计算Hurst指数"""
        lags = range(2, min(len(data) // 2, 20))
        if len(lags) == 0:
            return 0.5
        tau = [np.std(np.subtract(data[lag:], data[:-lag])) for lag in lags]

        if len(tau) > 0 and np.all(np.array(tau) > 0):
            m = np.polyfit(np.log(lags), np.log(tau), 1)
            return m[0]
        return 0.5

    def compare_sessions(self, sessions_data, comparison_type='eeg'):
        """
        对比多个session的数据

        参数:
            sessions_data: 字典，key为session标识，value为session数据
            comparison_type: 'eeg', 'psd', 'stable', 'feature'

        返回:
            对比结果
        """
        result = {
            'sessions': list(sessions_data.keys()),
            'data': {}
        }

        if comparison_type == 'eeg':
            # 计算每个session的平均信号
            for session_name, session_d in sessions_data.items():
                if session_d.ndim == 3:
                    # 有多个trials，取平均
                    avg_signal = np.mean(session_d, axis=(0, 1))
                else:
                    avg_signal = np.mean(session_d, axis=0)
                result['data'][session_name] = avg_signal.tolist()

        elif comparison_type == 'psd':
            # 计算每个session的平均PSD
            for session_name, session_d in sessions_data.items():
                psd_result = self.calculate_psd(session_d)
                result['data'][session_name] = {
                    'frequencies': psd_result['frequencies'],
                    'psd': np.mean(psd_result['psd'], axis=0).tolist()
                }

        elif comparison_type == 'feature':
            # 提取每个session的特征
            for session_name, session_d in sessions_data.items():
                features = self.extract_features(session_d)
                # 计算所有通道的平均特征
                avg_features = {}
                for domain in features:
                    if features[domain]:
                        # 获取所有特征值
                        feature_values = []
                        for ch_features in features[domain]:
                            feature_values.append(list(ch_features.values()))
                        if feature_values:
                            avg_features[domain] = np.mean(feature_values, axis=0).tolist()
                result['data'][session_name] = avg_features

        return result

    def analyze_emotion_distribution(self, labels):
        """
        分析情绪标签分布

        参数:
            labels: 情绪标签数组

        返回:
            统计结果
        """
        # SEED数据集的情绪标签
        emotion_map = {
            0: '中立',
            1: '积极',
            -1: '消极'
        }

        # 统计各情绪数量
        unique, counts = np.unique(labels, return_counts=True)

        distribution = []
        for label, count in zip(unique, counts):
            emotion_name = emotion_map.get(int(label), f'未知({label})')
            distribution.append({
                'label': int(label),
                'emotion': emotion_name,
                'count': int(count),
                'percentage': float(count / len(labels) * 100) if len(labels) > 0 else 0
            })

        return {
            'distribution': distribution,
            'total': int(len(labels))
        }