import os

from django.http import JsonResponse
from django.template.loader import render_to_string
import numpy as np
import json
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from scipy.io import loadmat

from modelVisualization.services.SeedVisualization.singnal_processor import SignalProcessor
from modelVisualization.services.SeedVisualization.mat_loader import MATLoader


def load_topomap(request):
    """加载脑电拓扑页面"""
    html = render_to_string('SeedVisualization/topomap.html')
    return JsonResponse({'html': html})

def load_singleEEG(request):
    """加载模型注册页面"""
    html = render_to_string('SeedVisualization/singleEEG.html')
    return JsonResponse({'html': html})

def load_doubleEEG(request):
    """加载训练历史页面"""
    html = render_to_string('SeedVisualization/doubleEEG.html')
    return JsonResponse({'html': html})


# 62通道的标准名称（根据SEED数据集）
CHANNEL_NAMES = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2'
]  # 实际SEED数据集是62通道，这里列出部分示例，完整需要62个


# 62通道的标准2D坐标（简化版，实际需要根据10-20系统计算）
def generate_channel_coordinates():
    """生成62通道的2D坐标（模拟数据，实际应该使用真实坐标）"""
    coords = []
    for i in range(62):
        # 简化的坐标生成逻辑：将通道分布在椭圆上
        angle = (i / 62) * 2 * np.pi
        x = 0.5 + 0.4 * np.cos(angle)  # 范围0.1-0.9
        y = 0.5 + 0.4 * np.sin(angle)
        coords.append([x, y])
    return coords


@csrf_exempt
@require_http_methods(["POST"])
def load_eeg_data(request):
    """
    加载脑电数据文件
    请求格式: {'file_path': 'path/to/data.mat', 'data_key': 'data_key_in_mat'}
    """
    try:
        data = json.loads(request.body)
        file_path = data.get('file_path')
        data_key = data.get('data_key', 'data')  # mat文件中数据的关键字

        # 安全检查：确保文件路径在允许的目录内
        base_dir = 'D:/Work/Studio/PyCharm/PythonProject/DeepLearn/LibEER/data_utils/Dataset/SEED/ExtractedFeatures'  # 配置为实际数据目录
        full_path = os.path.join(base_dir, file_path)

        if not os.path.exists(full_path):
            return JsonResponse({'error': '文件不存在'}, status=404)

        # 读取.mat文件
        mat_data = loadmat(full_path)

        # 获取脑电数据（假设是62通道的数据）
        if data_key in mat_data:
            eeg_data = mat_data[data_key]
        else:
            # 尝试自动查找第一个合适的数组
            for key in mat_data:
                if isinstance(mat_data[key], np.ndarray) and mat_data[key].size >= 62:
                    eeg_data = mat_data[key]
                    break
            else:
                return JsonResponse({'error': '未找到合适的脑电数据'}, status=400)

        # 确保数据是2D的，并提取第一列或均值
        if eeg_data.ndim > 1:
            # 如果有多列数据，取第一列或均值
            eeg_values = np.mean(eeg_data, axis=1) if eeg_data.shape[1] > 1 else eeg_data.flatten()
        else:
            eeg_values = eeg_data.flatten()

        # 确保有62个通道的值
        if len(eeg_values) < 62:
            # 如果数据不足62，用0填充
            eeg_values = np.pad(eeg_values, (0, 62 - len(eeg_values)), 'constant')
        elif len(eeg_values) > 62:
            eeg_values = eeg_values[:62]

        # 生成通道坐标
        coordinates = generate_channel_coordinates()

        # 构建通道详细信息
        channel_details = []
        for i in range(62):
            # 模拟通道信息，实际应根据通道索引从数据库中获取
            channel_details.append({
                'index': i + 1,
                'name': CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else f'CH{i + 1}',
                'value': float(eeg_values[i]),
                'brain_region': get_brain_region(i),  # 根据通道索引确定脑区
                'description': f'这是{CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else "通道" + str(i + 1)}的描述信息',
                'analysis': {
                    'mean': float(np.mean(eeg_values)),
                    'std': float(np.std(eeg_values)),
                    'max': float(np.max(eeg_values)),
                    'min': float(np.min(eeg_values)),
                    'power_spectrum': [float(x) for x in np.random.randn(10)]  # 模拟功率谱数据
                }
            })

        # 准备返回数据
        response_data = {
            'success': True,
            'channels': channel_details,
            'coordinates': coordinates,
            'global_stats': {
                'mean': float(np.mean(eeg_values)),
                'std': float(np.std(eeg_values)),
                'max': float(np.max(eeg_values)),
                'min': float(np.min(eeg_values))
            }
        }

        return JsonResponse(response_data)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def get_brain_region(channel_index):
    """根据通道索引返回脑区（简化逻辑）"""
    if channel_index < 8:
        return '前额叶'
    elif channel_index < 16:
        return '额叶'
    elif channel_index < 24:
        return '运动皮层'
    elif channel_index < 32:
        return '中央区'
    elif channel_index < 40:
        return '顶叶'
    elif channel_index < 48:
        return '颞叶'
    else:
        return '枕叶'




# 初始化加载器和处理器
loader = MATLoader()
processor = SignalProcessor()

# 模拟数据缓存（用于没有真实数据时的备选）
mock_eeg_cache = {}


def get_mock_eeg_data():
    """生成或获取模拟EEG数据"""
    if 'default' not in mock_eeg_cache:
        np.random.seed(42)
        mock_eeg_cache['default'] = np.random.randn(62, 1000) * 10
    return mock_eeg_cache['default']


# ========== 数据浏览接口 ==========

@csrf_exempt
@require_http_methods(["GET"])
def get_datasets(request):
    """
    获取数据集列表
    GET /api/datasets
    """
    try:
        datasets = loader.get_available_datasets()
        # 如果没有找到真实数据集，返回一个默认选项
        if not datasets:
            datasets = [{
                'id': 'SEED',
                'name': 'SEED 数据集 (模拟)',
                'path': 'D:/Work/Studio/PyCharm/PythonProject/DeepLearn/LibEER/data_utils/Dataset/SEED'
            }]
        return JsonResponse({
            'code': 200,
            'message': 'success',
            'data': datasets
        })
    except Exception as e:
        return JsonResponse({
            'code': 500,
            'message': f'获取数据集失败: {str(e)}',
            'data': [{'id': 'SEED', 'name': 'SEED 数据集', 'path': 'D:/Work/Studio/PyCharm/PythonProject/DeepLearn/LibEER/data_utils/Dataset/SEED'}]
        })


@csrf_exempt
@require_http_methods(["GET"])
def get_files(request):
    """
    获取MAT文件列表
    GET /api/files?dataset={dataset}&type={type}
    """
    try:
        dataset = request.GET.get('dataset', 'SEED')
        file_type = request.GET.get('type', 'preprocessed')

        files = loader.get_mat_files(dataset, file_type)

        # 如果没有找到真实文件，返回模拟文件列表
        if not files:
            if file_type == 'preprocessed':
                files = [
                    {'id': '1_20131027', 'name': '1_20131027.mat', 'path': '', 'type': 'preprocessed', 'session': '1',
                     'subject': '20131027', 'is_label': False},
                    {'id': '2_20131027', 'name': '2_20131027.mat', 'path': '', 'type': 'preprocessed', 'session': '2',
                     'subject': '20131027', 'is_label': False},
                    {'id': '3_20131027', 'name': '3_20131027.mat', 'path': '', 'type': 'preprocessed', 'session': '3',
                     'subject': '20131027', 'is_label': False},
                    {'id': 'label', 'name': 'label.mat', 'path': '', 'type': 'preprocessed', 'session': None,
                     'subject': 'label', 'is_label': True},
                ]
            else:
                files = [
                    {'id': '1_20131027', 'name': '1_20131027.mat', 'path': '', 'type': 'extracted', 'session': '1',
                     'subject': '20131027', 'is_label': False},
                    {'id': '2_20131027', 'name': '2_20131027.mat', 'path': '', 'type': 'extracted', 'session': '2',
                     'subject': '20131027', 'is_label': False},
                    {'id': 'label', 'name': 'label.mat', 'path': '', 'type': 'extracted', 'session': None,
                     'subject': 'label', 'is_label': True},
                ]

        return JsonResponse({
            'code': 200,
            'message': 'success',
            'data': files
        })
    except Exception as e:
        return JsonResponse({
            'code': 500,
            'message': f'获取文件列表失败: {str(e)}',
            'data': []
        })


@csrf_exempt
@require_http_methods(["GET"])
def get_trials(request):
    """
    获取trial列表
    GET /api/trials?file_path={file_path}
    """
    try:
        file_path = request.GET.get('file_path')

        if file_path:
            trials = loader.get_trial_list(file_path)
        else:
            # 默认返回5个模拟trials
            trials = [{'index': i, 'name': f'Trial {i + 1}'} for i in range(5)]

        return JsonResponse({
            'code': 200,
            'message': 'success',
            'data': trials
        })
    except Exception as e:
        return JsonResponse({
            'code': 500,
            'message': f'获取trial列表失败: {str(e)}',
            'data': [{'index': i, 'name': f'Trial {i + 1}'} for i in range(5)]
        })


# ========== 数据加载接口 ==========

@csrf_exempt
@require_http_methods(["POST"])
def load_eeg_data(request):
    """
    加载EEG数据
    POST /api/load_eeg
    请求体: {"file_path": "/path/to/file.mat", "trial_index": 0}
    """
    try:
        body = json.loads(request.body)
        file_path = body.get('file_path')
        trial_index = body.get('trial_index', 0)

        # 如果有文件路径，尝试加载真实数据
        if file_path:
            try:
                result = loader.load_preprocessed_eeg(file_path, trial_index)
                eeg_data = result['eeg_data']
                channels = result['channels']
                metadata = result['metadata']
                sampling_rate = result['sampling_rate']

                # 获取trial列表
                trials = loader.get_trial_list(file_path)
            except Exception as e:
                # 加载失败，使用模拟数据
                print(f"加载真实数据失败: {e}，使用模拟数据")
                mock_data = get_mock_eeg_data()
                eeg_data = mock_data
                channels = loader.ch_names[:62]
                metadata = {'format': '2d', 'n_channels': 62, 'n_times': 1000, 'n_trials': 1}
                sampling_rate = 200
                trials = [{'index': 0, 'name': 'Trial 1'}]
        else:
            # 没有文件路径，使用模拟数据
            mock_data = get_mock_eeg_data()
            eeg_data = mock_data
            channels = loader.ch_names[:62]
            metadata = {'format': '2d', 'n_channels': 62, 'n_times': 1000, 'n_trials': 1}
            sampling_rate = 200
            trials = [{'index': 0, 'name': 'Trial 1'}]

        return JsonResponse({
            'code': 200,
            'message': 'success',
            'data': {
                'eeg_data': eeg_data.tolist(),
                'channels': channels,
                'metadata': metadata,
                'sampling_rate': sampling_rate,
                'trials': trials
            }
        })
    except Exception as e:
        return JsonResponse({
            'code': 500,
            'message': f'加载EEG数据失败: {str(e)}',
            'data': None
        })


@csrf_exempt
@require_http_methods(["POST"])
def load_features(request):
    """
    加载特征数据
    POST /api/load_features
    请求体: {"file_path": "/path/to/file.mat", "trial_index": 0}
    """
    try:
        body = json.loads(request.body)
        file_path = body.get('file_path')
        trial_index = body.get('trial_index', 0)

        if file_path:
            try:
                result = loader.load_extracted_features(file_path, trial_index)
            except:
                # 使用模拟特征
                result = {
                    'features': np.random.randn(62, 10),
                    'metadata': {'n_channels': 62, 'n_features': 10},
                    'channels': loader.ch_names[:62]
                }
        else:
            # 使用模拟特征
            result = {
                'features': np.random.randn(62, 10),
                'metadata': {'n_channels': 62, 'n_features': 10},
                'channels': loader.ch_names[:62]
            }

        return JsonResponse({
            'code': 200,
            'message': 'success',
            'data': {
                'features': result['features'].tolist() if hasattr(result['features'], 'tolist') else result[
                    'features'],
                'channels': result['channels'],
                'metadata': result['metadata']
            }
        })
    except Exception as e:
        return JsonResponse({
            'code': 500,
            'message': f'加载特征数据失败: {str(e)}',
            'data': None
        })


@csrf_exempt
@require_http_methods(["POST"])
def load_labels(request):
    """
    加载情绪标签
    POST /api/load_labels
    请求体: {"file_path": "/path/to/label.mat"}
    """
    try:
        body = json.loads(request.body)
        file_path = body.get('file_path')

        if file_path:
            try:
                result = loader.load_label_file(file_path)
            except:
                # 使用模拟标签
                labels = np.random.choice([-1, 0, 1], size=15)
                unique, counts = np.unique(labels, return_counts=True)
                distribution = []
                for label, count in zip(unique, counts):
                    emotion = {1: '积极', 0: '中立', -1: '消极'}.get(int(label), '未知')
                    distribution.append({
                        'label': int(label),
                        'emotion': emotion,
                        'count': int(count),
                        'percentage': float(count / 15 * 100)
                    })
                result = {
                    'labels': labels.tolist(),
                    'distribution': distribution,
                    'total': 15
                }
        else:
            # 使用模拟标签
            labels = np.random.choice([-1, 0, 1], size=15)
            unique, counts = np.unique(labels, return_counts=True)
            distribution = []
            for label, count in zip(unique, counts):
                emotion = {1: '积极', 0: '中立', -1: '消极'}.get(int(label), '未知')
                distribution.append({
                    'label': int(label),
                    'emotion': emotion,
                    'count': int(count),
                    'percentage': float(count / 15 * 100)
                })
            result = {
                'labels': labels.tolist(),
                'distribution': distribution,
                'total': 15
            }

        return JsonResponse({
            'code': 200,
            'message': 'success',
            'data': result
        })
    except Exception as e:
        return JsonResponse({
            'code': 500,
            'message': f'加载标签数据失败: {str(e)}',
            'data': None
        })


# ========== 信号处理接口 ==========

@csrf_exempt
@require_http_methods(["POST"])
def calculate_psd(request):
    """
    计算PSD功率谱密度
    POST /api/psd
    请求体: {"eeg_data": [...], "trial_index": 0} 或 {"file_path": "...", "trial_index": 0}
    """
    try:
        body = json.loads(request.body)

        # 优先使用传入的EEG数据
        if 'eeg_data' in body:
            eeg_data = np.array(body['eeg_data'])
        elif 'file_path' in body:
            # 从文件加载
            file_path = body['file_path']
            trial_index = body.get('trial_index', 0)
            load_result = loader.load_preprocessed_eeg(file_path, trial_index)
            eeg_data = load_result['eeg_data']
        else:
            # 使用模拟数据
            eeg_data = get_mock_eeg_data()

        # 计算PSD
        psd_result = processor.calculate_psd(eeg_data)

        return JsonResponse({
            'code': 200,
            'message': 'success',
            'data': psd_result
        })
    except Exception as e:
        return JsonResponse({
            'code': 500,
            'message': f'PSD计算失败: {str(e)}',
            'data': None
        })


@csrf_exempt
@require_http_methods(["POST"])
def detect_stable(request):
    """
    检测稳定区域
    POST /api/stable
    请求体: {"eeg_data": [...], "trial_index": 0, "threshold": 0.1}
    """
    try:
        body = json.loads(request.body)

        if 'eeg_data' in body:
            eeg_data = np.array(body['eeg_data'])
        elif 'file_path' in body:
            file_path = body['file_path']
            trial_index = body.get('trial_index', 0)
            load_result = loader.load_preprocessed_eeg(file_path, trial_index)
            eeg_data = load_result['eeg_data']
        else:
            eeg_data = get_mock_eeg_data()

        threshold = body.get('threshold', 0.1)
        window_size = body.get('window_size', 50)

        stable_result = processor.detect_stable_region(
            eeg_data,
            threshold=threshold,
            window_size=window_size
        )

        return JsonResponse({
            'code': 200,
            'message': 'success',
            'data': stable_result
        })
    except Exception as e:
        return JsonResponse({
            'code': 500,
            'message': f'稳定区域检测失败: {str(e)}',
            'data': None
        })


@csrf_exempt
@require_http_methods(["POST"])
def generate_topomap(request):
    """
    生成Topomap
    POST /api/topomap
    请求体: {"eeg_data": [...], "trial_index": 0, "time_point": 500}
    """
    try:
        body = json.loads(request.body)

        if 'eeg_data' in body:
            eeg_data = np.array(body['eeg_data'])
        elif 'file_path' in body:
            file_path = body['file_path']
            trial_index = body.get('trial_index', 0)
            load_result = loader.load_preprocessed_eeg(file_path, trial_index)
            eeg_data = load_result['eeg_data']
        else:
            eeg_data = get_mock_eeg_data()

        time_point = body.get('time_point')
        time_window = body.get('time_window')

        img_base64 = processor.generate_topomap(
            eeg_data,
            time_point=time_point,
            time_window=time_window
        )

        return JsonResponse({
            'code': 200,
            'message': 'success',
            'data': {
                'image': img_base64
            }
        })
    except Exception as e:
        return JsonResponse({
            'code': 500,
            'message': f'Topomap生成失败: {str(e)}',
            'data': None
        })


@csrf_exempt
@require_http_methods(["POST"])
def extract_features_api(request):
    """
    提取EEG特征
    POST /api/features
    请求体: {"eeg_data": [...], "trial_index": 0}
    """
    try:
        body = json.loads(request.body)

        if 'eeg_data' in body:
            eeg_data = np.array(body['eeg_data'])
        elif 'file_path' in body:
            file_path = body['file_path']
            trial_index = body.get('trial_index', 0)
            load_result = loader.load_preprocessed_eeg(file_path, trial_index)
            eeg_data = load_result['eeg_data']
        else:
            eeg_data = get_mock_eeg_data()

        features = processor.extract_features(eeg_data)

        return JsonResponse({
            'code': 200,
            'message': 'success',
            'data': {
                'channels': loader.ch_names[:len(eeg_data)],
                'features': features
            }
        })
    except Exception as e:
        return JsonResponse({
            'code': 500,
            'message': f'特征提取失败: {str(e)}',
            'data': None
        })


@csrf_exempt
@require_http_methods(["POST"])
def compare_sessions(request):
    """
    对比多个session
    POST /api/session_compare
    请求体: {"sessions": {"session1": {"file_path": "...", "trial_index": 0}, ...}, "type": "eeg"}
    """
    try:
        body = json.loads(request.body)
        sessions_config = body.get('sessions', {})
        comparison_type = body.get('type', 'eeg')

        sessions_data = {}
        for session_name, config in sessions_config.items():
            file_path = config.get('file_path')
            trial_index = config.get('trial_index', 0)

            if file_path:
                load_result = loader.load_preprocessed_eeg(file_path, trial_index)
                sessions_data[session_name] = load_result['eeg_data']
            else:
                # 生成不同的模拟数据
                np.random.seed(hash(session_name) % 100)
                sessions_data[session_name] = np.random.randn(62, 1000) * (10 + len(sessions_data))

        result = processor.compare_sessions(sessions_data, comparison_type)

        return JsonResponse({
            'code': 200,
            'message': 'success',
            'data': result
        })
    except Exception as e:
        return JsonResponse({
            'code': 500,
            'message': f'Session对比失败: {str(e)}',
            'data': None
        })


@csrf_exempt
@require_http_methods(["POST"])
def analyze_emotion(request):
    """
    分析情绪标签分布
    POST /api/emotion_analysis
    请求体: {"labels": [1, 0, -1, ...]} 或 {"file_path": "..."}
    """
    try:
        body = json.loads(request.body)

        if 'labels' in body:
            labels = np.array(body['labels'])
        elif 'file_path' in body:
            file_path = body['file_path']
            result = loader.load_label_file(file_path)
            labels = np.array(result['labels'])
        else:
            # 模拟标签
            labels = np.random.choice([-1, 0, 1], size=15)

        result = processor.analyze_emotion_distribution(labels)

        return JsonResponse({
            'code': 200,
            'message': 'success',
            'data': result
        })
    except Exception as e:
        return JsonResponse({
            'code': 500,
            'message': f'情绪分析失败: {str(e)}',
            'data': None
        })


# ========== 整合处理接口 ==========

@csrf_exempt
@require_http_methods(["POST"])
def process_eeg(request):
    """
    整合的EEG处理接口
    POST /api/process_eeg
    请求体: {
        "file_path": "/path/to/file.mat",
        "trial_index": 0,
        "process_type": "psd|stable|topomap|features|emotion",
        "params": {...}
    }
    """
    try:
        body = json.loads(request.body)
        file_path = body.get('file_path')
        trial_index = body.get('trial_index', 0)
        process_type = body.get('process_type', 'psd')
        params = body.get('params', {})

        # 加载EEG数据（如果不是情绪分析）
        if process_type != 'emotion':
            if file_path:
                try:
                    load_result = loader.load_preprocessed_eeg(file_path, trial_index)
                    eeg_data = load_result['eeg_data']
                except:
                    eeg_data = get_mock_eeg_data()
            else:
                eeg_data = get_mock_eeg_data()

        # 根据处理类型调用对应方法
        result = None
        if process_type == 'psd':
            psd_result = processor.calculate_psd(eeg_data)
            result = {
                'type': 'psd',
                'data': psd_result
            }
        elif process_type == 'stable':
            threshold = params.get('threshold', 0.1)
            window_size = params.get('window_size', 50)
            stable_result = processor.detect_stable_region(
                eeg_data,
                threshold=threshold,
                window_size=window_size
            )
            result = {
                'type': 'stable',
                'data': stable_result
            }
        elif process_type == 'topomap':
            time_point = params.get('time_point')
            time_window = params.get('time_window')
            img_base64 = processor.generate_topomap(
                eeg_data,
                time_point=time_point,
                time_window=time_window
            )
            result = {
                'type': 'topomap',
                'data': {'image': img_base64}
            }
        elif process_type == 'features':
            features_result = processor.extract_features(eeg_data)
            result = {
                'type': 'features',
                'data': features_result
            }
        elif process_type == 'emotion':
            # 情绪分析需要标签文件
            if file_path:
                try:
                    label_result = loader.load_label_file(file_path)
                    labels = np.array(label_result['labels'])
                except:
                    labels = np.random.choice([-1, 0, 1], size=15)
            else:
                labels = np.random.choice([-1, 0, 1], size=15)

            emotion_result = processor.analyze_emotion_distribution(labels)
            result = {
                'type': 'emotion',
                'data': emotion_result
            }
        else:
            return JsonResponse({
                'code': 400,
                'message': f'不支持的处理类型: {process_type}',
                'data': None
            })

        return JsonResponse({
            'code': 200,
            'message': 'success',
            'data': result
        })
    except Exception as e:
        return JsonResponse({
            'code': 500,
            'message': f'处理失败: {str(e)}',
            'data': None
        })