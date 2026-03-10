# backend/views_with_loader.py
"""
使用MATLoader的API视图
"""

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from modelVisualization.services.SeedVisualization.mat_loader import MATLoader
from modelVisualization.services.SeedVisualization.singnal_processor import SignalProcessor

# 初始化加载器和处理器
loader = MATLoader(data_root='D:/Work/Studio/PyCharm/PythonProject/DeepLearn/LibEER/data_utils/Dataset/SEED')  # 根据实际路径调整
processor = SignalProcessor()


@csrf_exempt
@require_http_methods(["GET"])
def get_datasets(request):
    """
    获取数据集列表
    GET /api/datasets
    """
    try:
        datasets = loader.get_available_datasets()
        return JsonResponse({
            'code': 200,
            'message': 'success',
            'data': datasets
        })
    except Exception as e:
        return JsonResponse({
            'code': 500,
            'message': f'获取数据集失败: {str(e)}',
            'data': None
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

        return JsonResponse({
            'code': 200,
            'message': 'success',
            'data': files
        })
    except Exception as e:
        return JsonResponse({
            'code': 500,
            'message': f'获取文件列表失败: {str(e)}',
            'data': None
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
        if not file_path:
            return JsonResponse({
                'code': 400,
                'message': '缺少file_path参数',
                'data': None
            })

        trials = loader.get_trial_list(file_path)

        return JsonResponse({
            'code': 200,
            'message': 'success',
            'data': trials
        })
    except Exception as e:
        return JsonResponse({
            'code': 500,
            'message': f'获取trial列表失败: {str(e)}',
            'data': None
        })


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
        trial_index = body.get('trial_index')

        if not file_path:
            return JsonResponse({
                'code': 400,
                'message': '缺少file_path参数',
                'data': None
            })

        # 加载EEG数据
        result = loader.load_preprocessed_eeg(file_path, trial_index)

        # 获取trial列表
        trials = loader.get_trial_list(file_path)

        return JsonResponse({
            'code': 200,
            'message': 'success',
            'data': {
                'eeg_data': result['eeg_data'].tolist(),
                'channels': result['channels'],
                'metadata': result['metadata'],
                'sampling_rate': result['sampling_rate'],
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
        trial_index = body.get('trial_index')

        if not file_path:
            return JsonResponse({
                'code': 400,
                'message': '缺少file_path参数',
                'data': None
            })

        # 加载特征数据
        result = loader.load_extracted_features(file_path, trial_index)

        return JsonResponse({
            'code': 200,
            'message': 'success',
            'data': {
                'features': result['features'].tolist(),
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

        if not file_path:
            return JsonResponse({
                'code': 400,
                'message': '缺少file_path参数',
                'data': None
            })

        # 加载标签数据
        result = loader.load_label_file(file_path)

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


@csrf_exempt
@require_http_methods(["POST"])
def process_eeg_with_loader(request):
    """
    使用加载的数据进行EEG处理
    POST /api/process_eeg
    请求体: {
        "file_path": "/path/to/file.mat",
        "trial_index": 0,
        "process_type": "psd|stable|topomap|features"
    }
    """
    try:
        body = json.loads(request.body)
        file_path = body.get('file_path')
        trial_index = body.get('trial_index', 0)
        process_type = body.get('process_type', 'psd')

        if not file_path:
            return JsonResponse({
                'code': 400,
                'message': '缺少file_path参数',
                'data': None
            })

        # 加载EEG数据
        load_result = loader.load_preprocessed_eeg(file_path, trial_index)
        eeg_data = load_result['eeg_data']

        # 根据处理类型调用不同的处理方法
        result = None
        if process_type == 'psd':
            psd_result = processor.calculate_psd(eeg_data)
            result = {
                'type': 'psd',
                'data': psd_result,
                'channels': load_result['channels']
            }
        elif process_type == 'stable':
            stable_result = processor.detect_stable_region(eeg_data)
            result = {
                'type': 'stable',
                'data': stable_result,
                'channels': load_result['channels']
            }
        elif process_type == 'topomap':
            topomap_result = processor.generate_topomap(eeg_data)
            result = {
                'type': 'topomap',
                'data': topomap_result,
                'channels': load_result['channels']
            }
        elif process_type == 'features':
            features_result = processor.extract_features(eeg_data)
            result = {
                'type': 'features',
                'data': features_result,
                'channels': load_result['channels']
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