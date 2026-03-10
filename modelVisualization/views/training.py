import json
import os
import tempfile

from django.http import JsonResponse
from django.template import TemplateDoesNotExist
from django.template.loader import render_to_string
from django.views.decorators.csrf import csrf_exempt

from modelVisualization.utils.model_registry import ModelRegistry


def _is_float(value):
    """判断是否为浮点数"""
    try:
        float(value)
        return True
    except ValueError:
        return False

def load_model_intro(request):
    """加载模型简介页面"""
    html = render_to_string('training/introduction.html')
    return JsonResponse({'html': html})


def load_model_registration(request):
    """加载模型注册页面"""
    html = render_to_string('training/register_model.html')
    return JsonResponse({'html': html})

def load_model_history(request):
    """加载训练历史页面"""
    html = render_to_string('training/history.html')
    return JsonResponse({'html': html})


def load_model_interface(request):
    """加载模型使用界面"""
    model_name = request.GET.get('model', '')
    category = request.GET.get('category', '')

    # 获取模型配置和预设
    registry = ModelRegistry()

    # 获取默认配置
    config = registry.get_model_config(model_name)
    if not config:
        config = {}

    # 获取所有预设
    presets = registry.get_model_presets(model_name)

    context = {
        'model_name': model_name,
        'category': category,
        'config': config,
        'presets': presets,
        'current_preset': 'default' if 'default' in presets else (presets[0] if presets else '')
    }

    html = render_to_string('training/model_general.html', context)
    return JsonResponse({'html': html})


@csrf_exempt
def upload_model_code(request):
    """上传模型代码文件"""
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']

        # 保存到临时目录
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(file_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        return JsonResponse({
            'code': 0,
            'filename': uploaded_file.name,
            'filepath': file_path
        })

    return JsonResponse({'code': 1, 'msg': '上传失败'})


@csrf_exempt
def upload_train_code(request):
    """上传训练代码文件"""
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']

        # 保存到临时目录
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(file_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        return JsonResponse({
            'code': 0,
            'filename': uploaded_file.name,
            'filepath': file_path
        })

    return JsonResponse({'code': 1, 'msg': '上传失败'})


@csrf_exempt
def register_model(request):
    """注册新模型"""
    if request.method == 'POST':
        model_name = request.POST.get('model_name')
        category = request.POST.get('category', '')
        description = request.POST.get('description', '')
        model_code_path = request.POST.get('model_code_path')
        train_code_path = request.POST.get('train_code_path')

        # 解析配置覆盖项
        config_overrides = {}
        config_str = request.POST.get('config_overrides')
        if config_str:
            try:
                config_overrides = json.loads(config_str)
            except:
                pass

        # 参数验证
        if not model_name:
            return JsonResponse({'success': False, 'message': '模型名称不能为空'})

        # 注册模型
        registry = ModelRegistry()
        success, message = registry.register_model(
            model_name=model_name,
            category=category,
            description=description,
            model_code_file=model_code_path if model_code_path else None,
            train_code_file=train_code_path if train_code_path else None,
            config_overrides=config_overrides
        )

        return JsonResponse({'success': success, 'message': message})

    return JsonResponse({'success': False, 'message': '无效的请求'})


@csrf_exempt
def get_preset_config(request):
    """获取预设配置"""
    model_name = request.GET.get('model_name', '')
    preset_name = request.GET.get('preset_name', 'default')

    registry = ModelRegistry()
    config = registry.get_preset_config(model_name, preset_name)

    if config:
        return JsonResponse({'success': True, 'config': config})
    return JsonResponse({'success': False, 'message': '配置不存在'})


@csrf_exempt
def get_model_presets(request):
    """获取模型的所有预设"""
    model_name = request.GET.get('model_name', '')

    registry = ModelRegistry()
    presets = registry.get_model_presets(model_name)

    return JsonResponse({'success': True, 'presets': presets})


@csrf_exempt
def save_preset(request):
    """保存预设"""
    if request.method == 'POST':
        model_name = request.POST.get('model_name', '')
        preset_name = request.POST.get('preset_name', '')
        if not model_name or not preset_name:
            return JsonResponse({'success': False, 'message': '参数缺失'})

        # 收集配置数据
        config_data = {}
        for key in request.POST:
            if key not in ['model_name', 'preset_name', 'csrfmiddlewaretoken']:
                # 关键修改：使用 getlist 而不是 get
                values = request.POST.getlist(key)
                if key[-1] == ']' :
                    key = key[:-2]
                if len(values) == 1:
                    config_data[key] = values[0]
                elif len(values) > 1:
                    config_data[key] = values

        registry = ModelRegistry()
        print(config_data)
        if registry.save_preset(model_name, preset_name, config_data):
            return JsonResponse({'success': True, 'message': '预设保存成功'})

        return JsonResponse({'success': False, 'message': '保存失败'})

    return JsonResponse({'success': False, 'message': '无效请求'})


@csrf_exempt
def delete_preset(request):
    """删除预设"""
    if request.method == 'POST':
        model_name = request.POST.get('model_name', '')
        preset_name = request.POST.get('preset_name', '')

        registry = ModelRegistry()

        success, message = registry.delete_preset(model_name, preset_name)
        return JsonResponse({'success': success, 'message': message})

    return JsonResponse({'success': False, 'message': '无效请求'})



@csrf_exempt
def start_training(request):
    """开始训练"""
    if request.method == 'POST':
        model_name = request.POST.get('model_name', '')

        # 收集所有配置参数
        config_data = {}
        for key in request.POST:
            if key not in ['model_name', 'category', 'csrfmiddlewaretoken']:
                values = request.POST.getlist(key)
                if key[-1] == ']':
                    key = key[:-2]
                if len(values) == 1:
                    config_data[key] = values[0]
                elif len(values) > 1:
                    config_data[key] = values

        print(config_data)
        # 这里需要实现具体的训练逻辑
        # 可以调用 subprocess 运行训练脚本
        # 暂时返回成功
        import uuid
        task_id = str(uuid.uuid4())[:8]

        # 找到训练文件
        registry = ModelRegistry()

        for model in registry.get_all_models():
            if model['name'] == model_name and model.get('train_file'):
                train_path = os.path.join(registry.code_dir, model['train_file'])

                # 导入模块
                import importlib.util
                spec = importlib.util.spec_from_file_location("train", train_path)
                train_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(train_module)

                # 创建 ModelConfig 对象
                from modelVisualization.libeer.config_class import ModelConfig
                config_obj = ModelConfig()

                # 用收集的数据更新配置
                for key, value in config_data.items():
                    if hasattr(config_obj, key):
                        if type(value) != list:
                            if value.isdigit():
                                value = int(value)
                            elif _is_float(value):
                                value = float(value)
                        setattr(config_obj, key, value)

                # 关键：传递 config_data 给 main 函数
                import threading
                threading.Thread(target=train_module.main, args=(config_obj,)).start()

                return JsonResponse({'success': True, 'message': '训练开始'})

    return JsonResponse({'success': False, 'message': '无效请求'})

