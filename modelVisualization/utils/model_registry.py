# utils/model_registry.py
import os
import yaml
from datetime import datetime
import shutil

from modelVisualization.libeer.config_class import ModelConfig


class ModelRegistry:
    """模型注册管理器"""

    def __init__(self, registry_path='./models_registry/'):
        self.registry_path = registry_path
        self.models_file = os.path.join(registry_path, 'models.yaml')
        self.configs_dir = os.path.join(registry_path, 'configs')
        self.code_dir = os.path.join(registry_path, 'code')

        # 初始化目录结构
        self._init_dirs()

    def _init_dirs(self):
        """初始化目录结构"""
        os.makedirs(self.registry_path, exist_ok=True)
        os.makedirs(self.configs_dir, exist_ok=True)
        os.makedirs(self.code_dir, exist_ok=True)

        # 如果models.yaml不存在，创建空文件
        if not os.path.exists(self.models_file):
            with open(self.models_file, 'w', encoding='utf-8') as f:
                yaml.dump({'models': []}, f, allow_unicode=True)

    def register_model(self, model_name, category, description,
                       model_code_file=None, train_code_file=None, config_overrides=None):
        """
        注册新模型

        Args:
            model_name: 模型名称（必填）
            category: 模型分类
            description: 模型描述
            model_code_file: 模型代码文件路径
            train_code_file: 训练代码文件路径
            config_overrides: 配置覆盖项
        """
        # 检查是否已存在
        existing_models = self.get_all_models()
        for model in existing_models:
            if model['name'] == model_name:
                return False, f"模型 '{model_name}' 已存在"

        # 创建配置文件（使用ModelConfig默认值 + 用户覆盖）
        config = ModelConfig()

        # 设置模型名称
        config.model = model_name

        # 应用用户覆盖的配置
        if config_overrides:
            config.update(**config_overrides)

        # 保存配置文件
        config_filename = f"{model_name}.yaml"
        config_path = os.path.join(self.configs_dir, config_filename)

        config_dict = config.__dict__
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True)

        # 复制代码文件（如果有）
        model_code_filename = None
        train_code_filename = None

        if model_code_file and os.path.exists(model_code_file):
            model_code_filename = f"{model_name}.py"
            shutil.copy(model_code_file, os.path.join(self.code_dir, model_code_filename))

        if train_code_file and os.path.exists(train_code_file):
            train_code_filename = f"{model_name}_train.py"
            shutil.copy(train_code_file, os.path.join(self.code_dir, train_code_filename))

        # 添加到注册表
        new_model = {
            'name': model_name,
            'category': category,
            'description': description,
            'code_file': model_code_filename,
            'train_file': train_code_filename,
            'config_file': config_filename,
            'registered_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        existing_models.append(new_model)

        # 保存更新
        with open(self.models_file, 'w', encoding='utf-8') as f:
            yaml.dump({'models': existing_models}, f, allow_unicode=True)

        return True, "模型注册成功"

    def get_all_models(self):
        """获取所有已注册模型"""
        if not os.path.exists(self.models_file):
            return []

        with open(self.models_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return data.get('models', [])

    def get_models_by_category(self):
        """按分类获取模型"""
        models = self.get_all_models()
        categorized = {}

        for model in models:
            category = model['category']
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(model['name'])

        return categorized

    def get_model_config(self, model_name):
        """获取模型的配置"""
        models = self.get_all_models()
        for model in models:
            if model['name'] == model_name:
                config_path = os.path.join(self.configs_dir, model['config_file'])
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        return yaml.safe_load(f)
        return None

    def delete_model(self, model_name):
        """删除模型"""
        models = self.get_all_models()
        updated_models = []
        deleted = False

        for model in models:
            if model['name'] == model_name:
                # 删除配置文件
                config_path = os.path.join(self.configs_dir, model['config_file'])
                if os.path.exists(config_path):
                    os.remove(config_path)

                # 删除代码文件
                if model['code_file']:
                    code_path = os.path.join(self.code_dir, model['code_file'])
                    if os.path.exists(code_path):
                        os.remove(code_path)

                if model['train_file']:
                    train_path = os.path.join(self.code_dir, model['train_file'])
                    if os.path.exists(train_path):
                        os.remove(train_path)

                deleted = True
            else:
                updated_models.append(model)

        if deleted:
            with open(self.models_file, 'w', encoding='utf-8') as f:
                yaml.dump({'models': updated_models}, f, allow_unicode=True)
            return True
        return False

    def get_model_presets(self, model_name):
        """获取模型的所有预设"""
        preset_dir = os.path.join(self.configs_dir, model_name)
        if not os.path.exists(preset_dir):
            return []

        presets = []
        for filename in os.listdir(preset_dir):
            if filename.endswith('.yaml'):
                preset_name = filename.replace('.yaml', '')
                presets.append(preset_name)

        return sorted(presets)

    def get_preset_config(self, model_name, preset_name):
        """获取指定预设的配置"""
        preset_file = os.path.join(self.configs_dir, model_name, f"{preset_name}.yaml")
        if os.path.exists(preset_file):
            with open(preset_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return None

    def save_preset(self, model_name, preset_name, config_data):
        """保存预设配置"""
        preset_dir = os.path.join(self.configs_dir, model_name)
        os.makedirs(preset_dir, exist_ok=True)

        preset_file = os.path.join(preset_dir, f"{preset_name}.yaml")
        with open(preset_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, allow_unicode=True)

        return True

    def delete_preset(self, model_name, preset_name):
        """删除预设"""
        if preset_name == 'default':
            return False, "不能删除默认预设"

        preset_dir = os.path.join(self.configs_dir, model_name)
        preset_file = os.path.join(preset_dir, f"{preset_name}.yaml")

        if os.path.exists(preset_file):
            try:
                os.remove(preset_file)
                return True, "删除成功"
            except Exception as e:
                return False, f"删除失败: {e}"
        else:
            return False, "预设文件不存在"

    def get_all_config_keys(self):
        """获取所有可能的配置键（从ModelConfig）"""

        config = ModelConfig()
        return list(config.__dict__.keys())