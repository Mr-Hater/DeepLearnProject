from django.urls import path

from modelVisualization.views import training

urlpatterns = [
    # 模型注册相关
    path('upload-model-code/', training.upload_model_code, name='upload_model_code'),
    path('upload-train-code/', training.upload_train_code, name='upload_train_code'),
    path('register-model/', training.register_model, name='register_model'),

    # 模型注册相关
    # path('load-model-registration/', training.load_model_registration, name='load_model_registration'),
    path('upload-model-code/', training.upload_model_code, name='upload_model_code'),
    path('upload-train-code/', training.upload_train_code, name='upload_train_code'),
    path('register-model/', training.register_model, name='register_model'),

    # 模型使用界面相关
    # path('load-model-interface/', training.load_model_interface, name='load_model_interface'),

    # 预设管理相关
    path('get-preset-config/', training.get_preset_config, name='get_preset_config'),
    path('get-model-presets/', training.get_model_presets, name='get_model_presets'),
    path('save-preset/', training.save_preset, name='save_preset'),
    path('delete-preset/', training.delete_preset, name='delete_preset'),

    # 训练相关
    path('start-training/', training.start_training, name='start_training'),

    # 其他辅助功能
    # path('get-model-config/', training.get_model_config, name='get_model_config'),
    # path('save-model-config/', training.save_model_config, name='save_model_config'),
    # path('get-model-code/', training.get_model_code, name='get_model_code'),
    # path('quick-train/', training.quick_train, name='quick_train'),
]