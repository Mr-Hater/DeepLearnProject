from django.urls import path

from modelVisualization.views import seedVisualization
from modelVisualization.views import views_with_loader

urlpatterns = [
    path('load-topomap/', seedVisualization.load_topomap, name='load-topomap'),
    path('load-singleEEG/', seedVisualization.load_singleEEG, name='load-singleEEG'),
    path('load-doubleEEG/', seedVisualization.load_doubleEEG, name='load-doubleEEG'),

    # path('load-eeg-data/', seedVisualization.load_eeg_data, name='load_eeg_data'),

    # ========== 数据浏览接口 ==========
    path('datasets/', seedVisualization.get_datasets, name='get_datasets'),
    path('files/', seedVisualization.get_files, name='get_files'),
    path('trials/', seedVisualization.get_trials, name='get_trials'),

    # ========== 数据加载接口 ==========
    path('load_eeg/', seedVisualization.load_eeg_data, name='load_eeg_data'),
    path('load_features/', seedVisualization.load_features, name='load_features'),
    path('load_labels/', seedVisualization.load_labels, name='load_labels'),

    # ========== 信号处理接口 ==========
    path('psd/', seedVisualization.calculate_psd, name='calculate_psd'),
    path('stable/', seedVisualization.detect_stable, name='detect_stable'),
    path('topomap/', seedVisualization.generate_topomap, name='generate_topomap'),
    path('features/', seedVisualization.extract_features_api, name='extract_features'),
    path('session_compare/', seedVisualization.compare_sessions, name='compare_sessions'),
    path('emotion_analysis/', seedVisualization.analyze_emotion, name='analyze_emotion'),

    # ========== 整合处理接口 ==========
    path('process_eeg', seedVisualization.process_eeg, name='process_eeg'),
]