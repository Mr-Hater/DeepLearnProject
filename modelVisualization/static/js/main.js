// frontend/js/main.js
/**
 * 脑电信号分析平台 - 整合后的核心交互逻辑
 * 支持真实数据加载和模拟数据备选
 */

layui.use(['layer', 'form'], function() {
    var layer = layui.layer;
    var form = layui.form;

    // API基础URL
    const API_BASE = 'http://localhost:8000/api';

    // 图表实例
    let charts = {
        eeg: null,
        psd: null,
        stable: null,
        topomap: null,
        emotion: null
    };

    // 当前选中的数据
    let currentData = {
        dataset: 'SEED',
        file: null,
        filePath: null,
        fileType: 'preprocessed',
        trialIndex: 0,
        eegData: null,
        channels: [],
        samplingRate: 200
    };

    // 文件列表缓存
    let fileList = [];

    /**
     * 更新状态栏
     */
    function updateStatus(message, type = 'info') {
        const statusBar = document.getElementById('statusBar');
        const statusText = document.getElementById('statusText');

        let icon = 'layui-icon-tips';
        let color = '#0050b3';
        let bgColor = '#e6f7ff';
        let borderColor = '#91d5ff';

        if (type === 'success') {
            icon = 'layui-icon-ok-circle';
            color = '#52c41a';
            bgColor = '#f6ffed';
            borderColor = '#b7eb8f';
        } else if (type === 'error') {
            icon = 'layui-icon-close-fill';
            color = '#f5222d';
            bgColor = '#fff1f0';
            borderColor = '#ffa39e';
        } else if (type === 'warning') {
            icon = 'layui-icon-help';
            color = '#faad14';
            bgColor = '#fffbe6';
            borderColor = '#ffe58f';
        }

        statusText.innerHTML = message;
        statusBar.style.color = color;
        statusBar.style.backgroundColor = bgColor;
        statusBar.style.borderColor = borderColor;

        // 更新图标
        const iconElement = statusBar.querySelector('i');
        if (iconElement) {
            iconElement.className = `layui-icon ${icon}`;
        }
    }

    /**
     * 初始化所有图表
     */
    function initCharts() {

        // EEG时域图
        charts.eeg = echarts.init(document.getElementById('chartEeg'));
        charts.eeg.setOption({
            title: { show: false },
            tooltip: { trigger: 'axis' },
            legend: { show: true, type: 'scroll', pageIconColor: '#667eea' },
            grid: { left: '5%', right: '5%', bottom: '5%', top: '15%', containLabel: true },
            xAxis: { type: 'category', name: '时间点' },
            yAxis: { type: 'value', name: '幅值 (μV)' },
            series: []
        });

        // PSD频谱图
        charts.psd = echarts.init(document.getElementById('chartPsd'));
        charts.psd.setOption({
            title: { show: false },
            tooltip: { trigger: 'axis' },
            legend: { show: true, type: 'scroll', pageIconColor: '#667eea' },
            grid: { left: '5%', right: '5%', bottom: '8%', top: '15%', containLabel: true },
            xAxis: { type: 'category', name: '频率 (Hz)' },
            yAxis: { type: 'value', name: '功率 (μV²/Hz)' },
            series: []
        });

        // 稳定区检测图
        charts.stable = echarts.init(document.getElementById('chartStable'));
        charts.stable.setOption({
            title: { show: false },
            tooltip: { trigger: 'axis' },
            legend: { data: ['稳定得分', '稳定阈值'] },
            grid: { left: '5%', right: '5%', bottom: '5%', top: '10%', containLabel: true },
            xAxis: { type: 'category', name: '时间点' },
            yAxis: { type: 'value', name: '稳定得分' },
            series: [
                { name: '稳定得分', type: 'line', data: [], smooth: true, lineStyle: { color: '#5470c6' } },
                { name: '稳定阈值', type: 'line', data: [], smooth: false, lineStyle: { color: '#fc8452', type: 'dashed' } }
            ]
        });

        // Topomap图（使用图片）
        charts.topomap = {
            container: document.getElementById('chartTopomap'),
            setImage: function(imgData) {
                this.container.innerHTML = `<img src="${imgData}" style="width:100%;height:100%;object-fit:contain;">`;
            },
            clear: function() {
                this.container.innerHTML = '<div style="text-align:center;line-height:300px;color:#999;">点击"Topomap"按钮生成</div>';
            }
        };
        charts.topomap.clear();

        // 情绪统计图
        charts.emotion = echarts.init(document.getElementById('chartEmotion'));
        charts.emotion.setOption({
            title: { show: false },
            tooltip: { trigger: 'item' },
            legend: { orient: 'horizontal', bottom: 10 },
            series: [
                {
                    name: '情绪分布',
                    type: 'pie',
                    radius: ['40%', '70%'],
                    avoidLabelOverlap: false,
                    label: { show: true, formatter: '{b}: {d}%' },
                    emphasis: { scale: true },
                    data: []
                }
            ]
        });
    }

    /**
     * 显示加载中
     */
    function showLoading() {
        for (let key in charts) {
            if (charts[key] && charts[key].showLoading) {
                charts[key].showLoading({
                    text: '加载中...',
                    color: '#667eea',
                    maskColor: 'rgba(255,255,255,0.3)'
                });
            }
        }
    }

    /**
     * 隐藏加载中
     */
    function hideLoading() {
        for (let key in charts) {
            if (charts[key] && charts[key].hideLoading) {
                charts[key].hideLoading();
            }
        }
    }

    /**
     * 加载数据集列表
     */
    async function loadDatasets() {
        try {
            const response = await fetch(`${API_BASE}/datasets`);
            const result = await response.json();

            let options = '<option value="">请选择数据集</option>';
            if (result.code === 200 && result.data && result.data.length > 0) {
                result.data.forEach(ds => {
                    options += `<option value="${ds.id}">${ds.name}</option>`;
                });
            } else {
                options += '<option value="SEED" selected>SEED 数据集</option>';
            }

            document.getElementById('datasetSelect').innerHTML = options;
            form.render('select');

            // 默认选择第一个
            if (result.data && result.data.length > 0) {
                document.getElementById('datasetSelect').value = result.data[0].id;
                form.render('select');
            }
            loadFiles();
        } catch (error) {
            console.error('加载数据集失败:', error);
            document.getElementById('datasetSelect').innerHTML = '<option value="SEED" selected>SEED 数据集</option>';
            form.render('select');
            loadFiles();
        }
    }

    /**
     * 加载文件列表
     */
    async function loadFiles() {
        const dataset = document.getElementById('datasetSelect').value || 'SEED';
        const fileType = document.getElementById('fileTypeSelect').value;

        try {
            const response = await fetch(`${API_BASE}/files?dataset=${dataset}&type=${fileType}`);
            const result = await response.json();

            let options = '<option value="">选择MAT文件</option>';
            if (result.code === 200 && result.data && result.data.length > 0) {
                fileList = result.data;
                result.data.forEach(file => {
                    const label = file.is_label ? '📄 ' : '📊 ';
                    options += `<option value="${file.id}" data-path="${file.path || ''}">${label}${file.name}</option>`;
                });
            } else {
                // 默认文件列表
                fileList = [
                    { id: '1_20131027', name: '1_20131027.mat', path: '', is_label: false },
                    { id: '2_20131027', name: '2_20131027.mat', path: '', is_label: false },
                    { id: '3_20131027', name: '3_20131027.mat', path: '', is_label: false },
                    { id: 'label', name: 'label.mat', path: '', is_label: true }
                ];
                fileList.forEach(file => {
                    const label = file.is_label ? '📄 ' : '📊 ';
                    options += `<option value="${file.id}" data-path="${file.path}">${label}${file.name}</option>`;
                });
            }

            document.getElementById('fileSelect').innerHTML = options;
            form.render('select');

            updateStatus(`已加载 ${fileList.length} 个文件`, 'success');

            // 如果有文件，默认选择第一个并加载trials
            if (fileList.length > 0) {
                setTimeout(() => {
                    loadTrials();
                }, 100);
            }
        } catch (error) {
            console.error('加载文件列表失败:', error);
            updateStatus('加载文件列表失败，使用默认文件', 'warning');
        }
    }

    /**
     * 加载trial列表
     */
    async function loadTrials() {
        const fileSelect = document.getElementById('fileSelect');
        const selectedIndex = fileSelect.selectedIndex;

        // 如果没有选中任何选项，选择第一个
        if (selectedIndex === -1 || !fileSelect.options[selectedIndex]) {
            if (fileSelect.options.length > 1) {
                fileSelect.selectedIndex = 1; // 跳过第一个提示选项
            } else {
                return;
            }
        }

        const selectedOption = fileSelect.options[fileSelect.selectedIndex];
        const filePath = selectedOption ? selectedOption.dataset.path : '';
        const fileId = selectedOption ? selectedOption.value : '';

        // 更新currentData
        currentData.file = fileId;
        currentData.filePath = filePath;

        // 如果是标签文件，提示
        const selectedFile = fileList.find(f => f.id === fileId);
        if (selectedFile && selectedFile.is_label) {
            updateStatus('当前选择的是标签文件，只能进行情绪统计', 'warning');
        }

        if (!filePath) {
            // 没有真实路径，返回默认trials
            let options = '<option value="0">Trial 1</option>';
            for (let i = 1; i < 5; i++) {
                options += `<option value="${i}">Trial ${i+1}</option>`;
            }
            document.getElementById('trialSelect').innerHTML = options;
            form.render('select');

            // 自动加载EEG数据（如果不是标签文件）
            if (!(selectedFile && selectedFile.is_label)) {
                setTimeout(() => {
                    loadEEGData();
                }, 200);
            }
            return;
        }

        try {
            const response = await fetch(`${API_BASE}/trials?file_path=${encodeURIComponent(filePath)}`);
            const result = await response.json();

            let options = '';
            if (result.code === 200 && result.data && result.data.length > 0) {
                result.data.forEach(trial => {
                    options += `<option value="${trial.index}">${trial.name}</option>`;
                });
            } else {
                options = '<option value="0">Trial 1</option>';
                for (let i = 1; i < 5; i++) {
                    options += `<option value="${i}">Trial ${i+1}</option>`;
                }
            }

            document.getElementById('trialSelect').innerHTML = options;
            form.render('select');

            // 自动加载EEG数据（如果不是标签文件）
            if (!(selectedFile && selectedFile.is_label)) {
                setTimeout(() => {
                    loadEEGData();
                }, 200);
            }
        } catch (error) {
            console.error('加载trial列表失败:', error);
            // 默认返回5个trials
            let options = '<option value="0">Trial 1</option>';
            for (let i = 1; i < 5; i++) {
                options += `<option value="${i}">Trial ${i+1}</option>`;
            }
            document.getElementById('trialSelect').innerHTML = options;
            form.render('select');
        }
    }

    /**
     * 加载EEG数据
     */
    async function loadEEGData() {
        showLoading();

        const fileSelect = document.getElementById('fileSelect');
        const selectedIndex = fileSelect.selectedIndex;

        if (selectedIndex === -1 || !fileSelect.options[selectedIndex]) {
            hideLoading();
            updateStatus('请先选择文件', 'warning');
            return;
        }

        const selectedOption = fileSelect.options[selectedIndex];
        const fileId = selectedOption.value;
        const filePath = selectedOption.dataset.path;
        const trialSelect = document.getElementById('trialSelect');
        const trialIndex = trialSelect ? trialSelect.value : '0';

        // 如果是标签文件，提示不能加载EEG
        const selectedFile = fileList.find(f => f.id === fileId);
        if (selectedFile && selectedFile.is_label) {
            hideLoading();
            updateStatus('标签文件不能加载EEG数据', 'warning');
            return;
        }

        try {
            const response = await fetch(`${API_BASE}/load_eeg`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    file_path: filePath,
                    trial_index: parseInt(trialIndex || '0')
                })
            });

            const result = await response.json();

            if (result.code === 200) {
                currentData.eegData = result.data.eeg_data;
                currentData.channels = result.data.channels;
                currentData.samplingRate = result.data.sampling_rate;

                document.getElementById('eegInfo').innerHTML =
                    `通道: ${result.data.channels.length} | 采样率: ${result.data.sampling_rate}Hz | 时间点: ${result.data.metadata.n_times || 1000}`;

                updateStatus(`EEG数据加载成功: ${result.data.metadata.n_times || 1000} 个时间点`, 'success');

                // 自动绘制EEG时域图
                plotEegTime();
            } else {
                updateStatus('加载失败: ' + result.message, 'error');
            }
        } catch (error) {
            console.error('加载EEG数据失败:', error);
            updateStatus('网络请求失败: ' + error.message, 'error');
        } finally {
            hideLoading();
        }
    }

    /**
     * 绘制EEG时域图
     */
    function plotEegTime() {
        if (!currentData.eegData) {
            layer.msg('请先加载EEG数据', { icon: 2, time: 2000 });
            return;
        }

        const data = currentData.eegData;
        const channels = currentData.channels;

        // 取前8个通道的数据展示
        const channelsToShow = Math.min(8, data.length);
        const series = [];
        const colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452', '#9a60b4'];

        // 确定要显示的时间点数量
        const timePoints = Math.min(500, data[0]?.length || 500);

        for (let i = 0; i < channelsToShow; i++) {
            series.push({
                name: channels[i],
                type: 'line',
                data: data[i] ? data[i].slice(0, timePoints) : [],
                smooth: true,
                showSymbol: false,
                lineStyle: { width: 1.5, color: colors[i % colors.length] }
            });
        }

        charts.eeg.setOption({
            xAxis: {
                data: Array.from({ length: timePoints }, (_, i) => i),
                name: '时间点'
            },
            series: series,
            legend: {
                data: channels.slice(0, channelsToShow),
                type: 'scroll',
                pageIconColor: '#667eea'
            }
        });

        updateStatus('EEG时域图已更新', 'success');
    }

    /**
     * 计算PSD并绘制
     */
    async function plotPsd() {
        if (!currentData.eegData) {
            layer.msg('请先加载EEG数据', { icon: 2, time: 2000 });
            return;
        }

        showLoading();

        try {
            const response = await fetch(`${API_BASE}/psd`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    eeg_data: currentData.eegData
                })
            });

            const result = await response.json();

            if (result.code === 200) {
                const data = result.data;
                // 取前8个通道展示
                const channelsToShow = Math.min(8, data.psd.length);
                const series = [];
                const colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452', '#9a60b4'];

                for (let i = 0; i < channelsToShow; i++) {
                    series.push({
                        name: data.channels[i],
                        type: 'line',
                        data: data.psd[i],
                        smooth: true,
                        showSymbol: false,
                        lineStyle: { color: colors[i % colors.length] }
                    });
                }

                charts.psd.setOption({
                    xAxis: {
                        data: data.frequencies.map(f => f.toFixed(1)),
                        name: '频率 (Hz)'
                    },
                    series: series,
                    legend: {
                        data: data.channels.slice(0, channelsToShow),
                        type: 'scroll'
                    }
                });

                updateStatus('PSD计算完成', 'success');
            } else {
                updateStatus('PSD计算失败: ' + result.message, 'error');
            }
        } catch (error) {
            console.error('PSD计算失败:', error);
            updateStatus('网络请求失败: ' + error.message, 'error');
        } finally {
            hideLoading();
        }
    }

    /**
     * 检测稳定区域并绘制
     */
    async function plotStable() {
        if (!currentData.eegData) {
            layer.msg('请先加载EEG数据', { icon: 2, time: 2000 });
            return;
        }

        showLoading();

        try {
            const response = await fetch(`${API_BASE}/stable`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    eeg_data: currentData.eegData,
                    threshold: 0.1,
                    window_size: 50
                })
            });

            const result = await response.json();

            if (result.code === 200) {
                const data = result.data;

                // 创建阈值线
                const thresholdLine = Array(data.stability_scores.length).fill(0.1);

                // 创建稳定区域标记
                const markAreas = [];
                if (data.stable_regions && data.stable_regions.length > 0) {
                    data.stable_regions.forEach(region => {
                        if (region && region.length >= 2) {
                            markAreas.push([{
                                name: '稳定区',
                                xAxis: region[0],
                                itemStyle: { color: 'rgba(84, 112, 198, 0.2)' }
                            }, {
                                xAxis: region[1]
                            }]);
                        }
                    });
                }

                // 更新图表
                charts.stable.setOption({
                    xAxis: {
                        data: data.time_points,
                        name: '时间点'
                    },
                    series: [
                        {
                            name: '稳定得分',
                            type: 'line',
                            data: data.stability_scores,
                            smooth: true,
                            lineStyle: { color: '#5470c6' }
                        },
                        {
                            name: '稳定阈值',
                            type: 'line',
                            data: thresholdLine,
                            smooth: false,
                            lineStyle: { color: '#fc8452', type: 'dashed' }
                        },
                        {
                            name: '稳定区域',
                            type: 'line',
                            data: [],
                            markArea: {
                                data: markAreas,
                                itemStyle: { color: 'rgba(84, 112, 198, 0.1)' }
                            }
                        }
                    ]
                });

                updateStatus(`检测到 ${data.stable_regions.length} 个稳定区域`, 'success');
            } else {
                updateStatus('稳定区检测失败: ' + result.message, 'error');
            }
        } catch (error) {
            console.error('稳定区检测失败:', error);
            updateStatus('网络请求失败: ' + error.message, 'error');
        } finally {
            hideLoading();
        }
    }

    /**
     * 生成Topomap
     */
    async function plotTopomap() {
        if (!currentData.eegData) {
            layer.msg('请先加载EEG数据', { icon: 2, time: 2000 });
            return;
        }

        showLoading();

        try {
            const timePoint = Math.floor((currentData.eegData[0]?.length || 1000) / 2);

            const response = await fetch(`${API_BASE}/topomap`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    eeg_data: currentData.eegData,
                    time_point: timePoint
                })
            });

            const result = await response.json();

            if (result.code === 200) {
                charts.topomap.setImage(result.data.image);
                updateStatus('Topomap生成成功', 'success');
            } else {
                updateStatus('Topomap生成失败: ' + result.message, 'error');
            }
        } catch (error) {
            console.error('Topomap生成失败:', error);
            updateStatus('网络请求失败: ' + error.message, 'error');
        } finally {
            hideLoading();
        }
    }

    /**
     * 分析情绪分布
     */
    async function plotEmotion() {
        showLoading();

        const fileSelect = document.getElementById('fileSelect');
        const selectedIndex = fileSelect.selectedIndex;

        let filePath = '';
        if (selectedIndex !== -1 && fileSelect.options[selectedIndex]) {
            filePath = fileSelect.options[selectedIndex].dataset.path;
        }

        // 查找标签文件
        const labelFile = fileList.find(f => f.is_label);
        const labelPath = labelFile ? labelFile.path : '';

        try {
            const response = await fetch(`${API_BASE}/emotion_analysis`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    file_path: labelPath || filePath
                })
            });

            const result = await response.json();

            if (result.code === 200) {
                const data = result.data;
                const pieData = data.distribution.map(item => ({
                    name: item.emotion,
                    value: item.count
                }));

                charts.emotion.setOption({
                    series: [{
                        type: 'pie',
                        data: pieData,
                        label: { show: true, formatter: '{b}: {d}%' }
                    }]
                });

                updateStatus(`情绪统计完成，共 ${data.total} 个样本`, 'success');
            } else {
                updateStatus('情绪分析失败: ' + result.message, 'error');
            }
        } catch (error) {
            console.error('情绪分析失败:', error);
            updateStatus('网络请求失败: ' + error.message, 'error');
        } finally {
            hideLoading();
        }
    }

    /**
     * 清空所有图表
     */
    function clearCharts() {
        charts.eeg.setOption({ series: [] });
        charts.psd.setOption({ series: [] });
        charts.stable.setOption({
            series: [
                { data: [] },
                { data: [] },
                { data: [], markArea: { data: [] } }
            ]
        });
        charts.topomap.clear();
        charts.emotion.setOption({ series: [{ data: [] }] });
        updateStatus('已清空所有图表', 'info');
    }

    /**
     * 表单监听
     */
    form.on('select(datasetSelect)', function() {
        loadFiles();
    });

    form.on('select(fileTypeSelect)', function() {
        loadFiles();
    });

    form.on('select(fileSelect)', function() {
        loadTrials();
    });

    form.on('select(trialSelect)', function(data) {
        currentData.trialIndex = parseInt(data.value);
        // 切换trial时重新加载数据
        loadEEGData();
    });

    /**
     * 按钮事件绑定
     */
    document.getElementById('btnEegTime').addEventListener('click', plotEegTime);
    document.getElementById('btnPsd').addEventListener('click', plotPsd);
    document.getElementById('btnStable').addEventListener('click', plotStable);
    document.getElementById('btnTopomap').addEventListener('click', plotTopomap);
    document.getElementById('btnEmotionStats').addEventListener('click', plotEmotion);
    document.getElementById('btnClear').addEventListener('click', clearCharts);

    document.getElementById('btnFeatures').addEventListener('click', function() {
        if (!currentData.eegData) {
            layer.msg('请先加载EEG数据', { icon: 2, time: 2000 });
            return;
        }
        layer.msg('特征分析功能开发中...', { icon: 6, time: 2000 });
    });

    document.getElementById('btnSessionCompare').addEventListener('click', function() {
        layer.msg('Session对比功能开发中...', { icon: 6, time: 2000 });
    });

    document.getElementById('btnRefresh').addEventListener('click', function() {
        loadFiles();
        layer.msg('数据刷新中...', { icon: 16, time: 1000, shade: 0.3 });
    });

    // 初始化
    initCharts();
    loadDatasets();
});