// frontend/js/load_data_example.js
/**
 * 前端数据加载示例
 */

// 使用fetch加载数据
async function loadDataExample() {

    // 1. 获取数据集列表
    async function fetchDatasets() {
        try {
            const response = await fetch('/api/datasets');
            const result = await response.json();
            if (result.code === 200) {
                console.log('数据集列表:', result.data);
                return result.data;
            }
        } catch (error) {
            console.error('获取数据集失败:', error);
        }
    }

    // 2. 获取文件列表
    async function fetchFiles(dataset = 'SEED', type = 'preprocessed') {
        try {
            const response = await fetch(`/api/files?dataset=${dataset}&type=${type}`);
            const result = await response.json();
            if (result.code === 200) {
                console.log(`${type}文件列表:`, result.data);
                return result.data;
            }
        } catch (error) {
            console.error('获取文件列表失败:', error);
        }
    }

    // 3. 加载EEG数据
    async function loadEEG(filePath, trialIndex = 0) {
        try {
            const response = await fetch('/api/load_eeg', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_path: filePath, trial_index: trialIndex })
            });
            const result = await response.json();
            if (result.code === 200) {
                console.log('EEG数据加载成功:', result.data);
                return result.data;
            } else {
                console.error('加载失败:', result.message);
            }
        } catch (error) {
            console.error('请求失败:', error);
        }
    }

    // 4. 加载标签数据
    async function loadLabels(filePath) {
        try {
            const response = await fetch('/api/load_labels', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_path: filePath })
            });
            const result = await response.json();
            if (result.code === 200) {
                console.log('标签数据加载成功:', result.data);
                return result.data;
            }
        } catch (error) {
            console.error('请求失败:', error);
        }
    }

    // 使用示例
    (async () => {
        // 获取文件列表
        const files = await fetchFiles('SEED', 'preprocessed');

        if (files && files.length > 0) {
            // 加载第一个文件的EEG数据
            const eegData = await loadEEG(files[0].path, 0);

            // 如果有eegData，可以在ECharts中绘制
            if (eegData) {
                // 绘制EEG时域图
                const chart = echarts.init(document.getElementById('eegChart'));
                chart.setOption({
                    xAxis: { data: Array.from({ length: eegData.eeg_data[0].length }, (_, i) => i) },
                    series: eegData.channels.slice(0, 5).map((ch, idx) => ({
                        name: ch,
                        type: 'line',
                        data: eegData.eeg_data[idx]
                    }))
                });
            }
        }

        // 加载标签数据
        const labelFiles = files?.filter(f => f.name === 'label');
        if (labelFiles && labelFiles.length > 0) {
            await loadLabels(labelFiles[0].path);
        }
    })();
}