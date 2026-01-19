from modelVisualization.utils.model_registry import ModelRegistry

def generate_use_model_menu():
    """生成算法使用页面的左侧菜单"""
    # 这里需要从数据库获取已注册的模型
    # 假设有以下数据结构（实际应该从数据库查询）

    registry = ModelRegistry()
    categorized_models = registry.get_models_by_category()

    # 1. 固定菜单项
    fixed_items = [
        {'name': '简介', 'type': 'intro', 'id': 'intro'},
        {'name': '注册模型', 'type': 'register', 'id': 'register'},
        {'name': '训练历史', 'type': 'history', 'id': 'history'}
    ]

    # 生成HTML
    html = '<ul class="layui-nav layui-nav-tree" lay-filter="useModelMenu">\n'

    # 添加固定项目
    for item in fixed_items:
        html += f'    <li class="layui-nav-item">\n'
        html += f'        <a href="javascript:;" data-type="{item["type"]}" data-id="{item["id"]}">{item["name"]}</a>\n'
        html += f'    </li>\n'

    # 添加已注册的模型（按分类）
    for category, models in categorized_models.items():
        if models:  # 只显示有模型的分类
            html += f'    <li class="layui-nav-item">\n'
            html += f'        <a href="javascript:;">{category}</a>\n'
            html += f'        <dl class="layui-nav-child">\n'

            for model in models:
                html += f'            <dd><a href="javascript:;" data-model="{model}" data-category="{category}">{model}</a></dd>\n'

            html += f'        </dl>\n'
        html += f'    </li>\n'

    # 显示空分类提示（如果需要）
    html += '</ul>'
    return html