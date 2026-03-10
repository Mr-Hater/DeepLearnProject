from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

import os

from django.template.loader import render_to_string
from django.views.decorators.csrf import csrf_exempt

from DeepLearn import settings
from modelVisualization.views.html import createHtml

def admin(request):
    print("fdsfadfasdf")
    return render(request, 'admin.html')

def get_Index(request):
    text = request.GET.get("getText","")
    """获取PDF文件列表"""
    pdf_name = text + ".pdf"
    pdf_path = os.path.join(settings.MEDIA_ROOT, 'pdfs',pdf_name)
    if os.path.exists(pdf_path) and pdf_name.lower().endswith('.pdf'):
        print(pdf_name)
        print( f'{settings.MEDIA_URL}pdfs/{pdf_name}')
        context = {
            'pdf_name': text,
            'pdf_url': f'{settings.MEDIA_URL}pdfs/{pdf_name}'
        }
        return JsonResponse(context)
    return JsonResponse({'error': 'PDF not found'}, status=404)

def display(request):
    return render(request, 'public/leftList.html')


def load_left_menu(request):
    """加载左侧菜单"""
    menu_type = request.GET.get('type', 'home')

    if menu_type == 'introduction':
        # 返回 leftList.html 的内容
        html = render_to_string('introduction/leftList.html')
    elif menu_type == 'useModel':
        html = render_to_string('training/leftList.html')
        # 动态生成
        html = createHtml.generate_use_model_menu()
    elif menu_type == 'visualization':
        html = render_to_string('visualization/leftList.html')
    elif menu_type == 'home':
        # 返回简化菜单
        html = render_to_string('public/leftList.html')
        # html = """
        # <ul class="layui-nav layui-nav-tree" lay-filter="test">
        #     <li class="layui-nav-item"><a href="javascript:;">首页菜单1</a></li>
        #     <li class="layui-nav-item"><a href="javascript:;">首页菜单2</a></li>
        # </ul>
        # """
    elif menu_type == 'SEEDVisualization':
        html = render_to_string('SeedVisualization/leftList.html')
    else:
        # 其他类型的菜单
        html = """
        <ul class="layui-nav layui-nav-tree" lay-filter="test">
            <li class="layui-nav-item"><a href="javascript:;">默认菜单项</a></li>
        </ul>
        """

    return JsonResponse({'html': html})

def load_content(request):
    """加载右侧内容"""
    content_type = request.GET.get('type', 'home')

    if content_type == 'introduction':
        # 返回 first.html 的内容
        html = render_to_string('introduction/content.html')
    elif content_type == 'useModel':
        html = render_to_string('training/content.html')
    elif content_type == 'visualization':
        html = render_to_string('visualization/content.html')
    elif content_type == 'home':
        # 返回首页内容
        html = render_to_string('public/content.html')

        # html = """
        # <blockquote class="layui-elem-quote layui-text">
        #     <i class="layui-icon layui-icon-face-smile" style="font-size: 30px; color: #1E9FFF;"></i>
        #     <span style="font-size: 25px">LibEER算法使用系统</span>
        # </blockquote>
        # <div><img src="/static/img/2.gif" style="margin: 0 auto; height: auto;width: 100%"></div>
        # """
    elif content_type == 'SEEDVisualization':
        html = render_to_string('SeedVisualization/index.html')
    else:
        html = f"""
        <div class="layui-card">
            <div class="layui-card-header">{content_type} 页面</div>
            <div class="layui-card-body">
                这是 {content_type} 页面的内容
            </div>
        </div>
        """

    return JsonResponse({'html': html})