"""
URL configuration for DeepLearn project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include

from django.conf import settings
from django.conf.urls.static import static

from modelVisualization.views import admin
from modelVisualization.views import training

from django.contrib.auth import views as auth_views  # 添加这行

urlpatterns = [
    # path("admin/", admin.site.urls),
    path("", admin.admin, name='admin'),

    path("get/Index/", admin.get_Index),

    path("display/",admin.display),

    path('load-left-menu/', admin.load_left_menu, name='load_left_menu'),
    path('load-content/', admin.load_content, name='load_content'),


    path('', include('modelVisualization.urls.training_urls')),
    path('seed/', include('modelVisualization.urls.seedVisualization_urls')),

    # 添加认证相关的URL
    path('accounts/login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('accounts/logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),

]

# 仅在 DEBUG 模式下提供媒体文件服务
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)