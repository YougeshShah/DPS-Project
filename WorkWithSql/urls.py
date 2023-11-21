"""WorkWithSql URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
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
from re import template
from django.contrib import admin
from django.urls import path,include
from django.contrib.auth import views
from authapp import views
from django.conf import settings
from django.conf.urls.static import static

import os


urlpatterns = [
    path('admin/', admin.site.urls),
    path('',include("authapp.urls")),
    path("register", views.register_request, name="register"),
    path("", views.login_request, name="login"),
    path("login", views.login_request, name="login"),
    path("logout", views.logout_request, name= "logout"),
    path('preprocessing', views.preprocessing, name='preprocessing'),
    path('checker_page', views.checker_page, name='checker_page'),
    path('chooseMethod', views.chooseMethod, name='chooseMethod'),
    path('classification', views.classification, name='classification'),
    path('clustering', views.clustering, name='clustering'),
    
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
