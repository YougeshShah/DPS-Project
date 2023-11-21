from .views import Home, Home,Pickle, diabetes_pre,preprocessing,checker_page,chooseMethod,classification,clustering
from django.urls import path

urlpatterns = [
    path('pickle',Pickle,name='Pickle'),
    path('diabetes_pre/', diabetes_pre, name='diabetesprediction'),
    path('home',Home),
    path('preprocessing/', preprocessing, name='preprocessing'),
    path('checker_page/', checker_page, name='checker_page'),
    path('chooseMethod/', chooseMethod, name='chooseMethod'),
    path('classification/', classification, name='classification'),
    path('clustering/', clustering, name='clustering'),

]
