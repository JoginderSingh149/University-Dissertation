from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_file, name='upload'),
    path('dashboard/', views.dashboard, name='dashboard'),
]