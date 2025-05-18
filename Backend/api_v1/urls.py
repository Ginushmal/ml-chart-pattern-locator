from django.urls import path
from .views import PatternDetectionView

urlpatterns = [
    path('detect-patterns/', PatternDetectionView.as_view(), name='detect-patterns'),
] 