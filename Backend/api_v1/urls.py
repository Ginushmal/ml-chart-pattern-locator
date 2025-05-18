from django.urls import path
from .views import PatternDetectionView, OHLCDataAPIView

urlpatterns = [
    path('detect-patterns/', PatternDetectionView.as_view(), name='detect-patterns'),
    path('ohlc-data/', OHLCDataAPIView.as_view(), name='ohlc-data'), 
] 