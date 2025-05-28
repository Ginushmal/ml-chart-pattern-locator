from django.urls import path
from .views import PatternDetectionStartView, PatternDetectionProgressView, OHLCDataAPIView

urlpatterns = [
    path('detect-patterns/start/', PatternDetectionStartView.as_view(), name='detect-patterns-start'),
    path('detect-patterns/progress/<str:request_id>/', PatternDetectionProgressView.as_view(), name='detect-patterns-progress'),
    path('ohlc-data/', OHLCDataAPIView.as_view(), name='ohlc-data'), 
] 