from django.urls import path
from . import views

urlpatterns = [
    path('fetch-data/', views.fetch_stock_data, name='fetch_stock_data'),
    path('backtest/', views.backtest_view, name='backtest_view'),
    path('predict/', views.predict_stock_prices, name='predict_stock_prices'),
    path('report/', views.generate_report, name='generate_report'),
]
