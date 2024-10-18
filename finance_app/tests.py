from django.test import TestCase, Client
from .models import StockData
from django.urls import reverse

class BacktestTestCase(TestCase):
    def setUp(self):
        # Create sample data
        StockData.objects.create(symbol='AAPL', date='2023-01-01', close_price=150, open_price=145, high_price=151, low_price=144, volume=1000000)
        # ... Add more data points as needed

    def test_backtest_view(self):
        client = Client()
        response = client.get(reverse('backtest_view'))
        self.assertEqual(response.status_code, 200)
        # Test POST request with form data
        response = client.post(reverse('backtest_view'), {
            'initial_investment': 10000,
            'short_window': 5,
            'long_window': 10,
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Total Return')
