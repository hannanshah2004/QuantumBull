from django.shortcuts import render
from django.http import HttpResponse

def home(request):
    return render(request, 'finance_app/home.html')

def error(request):
    return render(request, 'finance_app/error.html')

def fetch_stock_data(request):
    if request.method == 'POST':
        symbol = request.POST.get('symbol', '').upper().strip()
        if not symbol:
            return render(request, 'finance_app/error.html', {'message': 'Please enter a stock symbol.'})
        return render(request, 'finance_app/success.html', {'message': f'Data for {symbol} fetched successfully.'})
    else:
        return render(request, 'finance_app/fetch_data.html')

def backtest_view(request):
    if request.method == 'POST':
        # Fake form handling
        # In a real scenario, you'd have `form = BacktestForm(request.POST)` 
        # and some validation, but for now, just skip it.
        return render(request, 'finance_app/backtest_results.html', {
            'total_return': 0.00,
            'max_drawdown': 0.00,
            'num_trades': 0,
            'symbol': 'FAKE_SYMBOL',
        })
    else:
        # In a real scenario, you'd pass a form instance here.
        return render(request, 'finance_app/backtest.html', {'form': None})

def predict_stock_prices(request):
    if request.method == 'POST':
        symbol = request.POST.get('symbol', '').upper().strip()
        if not symbol:
            return render(request, 'finance_app/error.html', {'message': 'Please enter stock symbol.'})
        predictions_list = []
        return render(request, 'finance_app/predictions.html', {'predictions': predictions_list, 'symbol': symbol})
    else:
        return render(request, 'finance_app/predictions.html')

def get_predicted_prices(symbol):
    # Placeholder for predicted prices
    return []

def generate_report(request):
    if request.method == 'POST':
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="report.pdf"'
        response.write(b"%PDF-1.4\n% Bare-bones PDF content")
        return response
    else:
        return render(request, 'finance_app/generate_report.html')
