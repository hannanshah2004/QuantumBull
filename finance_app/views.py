import os
import time
import requests
import pandas as pd
import numpy as np
import pmdarima
import pickle
import io
import base64
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime, timedelta
from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.template.loader import render_to_string
from django.utils import timezone
from .models import StockData
from .forms import BacktestForm
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
import tensorflow as tf
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from sklearn.preprocessing import MinMaxScaler
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.charts.lineplots import LinePlot
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM

def home(request):
    return render(request, 'finance_app/home.html')

def fetch_stock_data(request):
    if request.method == 'POST':
        # Get the stock symbol from the POST data
        symbol = request.POST.get('symbol', '').upper().strip()

        if not symbol:
            return render(request, 'finance_app/error.html', {'message': 'Please enter a stock symbol.'})

        function = 'TIME_SERIES_MONTHLY_ADJUSTED'
        api_key = os.environ.get('ALPHA_VANTAGE_API_KEY', 'YOUR_API_KEY')  # Replace 'YOUR_API_KEY' with your actual API key

        # Delete existing entries for the symbol in the StockData model
        StockData.objects.filter(symbol=symbol).delete()

        url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api_key}'

        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            return render(request, 'finance_app/error.html', {'message': f'HTTP error occurred: {http_err}'})
        except requests.exceptions.ConnectionError as conn_err:
            return render(request, 'finance_app/error.html', {'message': f'Connection error occurred: {conn_err}'})
        except Exception as err:
            return render(request, 'finance_app/error.html', {'message': f'An error occurred: {err}'})

        data = response.json()

        # Check for rate limit message
        if 'Note' in data:
            note_message = data['Note']
            return render(request, 'finance_app/error.html', {'message': note_message})

        # Check for error message in the response
        if 'Error Message' in data:
            error_message = data['Error Message']
            return render(request, 'finance_app/error.html', {'message': error_message})

        if 'Monthly Adjusted Time Series' in data:
            time_series = data['Monthly Adjusted Time Series']
            today_date = timezone.now().date()
            two_years_ago = today_date - timedelta(days=730)

            for date_str, monthly_data in time_series.items():
                date = datetime.strptime(date_str, '%Y-%m-%d').date()

                if date > today_date:
                    continue

                if two_years_ago <= date <= today_date:
                    StockData.objects.update_or_create(
                        symbol=symbol,
                        date=date,
                        defaults={
                            'open_price': monthly_data['1. open'],
                            'high_price': monthly_data['2. high'],
                            'low_price': monthly_data['3. low'],
                            'close_price': monthly_data['4. close'],
                            'adjusted_close_price': monthly_data['5. adjusted close'],
                            'volume': monthly_data['6. volume'],
                            'dividend_amount': monthly_data['7. dividend amount'],
                        }
                    )
            return render(request, 'finance_app/success.html', {'message': f'Data for {symbol} fetched successfully.'})
        else:
            return render(request, 'finance_app/error.html', {'message': 'Unexpected data format received from API.'})
    else:
        return render(request, 'finance_app/fetch_data.html')

def backtest_view(request):
    if request.method == 'POST':
        form = BacktestForm(request.POST)
        if form.is_valid():
            # Convert initial_investment to float
            initial_investment = float(form.cleaned_data['initial_investment'])
            short_window = form.cleaned_data['short_window']
            long_window = form.cleaned_data['long_window']
            symbol = form.cleaned_data['symbol'].upper()  # Get symbol from the form

            # Fetch data from database
            stock_qs = StockData.objects.filter(symbol=symbol).order_by('date')
            if not stock_qs.exists():
                return render(request, 'finance_app/error.html', {'message': f"No data found for symbol: {symbol}"})

            data = pd.DataFrame.from_records(stock_qs.values('date', 'close_price'))
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            data['close_price'] = data['close_price'].astype(float)

            # Check if we have enough data points
            data_length = len(data)
            if data_length < long_window:
                error_message = f"Not enough data points ({data_length}) for the specified window sizes."
                return render(request, 'finance_app/error.html', {'message': error_message})

            # Calculate moving averages
            data['short_ma'] = data['close_price'].rolling(window=short_window).mean()
            data['long_ma'] = data['close_price'].rolling(window=long_window).mean()

            # Generate signals
            data['signal'] = 0.0
            valid_indices = data.index[max(short_window, long_window):]
            data.loc[valid_indices, 'signal'] = np.where(
                data['short_ma'][valid_indices] > data['long_ma'][valid_indices], 1.0, 0.0)
            data['positions'] = data['signal'].diff()

            # Backtesting logic
            positions = pd.DataFrame(index=data.index).fillna(0.0)
            positions[symbol] = 100 * data['signal']

            # Ensure numeric data is float
            positions = positions.astype(float)
            pos_diff = positions.diff()

            # Calculate portfolio value
            portfolio = positions.multiply(data['close_price'], axis=0)
            portfolio['holdings'] = portfolio.sum(axis=1)

            # Ensure cash_outflow is float
            cash_outflow = (pos_diff.multiply(data['close_price'], axis=0)).sum(axis=1).cumsum()
            cash_outflow = cash_outflow.astype(float)

            # Calculate cash position
            portfolio['cash'] = initial_investment - cash_outflow

            # Calculate total portfolio value
            portfolio['total'] = portfolio['cash'] + portfolio['holdings']
            portfolio['returns'] = portfolio['total'].pct_change()

            # Performance metrics
            total_return = (portfolio['total'][-1] - initial_investment) / initial_investment * 100
            max_drawdown = ((portfolio['total'].cummax() - portfolio['total']) / portfolio['total'].cummax()).max() * 100
            num_trades = data['positions'].abs().sum()

            context = {
                'form': form,
                'total_return': round(total_return, 2),
                'max_drawdown': round(max_drawdown, 2),
                'num_trades': int(num_trades),
                'symbol': symbol,
            }
            return render(request, 'finance_app/backtest_results.html', context)
    else:
        form = BacktestForm()
    return render(request, 'finance_app/backtest.html', {'form': form})

def predict_stock_prices(request):
    if request.method == 'POST':
        # Get the stock symbol from the POST data
        symbol = request.POST.get('symbol', '').upper().strip()
        if not symbol:
            return render(request, 'finance_app/error.html', {'message': 'Please enter stock symbol.'})

        predictions_df = get_predicted_prices(symbol)
        if predictions_df is None or predictions_df.empty:
            return render(request, 'finance_app/error.html', {'message': f"No data found for symbol: {symbol}"})

        # Convert predictions to list of tuples for template rendering
        predictions_list = list(zip(predictions_df['date'], predictions_df['predicted_price']))

        return render(request, 'finance_app/predictions.html', {'predictions': predictions_list, 'symbol': symbol})
    else:
        # If GET request, render the form without predictions
        return render(request, 'finance_app/predictions.html')

def get_predicted_prices(symbol):
    # Load historical data
    stock_qs = StockData.objects.filter(symbol=symbol).order_by('date')
    if not stock_qs.exists():
        return None  # Or raise an exception

    data = pd.DataFrame.from_records(stock_qs.values('date', 'close_price'))
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data['close_price'] = data['close_price'].astype(float)

    # Prepare data
    dataset = data['close_price'].values
    dataset = dataset.reshape(-1, 1)

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create training data
    look_back = min(60, len(scaled_data) - 1)
    X_train = []
    y_train = []
    for i in range(look_back, len(scaled_data)):
        X_train.append(scaled_data[i - look_back:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    # Compile and train the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Prepare for prediction
    last_look_back = scaled_data[-look_back:]

    predictions = []

    for _ in range(30):
        X_input = last_look_back.reshape((1, look_back, 1))
        pred_price = model.predict(X_input, verbose=0)

        # Flatten and reshape pred_price to ensure it has shape (1, 1)
        pred_price = pred_price.flatten().reshape(-1, 1)

        # Append prediction
        predictions.append(pred_price[0, 0])

        # Update last_look_back
        last_look_back = np.concatenate((last_look_back[1:], pred_price), axis=0)

    # Inverse transform predictions
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Generate future dates
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')

    # Prepare predictions DataFrame
    predictions_df = pd.DataFrame({
        'date': future_dates,
        'predicted_price': predicted_prices.flatten(),
    })

    return predictions_df

pdfmetrics.registerFont(TTFont('Roboto', os.path.join(settings.BASE_DIR, 'finance_app/fonts/finance_app/Roboto-Black.ttf')))
pdfmetrics.registerFont(TTFont('Roboto-Bold', os.path.join(settings.BASE_DIR, 'finance_app/fonts/finance_app/Roboto-Bold.ttf')))

def generate_report(request):
    if request.method == 'POST':
        # Get the form data from POST request
        symbol = request.POST.get('symbol', 'AAPL').upper()
        initial_investment = request.POST.get('initial_investment', 10000)
        # In generate_report function
        short_window = int(request.POST.get('short_window', 20))
        long_window = int(request.POST.get('long_window', 50))

        # Validate and convert input data
        try:
            initial_investment = float(initial_investment)
            short_window = int(short_window)
            long_window = int(long_window)

            if short_window >= long_window:
                return render(request, 'finance_app/error.html', {'message': "Short term window must be less than long term window."})
            if initial_investment <= 0:
                return render(request, 'finance_app/error.html', {'message': "Initial investment must be greater than zero."})
            if short_window <= 0 or long_window <= 0:
                return render(request, 'finance_app/error.html', {'message': "Window sizes must be positive integers."})

        except ValueError:
            return render(request, 'finance_app/error.html', {'message': "Invalid input. Please enter valid numbers."})

        # Load historical data from the past 2 years
        stock_qs = StockData.objects.filter(symbol=symbol).order_by('date')
        if not stock_qs.exists():
            return render(request, 'finance_app/error.html', {'message': f"No data found for symbol: {symbol}"})

        # Prepare the data for the last 2 years
        today = datetime.today()
        two_years_ago = today - timedelta(days=730)
        stock_qs = stock_qs.filter(date__gte=two_years_ago.date())

        data_df = pd.DataFrame.from_records(stock_qs.values('date', 'close_price'))
        data_df['date'] = pd.to_datetime(data_df['date'])
        data_df.set_index('date', inplace=True)
        data_df['close_price'] = data_df['close_price'].astype(float)

        # Check if we have enough data points
        data_length = len(data_df)
        if data_length < long_window:
            return render(request, 'finance_app/error.html', {'message': f"Not enough data points ({data_length}) for the specified window sizes."})

        # Calculate moving averages
        data_df['short_ma'] = data_df['close_price'].rolling(window=short_window).mean()
        data_df['long_ma'] = data_df['close_price'].rolling(window=long_window).mean()

        # Generate signals
        data_df['signal'] = 0.0
        valid_indices = data_df.index[max(short_window, long_window):]
        data_df.loc[valid_indices, 'signal'] = np.where(
            data_df['short_ma'][valid_indices] > data_df['long_ma'][valid_indices], 1.0, 0.0)
        data_df['positions'] = data_df['signal'].diff()

        # Backtesting logic
        positions = pd.DataFrame(index=data_df.index).fillna(0.0)
        positions[symbol] = 100 * data_df['signal']

        # Ensure numeric data is float
        positions = positions.astype(float)
        pos_diff = positions.diff()

        # Calculate portfolio value
        portfolio = positions.multiply(data_df['close_price'], axis=0)
        portfolio['holdings'] = portfolio.sum(axis=1)

        # Calculate cash outflow
        cash_outflow = (pos_diff.multiply(data_df['close_price'], axis=0)).sum(axis=1).cumsum()
        cash_outflow = cash_outflow.astype(float)

        # Calculate cash position
        portfolio['cash'] = initial_investment - cash_outflow

        # Calculate total portfolio value
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()

        # Performance metrics
        total_return = (portfolio['total'][-1] - initial_investment) / initial_investment * 100
        max_drawdown = ((portfolio['total'].cummax() - portfolio['total']) / portfolio['total'].cummax()).max() * 100
        num_trades = data_df['positions'].abs().sum()

        # Round the metrics
        total_return = round(total_return, 2)
        max_drawdown = round(max_drawdown, 2)
        num_trades = int(num_trades)

        # Get predicted prices
        predictions_df = get_predicted_prices(symbol)
        if predictions_df is None:
            return render(request, 'finance_app/error.html', {'message': f"Could not generate predictions for symbol: {symbol}"})

        # Ensure that the predictions start after the last historical date
        predictions_df = predictions_df[predictions_df['date'] > data_df.index[-1]]

        # Create a BytesIO buffer for the PDF
        buffer = BytesIO()

        # Create the PDF object
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=18)

        # Container for the 'Flowable' objects
        elements = []

        # Define styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='CustomTitle', fontName='Roboto-Bold', fontSize=24, leading=30, alignment=1, spaceAfter=12))
        styles.add(ParagraphStyle(name='CustomHeading2', fontName='Roboto-Bold', fontSize=18, leading=22, spaceBefore=12, spaceAfter=6))
        styles.add(ParagraphStyle(name='CustomBodyText', fontName='Roboto', fontSize=10, leading=14, spaceBefore=6, spaceAfter=6))

        # Add title
        elements.append(Paragraph(f"Stock Report for {symbol}", styles['CustomTitle']))
        elements.append(Paragraph(f"Generated on {today.strftime('%B %d, %Y')}", styles['CustomBodyText']))
        elements.append(Spacer(1, 12))

        # Plot using Matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        import matplotlib.dates as mdates

        fig, ax = plt.subplots(figsize=(10, 5))

        # Convert index and columns to NumPy arrays
        historical_dates = data_df.index.to_numpy()
        historical_prices = data_df['close_price'].to_numpy()

        predicted_dates = predictions_df['date'].to_numpy()
        predicted_prices = predictions_df['predicted_price'].to_numpy()

        # Plot historical prices
        ax.plot(historical_dates, historical_prices, label='Historical Prices', marker='o', color='blue')

        # Plot predicted prices
        ax.plot(predicted_dates, predicted_prices, label='Predicted Prices', marker='o', linestyle='--', color='red')

        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f"{symbol} Stock Prices")
        ax.legend()
        ax.grid(True)

        # In plotting code
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate(rotation=45)

        # Save the plot to a BytesIO object
        img_buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(img_buffer, format='PNG')
        plt.close(fig)
        img_buffer.seek(0)

        # Insert the image into the PDF
        elements.append(Image(img_buffer, width=500, height=250))
        elements.append(Spacer(1, 12))

        # Add predicted stock prices
        elements.append(Paragraph("Predicted Stock Prices for Next 30 Days", styles['CustomHeading2']))

        # Prepare table data
        table_data = [['Date', 'Predicted Price']] + [
            [date.strftime('%Y-%m-%d'), f"${price:.2f}"]
            for date, price in zip(predictions_df['date'], predictions_df['predicted_price'])
        ]

        # Create table
        t = Table(table_data, colWidths=[150, 150])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Roboto-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTNAME', (0, 1), (-1, -1), 'Roboto'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 12))

        # Backtesting results
        elements.append(Paragraph("Backtesting Results", styles['CustomHeading2']))
        backtest_table_data = [
            ['Metric', 'Value'],
            ['Total Return', f"{total_return:.2f}%"],
            ['Max Drawdown', f"{max_drawdown:.2f}%"],
            ['Number of Trades Executed', str(num_trades)]
        ]
        t = Table(backtest_table_data, colWidths=[200, 100])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Roboto-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTNAME', (0, 1), (-1, -1), 'Roboto'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(t)

        # Build the PDF
        doc.build(elements)

        # Get the value of the BytesIO buffer and write it to the response
        pdf = buffer.getvalue()
        buffer.close()
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{symbol}_stock_report.pdf"'
        response.write(pdf)

        return response

    else:
        # If GET request, render the form template
        return render(request, 'finance_app/generate_report.html')
