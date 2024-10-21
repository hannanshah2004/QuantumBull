# QuantumMaze

**QuantumMaze** is a Django-based backend for fetching, analyzing, and predicting stock market data. It integrates the Alpha Vantage API for data retrieval, TensorFlow's LSTM for price predictions, and Docker for containerized deployment with a CI/CD pipeline via GitHub Actions.

## Features
- **Stock Data Fetching**: Retrieves daily prices via Alpha Vantage API.
- **Data Optimization**: Limits data to the last two years.
- **Backtesting**: Simulates investment strategies using moving averages.
- **Price Predictions**: Leverages LSTM for stock price forecasting.
- **Reports**: Generates PDF and JSON performance reports.
- **CI/CD**: Automated testing and deployment with GitHub Actions.

## Installation
1. Clone the repository and set up a virtual environment:
    ```bash
    git clone https://github.com/hannanshah2004/QuantumBull.git
    cd QuantumBull
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2. Set up environment variables in a `.env` file:
    ```env
    ALPHAVANTAGE_API_KEY=your_api_key
    SECRET_KEY=your_secret_key
    DATABASE_URL=postgres://user:password@localhost:5432/quantumbull
    ```

3. Apply migrations and start the development server:
    ```bash
    python manage.py migrate
    python manage.py runserver
    ```

## Usage
- **Fetch Stock Data**: Submit a stock symbol via `/fetch_data/`.
- **Backtest Strategies**: Test strategies with `/backtest/`.
- **Predict Prices**: Get stock price forecasts using `/predict/`.
- **Generate Reports**: Access `/generate_report/` for detailed analysis.

## CI/CD Pipeline
The CI/CD pipeline automates deployment to AWS EC2 using Docker. GitHub Actions runs tests in containers and deploys the updated app upon successful builds.

## EC2 Instance Limitations
Due to memory and computational constraints on the EC2 instance, the full functionality of the application cannot be tested live on the website. However, you can [watch this video](#) showing the system in action. You can still navigate the website for an overview of the features.

## Docker Setup
To run locally:
```bash
docker build -t quantummaze:latest .
docker run -d -p 8000:8000 --env-file .env quantummaze:latest
