from django import forms

class BacktestForm(forms.Form):
    initial_investment = forms.DecimalField(label="Initial Investment", min_value=0)
    short_window = forms.IntegerField(label="Short Window", min_value=1)
    long_window = forms.IntegerField(label="Long Window", min_value=1)
    symbol = forms.CharField(label="Stock Symbol", max_length=10, initial='AAPL', help_text="Enter the stock symbol (e.g., AAPL)")
