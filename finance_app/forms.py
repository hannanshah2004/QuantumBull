from django import forms

class BacktestForm(forms.Form):
    initial_investment = forms.DecimalField(
        label="Initial Investment",
        min_value=0,
        widget=forms.NumberInput(attrs={
            'class': 'w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent',
            'placeholder': 'Enter initial investment amount'
        })
    )
    short_window = forms.IntegerField(
        label="Short Window",
        min_value=1,
        widget=forms.NumberInput(attrs={
            'class': 'w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent',
            'placeholder': 'Enter short window period'
        })
    )
    long_window = forms.IntegerField(
        label="Long Window",
        min_value=1,
        widget=forms.NumberInput(attrs={
            'class': 'w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent',
            'placeholder': 'Enter long window period'
        })
    )
    symbol = forms.CharField(
        label="Stock Symbol",
        max_length=10,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent',
            'placeholder': 'Enter stock symbol'
        })
    )
