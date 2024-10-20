# finance_app/models.py

from django.db import models

class StockData(models.Model):
    symbol = models.CharField(max_length=10)
    date = models.DateField()
    open_price = models.DecimalField(max_digits=15, decimal_places=4, null=True, blank=True)
    high_price = models.DecimalField(max_digits=15, decimal_places=4, null=True, blank=True)
    low_price = models.DecimalField(max_digits=15, decimal_places=4, null=True, blank=True)
    close_price = models.DecimalField(max_digits=15, decimal_places=4, null=True, blank=True)
    adjusted_close_price = models.DecimalField(max_digits=15, decimal_places=4, null=True, blank=True)
    volume = models.BigIntegerField(null=True, blank=True)
    dividend_amount = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)

    class Meta:
        unique_together = ('symbol', 'date')
        ordering = ['date']

    def __str__(self):
        return f"{self.symbol} on {self.date}"
