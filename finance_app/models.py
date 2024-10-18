from django.db import models

class StockData(models.Model):
    symbol = models.CharField(max_length=10)
    date = models.DateField()
    open_price = models.FloatField()
    high_price = models.FloatField()
    low_price = models.FloatField()
    close_price = models.FloatField()
    adjusted_close_price = models.FloatField(null=True, blank=True)
    volume = models.BigIntegerField()
    dividend_amount = models.FloatField(null=True, blank=True)

    class Meta:
        unique_together = ('symbol', 'date')
