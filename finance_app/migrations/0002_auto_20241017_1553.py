# Generated by Django 3.2.25 on 2024-10-17 19:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('finance_app', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='stockdata',
            name='adjusted_close_price',
            field=models.DecimalField(blank=True, decimal_places=4, max_digits=15, null=True),
        ),
        migrations.AddField(
            model_name='stockdata',
            name='dividend_amount',
            field=models.DecimalField(blank=True, decimal_places=4, max_digits=10, null=True),
        ),
        migrations.AlterField(
            model_name='stockdata',
            name='close_price',
            field=models.DecimalField(blank=True, decimal_places=4, max_digits=15, null=True),
        ),
    ]
