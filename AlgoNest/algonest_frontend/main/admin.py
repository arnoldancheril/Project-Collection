# from django.contrib import admin
# from .models import Bot

# admin.site.register(Bot)

from django.contrib import admin
from .models import Bot

@admin.register(Bot)
class BotAdmin(admin.ModelAdmin):
    list_display = ('name', 'risk_level', 'monthly_price')
    fields = ('name', 'description', 'monthly_price', 'risk_level', 'performance_data', 'greek_values', 'sample_trades')
