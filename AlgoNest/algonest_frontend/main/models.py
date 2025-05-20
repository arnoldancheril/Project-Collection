# from django.db import models
# # from django.contrib.postgres.fields import ArrayField, JSONField

# class Bot(models.Model):
#     name = models.CharField(max_length=100)
#     description = models.TextField()
#     monthly_price = models.DecimalField(max_digits=6, decimal_places=2)
#     performance_data = models.JSONField(default=list)  # Store historical performance as a JSON field
#     risk_level = models.CharField(max_length=20, choices=[
#         ('Low Risk', 'Low Risk'),
#         ('Medium Risk', 'Medium Risk'),
#         ('High Risk', 'High Risk'),
#     ],
#     default='Medium Risk')
#     greek_values = JSONField(default=list)  # List of dicts
#     sample_trades = JSONField(default=list)  # List of dicts

#     def __str__(self):
#         return self.name

from django.db import models

class Bot(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    monthly_price = models.DecimalField(max_digits=6, decimal_places=2)
    performance_data = models.JSONField(default=list)  # Store historical performance as a JSON field
    risk_level = models.CharField(max_length=20, choices=[
        ('Low Risk', 'Low Risk'),
        ('Medium Risk', 'Medium Risk'),
        ('High Risk', 'High Risk'),
    ],
    default='Medium Risk')
    greek_values = models.JSONField(default=list)  # List of dicts, import JSONField from models
    sample_trades = models.JSONField(default=list)  # List of dicts, import JSONField from models

    def __str__(self):
        return self.name
