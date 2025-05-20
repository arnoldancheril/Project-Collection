# highlight_code.py

from django import template
from django.utils.html import escape

register = template.Library()

@register.filter(name='escape_code')
def escape_code(value):
    return escape(value)
