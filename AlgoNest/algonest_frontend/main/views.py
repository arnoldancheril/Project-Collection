# views.py

from django.shortcuts import render, redirect, get_object_or_404
from .models import Bot
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
import os

def landing_page(request):
    return render(request, 'landing_page.html')

def bots_list(request):
    bots = Bot.objects.all()
    return render(request, 'bots_list.html', {'bots': bots})

def bot_detail(request, bot_id):
    bot = get_object_or_404(Bot, pk=bot_id)

    # Load the algorithm code if it exists for the bot
    algorithm_code = ''
    algorithm_filename = ''
    if bot.name == 'Aggressive Alpha':
        algorithm_filename = 'aggressive_alpha_algorithm.py'
    elif bot.name == 'Balanced Beta':
        algorithm_filename = 'balanced_beta_algorithm.py'
    elif bot.name == 'Steady Sigma':
        algorithm_filename = 'steady_sigma_algorithm.py'

    if algorithm_filename:
        # Construct the path to the algorithm file
        algorithm_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'algorithms',
            algorithm_filename
        )
        try:
            with open(algorithm_path, 'r') as file:
                algorithm_code = file.read()
        except FileNotFoundError:
            algorithm_code = 'Algorithm code not available.'

    context = {
        'bot': bot,
        'algorithm_code': algorithm_code,
    }
    return render(request, 'bot_detail.html', context)

@login_required
def user_dashboard(request):
    # Simulate user's investments in bots
    # In a real application, retrieve this data from the database based on the logged-in user
    user_investments = [
        {
            'bot': Bot.objects.get(name='Aggressive Alpha'),
            'amount_invested': 5000,
            'current_value': 6200,
            'profit_loss': 6200 - 5000,  # current_value - amount_invested
            'performance_data': [
                ['2024-09-01', 5000],
                ['2024-09-08', 5400],
                ['2024-09-15', 5800],
                ['2024-09-22', 6000],
                ['2024-09-29', 6200],
            ],
        },
        {
            'bot': Bot.objects.get(name='Balanced Beta'),
            'amount_invested': 3000,
            'current_value': 3300,
            'profit_loss': 3300 - 3000,  # current_value - amount_invested
            'performance_data': [
                ['2024-09-01', 3000],
                ['2024-09-08', 3100],
                ['2024-09-15', 3200],
                ['2024-09-22', 3250],
                ['2024-09-29', 3300],
            ],
        },
        {
            'bot': Bot.objects.get(name='Steady Sigma'),
            'amount_invested': 8000,
            'current_value': 8500,
            'profit_loss': 8500 - 8000,  # current_value - amount_invested
            'performance_data': [
                ['2024-09-01', 8000],
                ['2024-09-08', 8050],
                ['2024-09-15', 8100],
                ['2024-09-22', 8300],
                ['2024-09-29', 8500],
            ],
        },
    ]

    # Check if the user has investments
    if user_investments:
        # Calculate totals
        total_invested = sum(inv['amount_invested'] for inv in user_investments)
        total_current_value = sum(inv['current_value'] for inv in user_investments)
        total_profit_loss = total_current_value - total_invested

        # Prepare data for the overall performance chart
        performance_dates = [data[0] for data in user_investments[0]['performance_data']]
        performance_values = [
            sum(inv['performance_data'][i][1] for inv in user_investments)
            for i in range(len(performance_dates))
        ]
    else:
        # Default values when no investments
        total_invested = 0
        total_current_value = 0
        total_profit_loss = 0
        performance_dates = []
        performance_values = []

    context = {
        'user_investments': user_investments,
        'total_invested': total_invested,
        'total_current_value': total_current_value,
        'total_profit_loss': total_profit_loss,
        'performance_dates': performance_dates,
        'performance_values': performance_values,
    }

    return render(request, 'user_dashboard.html', context)

def profile(request):
    return render(request, 'profile.html')

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

def pricing(request):
    return render(request, 'pricing.html')

def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

def privacy_policy(request):
    return render(request, 'privacy_policy.html')

def terms_of_service(request):
    return render(request, 'terms_of_service.html')
