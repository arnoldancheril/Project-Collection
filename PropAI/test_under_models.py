#!/usr/bin/env python3
"""Test Under and Specialized Models"""
import sys
import time

START_DATE = '2025-12-01'
END_DATE = '2026-02-03'

results = {}

def test_model(name, module_name, func_name):
    print(f'\n{"="*70}')
    print(f'Testing: {name}')
    print(f'{"="*70}')
    try:
        module = __import__(f'src.nba_props.engine.{module_name}', fromlist=[func_name])
        func = getattr(module, func_name)
        start = time.time()
        r = func(START_DATE, END_DATE)
        duration = time.time() - start
        
        # Extract results
        total_picks = getattr(r, 'total_picks', 0) or getattr(r, 'picks_count', 0) or getattr(r, 'total', 0) or getattr(r, 'picks', 0) or 0
        total_hits = getattr(r, 'total_hits', 0) or getattr(r, 'hits_count', 0) or getattr(r, 'hits', 0) or 0
        hit_rate = (total_hits / total_picks * 100) if total_picks > 0 else 0
        
        sb_picks = getattr(r, 'sportsbook_picks', 0) or 0
        sb_hits = getattr(r, 'sportsbook_hits', 0) or 0
        der_picks = getattr(r, 'derived_picks', 0) or 0
        der_hits = getattr(r, 'derived_hits', 0) or 0
        
        pts_picks = getattr(r, 'pts_picks', 0) or 0
        pts_hits = getattr(r, 'pts_hits', 0) or 0
        reb_picks = getattr(r, 'reb_picks', 0) or 0
        reb_hits = getattr(r, 'reb_hits', 0) or 0
        
        results[name] = {
            'total_picks': total_picks,
            'total_hits': total_hits,
            'hit_rate': hit_rate,
            'sb_picks': sb_picks,
            'sb_hits': sb_hits,
            'der_picks': der_picks,
            'der_hits': der_hits,
            'pts_picks': pts_picks,
            'pts_hits': pts_hits,
            'reb_picks': reb_picks,
            'reb_hits': reb_hits,
            'duration': duration,
            'error': None
        }
        
        print(f'  Total: {total_hits}/{total_picks} ({hit_rate:.1f}%)')
        print(f'  Duration: {duration:.1f}s')
        
    except Exception as e:
        results[name] = {'error': str(e), 'total_picks': 0, 'total_hits': 0, 'hit_rate': 0}
        print(f'  ERROR: {str(e)[:100]}')

# Test Under models and specialized models
models = [
    ('V13 Under', 'model_v13_under', 'run_backtest_v13_under'),
    ('V14 Under', 'model_v14_under', 'run_backtest_v14_under'),
    ('V15 Under', 'model_v15_under', 'run_backtest_v15_under'),
    ('V17 Under', 'model_v17_under', 'run_backtest_v17_under'),
    ('V18 Under', 'model_v18_under', 'run_backtest_v18_under'),
    ('V19 Under', 'model_v19_under', 'run_backtest_v19_under'),
    ('Under V2', 'under_model_v2', 'backtest_under_model_v2'),
    ('RCM', 'regression_contribution_model', 'run_rcm_backtest'),
    ('Production', 'model_production', 'run_backtest'),
    ('Final', 'model_final', 'run_full_backtest'),
]

for name, module, func in models:
    test_model(name, module, func)

# Print summary
print('\n\n')
print('='*100)
print('UNDER & SPECIALIZED MODELS - BACKTEST SUMMARY')
print(f'Period: {START_DATE} to {END_DATE}')
print('='*100)
print(f'{"Model":<18} {"Picks":>8} {"Hits":>8} {"Rate":>8} {"SB Rate":>10} {"Der Rate":>10} {"PTS":>10} {"REB":>10}')
print('-'*100)

sorted_results = sorted([(k, v) for k, v in results.items()], key=lambda x: -x[1].get('hit_rate', 0))
for name, r in sorted_results:
    if r.get('error'):
        print(f'{name:<18} ERROR: {r["error"][:60]}')
    else:
        sb_rate = f"{r['sb_hits']/r['sb_picks']*100:.1f}%" if r['sb_picks'] > 0 else '-'
        der_rate = f"{r['der_hits']/r['der_picks']*100:.1f}%" if r['der_picks'] > 0 else '-'
        pts_rate = f"{r['pts_hits']/r['pts_picks']*100:.1f}%" if r['pts_picks'] > 0 else '-'
        reb_rate = f"{r['reb_hits']/r['reb_picks']*100:.1f}%" if r['reb_picks'] > 0 else '-'
        print(f'{name:<18} {r["total_picks"]:>8} {r["total_hits"]:>8} {r["hit_rate"]:>7.1f}% {sb_rate:>10} {der_rate:>10} {pts_rate:>10} {reb_rate:>10}')
