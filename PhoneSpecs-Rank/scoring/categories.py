# scoring/categories.py
from scoring.normalization import min_max_normalize, DISPLAY_SCORES

def score_performance(cpu, gpu, ram):
    return round((cpu * 0.5 + gpu * 0.3 + ram * 0.2), 2)

def score_display(dtype, refresh, ppi):
    return round((
        DISPLAY_SCORES.get(dtype, 60) * 0.4 +
        min_max_normalize(refresh, 60, 144) * 0.3 +
        min_max_normalize(ppi, 300, 600) * 0.3
    ), 2)

def score_battery(capacity, charging):
    return round((
        min_max_normalize(capacity, 3000, 6000) * 0.7 +
        min_max_normalize(charging, 10, 120) * 0.3
    ), 2)
