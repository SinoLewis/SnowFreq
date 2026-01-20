# scoring/normalization.py
def min_max_normalize(value, min_v, max_v):
    if value is None:
        return 0
    return round((value - min_v) / (max_v - min_v) * 100, 2)

# Categorical Mappings
DISPLAY_SCORES = {
    "AMOLED": 100,
    "OLED": 85,
    "IPS LCD": 70,
    "TFT": 50
}

STORAGE_SCORES = {
    "UFS 4.0": 100,
    "UFS 3.1": 90,
    "UFS 2.2": 75,
    "eMMC": 50
}
