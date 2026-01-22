"""
SHARED STATE - Monitoring va Django o'rtasida ma'lumot almashish
JSON fayl orqali ishlaydi
"""

import json
import os
from pathlib import Path

# Ma'lumotlar fayli
STATE_FILE = Path(__file__).parent / 'monitoring_state.json'

def save_state(barcha: int, faol: int, uxlayotgan: int = 0):
    """Monitoring natijalarini saqlash"""
    data = {
        'barcha': barcha,
        'faol': faol,
        'uxlayotgan': uxlayotgan
    }
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"⚠️ State saqlanmadi: {e}")


def get_state() -> dict:
    """Monitoring natijalarini olish (Django uchun)"""
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ State o'qilmadi: {e}")
    
    # Default qiymatlar
    return {'barcha': 0, 'faol': 0, 'uxlayotgan': 0}