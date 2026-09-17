"""
config.py — Synthetic data generation parameters.
Adjust these to change the shape of the test dataset.
"""
from datetime import date

# ── Scale ──────────────────────────────────────────────────────
NUM_CUSTOMERS          = 500
DATE_START             = date(2022, 1, 1)
DATE_END               = date(2024, 12, 31)
RANDOM_SEED            = 42

# ── Product catalogue ──────────────────────────────────────────
PRODUCT_TYPES = [
    {"name": "Whole Bean Coffee",  "price_range": (18, 45),  "repurchase_days": 28},
    {"name": "Ground Coffee",      "price_range": (14, 38),  "repurchase_days": 28},
    {"name": "Espresso Pods",      "price_range": (12, 30),  "repurchase_days": 21},
    {"name": "Loose Leaf Tea",     "price_range": (10, 28),  "repurchase_days": 35},
    {"name": "Cold Brew Kits",     "price_range": (20, 55),  "repurchase_days": 45},
    {"name": "Accessories",        "price_range": (25, 120), "repurchase_days": 180},
]

# 6 variants per product type
VARIANTS_PER_PRODUCT = 6

# ── Customer behaviour profiles ────────────────────────────────
# Each profile controls: how many orders, how many campaigns needed, etc.
CUSTOMER_PROFILES = [
    {
        "name": "high_value",
        "weight": 0.15,        # 15% of customers
        "order_range": (8, 24),
        "campaigns_before_purchase_range": (1, 3),
        "open_rate": 0.55,
        "click_rate": 0.25,
    },
    {
        "name": "campaign_driven",
        "weight": 0.25,
        "order_range": (3, 10),
        "campaigns_before_purchase_range": (3, 6),
        "open_rate": 0.40,
        "click_rate": 0.18,
    },
    {
        "name": "organic",
        "weight": 0.20,
        "order_range": (2, 8),
        "campaigns_before_purchase_range": (0, 1),
        "open_rate": 0.20,
        "click_rate": 0.05,
    },
    {
        "name": "occasional",
        "weight": 0.25,
        "order_range": (1, 3),
        "campaigns_before_purchase_range": (5, 12),
        "open_rate": 0.25,
        "click_rate": 0.08,
    },
    {
        "name": "lapsed",
        "weight": 0.15,
        "order_range": (0, 1),
        "campaigns_before_purchase_range": (10, 20),
        "open_rate": 0.10,
        "click_rate": 0.02,
    },
]

# ── Campaign schedule ──────────────────────────────────────────
# Campaigns per month (min, max)
CAMPAIGNS_PER_MONTH_RANGE = (2, 5)

CAMPAIGN_TYPE_WEIGHTS = {
    "promotional":  0.35,
    "newsletter":   0.25,
    "reactivation": 0.10,
    "welcome":      0.05,
    "seasonal":     0.15,
    "loyalty":      0.10,
}

# ── Seasonal order boost ───────────────────────────────────────
# Multiplier on order probability by month (1.0 = baseline)
SEASONAL_ORDER_MULTIPLIER = {
    1: 1.3,   # Jan (New Year coffee resolutions)
    2: 0.9,
    3: 1.0,
    4: 1.0,
    5: 1.1,
    6: 1.2,   # Summer cold brew launch
    7: 1.2,
    8: 1.0,
    9: 1.1,
    10: 1.0,
    11: 1.4,  # Pre-holiday
    12: 1.6,  # Holiday peak
}
