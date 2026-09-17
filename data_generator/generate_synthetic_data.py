#!/usr/bin/env python3
"""
generate_synthetic_data.py — synthetic Shopify + Klaviyo data for merchant "abc".

Populates the RAW layers only (schema.sql Layer 1A + 1B):
    products, product_variants, customers, orders, order_lines,
    klaviyo_profiles, campaigns, email_events

It deliberately does NOT touch customer_identity_map, customer_events or any
derived table. Those belong to ingestion/ and transformation/ — the pipeline
has to earn its results from raw data, exactly as it would with a real merchant.

────────────────────────────────────────────────────────────────────────────
THE BEHAVIOURAL MODEL  (why campaign-to-purchase patterns come out realistic)
────────────────────────────────────────────────────────────────────────────
Every purchase is the product of two things:

  1. NEED   — a repurchase clock. After each order the customer becomes
              "in-market" again after a personal interval (derived from the
              profile's order_range), with lognormal noise, shortened in
              high-season months via SEASONAL_ORDER_MULTIPLIER.

  2. NUDGE  — once in-market, a subscribed customer needs k campaign touches
              before converting, where k is drawn per purchase cycle from the
              profile's campaigns_before_purchase_range. From the k-th touch
              on, each delivered campaign converts with a probability that
              depends on engagement (clicked > opened > ignored) and on the
              campaign type (discount-led beats newsletter).

  k = 0, a non-subscriber, or a purchase cycle that began before Klaviyo
  launched → the customer simply buys when the need arises (organic purchase).

The simulation runs chronologically, campaign by campaign, because targeting
depends on customer state at send time (reactivation → 60+ days silent,
loyalty → 3+ orders, welcome → joined in the last 30 days) and customer state
depends on earlier campaigns.

Timeline
    DATE_START ─────────── CAMPAIGN_START ─────────────── DATE_END
    │  orders only (organic) │  orders + Klaviyo campaigns  │
    Campaigns cover the final CAMPAIGN_MONTHS (default 18) of the window.

Deliberate real-world wrinkles (all switchable in the constants below)
    • Klaviyo-only profiles with no Shopify customer      → identity resolution
    • A few Klaviyo emails stored in different letter case → LOWER() on the join
    • ~0.5% of orders arrive with no order_lines rows      → LEFT JOIN discipline
    • Unsubscribes flip Klaviyo consent but not Shopify's  → systems disagree

Ground truth (profile per customer, trigger per order) is written to
data_generator/output/ (git-ignored) and never to the database, so the
recommendation agent can be back-tested without being able to cheat.

Usage
    python generate_synthetic_data.py                 # generate + load
    python generate_synthetic_data.py --reset         # wipe existing data first
    python generate_synthetic_data.py --dry-run       # no DB, just the report
    python generate_synthetic_data.py --csv           # also dump tables as CSV
    python generate_synthetic_data.py --port 5433     # override DB port
"""
from __future__ import annotations

import argparse
import bisect
import csv
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from statistics import mean, median

import numpy as np
from dateutil.relativedelta import relativedelta
from faker import Faker

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import config as cfg  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(SCRIPT_DIR.parent / ".env")
except ImportError:  # dotenv is optional for --dry-run
    pass


# ══════════════════════════════════════════════════════════════════════════
# Parameters  (config.py is the source of truth; .env may override scale/dates)
# ══════════════════════════════════════════════════════════════════════════
def _env_date(key: str, default: date) -> date:
    raw = os.getenv(key)
    return date.fromisoformat(raw) if raw else default


NUM_CUSTOMERS = int(os.getenv("SYNTHETIC_NUM_CUSTOMERS", cfg.NUM_CUSTOMERS))
SEED = int(os.getenv("SYNTHETIC_RANDOM_SEED", cfg.RANDOM_SEED))
DATE_START = _env_date("SYNTHETIC_DATE_START", cfg.DATE_START)
DATE_END = _env_date("SYNTHETIC_DATE_END", cfg.DATE_END)
CAMPAIGN_MONTHS = getattr(cfg, "CAMPAIGN_MONTHS", 18)

START_DT = datetime.combine(DATE_START, time.min)
END_DT = datetime.combine(DATE_END, time(23, 59, 59))
CAMPAIGN_START_DT = datetime.combine(
    DATE_END + timedelta(days=1) - relativedelta(months=CAMPAIGN_MONTHS), time.min
)
KLAVIYO_LAUNCH_DT = CAMPAIGN_START_DT - timedelta(days=21)  # bulk profile sync
TOTAL_SPAN_DAYS = (END_DT - START_DT).days

# Behaviour knobs not covered by config.py
KLAVIYO_ONLY_PROFILES = 40        # subscribers who never became Shopify customers
EMAIL_CASE_VARIANT_PCT = 0.03     # Klaviyo email stored in different letter case
ORDERS_WITHOUT_LINES_PCT = 0.005  # orders whose line items never synced
CHECKOUT_FIRST_PCT = 0.70         # customer record created by their first checkout
PROMO_SEGMENT_PCT = 0.75          # share of subscribers a promo is sent to
BOUNCE_RATE = 0.012
REFUND_RATE = 0.03
ATTRIBUTION_WINDOW_DAYS = 30      # only used for the sanity report

ACCEPTS_MARKETING_RATE = {"organic": 0.55, "lapsed": 0.85}  # others: 0.90
CONVERT_P = {"clicked": 0.85, "opened": 0.55, "ignored": 0.04}
TYPE_LIFT = {"promotional": 1.0, "seasonal": 1.1, "reactivation": 1.2,
             "welcome": 1.1, "loyalty": 1.0, "newsletter": 0.6}
LAG_MEAN_DAYS = {"clicked": (1.0, 5), "opened": (3.0, 10), "ignored": (6.0, 14)}

MEAN_SEASONAL = float(np.mean(list(cfg.SEASONAL_ORDER_MULTIPLIER.values())))  # keeps yearly volume on target

rng = np.random.default_rng(SEED)
Faker.seed(SEED)
fake = Faker("en_US")


# ── tiny RNG helpers (always return plain Python types) ───────────────────
def rf() -> float:
    return float(rng.random())


def ri(a: int, b: int) -> int:
    """Inclusive integer."""
    return int(rng.integers(a, b + 1))


def pick(n_or_seq, p=None):
    if isinstance(n_or_seq, int):
        return int(rng.choice(n_or_seq, p=p))
    return n_or_seq[int(rng.choice(len(n_or_seq), p=p))]


def id_series(base: int, max_step: int):
    cur = base
    while True:
        cur += ri(1, max_step)
        yield str(cur)


_B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def klaviyo_id(prefix: str) -> str:
    return prefix + "".join(_B32[ri(0, 31)] for _ in range(24))


def daytime(ts: datetime, not_before: datetime | None = None) -> datetime:
    """Move a timestamp to a plausible shopping hour on the same day."""
    out = ts.replace(hour=ri(7, 22), minute=ri(0, 59), second=ri(0, 59), microsecond=0)
    if not_before and out <= not_before:
        out = not_before + timedelta(minutes=ri(10, 180))
    if out > END_DT:
        out = END_DT - timedelta(minutes=ri(1, 120))
    return out


# ══════════════════════════════════════════════════════════════════════════
# Catalogue
# ══════════════════════════════════════════════════════════════════════════
# variant tuple: (title, position in the type's price range 0..1, weight in grams)
CATALOGUE = {
    "Whole Bean Coffee": {
        "code": "WB",
        "products": ["Ethiopia Yirgacheffe Single Origin", "Colombia Huila Single Origin",
                     "House Espresso Blend"],
        "variants": [("250g / Light Roast", 0.00, 250), ("250g / Medium Roast", 0.00, 250),
                     ("250g / Dark Roast", 0.00, 250), ("500g / Medium Roast", 0.45, 500),
                     ("500g / Dark Roast", 0.45, 500), ("1kg / Medium Roast", 1.00, 1000)],
    },
    "Ground Coffee": {
        "code": "GR",
        "products": ["Breakfast Blend", "Guatemala Antigua", "Decaf Swiss Water Blend"],
        "variants": [("250g / Drip Grind", 0.00, 250), ("250g / French Press Grind", 0.00, 250),
                     ("250g / Moka Pot Grind", 0.00, 250), ("500g / Drip Grind", 0.50, 500),
                     ("500g / French Press Grind", 0.50, 500), ("1kg / Drip Grind", 1.00, 1000)],
    },
    "Espresso Pods": {
        "code": "EP",
        "products": ["Ristretto Pods", "Lungo Pods", "Decaf Pods"],
        "variants": [("10 Pods / Intensity 6", 0.00, 60), ("10 Pods / Intensity 9", 0.00, 60),
                     ("30 Pods / Intensity 6", 0.45, 180), ("30 Pods / Intensity 9", 0.45, 180),
                     ("50 Pods / Variety Pack", 0.80, 300), ("100 Pods / Variety Pack", 1.00, 600)],
    },
    "Loose Leaf Tea": {
        "code": "LT",
        "products": ["Jasmine Green", "Assam Breakfast Black", "Milk Oolong"],
        "variants": [("50g Tin", 0.00, 120), ("100g Refill Pouch", 0.30, 110),
                     ("100g Tin", 0.40, 200), ("Sampler / 4 x 25g", 0.50, 150),
                     ("250g Refill Pouch", 0.80, 260), ("500g Bulk Bag", 1.00, 520)],
    },
    "Cold Brew Kits": {
        "code": "CB",
        "products": ["Classic Cold Brew", "Vanilla Cold Brew", "Nitro-Style Cold Brew"],
        "variants": [("4-Pouch Pack", 0.00, 400), ("Concentrate / 1L", 0.20, 1100),
                     ("8-Pouch Pack", 0.40, 800), ("Starter Kit / 1L Brewer", 0.60, 900),
                     ("12-Pouch Pack", 0.70, 1200), ("Starter Kit / 2L Brewer", 1.00, 1400)],
    },
    "Accessories": {
        "code": "AC",
        "products": ["Burr Hand Grinder", "Pour-Over Dripper Set", "Precision Brew Scale"],
        "variants": [("Matte Black", 0.20, 700), ("Stainless Steel", 0.35, 750),
                     ("Copper", 0.50, 750), ("Matte Black / Gift Box", 0.60, 900),
                     ("Stainless Steel / Gift Box", 0.80, 950), ("Pro Edition", 1.00, 1000)],
    },
}
TYPE_POPULARITY = {"Whole Bean Coffee": 0.30, "Ground Coffee": 0.22, "Espresso Pods": 0.20,
                   "Loose Leaf Tea": 0.13, "Cold Brew Kits": 0.10, "Accessories": 0.05}


def type_weights(month: int) -> list[float]:
    """Product-type mix shifts with the season (cold brew in summer, gifts in Nov–Dec)."""
    w = dict(TYPE_POPULARITY)
    if month in (6, 7, 8):
        w["Cold Brew Kits"] *= 2.5
    if month in (11, 12, 1, 2):
        w["Cold Brew Kits"] *= 0.5
        w["Loose Leaf Tea"] *= 1.3
    if month in (11, 12):
        w["Accessories"] *= 2.5
    names = [t["name"] for t in cfg.PRODUCT_TYPES]
    vals = [w.get(n, 0.05) for n in names]
    s = sum(vals)
    return [v / s for v in vals]


def build_catalogue(category_ids: dict[str, int]):
    products, variants, by_type = [], [], {}
    pid_gen = id_series(8_100_000_000_000, 900_000)
    vid_gen = id_series(44_200_000_000_000, 900_000)
    for ptype in cfg.PRODUCT_TYPES:
        name = ptype["name"]
        lo, hi = ptype["price_range"]
        spec = CATALOGUE.get(name) or {
            "code": "".join(w[0] for w in name.split())[:3].upper(),
            "products": [f"{name} No.{i}" for i in (1, 2, 3)],
            "variants": [(f"Option {i + 1}", i / 5, 300) for i in range(6)],
        }
        by_type[name] = []
        for p_i, title in enumerate(spec["products"]):
            product_id = next(pid_gen)
            created = START_DT - timedelta(days=ri(60, 400))
            prices = []
            for v_i, (v_title, frac, grams) in enumerate(spec["variants"][: cfg.VARIANTS_PER_PRODUCT]):
                raw = lo + (hi - lo) * min(1.0, frac * 0.92 + p_i * 0.04)
                price = float(min(hi, math.floor(raw) + 0.99))
                prices.append(price)
                v = {
                    "variant_id": next(vid_gen), "product_id": product_id, "title": v_title,
                    "sku": f"{spec['code']}-{p_i + 1:02d}-{v_i + 1:02d}", "price": price,
                    "compare_at_price": round(price * 1.15, 2) if rf() < 0.2 else None,
                    "inventory_quantity": ri(0, 400), "created": created,
                    "grams": grams, "product_title": title, "product_type": name,
                }
                variants.append(v)
                by_type[name].append(v)
            products.append({
                "product_id": product_id, "title": title, "product_type": name,
                "vendor": "abc Roasters" if name != "Accessories" else "abc Brew Gear",
                "category_id": category_ids.get(name),
                "tags": ", ".join([name.lower(), "synthetic"]),
                "price_min": min(prices), "price_max": max(prices), "created": created,
            })
    return products, variants, by_type


# ══════════════════════════════════════════════════════════════════════════
# Customers
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Customer:
    profile: dict
    is_shopify: bool
    customer_id: str | None
    email: str
    first_name: str
    last_name: str
    phone: str
    city: str
    state: str
    zip: str
    created_at: datetime
    accepts_marketing: bool
    # Klaviyo side
    has_profile: bool = False
    klaviyo_profile_id: str | None = None
    klaviyo_email: str | None = None
    klaviyo_created_at: datetime | None = None
    subscribed: bool = False
    # stable preferences
    open_rate: float = 0.2
    click_rate: float = 0.05
    fav_type: str = ""
    fav_variant: dict | None = None
    tax_rate: float = 0.0
    device: str = "mobile"
    client: str = "gmail"
    source: str = "web"
    # simulation state
    target_orders: int = 0               # sets the repurchase rhythm
    max_orders: int = 0                  # hard ceiling (only binds for low-activity profiles)
    interval_days: float = 60.0
    orders_done: int = 0
    total_spent: float = 0.0
    last_order_at: datetime | None = None
    need_at: datetime | None = None      # when the customer is next in-market
    cycle_start: datetime | None = None  # when the current purchase cycle began
    k_needed: int = 0                    # campaign touches required this cycle
    touches: int = 0                     # touches received since need_at
    pending: tuple | None = None         # (order_ts, campaign, engaged)
    received: list = field(default_factory=list)   # delivered send_times (sorted)
    n_sent: int = 0
    n_opened: int = 0
    n_clicked: int = 0
    last_event: datetime | None = None


def draw_profile() -> dict:
    weights = np.array([p["weight"] for p in cfg.CUSTOMER_PROFILES], dtype=float)
    return pick(cfg.CUSTOMER_PROFILES, p=weights / weights.sum())


_used_emails: set[str] = set()


def unique_email(first: str, last: str) -> str:
    clean = lambda s: "".join(ch for ch in s.lower() if ch.isalpha())  # noqa: E731
    while True:
        # example.* domains are reserved (RFC 2606) — no real inbox can ever match
        email = f"{clean(first)}.{clean(last)}{ri(1, 99)}@example.{pick(['com', 'org', 'net'])}"
        if email not in _used_emails:
            _used_emails.add(email)
            return email


def person() -> dict:
    first, last = fake.first_name(), fake.last_name()
    return {
        "first_name": first, "last_name": last, "email": unique_email(first, last),
        "phone": f"+1 ({ri(201, 989)}) 555-{ri(0, 9999):04d}",
        "city": fake.city()[:100], "state": fake.state()[:100], "zip": fake.zipcode(),
    }


def jitter_rate(base: float, cap: float) -> float:
    return float(min(cap, max(0.01, base * rng.lognormal(0, 0.25))))


def attach_klaviyo(c: Customer) -> None:
    c.has_profile = True
    c.klaviyo_profile_id = klaviyo_id("01H")
    c.klaviyo_email = c.email
    if rf() < EMAIL_CASE_VARIANT_PCT:
        local, domain = c.email.split("@")
        c.klaviyo_email = ".".join(p.capitalize() for p in local.split(".")) + "@" + domain.capitalize()
    if c.created_at < KLAVIYO_LAUNCH_DT:      # existing customers bulk-synced at launch
        c.klaviyo_created_at = KLAVIYO_LAUNCH_DT + timedelta(seconds=ri(0, 6 * 3600))
    else:
        c.klaviyo_created_at = c.created_at + timedelta(minutes=ri(1, 30))
    c.subscribed = c.accepts_marketing


def build_customers(by_type: dict, cid_gen) -> list[Customer]:
    customers: list[Customer] = []
    type_names = [t["name"] for t in cfg.PRODUCT_TYPES]
    base_w = [TYPE_POPULARITY.get(n, 0.05) for n in type_names]
    base_w = [w / sum(base_w) for w in base_w]
    signup_span = max(1, (TOTAL_SPAN_DAYS - 45) * 86400)

    for _ in range(NUM_CUSTOMERS):
        prof = draw_profile()
        created = daytime(START_DT + timedelta(seconds=ri(0, signup_span)))
        accepts = rf() < ACCEPTS_MARKETING_RATE.get(prof["name"], 0.90)
        c = Customer(profile=prof, is_shopify=True, customer_id=next(cid_gen),
                     created_at=created, accepts_marketing=accepts, **person())
        c.open_rate = jitter_rate(prof["open_rate"], 0.85)
        c.click_rate = min(c.open_rate * 0.9, jitter_rate(prof["click_rate"], 0.6))
        c.fav_type = pick(type_names, p=base_w)
        c.fav_variant = pick(by_type[c.fav_type])
        c.tax_rate = pick([0.0, 0.04, 0.06, 0.0625, 0.0725, 0.0825, 0.0875])
        c.device = pick(["mobile", "desktop", "tablet"], p=[0.58, 0.36, 0.06])
        c.client = pick(["gmail", "apple_mail", "outlook", "yahoo"], p=[0.45, 0.33, 0.15, 0.07])
        src_p = [0.45, 0.20, 0.32, 0.03] if prof["name"] == "high_value" else [0.66, 0.27, 0.04, 0.03]
        c.source = pick(["web", "mobile_app", "subscription", "pos"], p=src_p)

        # Order budget: order_range describes a full-window customer; scale by tenure.
        tenure_days = max(1, (END_DT - created).days)
        n_full = ri(*prof["order_range"])
        if prof["name"] == "lapsed":
            c.target_orders = n_full
        else:
            c.target_orders = max(1, round(n_full * max(tenure_days / TOTAL_SPAN_DAYS, 0.15)))
        # Only profiles defined by *not* buying get a hard ceiling. A ceiling on active
        # buyers makes everyone run out of budget late in the window and fakes a churn wave.
        c.max_orders = {"lapsed": c.target_orders, "occasional": c.target_orders + 1}.get(prof["name"], 10**6)
        if c.target_orders:
            c.interval_days = float(np.clip(tenure_days / c.target_orders, 14, 540)) * MEAN_SEASONAL

        if accepts or rf() < 0.5:   # non-consenting customers are only sometimes synced
            attach_klaviyo(c)
        customers.append(c)

    # Klaviyo-only subscribers: exist in Klaviyo, never checked out on Shopify.
    lapsed_like = {"name": "klaviyo_only", "order_range": (0, 0),
                   "campaigns_before_purchase_range": (0, 0), "open_rate": 0.15, "click_rate": 0.03}
    span = max(1, int((END_DT - timedelta(days=30) - KLAVIYO_LAUNCH_DT).total_seconds()))
    for _ in range(KLAVIYO_ONLY_PROFILES):
        created = daytime(KLAVIYO_LAUNCH_DT + timedelta(seconds=ri(0, span)))
        c = Customer(profile=lapsed_like, is_shopify=False, customer_id=None,
                     created_at=created, accepts_marketing=True, **person())
        c.open_rate = jitter_rate(0.15, 0.85)
        c.click_rate = min(c.open_rate * 0.9, jitter_rate(0.03, 0.6))
        attach_klaviyo(c)
        customers.append(c)
    return customers


# ══════════════════════════════════════════════════════════════════════════
# Orders
# ══════════════════════════════════════════════════════════════════════════
ORDERS: list[dict] = []


def schedule_next_need(c: Customer, from_ts: datetime) -> None:
    c.need_at, c.touches, c.k_needed, c.cycle_start = None, 0, 0, from_ts
    if c.orders_done >= c.max_orders:
        return
    gap = c.interval_days * float(rng.lognormal(0, 0.35))
    tentative = from_ts + timedelta(days=gap)
    gap = max(7.0, gap / cfg.SEASONAL_ORDER_MULTIPLIER.get(tentative.month, 1.0))
    need = from_ts + timedelta(days=gap)
    if need <= END_DT:
        c.need_at = need
        c.k_needed = ri(*c.profile["campaigns_before_purchase_range"])


def needs_campaign(c: Customer) -> bool:
    # Purchase cycles already in flight when Klaviyo launched complete organically —
    # turning on email does not make existing customers stop buying overnight.
    return (c.has_profile and c.subscribed and c.k_needed > 0
            and c.cycle_start is not None and c.cycle_start >= CAMPAIGN_START_DT)


def place_order(c: Customer, ts: datetime, by_type: dict, campaign=None, engaged=False) -> None:
    type_names = [t["name"] for t in cfg.PRODUCT_TYPES]
    line_p = [0.40, 0.40, 0.20] if c.profile["name"] == "high_value" else [0.62, 0.28, 0.10]
    reorder_p = 0.15 if c.fav_type == "Accessories" else 0.70
    chosen: dict[str, dict] = {}
    for i in range(pick([1, 2, 3], p=line_p)):
        if i == 0 and c.orders_done > 0 and rf() < reorder_p:
            v = c.fav_variant
        elif i == 0 and c.orders_done == 0 and c.fav_type != "Accessories":
            v = c.fav_variant
        else:
            v = pick(by_type[pick(type_names, p=type_weights(ts.month))])
        qty = pick([1, 2, 3], p=[0.75, 0.20, 0.05])
        if v["variant_id"] in chosen:
            chosen[v["variant_id"]]["quantity"] += qty
        else:
            chosen[v["variant_id"]] = {"variant": v, "quantity": qty}

    code, pct = None, 0.0
    if campaign is not None and campaign.discount_pct and rf() < (0.80 if engaged else 0.30):
        code, pct = campaign.discount_code, campaign.discount_pct
    elif c.orders_done == 0 and rf() < 0.25:
        code, pct = "WELCOME10", 10.0

    lines, gross, discounts, grams = [], 0.0, 0.0, 0
    for item in chosen.values():
        v, qty = item["variant"], item["quantity"]
        disc = round(v["price"] * qty * pct / 100, 2)
        gross += v["price"] * qty
        discounts += disc
        grams += v["grams"] * qty
        lines.append({"variant": v, "quantity": qty, "total_discount": disc})

    subtotal = round(gross - discounts, 2)          # Shopify: subtotal is after discounts
    tax = round(subtotal * c.tax_rate, 2)
    total = round(subtotal + tax, 2)

    recent = ts > END_DT - timedelta(days=2)
    financial = "pending" if recent and rf() < 0.5 else ("refunded" if rf() < REFUND_RATE else "paid")
    fulfillment = "unfulfilled" if recent else "fulfilled"

    ORDERS.append({
        "customer": c, "ts": ts, "lines": lines,
        "drop_lines": rf() < ORDERS_WITHOUT_LINES_PCT,
        "financial_status": financial, "fulfillment_status": fulfillment,
        "updated": min(END_DT, ts + timedelta(days=ri(1, 4), hours=ri(0, 10))),
        "subtotal": subtotal, "discounts": round(discounts, 2), "total": total, "tax": tax,
        "grams": grams, "discount_code": code,
        "trigger_campaign": campaign, "touches": c.touches if campaign is not None else 0,
    })
    c.orders_done += 1
    if financial != "refunded":
        c.total_spent = round(c.total_spent + total, 2)
    c.last_order_at = ts
    c.pending = None
    schedule_next_need(c, ts)


def advance(c: Customer, until: datetime, by_type: dict) -> None:
    """Realise everything that happens to this customer strictly before `until`."""
    while c.is_shopify:
        if c.pending is not None:
            if c.pending[0] >= until:
                return
            ts, camp, engaged = c.pending
            place_order(c, ts, by_type, campaign=camp, engaged=engaged)
            continue
        if c.need_at is None or c.need_at >= until or c.orders_done >= c.max_orders:
            return
        if needs_campaign(c):
            return                                   # in-market, waiting for the nudge
        place_order(c, daytime(c.need_at, not_before=c.last_order_at), by_type)


def initialise_purchasing(customers: list[Customer], by_type: dict) -> None:
    for c in customers:
        if not c.is_shopify or c.target_orders == 0:
            continue
        checkout_first = (not c.accepts_marketing) or rf() < CHECKOUT_FIRST_PCT
        if checkout_first:
            place_order(c, c.created_at, by_type)    # account created by this checkout
        else:                                        # signed up first, buys later
            c.cycle_start = c.created_at
            c.need_at = c.created_at + timedelta(days=ri(3, 30))
            c.k_needed = ri(*c.profile["campaigns_before_purchase_range"])
            if c.need_at > END_DT:
                c.need_at = None


# ══════════════════════════════════════════════════════════════════════════
# Campaigns
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Campaign:
    campaign_id: str
    name: str
    type_name: str
    type_id: int | None
    subject: str
    send_time: datetime
    created_at: datetime
    discount_code: str | None
    discount_pct: float | None
    recipients: int = 0
    delivered: int = 0
    opens: int = 0
    clicks: int = 0
    unsubs: int = 0


TITLES = {
    "promotional": ["Flash Sale — {pct}% Off All Beans", "Stock Up & Save {pct}%",
                    "Weekend Brew Deal — {pct}% Off", "Pods Restock Event — {pct}% Off",
                    "Mid-Month Refill — {pct}% Off Your Usual"],
    "newsletter": ["Brew Guide: Dialing In Your Pour-Over", "Origin Story: Meet Our Huila Growers",
                   "The Roast Report", "Tea Notes: Steeping Oolong Right", "Grind Size 101",
                   "Water Matters: The Overlooked Ingredient"],
    "reactivation": ["We Miss You — {pct}% Off Your Next Bag",
                     "Your Grinder Is Getting Lonely ({pct}% Inside)",
                     "Come Back for a Fresh Roast — {pct}% Off"],
    "welcome": ["Welcome to abc — {pct}% Off Your First Order"],
    "loyalty": ["Thank You Reward — {pct}% for Our Regulars", "Members' Early Access + {pct}% Off"],
}
SEASONAL_TITLES = {1: "New Year, New Brew", 2: "Valentine's Gift Sets", 3: "Spring Roast Release",
                   4: "Spring Refresh Bundles", 5: "Mother's Day Gift Guide",
                   6: "Cold Brew Summer Launch", 7: "Iced Coffee Season",
                   8: "Back-to-Routine Bundles", 9: "Fall Flavors Are Here",
                   10: "Cozy Season Blends", 11: "Black Friday / Cyber Week",
                   12: "Holiday Gift Sets"}
DISCOUNT_OPTIONS = {"promotional": [10, 15, 20], "seasonal": [15, 20, 25, 30],
                    "reactivation": [20, 25], "welcome": [10], "loyalty": [15, 20]}
CODE_PREFIX = {"promotional": "BREW", "seasonal": "SEASON", "reactivation": "COMEBACK",
               "welcome": "HELLO", "loyalty": "VIP"}
FORCE_SEASONAL_MONTHS = (1, 6, 11, 12)
WEEKDAY_W = [0.15, 0.25, 0.20, 0.25, 0.10, 0.025, 0.025]   # Mon..Sun


def build_campaign_schedule(type_lookup: dict[str, tuple[int | None, bool]]) -> list[Campaign]:
    names = list(cfg.CAMPAIGN_TYPE_WEIGHTS)
    weights = np.array([cfg.CAMPAIGN_TYPE_WEIGHTS[n] for n in names], dtype=float)
    weights /= weights.sum()
    campaigns, seq = [], 0
    month = CAMPAIGN_START_DT.date().replace(day=1)
    while month <= DATE_END:
        next_month = month + relativedelta(months=1)
        days = [month + timedelta(days=d) for d in range((next_month - month).days)]
        days = [d for d in days if CAMPAIGN_START_DT.date() <= d <= DATE_END - timedelta(days=1)]
        n = min(len(days), ri(*cfg.CAMPAIGNS_PER_MONTH_RANGE))
        chosen: list[date] = []
        for _ in range(25):                          # prefer sends ≥ 3 days apart
            w = np.array([WEEKDAY_W[d.weekday()] for d in days])
            idx = rng.choice(len(days), size=n, replace=False, p=w / w.sum())
            chosen = sorted(days[int(i)] for i in idx)
            if all((b - a).days >= 3 for a, b in zip(chosen, chosen[1:])):
                break
        for j, day in enumerate(chosen):
            ctype = "seasonal" if (j == 0 and month.month in FORCE_SEASONAL_MONTHS) else pick(names, p=weights)
            send = datetime.combine(day, time(pick([9, 10, 11, 17, 18]), pick([0, 15, 30, 45])))
            _, type_has_discount = type_lookup.get(ctype, (None, ctype != "newsletter"))
            pct = None
            if type_has_discount and ctype in DISCOUNT_OPTIONS and (ctype != "seasonal" or rf() < 0.85):
                pct = float(pick(DISCOUNT_OPTIONS[ctype]))
            if ctype == "seasonal":
                title = SEASONAL_TITLES[month.month] + (f" — {pct:.0f}% Off" if pct else "")
            else:
                title = pick(TITLES[ctype]).format(pct=f"{pct:.0f}" if pct else "")
            seq += 1
            campaigns.append(Campaign(
                campaign_id=klaviyo_id("01J"), name=f"{send:%Y-%m} | {title}", type_name=ctype,
                type_id=type_lookup.get(ctype, (None, False))[0], subject=title, send_time=send,
                created_at=send - timedelta(days=ri(1, 5), hours=ri(0, 6)),
                discount_code=f"{CODE_PREFIX[ctype]}{pct:.0f}-{send:%m%y}{chr(64 + j + 1)}" if pct else None,
                discount_pct=pct,
            ))
        month = next_month
    return sorted(campaigns, key=lambda c: c.send_time)


def is_recipient(c: Customer, camp: Campaign) -> bool:
    if not (c.has_profile and c.subscribed and c.klaviyo_created_at <= camp.send_time):
        return False
    t = camp.type_name
    if t == "promotional":
        return rf() < PROMO_SEGMENT_PCT
    if t == "reactivation":
        anchor = c.last_order_at or c.created_at
        return (camp.send_time - anchor).days >= 60
    if t == "welcome":
        return (camp.send_time - c.created_at).days <= 30
    if t == "loyalty":
        return c.orders_done >= 3
    return True                                       # newsletter, seasonal


EMAIL_EVENTS: list[tuple] = []


def send_campaign(c: Customer, camp: Campaign) -> None:
    def log(kind, ts, url=None, engaged=False):
        EMAIL_EVENTS.append((camp.campaign_id, c.klaviyo_profile_id, c.klaviyo_email, kind, ts, url,
                             c.device if engaged else None, c.client if engaged else None))
        c.last_event = ts if c.last_event is None else max(c.last_event, ts)

    camp.recipients += 1
    c.n_sent += 1
    sent_ts = camp.send_time + timedelta(seconds=ri(0, 900))
    log("sent", sent_ts)
    if rf() < BOUNCE_RATE:
        log("bounced", sent_ts + timedelta(seconds=ri(1, 60)))
        return
    delivered_ts = sent_ts + timedelta(seconds=ri(1, 120))
    log("delivered", delivered_ts)
    camp.delivered += 1
    c.received.append(camp.send_time)

    in_market = (c.is_shopify and c.pending is None and c.need_at is not None
                 and c.need_at <= camp.send_time and c.orders_done < c.max_orders
                 and needs_campaign(c))

    # People who need coffee pay more attention to coffee emails.
    p_open = min(0.92, c.open_rate * (1.3 if in_market else 1.0))
    opened = rf() < p_open
    clicked, engage_ts = False, delivered_ts
    if opened:
        open_ts = min(END_DT, delivered_ts + timedelta(hours=min(96.0, float(rng.exponential(5.0)) + 0.02)))
        log("opened", open_ts, engaged=True)
        camp.opens += 1
        c.n_opened += 1
        engage_ts = open_ts
        p_click = min(0.9, (c.click_rate / c.open_rate) * (1.25 if in_market else 1.0))
        if rf() < p_click:
            clicked = True
            click_ts = min(END_DT, open_ts + timedelta(seconds=ri(20, 900)))
            slug = camp.type_name if camp.type_name != "newsletter" else "journal"
            log("clicked", click_ts, engaged=True,
                url=f"https://shop.abc-coffee.example/{slug}?utm_source=klaviyo&utm_campaign={camp.campaign_id}")
            camp.clicks += 1
            c.n_clicked += 1
            engage_ts = click_ts
        p_unsub = 0.006 if c.profile["name"] in ("lapsed", "occasional", "klaviyo_only") else 0.0015
        if not clicked and rf() < p_unsub:
            unsub_ts = min(END_DT, open_ts + timedelta(seconds=ri(30, 240)))
            log("unsubscribed", unsub_ts, engaged=True)
            camp.unsubs += 1
            c.subscribed = False
            if c.need_at is not None:     # they now buy on their own clock, but not in the past
                c.need_at = max(c.need_at, unsub_ts + timedelta(days=ri(5, 40)))
                if c.need_at > END_DT:
                    c.need_at = None
            return

    if not in_market:
        return
    c.touches += 1
    if c.touches < c.k_needed:
        return
    level = "clicked" if clicked else "opened" if opened else "ignored"
    if rf() < min(0.97, CONVERT_P[level] * TYPE_LIFT.get(camp.type_name, 1.0)):
        lag_mean, lag_cap = LAG_MEAN_DAYS[level]
        lag = min(float(lag_cap), float(rng.exponential(lag_mean)))
        order_ts = daytime(engage_ts + timedelta(days=lag), not_before=engage_ts)
        if engage_ts < order_ts <= END_DT:
            c.pending = (order_ts, camp, opened)


# ══════════════════════════════════════════════════════════════════════════
# Simulation driver
# ══════════════════════════════════════════════════════════════════════════
def simulate(category_ids: dict, type_lookup: dict) -> dict:
    products, variants, by_type = build_catalogue(category_ids)
    customers = build_customers(by_type, id_series(7_300_000_000_000, 4_000_000))
    campaigns = build_campaign_schedule(type_lookup)
    initialise_purchasing(customers, by_type)

    sent_campaigns = []
    for camp in campaigns:
        for c in customers:
            advance(c, camp.send_time, by_type)
        recipients = [c for c in customers if is_recipient(c, camp)]
        if not recipients:                            # e.g. a welcome send with no new signups
            continue
        for c in recipients:
            send_campaign(c, camp)
        sent_campaigns.append(camp)
    for c in customers:
        advance(c, END_DT + timedelta(seconds=1), by_type)

    # Shopify-style IDs and order numbers, assigned in chronological order.
    ORDERS.sort(key=lambda o: o["ts"])
    oid_gen, lid_gen = id_series(5_400_000_000_000, 600_000), id_series(13_800_000_000_000, 300_000)
    for n, o in enumerate(ORDERS, start=1001):
        o["order_id"], o["order_number"] = next(oid_gen), f"#{n}"
        for ln in o["lines"]:
            ln["order_line_id"] = next(lid_gen)
    EMAIL_EVENTS.sort(key=lambda e: e[4])
    return {"products": products, "variants": variants, "customers": customers,
            "campaigns": sent_campaigns, "orders": ORDERS, "email_events": EMAIL_EVENTS}


# ══════════════════════════════════════════════════════════════════════════
# Row builders  (column order must match the INSERT statements in TABLES)
# ══════════════════════════════════════════════════════════════════════════
def build_rows(d: dict) -> dict[str, list[tuple]]:
    shopify = [c for c in d["customers"] if c.is_shopify]
    rows: dict[str, list[tuple]] = {}
    rows["products"] = [(p["product_id"], p["title"], p["product_type"], p["vendor"], p["category_id"],
                         p["tags"], p["price_min"], p["price_max"], p["created"], p["created"], p["created"])
                        for p in d["products"]]
    rows["product_variants"] = [(v["variant_id"], v["product_id"], v["title"], v["sku"], v["price"],
                                 v["compare_at_price"], v["inventory_quantity"], v["created"], v["created"])
                                for v in d["variants"]]
    rows["customers"] = [(c.customer_id, c.email, c.first_name, c.last_name, c.phone, c.city, c.state,
                          "US", c.zip, c.accepts_marketing, c.created_at, c.orders_done, c.total_spent,
                          "subscription" if c.source == "subscription" else None, True)
                         for c in shopify]
    rows["orders"] = [(o["order_id"], o["customer"].customer_id, o["order_number"], o["financial_status"],
                       o["fulfillment_status"], o["ts"], o["updated"], o["subtotal"], o["discounts"],
                       o["total"], o["tax"], o["grams"], o["discount_code"], o["customer"].source,
                       o["customer"].city, "US", o["customer"].zip,
                       "subscription" if o["customer"].source == "subscription" else None)
                      for o in d["orders"]]
    rows["order_lines"] = [(ln["order_line_id"], o["order_id"], ln["variant"]["product_id"],
                            ln["variant"]["variant_id"], ln["variant"]["sku"], ln["variant"]["product_title"],
                            ln["variant"]["product_type"], ln["variant"]["title"], ln["quantity"],
                            ln["variant"]["price"], ln["total_discount"])
                           for o in d["orders"] if not o["drop_lines"] for ln in o["lines"]]
    rows["klaviyo_profiles"] = [(c.klaviyo_profile_id, c.klaviyo_email, c.first_name, c.last_name, c.phone,
                                 c.city, "United States", c.subscribed, c.klaviyo_created_at, c.last_event,
                                 c.n_sent, c.n_opened, c.n_clicked)
                                for c in d["customers"] if c.has_profile]
    rows["campaigns"] = [(k.campaign_id, k.name, k.type_id, k.subject, k.send_time, "sent", k.recipients,
                          k.recipients, k.delivered, k.opens, k.clicks, k.unsubs, k.discount_code is not None,
                          k.discount_code, k.discount_pct, k.created_at)
                         for k in d["campaigns"]]
    rows["email_events"] = list(d["email_events"])
    return rows


# Insert order respects foreign keys. line_total (generated) and event_id (UUID default) are omitted.
TABLES = {
    "products": "product_id, title, product_type, vendor, category_id, tags, price_min, price_max, "
                "published_at, product_created_at, product_updated_at",
    "product_variants": "variant_id, product_id, title, sku, price, compare_at_price, inventory_quantity, "
                        "variant_created_at, variant_updated_at",
    "customers": "customer_id, email, first_name, last_name, phone, city, state_province, country_code, zip, "
                 "accepts_marketing, customer_created_at, orders_count, total_spent_usd, tags, verified_email",
    "orders": "order_id, customer_id, order_number, financial_status, fulfillment_status, order_created_at, "
              "order_updated_at, subtotal_price, total_discounts, total_price_usd, total_tax, "
              "total_weight_grams, discount_code, source_name, shipping_city, shipping_country_code, "
              "shipping_zip, order_tags",
    "order_lines": "order_line_id, order_id, product_id, variant_id, sku, product_title, product_type, "
                   "variant_title, quantity, price, total_discount",
    "klaviyo_profiles": "klaviyo_profile_id, email, first_name, last_name, phone, city, country, "
                        "accepts_marketing, klaviyo_created_at, last_event_date, total_emails_sent, "
                        "total_emails_opened, total_emails_clicked",
    "campaigns": "campaign_id, campaign_name, campaign_type_id, subject_line, send_time, status, "
                 "total_recipients, total_sent, total_delivered, total_opens, total_clicks, "
                 "total_unsubscribes, has_discount_code, discount_code, discount_pct, campaign_created_at",
    "email_events": "campaign_id, klaviyo_profile_id, email, event_type, event_timestamp, url_clicked, "
                    "device_type, email_client",
}
# Everything --reset clears. Lookups (product_category, campaign_type) and pipeline_runs are kept.
RESET_TABLES = ["customer_recommendations", "customer_kpis", "campaign_attribution", "customer_derivatives",
                "customer_events", "customer_identity_map", "email_events", "campaigns", "klaviyo_profiles",
                "order_lines", "orders", "product_variants", "products", "customers"]


# ══════════════════════════════════════════════════════════════════════════
# Database
# ══════════════════════════════════════════════════════════════════════════
def connect(args):
    import psycopg2
    # Note: POSTGRES_HOST/PORT in .env describe the network *inside* Docker (db:5432).
    # This script runs on the host, so it uses the published port instead (5433).
    params = dict(
        host=args.host or os.getenv("SYNTHETIC_DB_HOST", "localhost"),
        port=int(args.port or os.getenv("SYNTHETIC_DB_PORT", 5433)),
        dbname=os.getenv("POSTGRES_DB", "merchant_abc"),
        user=os.getenv("POSTGRES_USER", "abc_user"),
        password=os.getenv("POSTGRES_PASSWORD", "abc_dev_password"),
    )
    try:
        conn = psycopg2.connect(**params)
    except psycopg2.OperationalError as exc:
        sys.exit(f"\n✗ Could not connect to Postgres at {params['host']}:{params['port']} "
                 f"(db={params['dbname']}, user={params['user']}).\n  {exc}\n"
                 "  Is the container up?  →  docker compose ps\n")
    print(f"✓ Connected to {params['dbname']} at {params['host']}:{params['port']}")
    return conn


def read_lookups(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT category_name, category_id FROM product_category")
        categories = dict(cur.fetchall())
        cur.execute("SELECT type_name, campaign_type_id, has_discount FROM campaign_type")
        types = {name: (tid, disc) for name, tid, disc in cur.fetchall()}
    if not categories or not types:
        sys.exit("✗ Lookup tables are empty — has db/schema.sql been applied?")
    return categories, types


def fallback_lookups():
    """IDs as seeded by schema.sql — only used for --dry-run."""
    categories = {t["name"]: i for i, t in enumerate(cfg.PRODUCT_TYPES, start=1)}
    order = ["promotional", "newsletter", "reactivation", "welcome", "seasonal", "loyalty"]
    return categories, {n: (i, n != "newsletter") for i, n in enumerate(order, start=1)}


def _py(v):
    return v.item() if isinstance(v, np.generic) else v


def load(conn, rows: dict[str, list[tuple]], reset: bool) -> None:
    from psycopg2.extras import execute_values
    with conn:                                        # one transaction: all or nothing
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM customers")
            existing = cur.fetchone()[0]
            if existing and not reset:
                sys.exit(f"✗ customers already holds {existing} rows. Re-run with --reset to replace them.")
            if reset:
                cur.execute("SELECT table_name FROM information_schema.tables "
                            "WHERE table_schema = current_schema() AND table_name = ANY(%s)", (RESET_TABLES,))
                present = [r[0] for r in cur.fetchall()]
                cur.execute(f"TRUNCATE {', '.join(present)} RESTART IDENTITY CASCADE")
                print(f"✓ Reset {len(present)} tables")
            for table, columns in TABLES.items():
                data = [tuple(_py(v) for v in r) for r in rows[table]]
                execute_values(cur, f"INSERT INTO {table} ({columns}) VALUES %s", data, page_size=2000)
                print(f"  → {table:<18} {len(data):>7,} rows")
    print("✓ Load committed")


# ══════════════════════════════════════════════════════════════════════════
# Output files + sanity report
# ══════════════════════════════════════════════════════════════════════════
def write_csvs(d: dict, rows: dict, all_tables: bool) -> Path:
    out = SCRIPT_DIR / "output"
    out.mkdir(exist_ok=True)

    def dump(name, header, data):
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(data)

    dump("ground_truth_customers.csv",
         ["customer_id", "email", "klaviyo_profile_id", "profile", "accepts_marketing",
          "still_subscribed", "target_orders", "orders_placed"],
         [(c.customer_id, c.email, c.klaviyo_profile_id, c.profile["name"], c.accepts_marketing,
           c.subscribed, c.target_orders, c.orders_done) for c in d["customers"]])
    dump("ground_truth_orders.csv",
         ["order_id", "customer_id", "order_created_at", "profile", "trigger", "trigger_campaign_id",
          "campaign_touches_this_cycle"],
         [(o["order_id"], o["customer"].customer_id, o["ts"], o["customer"].profile["name"],
           "campaign" if o["trigger_campaign"] else "organic",
           o["trigger_campaign"].campaign_id if o["trigger_campaign"] else None, o["touches"])
          for o in d["orders"]])
    if all_tables:
        for table, columns in TABLES.items():
            dump(f"{table}.csv", [c.strip() for c in columns.split(",")], rows[table])
    return out


def report(d: dict, rows: dict) -> None:
    print("\n" + "═" * 78)
    print(f" Window {DATE_START} → {DATE_END}   |   campaigns from {CAMPAIGN_START_DT.date()}"
          f"   |   seed {SEED}")
    print("═" * 78)
    for table in TABLES:
        print(f"  {table:<18} {len(rows[table]):>7,}")
    no_match = sum(1 for c in d["customers"] if not c.is_shopify)
    no_profile = sum(1 for c in d["customers"] if c.is_shopify and not c.has_profile)
    case_var = sum(1 for c in d["customers"] if c.has_profile and c.klaviyo_email != c.email)
    no_lines = sum(1 for o in d["orders"] if o["drop_lines"])
    print(f"\n  Identity wrinkles : {no_match} Klaviyo-only profiles · {no_profile} Shopify customers "
          f"with no Klaviyo profile · {case_var} case-variant emails")
    print(f"  LEFT JOIN wrinkle : {no_lines} orders with no order_lines rows")

    era_floor = CAMPAIGN_START_DT + timedelta(days=ATTRIBUTION_WINDOW_DAYS)
    print("\n  Does the data carry the intended signal?  (campaign-era orders only)")
    print(f"  {'profile':<16}{'cust':>5}{'orders':>8}{'ord/cust':>9}{'never':>7}"
          f"{'camp-trig':>10}{'touches~':>9}{'in 30d':>8}{'open':>7}{'click':>7}")
    for prof in cfg.CUSTOMER_PROFILES:
        cs = [c for c in d["customers"] if c.profile["name"] == prof["name"]]
        if not cs:
            continue
        ids = {id(c) for c in cs}
        orders = [o for o in d["orders"] if id(o["customer"]) in ids]
        era = [o for o in orders if o["ts"] >= era_floor]
        trig = [o for o in era if o["trigger_campaign"]]
        window = []
        for o in era:
            rec = o["customer"].received
            lo = bisect.bisect_left(rec, o["ts"] - timedelta(days=ATTRIBUTION_WINDOW_DAYS))
            window.append(bisect.bisect_left(rec, o["ts"]) - lo)
        sent = sum(c.n_sent for c in cs)
        print(f"  {prof['name']:<16}{len(cs):>5}{len(orders):>8}{len(orders) / len(cs):>9.1f}"
              f"{sum(1 for c in cs if c.orders_done == 0):>7}"
              f"{(len(trig) / len(era) if era else 0):>10.0%}"
              f"{(median(o['touches'] for o in trig) if trig else 0):>9.1f}"
              f"{(mean(window) if window else 0):>8.1f}"
              f"{(sum(c.n_opened for c in cs) / sent if sent else 0):>7.0%}"
              f"{(sum(c.n_clicked for c in cs) / sent if sent else 0):>7.0%}")
    print("\n  camp-trig = share of orders triggered by a campaign   touches~ = median touches before"
          "\n  those orders   in 30d = avg campaigns received in the 30 days before an order")
    print("═" * 78)


# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic Shopify + Klaviyo data for merchant abc.")
    ap.add_argument("--dry-run", action="store_true", help="generate and report, but do not touch the database")
    ap.add_argument("--reset", action="store_true", help="truncate raw + derived tables before loading")
    ap.add_argument("--csv", action="store_true", help="also write every table to data_generator/output/")
    ap.add_argument("--host", help="DB host (default: localhost)")
    ap.add_argument("--port", help="DB port (default: 5433, the published Docker port)")
    args = ap.parse_args()

    conn = None
    if args.dry_run:
        categories, types = fallback_lookups()
    else:
        conn = connect(args)
        categories, types = read_lookups(conn)

    print(f"Simulating {NUM_CUSTOMERS} customers …")
    data = simulate(categories, types)
    rows = build_rows(data)
    report(data, rows)
    out = write_csvs(data, rows, all_tables=args.csv)
    print(f"✓ Ground truth written to {out}/")

    if conn is not None:
        load(conn, rows, reset=args.reset)
        conn.close()
    else:
        print("Dry run — database untouched.")


if __name__ == "__main__":
    main()
