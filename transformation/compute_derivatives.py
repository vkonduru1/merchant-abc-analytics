#!/usr/bin/env python3
"""
transformation/compute_derivatives.py — build customer_derivatives.

Reads customer_events (the chronological overlay table) and computes
first-level derived features for every customer. Writes one row per
customer per pipeline run into customer_derivatives.

Key outputs
───────────
  Purchase behaviour   : first/second/latest purchase date, total orders,
                         total revenue, AOV, recency, frequency, tenure
  Customer type        : Prospect | One-time Purchaser | Repeat Purchaser
  Spend windows        : last 3 / 6 / 9 / 12 months
  Product behaviour    : distinct products, first/latest product type,
                         discount usage, season, holiday flag
  Email engagement     : campaigns received / opened / clicked, rates,
                         days since last campaign
  Attribution          : avg campaigns before purchase, campaigns before
                         first purchase, influence rate, attribution segment

Attribution segment definitions
────────────────────────────────
  no_campaign_influence : campaign_influence_rate == 0 AND orders > 0
                          (bought, but never with a campaign in the 30d window)
  campaign_1_2          : avg_campaigns_before_purchase in [1, 2]
  campaign_3_5          : avg_campaigns_before_purchase in [3, 5]
  campaign_gt5          : avg_campaigns_before_purchase > 5
  never_purchased       : 0 orders (prospect — may still have email events)

Attribution window
──────────────────
  For each ORDER event, count the number of CAMPAIGN_SENT events in the
  30 days before that order. If ≥ 1 campaign was sent in that window,
  the order is "campaign influenced." The average of those counts across
  all orders is avg_campaigns_before_purchase.

  Note: only orders after the campaign programme started (2023-07-01) are
  counted for attribution metrics. Earlier orders have no campaign data
  by construction and would inflate "no influence."

AS-OF date
──────────
  Recency and "days since last campaign" are measured against
  MAX(event_date) across all customer_events, not CURRENT_DATE.
  This keeps the numbers meaningful for a 2022-2024 historical dataset
  regardless of when the script is run.

Usage
    python transformation/compute_derivatives.py              # run
    python transformation/compute_derivatives.py --dry-run    # report, no DB writes
    python transformation/compute_derivatives.py --port 5433  # override port
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "data_generator"))

try:
    from dotenv import load_dotenv
    load_dotenv(SCRIPT_DIR.parent / ".env")
except ImportError:
    pass

ATTRIBUTION_WINDOW_DAYS = 30
CAMPAIGN_START = datetime(2023, 7, 1)   # keep in sync with config.py / CAMPAIGN_MONTHS


# ══════════════════════════════════════════════════════════════════════════
# DB helpers
# ══════════════════════════════════════════════════════════════════════════
def connect(args):
    import psycopg2
    params = dict(
        host=args.host or os.getenv("SYNTHETIC_DB_HOST", "localhost"),
        port=int(args.port or os.getenv("SYNTHETIC_DB_PORT", 5433)),
        dbname=os.getenv("POSTGRES_DB", "merchant_abc"),
        user=os.getenv("POSTGRES_USER", "abc_user"),
        password=os.getenv("POSTGRES_PASSWORD", "abc_dev_password"),
    )
    try:
        conn = psycopg2.connect(**params)
    except Exception as exc:
        sys.exit(f"\n✗ Cannot connect to Postgres at {params['host']}:{params['port']}.\n  {exc}\n")
    print(f"✓ Connected to {params['dbname']} at {params['host']}:{params['port']}")
    return conn


def latest_run_id(conn) -> str:
    """Re-use the most recent pipeline_runs row created by build_customer_events."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT run_id FROM pipeline_runs
            WHERE triggered_by = 'build_customer_events'
              AND status = 'completed'
            ORDER BY started_at DESC LIMIT 1
        """)
        row = cur.fetchone()
    if not row:
        sys.exit("✗ No completed build_customer_events run found. "
                 "Run transformation/build_customer_events.py first.")
    return row[0]


def new_run_id(conn) -> str:
    run_id = f"cd-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO pipeline_runs (run_id, run_type, status, started_at, triggered_by)
            VALUES (%s, 'full', 'running', NOW(), 'compute_derivatives')
        """, (run_id,))
    conn.commit()
    return run_id


def finish_run(conn, run_id: str, n: int, err: str | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE pipeline_runs
            SET status = %s, completed_at = NOW(), customers_processed = %s, error_message = %s
            WHERE run_id = %s
        """, ("failed" if err else "completed", n, err, run_id))
    conn.commit()


# ══════════════════════════════════════════════════════════════════════════
# Load customer_events into memory
# ══════════════════════════════════════════════════════════════════════════
def load_events(conn):
    """
    Returns (as_of_date, events_by_customer).

    events_by_customer: {customer_id: sorted list of event dicts}
    as_of_date: MAX(event_date) across all events — used as recency anchor
    """
    print("  Loading customer_events …")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                customer_id,
                event_date,
                event_type,
                revenue_amount,
                product_type,
                discount_code,
                has_discount,
                campaign_id,
                campaign_name,
                campaign_type,
                has_campaign_discount
            FROM customer_events
            ORDER BY customer_id, event_date
        """)
        rows = cur.fetchall()

    as_of = max(r[1] for r in rows) if rows else datetime.now()
    events: dict[str, list[dict]] = defaultdict(list)
    for (customer_id, event_date, event_type, revenue_amount, product_type,
         discount_code, has_discount, campaign_id, campaign_name,
         campaign_type, has_campaign_discount) in rows:
        events[customer_id].append({
            "date": event_date,
            "type": event_type,
            "revenue": float(revenue_amount) if revenue_amount else None,
            "product_type": product_type,
            "discount_code": discount_code,
            "has_discount": has_discount,
            "campaign_id": campaign_id,
            "campaign_name": campaign_name,
            "campaign_type": campaign_type,
            "has_campaign_discount": has_campaign_discount,
        })
    print(f"    {len(rows):,} events across {len(events):,} customers")
    print(f"    As-of date: {as_of.date()}")
    return as_of, dict(events)


# ══════════════════════════════════════════════════════════════════════════
# Per-customer computation
# ══════════════════════════════════════════════════════════════════════════
def season_of(dt: datetime) -> str:
    m = dt.month
    if m in (12, 1, 2):  return "WINTER"
    if m in (3, 4, 5):   return "SPRING"
    if m in (6, 7, 8):   return "SUMMER"
    return "FALL"


def compute_one(customer_id: str, evts: list[dict], as_of: datetime,
                run_id: str) -> dict:
    orders     = [e for e in evts if e["type"] == "ORDER"]
    campaigns  = [e for e in evts if e["type"] == "CAMPAIGN_SENT"]
    opens      = [e for e in evts if e["type"] == "EMAIL_OPENED"]
    clicks     = [e for e in evts if e["type"] == "EMAIL_CLICKED"]

    # ── Purchase behaviour ────────────────────────────────────────────────
    n_orders = len(orders)
    first_purchase   = orders[0]["date"]  if orders else None
    second_purchase  = orders[1]["date"]  if len(orders) >= 2 else None
    latest_purchase  = orders[-1]["date"] if orders else None

    total_revenue    = round(sum(o["revenue"] for o in orders if o["revenue"]), 2) if orders else 0.0
    aov              = round(total_revenue / n_orders, 2) if n_orders else None

    recency_days    = (as_of - latest_purchase).days  if latest_purchase else None
    recency_months  = round(recency_days / 30.44, 2)  if recency_days is not None else None

    if n_orders >= 2 and first_purchase and latest_purchase:
        span = (latest_purchase - first_purchase).days
        freq_months = round((span / (n_orders - 1)) / 30.44, 2)
    else:
        freq_months = None

    first_event   = evts[0]["date"]
    tenure_months = round((as_of - first_event).days / 30.44)

    if n_orders == 0:
        customer_type = "Prospect"
    elif n_orders == 1:
        customer_type = "One-time Purchaser"
    else:
        customer_type = "Repeat Purchaser"

    # ── Spend windows ─────────────────────────────────────────────────────
    def spend_in(months: int) -> float:
        cutoff = as_of - timedelta(days=months * 30)
        return round(sum(o["revenue"] for o in orders
                         if o["revenue"] and o["date"] >= cutoff), 2)
    spend_3m  = spend_in(3)
    spend_6m  = spend_in(6)
    spend_9m  = spend_in(9)
    spend_12m = spend_in(12)

    # ── Product behaviour ─────────────────────────────────────────────────
    product_types = [o["product_type"] for o in orders if o["product_type"]]
    distinct_products = len(set(product_types))
    first_product  = orders[0]["product_type"]  if orders else None
    latest_product = orders[-1]["product_type"] if orders else None
    discount_codes = set(o["discount_code"] for o in orders if o["discount_code"])
    n_discount_codes = len(discount_codes)
    season_last = season_of(latest_purchase) if latest_purchase else None
    holiday_shopper = any(o["date"].month in (11, 12) for o in orders)
    early_adopter_count = sum(
        1 for o in orders
        if o["date"] < (first_purchase + timedelta(days=14))  # type: ignore[operator]
    ) if first_purchase else 0

    # ── Email engagement ──────────────────────────────────────────────────
    n_sent    = len(campaigns)
    n_opened  = len(opens)
    n_clicked = len(clicks)
    open_rate  = round(n_opened  / n_sent, 4) if n_sent else None
    click_rate = round(n_clicked / n_sent, 4) if n_sent else None
    first_campaign  = campaigns[0]["date"]  if campaigns else None
    latest_campaign = campaigns[-1]["date"] if campaigns else None
    days_since_camp = (as_of - latest_campaign).days if latest_campaign else None

    # ── Campaign-to-purchase attribution ──────────────────────────────────
    # Only count orders that fell after the campaign programme began.
    # For each such order, count CAMPAIGN_SENT events in the 30-day window
    # before the order date.
    attr_orders = [o for o in orders if o["date"] >= CAMPAIGN_START]
    campaign_dates = [c["date"] for c in campaigns]

    windows: list[int] = []        # campaigns-in-window per attributable order
    for o in attr_orders:
        lo = o["date"] - timedelta(days=ATTRIBUTION_WINDOW_DAYS)
        n_in_window = sum(1 for cd in campaign_dates if lo <= cd < o["date"])
        windows.append(n_in_window)

    purchases_with_campaign    = sum(1 for w in windows if w >= 1)
    purchases_without_campaign = sum(1 for w in windows if w == 0)
    influence_rate = (
        round(purchases_with_campaign / len(attr_orders), 4)
        if attr_orders else None
    )
    avg_camps_before = (
        round(sum(windows) / len(windows), 2)
        if windows else None
    )
    camps_before_first = (
        windows[0] if attr_orders and orders and orders[0]["date"] >= CAMPAIGN_START
        else None
    )

    # ── Attribution segment ───────────────────────────────────────────────
    if n_orders == 0:
        segment = "never_purchased"
    elif not attr_orders or influence_rate == 0:
        segment = "no_campaign_influence"
    elif avg_camps_before is not None and avg_camps_before <= 2:
        segment = "campaign_1_2"
    elif avg_camps_before is not None and avg_camps_before <= 5:
        segment = "campaign_3_5"
    else:
        segment = "campaign_gt5"

    return {
        "customer_id":                      customer_id,
        "run_id":                           run_id,
        "first_purchase_date":              first_purchase,
        "second_purchase_date":             second_purchase,
        "latest_purchase_date":             latest_purchase,
        "total_orders":                     n_orders,
        "total_revenue":                    total_revenue,
        "average_order_value":              aov,
        "recency_days":                     recency_days,
        "recency_months":                   recency_months,
        "frequency_months":                 freq_months,
        "months_since_customer":            tenure_months,
        "customer_type":                    customer_type,
        "spend_last_3_months":              spend_3m,
        "spend_last_6_months":              spend_6m,
        "spend_last_9_months":              spend_9m,
        "spend_last_12_months":             spend_12m,
        "total_distinct_products":          distinct_products,
        "first_purchase_product_type":      first_product,
        "latest_purchase_product_type":     latest_product,
        "total_distinct_discount_codes":    n_discount_codes,
        "season_last_purchased":            season_last,
        "holiday_shopper":                  holiday_shopper,
        "early_adopter_count":              early_adopter_count,
        "total_campaigns_received":         n_sent,
        "total_campaigns_opened":           n_opened,
        "total_campaigns_clicked":          n_clicked,
        "first_campaign_date":              first_campaign,
        "latest_campaign_date":             latest_campaign,
        "email_open_rate":                  open_rate,
        "email_click_rate":                 click_rate,
        "days_since_last_campaign":         days_since_camp,
        "avg_campaigns_before_purchase":    avg_camps_before,
        "campaigns_before_first_purchase":  camps_before_first,
        "total_purchases_with_campaign":    purchases_with_campaign,
        "total_purchases_without_campaign": purchases_without_campaign,
        "campaign_influence_rate":          influence_rate,
        "attribution_segment":              segment,
    }


# ══════════════════════════════════════════════════════════════════════════
# Write
# ══════════════════════════════════════════════════════════════════════════
INSERT_SQL = """
INSERT INTO customer_derivatives (
    customer_id, run_id, run_date,
    first_purchase_date, second_purchase_date, latest_purchase_date,
    total_orders, total_revenue, average_order_value,
    recency_days, recency_months, frequency_months, months_since_customer,
    customer_type,
    spend_last_3_months, spend_last_6_months, spend_last_9_months, spend_last_12_months,
    total_distinct_products, first_purchase_product_type, latest_purchase_product_type,
    total_distinct_discount_codes, season_last_purchased, holiday_shopper, early_adopter_count,
    total_campaigns_received, total_campaigns_opened, total_campaigns_clicked,
    first_campaign_date, latest_campaign_date,
    email_open_rate, email_click_rate, days_since_last_campaign,
    avg_campaigns_before_purchase, campaigns_before_first_purchase,
    total_purchases_with_campaign, total_purchases_without_campaign,
    campaign_influence_rate, attribution_segment
) VALUES (
    %(customer_id)s, %(run_id)s, CURRENT_DATE,
    %(first_purchase_date)s, %(second_purchase_date)s, %(latest_purchase_date)s,
    %(total_orders)s, %(total_revenue)s, %(average_order_value)s,
    %(recency_days)s, %(recency_months)s, %(frequency_months)s, %(months_since_customer)s,
    %(customer_type)s,
    %(spend_last_3_months)s, %(spend_last_6_months)s, %(spend_last_9_months)s, %(spend_last_12_months)s,
    %(total_distinct_products)s, %(first_purchase_product_type)s, %(latest_purchase_product_type)s,
    %(total_distinct_discount_codes)s, %(season_last_purchased)s, %(holiday_shopper)s, %(early_adopter_count)s,
    %(total_campaigns_received)s, %(total_campaigns_opened)s, %(total_campaigns_clicked)s,
    %(first_campaign_date)s, %(latest_campaign_date)s,
    %(email_open_rate)s, %(email_click_rate)s, %(days_since_last_campaign)s,
    %(avg_campaigns_before_purchase)s, %(campaigns_before_first_purchase)s,
    %(total_purchases_with_campaign)s, %(total_purchases_without_campaign)s,
    %(campaign_influence_rate)s, %(attribution_segment)s
)
ON CONFLICT (customer_id, run_id) DO UPDATE SET
    total_orders                    = EXCLUDED.total_orders,
    total_revenue                   = EXCLUDED.total_revenue,
    attribution_segment             = EXCLUDED.attribution_segment,
    campaign_influence_rate         = EXCLUDED.campaign_influence_rate,
    avg_campaigns_before_purchase   = EXCLUDED.avg_campaigns_before_purchase,
    updated_at                      = NOW()
"""


def write(conn, records: list[dict]) -> None:
    from psycopg2.extras import execute_batch
    with conn:
        with conn.cursor() as cur:
            execute_batch(cur, INSERT_SQL, records, page_size=500)
    print(f"✓ Upserted {len(records):,} rows into customer_derivatives")


# ══════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════
def report_memory(records: list[dict]) -> None:
    from collections import Counter
    segs = Counter(r["attribution_segment"] for r in records)
    types = Counter(r["customer_type"] for r in records)
    total = len(records)
    influenced = [r for r in records if r["campaign_influence_rate"] and r["campaign_influence_rate"] > 0]

    print(f"\n{'═' * 62}")
    print(f"  customer_derivatives — {total:,} customers computed")
    print(f"{'═' * 62}")
    print(f"  Customer types")
    for t, n in types.most_common():
        print(f"    {t:<26} {n:>5}  ({n * 100 / total:.1f}%)")
    print(f"\n  Attribution segments (campaign-era orders only)")
    order = ["campaign_1_2", "campaign_3_5", "campaign_gt5",
             "no_campaign_influence", "never_purchased"]
    for s in order:
        n = segs.get(s, 0)
        print(f"    {s:<26} {n:>5}  ({n * 100 / total:.1f}%)")
    if influenced:
        rates = [r["campaign_influence_rate"] for r in influenced]
        avg_inf = round(sum(rates) / len(rates), 3)
        avgs = [r["avg_campaigns_before_purchase"] for r in influenced
                if r["avg_campaigns_before_purchase"] is not None]
        avg_touches = round(sum(avgs) / len(avgs), 1) if avgs else None
        print(f"\n  Among campaign-influenced buyers ({len(influenced):,})")
        print(f"    Avg influence rate          : {avg_inf:.1%}")
        print(f"    Avg campaigns before purchase: {avg_touches}")
    print(f"{'═' * 62}\n")


def report_db(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT attribution_segment, COUNT(*), ROUND(AVG(total_revenue),2),
                   ROUND(AVG(total_orders),1), ROUND(AVG(avg_campaigns_before_purchase),1)
            FROM customer_derivatives d
            JOIN (SELECT customer_id, MAX(run_id) rid FROM customer_derivatives GROUP BY 1) latest
              ON latest.customer_id = d.customer_id AND latest.rid = d.run_id
            GROUP BY attribution_segment
            ORDER BY AVG(total_revenue) DESC NULLS LAST
        """)
        rows = cur.fetchall()
    print(f"\n  {'segment':<26} {'cust':>5}  {'avg rev':>8}  {'avg ord':>7}  {'avg touches':>11}")
    print(f"  {'─' * 62}")
    for seg, n, rev, orders, touches in rows:
        print(f"  {seg:<26} {n:>5}  ${rev or 0:>7.2f}  {orders or 0:>7.1f}  {touches or 0:>11.1f}")


# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute customer_derivatives from customer_events.")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and report, do not write to DB")
    ap.add_argument("--host", help="DB host (default: localhost)")
    ap.add_argument("--port", help="DB port (default: 5433)")
    args = ap.parse_args()

    conn = connect(args)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM customer_events")
        if cur.fetchone()[0] == 0:
            sys.exit("✗ customer_events is empty. "
                     "Run transformation/build_customer_events.py first.")

    run_id = "dry-run" if args.dry_run else new_run_id(conn)

    as_of, events = load_events(conn)
    # Filter to Shopify customers only — Klaviyo-only profiles have no FK in customers
    with conn.cursor() as cur:
        cur.execute("SELECT customer_id FROM customers")
        shopify_ids = {r[0] for r in cur.fetchall()}
    events = {k: v for k, v in events.items() if k in shopify_ids}
    print(f"  Computing derivatives for {len(events):,} Shopify customers …")
    records = [compute_one(cid, evts, as_of, run_id)
               for cid, evts in events.items()]
    report_memory(records)

    if args.dry_run:
        print("Dry run — no rows written.")
        conn.close()
        return

    try:
        write(conn, records)
    except Exception as exc:
        finish_run(conn, run_id, 0, str(exc))
        raise
    finish_run(conn, run_id, len(records))
    report_db(conn)
    conn.close()


if __name__ == "__main__":
    main()
