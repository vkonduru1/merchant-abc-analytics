#!/usr/bin/env python3
"""
transformation/campaign_attribution.py — build campaign_attribution.

Writes one row per (customer, order) with the campaign context that
existed in the 30 days before that order was placed.

This is the per-order evidence table that backs:
  - The recommendation agent's decision per customer
  - The analytics dashboard's campaign performance view
  - Backtesting (compare predicted vs actual outcome)

Why a separate table from customer_derivatives?
  customer_derivatives has one row per customer summarising their
  entire history. campaign_attribution has one row per ORDER, so you
  can see which specific campaign preceded each specific purchase —
  essential for drill-down and for training a classifier later.

Attribution logic
──────────────────
  For each ORDER event in customer_events (after CAMPAIGN_START):
    1. Look back ATTRIBUTION_WINDOW_DAYS (30) before the order date.
    2. Count all CAMPAIGN_SENT events in that window for the same customer.
    3. Identify the most recent campaign sent before the order.
    4. Classify the order into an attribution_bucket.

Attribution buckets
────────────────────
  no_influence   : 0 campaigns in window
  campaign_1_2   : 1–2 campaigns in window
  campaign_3_5   : 3–5 campaigns in window
  campaign_gt5   : 6+ campaigns in window

order_number_in_lifecycle
──────────────────────────
  1 = customer's first-ever order, 2 = second, etc. Computed from
  customer_events order chronology, not from orders.orders_count
  (which may be stale for a recently joined customer).

Usage
    python transformation/campaign_attribution.py              # run
    python transformation/campaign_attribution.py --dry-run    # report only
    python transformation/campaign_attribution.py --port 5433  # override port
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "data_generator"))

try:
    from dotenv import load_dotenv
    load_dotenv(SCRIPT_DIR.parent / ".env")
except ImportError:
    pass

ATTRIBUTION_WINDOW_DAYS = 30
CAMPAIGN_START = datetime(2023, 7, 1)


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
        sys.exit(f"\n✗ Cannot connect at {params['host']}:{params['port']}.\n  {exc}\n")
    print(f"✓ Connected to {params['dbname']} at {params['host']}:{params['port']}")
    return conn


def get_run_id(conn) -> str:
    """Re-use the latest completed derivatives run, or create a new one."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT run_id FROM pipeline_runs
            WHERE triggered_by IN ('compute_derivatives', 'campaign_attribution')
              AND status = 'completed'
            ORDER BY started_at DESC LIMIT 1
        """)
        row = cur.fetchone()
    if row:
        return row[0]
    # Create a fresh run if nothing exists yet
    run_id = f"ca-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO pipeline_runs (run_id, run_type, status, started_at, triggered_by)
            VALUES (%s, 'full', 'running', NOW(), 'campaign_attribution')
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
# Load events
# ══════════════════════════════════════════════════════════════════════════
def load_events(conn) -> tuple[dict, dict]:
    """
    Returns:
      orders_by_customer   : {customer_id: [(order_date, order_id, revenue), ...]}
      campaigns_by_customer: {customer_id: [(send_date, campaign_id), ...]}
    Both sorted chronologically.
    """
    print("  Loading events from customer_events …")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT customer_id, event_date, event_ref_id, revenue_amount
            FROM customer_events
            WHERE event_type = 'ORDER'
            ORDER BY customer_id, event_date
        """)
        order_rows = cur.fetchall()

        cur.execute("""
            SELECT customer_id, event_date, campaign_id
            FROM customer_events
            WHERE event_type = 'CAMPAIGN_SENT'
              AND campaign_id IS NOT NULL
            ORDER BY customer_id, event_date
        """)
        camp_rows = cur.fetchall()

    orders: dict[str, list] = defaultdict(list)
    for cid, dt, ref_id, rev in order_rows:
        orders[cid].append((dt, ref_id, float(rev) if rev else 0.0))

    campaigns: dict[str, list] = defaultdict(list)
    for cid, dt, camp_id in camp_rows:
        campaigns[cid].append((dt, camp_id))

    print(f"    {len(order_rows):,} orders across {len(orders):,} customers")
    print(f"    {len(camp_rows):,} campaign sends across {len(campaigns):,} customers")
    return dict(orders), dict(campaigns)


def load_order_meta(conn) -> dict[str, dict]:
    """campaign_id → {name, type, discount_pct, has_discount_code}"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT k.campaign_id, k.campaign_name, ct.type_name,
                   k.discount_pct, k.has_discount_code
            FROM campaigns k
            LEFT JOIN campaign_type ct USING (campaign_type_id)
        """)
        return {
            row[0]: {"name": row[1], "type": row[2],
                     "discount_pct": row[3], "has_discount": row[4]}
            for row in cur.fetchall()
        }


# ══════════════════════════════════════════════════════════════════════════
# Compute per-order attribution
# ══════════════════════════════════════════════════════════════════════════
def bucket(n: int) -> str:
    if n == 0:   return "no_influence"
    if n <= 2:   return "campaign_1_2"
    if n <= 5:   return "campaign_3_5"
    return "campaign_gt5"


def compute(orders: dict, campaigns: dict,
            camp_meta: dict, run_id: str) -> list[dict]:
    records: list[dict] = []
    campaign_window = timedelta(days=ATTRIBUTION_WINDOW_DAYS)

    for customer_id, order_list in orders.items():
        camp_list = campaigns.get(customer_id, [])  # [(dt, camp_id), ...]
        camp_dates = [c[0] for c in camp_list]      # just the timestamps

        for order_num, (order_date, order_id, revenue) in enumerate(order_list, start=1):
            # Only attribute orders that fall after the campaign programme started
            if order_date < CAMPAIGN_START:
                # Still record the row, just with zeroed attribution columns
                records.append({
                    "customer_id":              customer_id,
                    "run_id":                   run_id,
                    "order_id":                 order_id,
                    "order_date":               order_date,
                    "order_revenue":            revenue,
                    "order_number_in_lifecycle": order_num,
                    "campaigns_in_30d_window":  0,
                    "campaigns_opened_30d":     0,
                    "campaigns_clicked_30d":    0,
                    "most_recent_campaign_id":  None,
                    "days_since_last_campaign": None,
                    "campaign_influenced":       False,
                    "attribution_bucket":        "pre_campaign_era",
                })
                continue

            window_start = order_date - campaign_window
            in_window = [(dt, cid) for dt, cid in camp_list
                         if window_start <= dt < order_date]
            n_in_window = len(in_window)

            most_recent_camp = in_window[-1] if in_window else None
            days_since = (
                (order_date - most_recent_camp[0]).days
                if most_recent_camp else None
            )

            records.append({
                "customer_id":               customer_id,
                "run_id":                    run_id,
                "order_id":                  order_id,
                "order_date":                order_date,
                "order_revenue":             revenue,
                "order_number_in_lifecycle": order_num,
                "campaigns_in_30d_window":   n_in_window,
                "campaigns_opened_30d":      0,   # placeholder — joins email_events in Phase 2
                "campaigns_clicked_30d":     0,
                "most_recent_campaign_id":   most_recent_camp[1] if most_recent_camp else None,
                "days_since_last_campaign":  days_since,
                "campaign_influenced":       n_in_window >= 1,
                "attribution_bucket":        bucket(n_in_window),
            })

    records.sort(key=lambda r: (r["customer_id"], r["order_date"]))
    return records


# ══════════════════════════════════════════════════════════════════════════
# Write
# ══════════════════════════════════════════════════════════════════════════
INSERT_SQL = """
INSERT INTO campaign_attribution (
    customer_id, run_id, order_id, order_date, order_revenue,
    order_number_in_lifecycle,
    campaigns_in_30d_window, campaigns_opened_30d, campaigns_clicked_30d,
    most_recent_campaign_id, days_since_last_campaign,
    campaign_influenced, attribution_bucket
) VALUES (
    %(customer_id)s, %(run_id)s, %(order_id)s, %(order_date)s, %(order_revenue)s,
    %(order_number_in_lifecycle)s,
    %(campaigns_in_30d_window)s, %(campaigns_opened_30d)s, %(campaigns_clicked_30d)s,
    %(most_recent_campaign_id)s, %(days_since_last_campaign)s,
    %(campaign_influenced)s, %(attribution_bucket)s
)
ON CONFLICT (customer_id, order_id, run_id) DO UPDATE SET
    campaigns_in_30d_window  = EXCLUDED.campaigns_in_30d_window,
    campaign_influenced      = EXCLUDED.campaign_influenced,
    attribution_bucket       = EXCLUDED.attribution_bucket
"""


def write(conn, records: list[dict]) -> None:
    from psycopg2.extras import execute_batch
    with conn:
        with conn.cursor() as cur:
            execute_batch(cur, INSERT_SQL, records, page_size=500)
    print(f"✓ Upserted {len(records):,} rows into campaign_attribution")


# ══════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════
def report_memory(records: list[dict]) -> None:
    from collections import Counter
    buckets = Counter(r["attribution_bucket"] for r in records)
    total = len(records)
    era = [r for r in records if r["attribution_bucket"] != "pre_campaign_era"]
    influenced = [r for r in era if r["campaign_influenced"]]
    rev_influenced = sum(r["order_revenue"] for r in influenced)
    rev_total = sum(r["order_revenue"] for r in era) if era else 0

    print(f"\n{'═' * 62}")
    print(f"  campaign_attribution — {total:,} order rows")
    print(f"{'═' * 62}")
    order = ["campaign_1_2", "campaign_3_5", "campaign_gt5",
             "no_influence", "pre_campaign_era"]
    for b in order:
        n = buckets.get(b, 0)
        pct = n * 100 / total if total else 0
        print(f"  {b:<24} {n:>5}  ({pct:.1f}%)")
    if era:
        print(f"\n  Campaign-era orders:        {len(era):,}")
        print(f"  Campaign-influenced orders: {len(influenced):,}  "
              f"({len(influenced) * 100 / len(era):.1f}%)")
        if rev_total:
            print(f"  Influenced revenue share:   "
                  f"${rev_influenced:,.2f} / ${rev_total:,.2f}  "
                  f"({rev_influenced * 100 / rev_total:.1f}%)")
    print(f"{'═' * 62}\n")


def report_db(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                attribution_bucket,
                COUNT(*)                        AS orders,
                COUNT(DISTINCT customer_id)     AS customers,
                ROUND(SUM(order_revenue), 2)    AS total_revenue,
                ROUND(AVG(days_since_last_campaign), 1) AS avg_days_since_camp
            FROM campaign_attribution
            WHERE attribution_bucket != 'pre_campaign_era'
            GROUP BY attribution_bucket
            ORDER BY total_revenue DESC NULLS LAST
        """)
        rows = cur.fetchall()
    print(f"\n  {'bucket':<24} {'orders':>7} {'customers':>10} "
          f"{'revenue':>10} {'days lag':>9}")
    print(f"  {'─' * 65}")
    for b, orders, custs, rev, lag in rows:
        print(f"  {b:<24} {orders:>7,} {custs:>10,} "
              f"${rev or 0:>9,.2f} {lag or 0:>9.1f}")


# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build campaign_attribution from customer_events.")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and report, do not write to DB")
    ap.add_argument("--host", help="DB host (default: localhost)")
    ap.add_argument("--port", help="DB port (default: 5433)")
    args = ap.parse_args()

    conn = connect(args)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM customer_events WHERE event_type='ORDER'")
        if cur.fetchone()[0] == 0:
            sys.exit("✗ No ORDER events in customer_events. "
                     "Run transformation/build_customer_events.py first.")

    run_id = "dry-run" if args.dry_run else get_run_id(conn)
    print(f"  run_id: {run_id}")

    orders, campaigns = load_events(conn)
    camp_meta = load_order_meta(conn)

    print("  Computing per-order attribution …")
    records = compute(orders, campaigns, camp_meta, run_id)
    report_memory(records)

    if args.dry_run:
        print("Dry run — no rows written.")
        conn.close()
        return

    try:
        write(conn, records)
    except Exception as exc:
        if not run_id.startswith("cd-"):   # only finish runs we created
            finish_run(conn, run_id, 0, str(exc))
        raise

    report_db(conn)
    conn.close()


if __name__ == "__main__":
    main()
