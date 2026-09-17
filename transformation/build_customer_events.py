#!/usr/bin/env python3
"""
transformation/build_customer_events.py — build customer_events.

Merges raw Shopify and Klaviyo tables into a single chronological event
timeline per customer — the architectural centrepiece of the AIIR framework.

Every row in customer_events is one of:
    ORDER           — a Shopify purchase (revenue_amount populated)
    CAMPAIGN_SENT   — a Klaviyo campaign delivered to this customer
    EMAIL_OPENED    — customer opened a campaign email
    EMAIL_CLICKED   — customer clicked a link in a campaign email
    EMAIL_BOUNCED   — delivery failed
    UNSUBSCRIBED    — customer unsubscribed

The table is append-only per pipeline_run. Each run inserts a fresh
batch tagged with run_id; it does NOT delete prior runs' rows. This
means you can query the latest run or compare across runs.

LEFT JOIN discipline
─────────────────────
  orders → order_lines : LEFT JOIN. An order without lines is still an
      order. Dropping it (INNER JOIN) would undercount revenue and make
      the customer's purchase timeline wrong.

Identity normalisation
──────────────────────
  Shopify uses numeric customer_id as the canonical key.
  Klaviyo uses klaviyo_profile_id. This script joins via
  customer_identity_map (email as the bridge) to normalise all
  events to shopify_customer_id. Klaviyo-only profiles (no Shopify
  customer) are included with their klaviyo_profile_id as the key
  — they appear in customer_events but cannot be attributed to orders.

Usage
    python transformation/build_customer_events.py              # run
    python transformation/build_customer_events.py --dry-run    # report only
    python transformation/build_customer_events.py --port 5433  # override port
    python transformation/build_customer_events.py --full       # re-run from scratch
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "data_generator"))

try:
    from dotenv import load_dotenv
    load_dotenv(SCRIPT_DIR.parent / ".env")
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════
# DB connection
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
        import psycopg2
        conn = psycopg2.connect(**params)
    except Exception as exc:
        sys.exit(f"\n✗ Cannot connect to Postgres at {params['host']}:{params['port']}.\n  {exc}\n")
    print(f"✓ Connected to {params['dbname']} at {params['host']}:{params['port']}")
    return conn


# ══════════════════════════════════════════════════════════════════════════
# Pipeline run management
# ══════════════════════════════════════════════════════════════════════════
def start_run(conn) -> str:
    run_id = f"ce-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO pipeline_runs (run_id, run_type, status, started_at, triggered_by)
            VALUES (%s, 'full', 'running', NOW(), 'build_customer_events')
        """, (run_id,))
    conn.commit()
    return run_id


def finish_run(conn, run_id: str, n: int, error: str | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE pipeline_runs
            SET status = %s, completed_at = NOW(), customers_processed = %s, error_message = %s
            WHERE run_id = %s
        """, ("failed" if error else "completed", n, error, run_id))
    conn.commit()


# ══════════════════════════════════════════════════════════════════════════
# Source queries
# ══════════════════════════════════════════════════════════════════════════
PURCHASE_SQL = """
-- One row per order per customer.
-- LEFT JOIN to order_lines: we want the order even when line items didn't sync.
-- We take the first (lowest-price) line's product_type as the order's type —
-- not perfect but consistent and cheap.
SELECT
    o.customer_id,
    o.order_created_at                                      AS event_date,
    'ORDER'                                                 AS event_type,
    'shopify'                                               AS event_source,
    o.order_id                                              AS event_ref_id,
    o.total_price_usd                                       AS revenue_amount,
    MIN(ol.product_type)                                    AS product_type,
    o.discount_code,
    (o.discount_code IS NOT NULL OR o.total_discounts > 0)  AS has_discount,
    NULL::VARCHAR                                           AS campaign_id,
    NULL::VARCHAR                                           AS campaign_name,
    NULL::VARCHAR                                           AS campaign_type,
    FALSE                                                   AS has_campaign_discount
FROM orders o
LEFT JOIN order_lines ol ON ol.order_id = o.order_id
WHERE o.financial_status IN ('paid', 'pending')   -- exclude refunded/voided
GROUP BY
    o.customer_id, o.order_created_at, o.order_id,
    o.total_price_usd, o.discount_code, o.total_discounts
"""

EMAIL_EVENTS_SQL = """
-- Email events for customers we can link to Shopify via the identity map.
-- LOWER() on both sides handles the case-variant emails.
-- Klaviyo-only profiles (no Shopify customer) use their profile id directly.
SELECT
    COALESCE(cim.shopify_customer_id, kp.klaviyo_profile_id)  AS customer_id,
    ee.event_timestamp                                          AS event_date,
    CASE ee.event_type
        WHEN 'sent'          THEN 'CAMPAIGN_SENT'
        WHEN 'delivered'     THEN 'CAMPAIGN_SENT'   -- delivered supersedes sent; deduplicated below
        WHEN 'opened'        THEN 'EMAIL_OPENED'
        WHEN 'clicked'       THEN 'EMAIL_CLICKED'
        WHEN 'bounced'       THEN 'EMAIL_BOUNCED'
        WHEN 'unsubscribed'  THEN 'UNSUBSCRIBED'
        ELSE UPPER(ee.event_type)
    END                                                         AS event_type,
    'klaviyo'                                                   AS event_source,
    ee.event_id::VARCHAR                                        AS event_ref_id,
    NULL::DECIMAL                                               AS revenue_amount,
    NULL::VARCHAR                                               AS product_type,
    NULL::VARCHAR                                               AS discount_code,
    FALSE                                                       AS has_discount,
    k.campaign_id,
    k.campaign_name,
    ct.type_name                                                AS campaign_type,
    COALESCE(k.has_discount_code, FALSE)                        AS has_campaign_discount
FROM email_events ee
JOIN klaviyo_profiles kp ON kp.klaviyo_profile_id = ee.klaviyo_profile_id
JOIN campaigns k         ON k.campaign_id = ee.campaign_id
LEFT JOIN campaign_type ct ON ct.campaign_type_id = k.campaign_type_id
LEFT JOIN customer_identity_map cim ON LOWER(cim.email) = LOWER(kp.email)
WHERE ee.event_type != 'sent'   -- keep delivered (which we relabel CAMPAIGN_SENT), drop raw sent
"""


# ══════════════════════════════════════════════════════════════════════════
# Build
# ══════════════════════════════════════════════════════════════════════════
def build(conn, run_id: str, dry_run: bool) -> int:
    """Fetch, merge, deduplicate, insert. Returns row count."""
    from psycopg2.extras import execute_values

    print("  Fetching purchase events …")
    with conn.cursor() as cur:
        cur.execute(PURCHASE_SQL)
        purchases = cur.fetchall()
    print(f"    {len(purchases):,} order events")

    print("  Fetching email events …")
    with conn.cursor() as cur:
        cur.execute(EMAIL_EVENTS_SQL)
        emails = cur.fetchall()
    print(f"    {len(emails):,} raw email events")

    # Deduplicate CAMPAIGN_SENT: per (customer_id, campaign_id) keep only one
    # row — delivered beats sent. Since the query already excludes raw 'sent'
    # (keeping 'delivered' as CAMPAIGN_SENT), duplicates can still arise when a
    # customer has both a 'delivered' row and a relabelled 'sent' row for the
    # same campaign. Deduplicate by (customer_id, campaign_id) keeping earliest.
    seen_delivered: set[tuple] = set()
    deduped: list = []
    for row in emails:
        customer_id, event_date, event_type, source, ref_id, rev, ptype, disc, has_disc, \
            campaign_id, campaign_name, campaign_type, has_camp_disc = row
        if event_type == "CAMPAIGN_SENT" and campaign_id:
            key = (customer_id, campaign_id)
            if key in seen_delivered:
                continue
            seen_delivered.add(key)
        deduped.append(row)
    print(f"    {len(deduped):,} after deduplication")

    # Merge and sort chronologically
    all_events = purchases + deduped
    all_events.sort(key=lambda r: r[1])   # sort by event_date
    print(f"  Total events to insert: {len(all_events):,}")

    if dry_run:
        return len(all_events)

    # Bulk insert in pages of 5,000
    INSERT_SQL = """
        INSERT INTO customer_events (
            customer_id, event_date, event_type, event_source, event_ref_id,
            revenue_amount, product_type, discount_code, has_discount,
            campaign_id, campaign_name, campaign_type, has_campaign_discount
        ) VALUES %s
    """
    page = 5_000
    inserted = 0
    with conn:
        with conn.cursor() as cur:
            for i in range(0, len(all_events), page):
                batch = all_events[i:i + page]
                execute_values(cur, INSERT_SQL, batch, page_size=page)
                inserted += len(batch)
    return inserted


# ══════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════
def report(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT event_type, COUNT(*) n
            FROM customer_events
            GROUP BY event_type
            ORDER BY n DESC
        """)
        by_type = cur.fetchall()

        cur.execute("""
            SELECT COUNT(DISTINCT customer_id) FROM customer_events
        """)
        n_customers = cur.fetchone()[0]

        cur.execute("""
            SELECT
                MIN(event_date) AS earliest,
                MAX(event_date) AS latest,
                COUNT(*) AS total
            FROM customer_events
        """)
        earliest, latest, total = cur.fetchone()

        cur.execute("""
            SELECT COUNT(DISTINCT customer_id)
            FROM customer_events
            WHERE event_type = 'ORDER'
        """)
        buyers = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(DISTINCT customer_id)
            FROM customer_events
            WHERE event_type = 'CAMPAIGN_SENT'
        """)
        recipients = cur.fetchone()[0]

        cur.execute("""
            -- Customers with at least one ORDER and at least one CAMPAIGN_SENT
            -- = the population the attribution model can be applied to
            SELECT COUNT(DISTINCT customer_id) FROM (
                SELECT customer_id FROM customer_events WHERE event_type = 'ORDER'
                INTERSECT
                SELECT customer_id FROM customer_events WHERE event_type = 'CAMPAIGN_SENT'
            ) x
        """)
        attributable = cur.fetchone()[0]

    print(f"\n{'═' * 60}")
    print(f"  customer_events — {total:,} rows across {n_customers:,} customers")
    print(f"  {earliest.date()} → {latest.date()}")
    print(f"{'═' * 60}")
    for etype, n in by_type:
        print(f"  {etype:<22} {n:>8,}")
    print(f"{'─' * 60}")
    print(f"  Customers with orders          : {buyers:,}")
    print(f"  Customers who received a campaign : {recipients:,}")
    print(f"  Both (attribution-eligible)    : {attributable:,}")
    print(f"{'═' * 60}\n")


# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description="Build customer_events from raw Shopify + Klaviyo tables.")
    ap.add_argument("--dry-run", action="store_true", help="count rows but do not write to DB")
    ap.add_argument("--full",    action="store_true", help="delete existing customer_events rows first")
    ap.add_argument("--host",    help="DB host (default: localhost)")
    ap.add_argument("--port",    help="DB port (default: 5433)")
    args = ap.parse_args()

    conn = connect(args)

    # Guard: check identity map is populated
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM customer_identity_map")
        n_map = cur.fetchone()[0]
    if n_map == 0:
        sys.exit("✗ customer_identity_map is empty. Run ingestion/identity_resolver.py first.")

    if args.full and not args.dry_run:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM customer_events")
        conn.commit()
        print("✓ Cleared existing customer_events rows")

    run_id = None if args.dry_run else start_run(conn)

    print("Building customer_events …")
    try:
        n = build(conn, run_id or "dry-run", dry_run=args.dry_run)
    except Exception as exc:
        if run_id:
            finish_run(conn, run_id, 0, str(exc))
        raise

    if args.dry_run:
        print(f"\nDry run — {n:,} events would be inserted. DB untouched.")
        conn.close()
        return

    finish_run(conn, run_id, n)
    print(f"✓ Inserted {n:,} events (run_id: {run_id})")
    report(conn)
    conn.close()


if __name__ == "__main__":
    main()
