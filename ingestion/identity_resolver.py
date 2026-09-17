#!/usr/bin/env python3
"""
ingestion/identity_resolver.py — build customer_identity_map.

Joins Shopify customers to Klaviyo profiles using email as the canonical key.
Handles the real-world messiness baked into the synthetic data:

  1. Exact match       — email strings are identical
  2. Case-variant      — Klaviyo stored mixed-case (e.g. John.Smith12@Example.Com)
                         resolved by LOWER() on both sides
  3. Klaviyo-only      — a Klaviyo profile exists with no Shopify counterpart
                         (subscriber who never checked out)
  4. Shopify-only      — a Shopify customer with no Klaviyo profile
                         (non-subscriber; no email consent)

All four outcomes land in customer_identity_map:
  - Rows 1 and 2 have both IDs populated (shopify + klaviyo)
  - Row 3 has shopify_customer_id = NULL
  - Row 4 has klaviyo_profile_id = NULL

The resolver is idempotent: run it multiple times and it produces
the same result. On conflict it updates identity_confidence and
last_verified_at but does not duplicate rows.

Usage
    python identity_resolver.py              # run
    python identity_resolver.py --dry-run    # report, no DB writes
    python identity_resolver.py --port 5433  # override DB port
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
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
        conn = psycopg2.connect(**params)
    except Exception as exc:
        sys.exit(f"\n✗ Cannot connect to Postgres at {params['host']}:{params['port']}.\n  {exc}\n")
    print(f"✓ Connected to {params['dbname']} at {params['host']}:{params['port']}")
    return conn


# ══════════════════════════════════════════════════════════════════════════
# Resolution logic
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class IdentityRow:
    shopify_customer_id: str | None
    klaviyo_profile_id:  str | None
    email:               str           # always lower-cased canonical form
    confidence:          str           # exact | case_variant | klaviyo_only | shopify_only


def resolve(conn) -> list[IdentityRow]:
    """
    Pure-Python resolution — reads raw tables, returns IdentityRow list.
    Three passes:

      Pass 1: exact email match (shopify.email == klaviyo.email)
      Pass 2: case-variant match (LOWER on both sides) for any still unmatched
      Pass 3: remaining Klaviyo-only and Shopify-only rows
    """
    with conn.cursor() as cur:
        cur.execute("SELECT customer_id, email FROM customers")
        shopify_rows = cur.fetchall()
        cur.execute("SELECT klaviyo_profile_id, email FROM klaviyo_profiles")
        klaviyo_rows = cur.fetchall()

    # Build lookup dicts  {email → id}
    sh_exact:  dict[str, str] = {email: cid  for cid,  email in shopify_rows}
    kl_exact:  dict[str, str] = {email: pid  for pid,  email in klaviyo_rows}
    sh_lower:  dict[str, str] = {email.lower(): cid  for cid,  email in shopify_rows}
    kl_lower:  dict[str, str] = {email.lower(): pid  for pid,  email in klaviyo_rows}
    # canonical email (lower) → original Klaviyo email (for the canonical field)
    kl_canon:  dict[str, str] = {email.lower(): email.lower() for _, email in klaviyo_rows}
    sh_canon:  dict[str, str] = {email.lower(): email.lower() for _, email in shopify_rows}

    resolved:      set[str] = set()   # lower-cased emails already handled
    rows: list[IdentityRow] = []

    # Pass 1 — exact match
    for email, cid in sh_exact.items():
        if email in kl_exact:
            rows.append(IdentityRow(cid, kl_exact[email], email.lower(), "exact"))
            resolved.add(email.lower())

    # Pass 2 — case-variant (LOWER matches but exact didn't)
    for email_low, cid in sh_lower.items():
        if email_low in resolved:
            continue
        if email_low in kl_lower:
            canonical = kl_canon.get(email_low, email_low)
            rows.append(IdentityRow(cid, kl_lower[email_low], canonical, "case_variant"))
            resolved.add(email_low)

    # Pass 3a — Klaviyo-only (no Shopify customer)
    for email_low, pid in kl_lower.items():
        if email_low not in resolved:
            rows.append(IdentityRow(None, pid, email_low, "klaviyo_only"))
            resolved.add(email_low)

    # Pass 3b — Shopify-only (no Klaviyo profile)
    for email_low, cid in sh_lower.items():
        if email_low not in resolved:
            rows.append(IdentityRow(cid, None, email_low, "shopify_only"))
            resolved.add(email_low)

    return rows


# ══════════════════════════════════════════════════════════════════════════
# Upsert
# ══════════════════════════════════════════════════════════════════════════
UPSERT = """
INSERT INTO customer_identity_map
    (shopify_customer_id, klaviyo_profile_id, email, identity_confidence,
     first_resolved_at, last_verified_at)
VALUES (%s, %s, %s, %s, NOW(), NOW())
ON CONFLICT (email) DO UPDATE SET
    shopify_customer_id = EXCLUDED.shopify_customer_id,
    klaviyo_profile_id  = EXCLUDED.klaviyo_profile_id,
    identity_confidence = EXCLUDED.identity_confidence,
    last_verified_at    = NOW();
"""


def write(conn, rows: list[IdentityRow]) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.executemany(
                UPSERT,
                [(r.shopify_customer_id, r.klaviyo_profile_id, r.email, r.confidence)
                 for r in rows],
            )
    print(f"✓ Upserted {len(rows)} rows into customer_identity_map")


# ══════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════
def report(rows: list[IdentityRow], conn=None) -> None:
    from collections import Counter
    counts = Counter(r.confidence for r in rows)
    total = len(rows)
    print(f"\n{'═' * 60}")
    print(f"  Identity resolution summary — {total} unique email identities")
    print(f"{'═' * 60}")
    print(f"  {'exact match':<22} {counts['exact']:>5}  "
          f"({counts['exact'] * 100 / total:.1f}%)  both IDs populated")
    print(f"  {'case-variant match':<22} {counts['case_variant']:>5}  "
          f"({counts['case_variant'] * 100 / total:.1f}%)  resolved by LOWER()")
    print(f"  {'Klaviyo-only':<22} {counts['klaviyo_only']:>5}  "
          f"({counts['klaviyo_only'] * 100 / total:.1f}%)  no Shopify customer")
    print(f"  {'Shopify-only':<22} {counts['shopify_only']:>5}  "
          f"({counts['shopify_only'] * 100 / total:.1f}%)  no Klaviyo profile")
    linked = counts["exact"] + counts["case_variant"]
    print(f"\n  Linked (both sides)  : {linked} / {total}  "
          f"({linked * 100 / total:.1f}%)")
    print(f"  Attribution-eligible : {linked}  "
          f"(customers whose campaign + purchase events can be overlaid)")

    if conn:
        # Verify what landed in the DB
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    SUM(CASE WHEN shopify_customer_id IS NOT NULL
                              AND klaviyo_profile_id  IS NOT NULL THEN 1 END) AS both,
                    SUM(CASE WHEN shopify_customer_id IS NULL     THEN 1 END) AS kl_only,
                    SUM(CASE WHEN klaviyo_profile_id  IS NULL     THEN 1 END) AS sh_only,
                    COUNT(*) AS total
                FROM customer_identity_map
            """)
            both, kl_only, sh_only, db_total = cur.fetchone()
        print(f"\n  DB verification ─ customer_identity_map has {db_total} rows")
        print(f"    both IDs present : {both}")
        print(f"    Klaviyo-only     : {kl_only}")
        print(f"    Shopify-only     : {sh_only}")
    print(f"{'═' * 60}\n")


# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description="Build customer_identity_map from Shopify + Klaviyo emails.")
    ap.add_argument("--dry-run", action="store_true", help="resolve and report, do not write to DB")
    ap.add_argument("--host", help="DB host (default: localhost)")
    ap.add_argument("--port", help="DB port (default: 5433)")
    args = ap.parse_args()

    conn = connect(args)
    print("Resolving identities …")
    rows = resolve(conn)
    report(rows, conn=None if args.dry_run else None)   # pre-write report always

    if args.dry_run:
        print("Dry run — no rows written.")
        report(rows)
        conn.close()
        return

    write(conn, rows)
    report(rows, conn=conn)   # post-write report with DB verification
    conn.close()


if __name__ == "__main__":
    main()
