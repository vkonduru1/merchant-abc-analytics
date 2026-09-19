-- ============================================================
-- merchant_abc  E-Commerce Analytics Platform
-- Schema version: 1.0.0
-- Vertical: Specialty Coffee & Beverages
-- Sources: Shopify (ERP) + Klaviyo (CRM/Email)
-- Architecture: AIIR — Analysis → Insights → Interpretations → Recommendations
-- ============================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- LAYER 0 — Lookup / Configuration Tables
-- ============================================================

CREATE TABLE product_category (
    category_id             SERIAL PRIMARY KEY,
    category_name           VARCHAR(100)    NOT NULL,
    description             TEXT,
    avg_repurchase_days     INTEGER,        -- expected days between repeat purchases
    created_at              TIMESTAMP       DEFAULT NOW()
);

COMMENT ON TABLE product_category IS 'Product category lookup. avg_repurchase_days drives synthetic data generation cadence.';

-- Seed: specialty coffee verticals
INSERT INTO product_category (category_name, description, avg_repurchase_days) VALUES
    ('Whole Bean Coffee',   'Single-origin and blend whole bean bags (250g, 500g, 1kg)',   28),
    ('Ground Coffee',       'Pre-ground bags for drip, french press, moka pot',            28),
    ('Espresso Pods',       'Nespresso-compatible and refillable pods',                    21),
    ('Loose Leaf Tea',      'Herbal, green, black and oolong loose leaf tins',             35),
    ('Cold Brew Kits',      'Cold brew concentrate pouches and brewing kits',              45),
    ('Accessories',         'Grinders, pour-over kits, scales, storage canisters',       180);

CREATE TABLE campaign_type (
    campaign_type_id        SERIAL PRIMARY KEY,
    type_name               VARCHAR(100)    NOT NULL,   -- promotional | newsletter | reactivation | welcome | seasonal
    description             TEXT,
    has_discount            BOOLEAN         DEFAULT FALSE,
    created_at              TIMESTAMP       DEFAULT NOW()
);

INSERT INTO campaign_type (type_name, description, has_discount) VALUES
    ('promotional',     'Discount-led campaign driving immediate purchase',         TRUE),
    ('newsletter',      'Product education, brew guides, origin stories',           FALSE),
    ('reactivation',    'Win-back campaign for lapsed customers (60+ days silent)', TRUE),
    ('welcome',         'Onboarding sequence for new subscribers',                  TRUE),
    ('seasonal',        'Holiday, New Year, summer cold brew themed campaigns',     TRUE),
    ('loyalty',         'Reward and milestone campaigns for repeat buyers',         TRUE);

CREATE TABLE pipeline_runs (
    run_id                  VARCHAR(50)     PRIMARY KEY,
    run_type                VARCHAR(50)     NOT NULL,       -- full | incremental | backtest
    status                  VARCHAR(50)     NOT NULL,       -- running | completed | failed
    started_at              TIMESTAMP       DEFAULT NOW(),
    completed_at            TIMESTAMP,
    customers_processed     INTEGER,
    error_message           TEXT,
    triggered_by            VARCHAR(100)    DEFAULT 'system'
);

-- ============================================================
-- LAYER 1A — Raw Shopify Tables  (Analysis — raw ingestion)
-- ============================================================

CREATE TABLE customers (
    customer_id             VARCHAR(50)     PRIMARY KEY,    -- Shopify numeric ID as string
    email                   VARCHAR(255)    UNIQUE NOT NULL,
    first_name              VARCHAR(100),
    last_name               VARCHAR(100),
    phone                   VARCHAR(30),
    city                    VARCHAR(100),
    state_province          VARCHAR(100),
    country_code            CHAR(2),
    zip                     VARCHAR(20),
    accepts_marketing       BOOLEAN         DEFAULT FALSE,
    customer_created_at     TIMESTAMP       NOT NULL,
    orders_count            INTEGER         DEFAULT 0,
    total_spent_usd         DECIMAL(12,2)   DEFAULT 0.00,
    tags                    TEXT,           -- comma-separated Shopify tags
    verified_email          BOOLEAN         DEFAULT TRUE,
    bq_loaded_date          DATE            DEFAULT CURRENT_DATE,
    created_at              TIMESTAMP       DEFAULT NOW(),
    updated_at              TIMESTAMP       DEFAULT NOW()
);

COMMENT ON TABLE customers IS 'Raw Shopify customer master. Source: Shopify Customers API.';
COMMENT ON COLUMN customers.customer_id IS 'Shopify numeric customer ID — primary join key to identity map.';
COMMENT ON COLUMN customers.accepts_marketing IS 'Email marketing consent — required for Klaviyo campaign eligibility.';

CREATE TABLE products (
    product_id              VARCHAR(50)     PRIMARY KEY,
    title                   VARCHAR(255)    NOT NULL,
    product_type            VARCHAR(100),
    vendor                  VARCHAR(100),
    category_id             INTEGER         REFERENCES product_category(category_id),
    tags                    TEXT,
    price_min               DECIMAL(10,2),
    price_max               DECIMAL(10,2),
    published_at            TIMESTAMP,
    product_created_at      TIMESTAMP       NOT NULL,
    product_updated_at      TIMESTAMP,
    bq_loaded_date          DATE            DEFAULT CURRENT_DATE,
    created_at              TIMESTAMP       DEFAULT NOW(),
    updated_at              TIMESTAMP       DEFAULT NOW()
);

CREATE TABLE product_variants (
    variant_id              VARCHAR(50)     PRIMARY KEY,
    product_id              VARCHAR(50)     NOT NULL REFERENCES products(product_id),
    title                   VARCHAR(255),               -- e.g. '250g / Light Roast'
    sku                     VARCHAR(100)    UNIQUE NOT NULL,
    price                   DECIMAL(10,2)   NOT NULL,
    compare_at_price        DECIMAL(10,2),              -- original price before discount
    inventory_quantity      INTEGER         DEFAULT 0,
    variant_created_at      TIMESTAMP,
    variant_updated_at      TIMESTAMP,
    created_at              TIMESTAMP       DEFAULT NOW(),
    updated_at              TIMESTAMP       DEFAULT NOW()
);

COMMENT ON COLUMN product_variants.sku IS 'Used as secondary join key with variant_id — dual-key join pattern prevents duplicate matches.';

CREATE TABLE orders (
    order_id                VARCHAR(50)     PRIMARY KEY,
    customer_id             VARCHAR(50)     REFERENCES customers(customer_id),
    order_number            VARCHAR(50)     UNIQUE NOT NULL,
    financial_status        VARCHAR(50),                -- paid | refunded | pending | voided
    fulfillment_status      VARCHAR(50),                -- fulfilled | unfulfilled | partial
    order_created_at        TIMESTAMP       NOT NULL,
    order_updated_at        TIMESTAMP,
    subtotal_price          DECIMAL(12,2),
    total_discounts         DECIMAL(12,2)   DEFAULT 0.00,
    total_price_usd         DECIMAL(12,2)   NOT NULL,
    total_tax               DECIMAL(12,2)   DEFAULT 0.00,
    total_weight_grams      INTEGER         DEFAULT 0,
    discount_code           VARCHAR(100),
    source_name             VARCHAR(50),                -- web | mobile_app | pos | subscription
    shipping_city           VARCHAR(100),
    shipping_country_code   CHAR(2),
    shipping_zip            VARCHAR(20),
    order_tags              TEXT,
    bq_loaded_date          DATE            DEFAULT CURRENT_DATE,
    created_at              TIMESTAMP       DEFAULT NOW(),
    updated_at              TIMESTAMP       DEFAULT NOW()
);

COMMENT ON TABLE orders IS 'Raw Shopify order headers. LEFT JOIN to order_lines, shipping, tax — do not use INNER JOIN (drops orders with missing sub-records).';
COMMENT ON COLUMN orders.order_created_at IS 'The operative timestamp for all purchase-date derivative calculations.';

CREATE TABLE order_lines (
    order_line_id           VARCHAR(50)     PRIMARY KEY,
    order_id                VARCHAR(50)     NOT NULL REFERENCES orders(order_id),
    product_id              VARCHAR(50)     REFERENCES products(product_id),
    variant_id              VARCHAR(50)     REFERENCES product_variants(variant_id),
    sku                     VARCHAR(100),
    product_title           VARCHAR(255),
    product_type            VARCHAR(100),
    variant_title           VARCHAR(255),
    quantity                INTEGER         NOT NULL DEFAULT 1,
    price                   DECIMAL(10,2)   NOT NULL,
    total_discount          DECIMAL(10,2)   DEFAULT 0.00,
    line_total              DECIMAL(12,2)   GENERATED ALWAYS AS
                                (price * quantity - total_discount) STORED,
    requires_shipping       BOOLEAN         DEFAULT TRUE,
    gift_card               BOOLEAN         DEFAULT FALSE,
    created_at              TIMESTAMP       DEFAULT NOW(),
    updated_at              TIMESTAMP       DEFAULT NOW()
);

COMMENT ON COLUMN order_lines.line_total IS 'Computed: price × quantity - total_discount. Line-level discount distinct from orders.total_discounts.';

-- ============================================================
-- LAYER 1B — Raw Klaviyo Tables  (Analysis — raw ingestion)
-- ============================================================

CREATE TABLE klaviyo_profiles (
    klaviyo_profile_id      VARCHAR(50)     PRIMARY KEY,
    email                   VARCHAR(255)    UNIQUE NOT NULL,
    first_name              VARCHAR(100),
    last_name               VARCHAR(100),
    phone                   VARCHAR(30),
    city                    VARCHAR(100),
    country                 VARCHAR(100),
    accepts_marketing       BOOLEAN         DEFAULT FALSE,
    klaviyo_created_at      TIMESTAMP       NOT NULL,
    last_event_date         TIMESTAMP,
    total_emails_sent       INTEGER         DEFAULT 0,
    total_emails_opened     INTEGER         DEFAULT 0,
    total_emails_clicked    INTEGER         DEFAULT 0,
    created_at              TIMESTAMP       DEFAULT NOW(),
    updated_at              TIMESTAMP       DEFAULT NOW()
);

CREATE TABLE campaigns (
    campaign_id             VARCHAR(50)     PRIMARY KEY,
    campaign_name           VARCHAR(255)    NOT NULL,
    campaign_type_id        INTEGER         REFERENCES campaign_type(campaign_type_id),
    subject_line            VARCHAR(500),
    send_time               TIMESTAMP       NOT NULL,
    status                  VARCHAR(50),                -- sent | draft | scheduled
    total_recipients        INTEGER         DEFAULT 0,
    total_sent              INTEGER         DEFAULT 0,
    total_delivered         INTEGER         DEFAULT 0,
    total_opens             INTEGER         DEFAULT 0,
    total_clicks            INTEGER         DEFAULT 0,
    total_unsubscribes      INTEGER         DEFAULT 0,
    has_discount_code       BOOLEAN         DEFAULT FALSE,
    discount_code           VARCHAR(100),
    discount_pct            DECIMAL(5,2),               -- e.g. 15.00 for 15% off
    campaign_created_at     TIMESTAMP       NOT NULL,
    created_at              TIMESTAMP       DEFAULT NOW(),
    updated_at              TIMESTAMP       DEFAULT NOW()
);

COMMENT ON TABLE campaigns IS 'Klaviyo campaign master — one row per campaign send event.';
COMMENT ON COLUMN campaigns.send_time IS 'Operative timestamp for campaign-to-purchase attribution window calculations.';

CREATE TABLE email_events (
    event_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id             VARCHAR(50)     NOT NULL REFERENCES campaigns(campaign_id),
    klaviyo_profile_id      VARCHAR(50)     REFERENCES klaviyo_profiles(klaviyo_profile_id),
    email                   VARCHAR(255)    NOT NULL,
    event_type              VARCHAR(50)     NOT NULL,   -- sent | delivered | opened | clicked | unsubscribed | bounced
    event_timestamp         TIMESTAMP       NOT NULL,
    url_clicked             TEXT,                       -- populated for clicked events only
    device_type             VARCHAR(50),                -- desktop | mobile | tablet
    email_client            VARCHAR(100),               -- gmail | outlook | apple_mail
    created_at              TIMESTAMP       DEFAULT NOW()
);

COMMENT ON TABLE email_events IS 'Klaviyo per-profile per-campaign event log. Primary source for email engagement derivatives.';

-- ============================================================
-- LAYER 2 — Identity Resolution
-- ============================================================

CREATE TABLE customer_identity_map (
    identity_id             SERIAL          PRIMARY KEY,
    shopify_customer_id     VARCHAR(50)     UNIQUE REFERENCES customers(customer_id),
    klaviyo_profile_id      VARCHAR(50)     UNIQUE REFERENCES klaviyo_profiles(klaviyo_profile_id),
    email                   VARCHAR(255)    UNIQUE NOT NULL,    -- the canonical join key
    identity_confidence     VARCHAR(20)     DEFAULT 'exact',    -- exact | fuzzy | manual
    first_resolved_at       TIMESTAMP       DEFAULT NOW(),
    last_verified_at        TIMESTAMP       DEFAULT NOW(),
    created_at              TIMESTAMP       DEFAULT NOW(),
    updated_at              TIMESTAMP       DEFAULT NOW()
);

COMMENT ON TABLE customer_identity_map IS
    'Cross-system identity resolution. Klaviyo uses email as PK; Shopify uses numeric customer_id.
     This table normalises both to a single customer identity keyed by email.
     All downstream layers join via shopify_customer_id after resolution here.';

-- ============================================================
-- LAYER 3 — Chronological Event Table  (THE CENTREPIECE)
-- Analysis layer: purchases + campaigns on a single timeline per customer
-- ============================================================

CREATE TABLE customer_events (
    event_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id             VARCHAR(50)     NOT NULL,   -- normalised Shopify customer_id
    event_date              TIMESTAMP       NOT NULL,
    event_type              VARCHAR(50)     NOT NULL,   -- ORDER | CAMPAIGN_SENT | EMAIL_OPENED | EMAIL_CLICKED | EMAIL_BOUNCED | UNSUBSCRIBED
    event_source            VARCHAR(20)     NOT NULL,   -- shopify | klaviyo
    event_ref_id            VARCHAR(100),               -- order_id OR campaign_id
    -- Order-event fields (null for campaign events)
    revenue_amount          DECIMAL(12,2),
    product_type            VARCHAR(100),
    discount_code           VARCHAR(100),
    has_discount            BOOLEAN         DEFAULT FALSE,
    -- Campaign-event fields (null for order events)
    campaign_id             VARCHAR(50),
    campaign_name           VARCHAR(255),
    campaign_type           VARCHAR(100),
    has_campaign_discount   BOOLEAN         DEFAULT FALSE,
    created_at              TIMESTAMP       DEFAULT NOW()
);

COMMENT ON TABLE customer_events IS
    'The chronological event overlay table — the architectural centrepiece of this use case.
     Merges Shopify purchase events and Klaviyo campaign events onto a single customer timeline.
     All attribution, recency, frequency, and campaign-influence metrics are derived from this table.
     Append-only per run. Query always filter by customer_id + date range.';

CREATE INDEX idx_customer_events_customer_id ON customer_events(customer_id);
CREATE INDEX idx_customer_events_event_date  ON customer_events(event_date);
CREATE INDEX idx_customer_events_event_type  ON customer_events(event_type);
CREATE INDEX idx_customer_events_composite   ON customer_events(customer_id, event_date, event_type);

-- ============================================================
-- LAYER 4 — First-Level Derivatives  (Insights of AIIR)
-- Ref: VS-PredictAlly-Schema-Reference Section 4A, 5B
-- ============================================================

CREATE TABLE customer_derivatives (
    derivative_id                   SERIAL          PRIMARY KEY,
    customer_id                     VARCHAR(50)     NOT NULL REFERENCES customers(customer_id),
    run_id                          VARCHAR(50)     NOT NULL REFERENCES pipeline_runs(run_id),
    run_date                        DATE            DEFAULT CURRENT_DATE,

    -- ── Purchase behaviour ───────────────────────────────────
    first_purchase_date             TIMESTAMP,
    second_purchase_date            TIMESTAMP,
    latest_purchase_date            TIMESTAMP,
    total_orders                    INTEGER         DEFAULT 0,
    total_revenue                   DECIMAL(12,2)   DEFAULT 0.00,
    average_order_value             DECIMAL(10,2),
    recency_days                    INTEGER,        -- days since last purchase
    recency_months                  DECIMAL(8,2),   -- months since last purchase
    frequency_months                DECIMAL(8,2),   -- avg months between purchases
    months_since_customer           INTEGER,        -- months since first purchase
    customer_type                   VARCHAR(50),    -- Prospect | One-time Purchaser | Repeat Purchaser

    -- ── Spend windows (rolling) ──────────────────────────────
    spend_last_3_months             DECIMAL(12,2)   DEFAULT 0.00,
    spend_last_6_months             DECIMAL(12,2)   DEFAULT 0.00,
    spend_last_9_months             DECIMAL(12,2)   DEFAULT 0.00,
    spend_last_12_months            DECIMAL(12,2)   DEFAULT 0.00,

    -- ── Product behaviour ────────────────────────────────────
    total_distinct_products         INTEGER         DEFAULT 0,
    first_purchase_product_type     VARCHAR(100),
    latest_purchase_product_type    VARCHAR(100),
    total_distinct_discount_codes   INTEGER         DEFAULT 0,
    season_last_purchased           VARCHAR(20),    -- WINTER | SPRING | SUMMER | FALL
    holiday_shopper                 BOOLEAN         DEFAULT FALSE,
    early_adopter_count             INTEGER         DEFAULT 0,

    -- ── Email / campaign engagement ──────────────────────────
    total_campaigns_received        INTEGER         DEFAULT 0,
    total_campaigns_opened          INTEGER         DEFAULT 0,
    total_campaigns_clicked         INTEGER         DEFAULT 0,
    first_campaign_date             TIMESTAMP,
    latest_campaign_date            TIMESTAMP,
    email_open_rate                 DECIMAL(5,4),   -- opened / received  [0.0000–1.0000]
    email_click_rate                DECIMAL(5,4),   -- clicked / received [0.0000–1.0000]
    days_since_last_campaign        INTEGER,

    -- ── Campaign-to-purchase attribution (new centrepiece) ───
    -- "How many campaigns were sent to this customer before each purchase?"
    avg_campaigns_before_purchase       DECIMAL(8,2),   -- avg across all purchase events
    campaigns_before_first_purchase     INTEGER,        -- campaigns sent before very first order
    total_purchases_with_campaign       INTEGER,        -- orders with >= 1 campaign in prior 30 days
    total_purchases_without_campaign    INTEGER,        -- orders with no campaign in prior 30 days
    campaign_influence_rate             DECIMAL(5,4),   -- purchases_with_campaign / total_orders

    -- ── Attribution segment ───────────────────────────────────
    attribution_segment             VARCHAR(50),
    -- no_campaign_influence | campaign_1_2 | campaign_3_5 | campaign_gt5 | never_purchased

    created_at                      TIMESTAMP       DEFAULT NOW(),
    updated_at                      TIMESTAMP       DEFAULT NOW(),

    UNIQUE(customer_id, run_id)
);

COMMENT ON TABLE customer_derivatives IS
    'First-level derivative features computed from customer_events.
     Mirrors vs_eng_attribute_derivatives_history from VS PredictAlly schema.
     Adds campaign-attribution columns not present in original VS schema.';

-- ============================================================
-- LAYER 5 — Campaign Attribution (per order, per customer)
-- ============================================================

CREATE TABLE campaign_attribution (
    attribution_id              SERIAL          PRIMARY KEY,
    customer_id                 VARCHAR(50)     NOT NULL REFERENCES customers(customer_id),
    run_id                      VARCHAR(50)     NOT NULL REFERENCES pipeline_runs(run_id),
    order_id                    VARCHAR(50)     REFERENCES orders(order_id),
    order_date                  TIMESTAMP,
    order_revenue               DECIMAL(12,2),
    order_number_in_lifecycle   INTEGER,        -- 1 = first order, 2 = second, etc.

    -- Campaigns in 30-day look-back window before this order
    campaigns_in_30d_window     INTEGER         DEFAULT 0,
    campaigns_opened_30d        INTEGER         DEFAULT 0,
    campaigns_clicked_30d       INTEGER         DEFAULT 0,
    most_recent_campaign_id     VARCHAR(50),
    days_since_last_campaign    INTEGER,        -- null if no campaign in window

    -- Attribution outcome
    campaign_influenced         BOOLEAN         DEFAULT FALSE,
    attribution_bucket          VARCHAR(50),
    -- no_influence | campaign_1_2 | campaign_3_5 | campaign_gt5

    created_at                  TIMESTAMP       DEFAULT NOW(),

    UNIQUE(customer_id, order_id, run_id)
);

-- ============================================================
-- LAYER 6 — Period KPIs  (Interpretations of AIIR)
-- ============================================================

CREATE TABLE customer_kpis (
    kpi_id                          SERIAL          PRIMARY KEY,
    customer_id                     VARCHAR(50)     NOT NULL REFERENCES customers(customer_id),
    run_id                          VARCHAR(50)     NOT NULL REFERENCES pipeline_runs(run_id),
    period_type                     VARCHAR(20)     NOT NULL,   -- monthly | quarterly | yearly
    period_label                    VARCHAR(20)     NOT NULL,   -- '2024-01' | '2024-Q1' | '2024'
    period_start                    DATE            NOT NULL,
    period_end                      DATE            NOT NULL,

    -- Revenue KPIs
    orders_in_period                INTEGER         DEFAULT 0,
    revenue_in_period               DECIMAL(12,2)   DEFAULT 0.00,
    avg_order_value_period          DECIMAL(10,2),

    -- Campaign KPIs
    campaigns_received_period       INTEGER         DEFAULT 0,
    campaigns_opened_period         INTEGER         DEFAULT 0,
    campaigns_clicked_period        INTEGER         DEFAULT 0,

    -- Attribution KPIs
    campaign_influenced_orders      INTEGER         DEFAULT 0,
    campaign_influenced_revenue     DECIMAL(12,2)   DEFAULT 0.00,
    non_influenced_orders           INTEGER         DEFAULT 0,
    non_influenced_revenue          DECIMAL(12,2)   DEFAULT 0.00,

    created_at                      TIMESTAMP       DEFAULT NOW(),

    UNIQUE(customer_id, run_id, period_type, period_label)
);

-- ============================================================
-- LAYER 7 — Recommendation Engine Output  (Recommendations of AIIR)
-- ============================================================

CREATE TABLE customer_recommendations (
    recommendation_id       SERIAL          PRIMARY KEY,
    customer_id             VARCHAR(50)     NOT NULL REFERENCES customers(customer_id),
    run_id                  VARCHAR(50)     NOT NULL REFERENCES pipeline_runs(run_id),
    run_date                DATE            DEFAULT CURRENT_DATE,

    -- Classification — multi-class output
    recommendation_label    VARCHAR(50)     NOT NULL,
   -- send_campaign | dont_send | no_campaign_needed | no_campaign_impact
    confidence_score        DECIMAL(5,4),               -- 0.0000–1.0000

    -- Supporting signals fed to the agent
    attribution_segment     VARCHAR(50),
    customer_type           VARCHAR(50),
    recency_days            INTEGER,
    campaign_influence_rate DECIMAL(5,4),
    email_open_rate         DECIMAL(5,4),
    email_click_rate        DECIMAL(5,4),
    avg_campaigns_before_purchase DECIMAL(8,2),

    -- LLM agent reasoning (free-text)
    reasoning               TEXT,

    -- Backtest fields (populated when run_type = 'backtest')
    backtest_actual_label   VARCHAR(50),
    backtest_correct        BOOLEAN,
    backtest_period_revenue DECIMAL(12,2),

    created_at              TIMESTAMP       DEFAULT NOW(),
    updated_at              TIMESTAMP       DEFAULT NOW(),

    UNIQUE(customer_id, run_id)
);

COMMENT ON TABLE customer_recommendations IS
    'Final output of the LangGraph recommendation agent.
     Multi-class classification: send_campaign | dont_send | no_campaign_impact.
     reasoning column stores the LLM agent narrative explaining the classification.
     Backtest columns populated when run against historical holdout data.';

-- ── Pipeline-specific recommendation tables (pipeline_v2) ──────────────
-- Separate tables prevent upsert collision between ML and Claude pipelines.
-- Both mirror customer_recommendations schema exactly.

CREATE TABLE IF NOT EXISTS customer_recommendations_ml (
    LIKE customer_recommendations INCLUDING ALL
);
ALTER TABLE customer_recommendations_ml
    DROP CONSTRAINT IF EXISTS cr_ml_customer_id_key;
ALTER TABLE customer_recommendations_ml
    ADD CONSTRAINT cr_ml_customer_id_key UNIQUE (customer_id);

CREATE TABLE IF NOT EXISTS customer_recommendations_claude (
    LIKE customer_recommendations INCLUDING ALL
);
ALTER TABLE customer_recommendations_claude
    DROP CONSTRAINT IF EXISTS cr_claude_customer_id_key;
ALTER TABLE customer_recommendations_claude
    ADD CONSTRAINT cr_claude_customer_id_key UNIQUE (customer_id);

-- ============================================================
-- Useful views
-- ============================================================

CREATE VIEW v_customer_timeline AS
SELECT
    ce.customer_id,
    cim.email,
    c.first_name,
    c.last_name,
    ce.event_date,
    ce.event_type,
    ce.event_source,
    ce.revenue_amount,
    ce.product_type,
    ce.campaign_name,
    ce.campaign_type,
    ce.has_discount,
    ce.event_ref_id
FROM customer_events ce
JOIN customer_identity_map cim ON cim.shopify_customer_id = ce.customer_id
JOIN customers c ON c.customer_id = ce.customer_id
ORDER BY ce.customer_id, ce.event_date;

COMMENT ON VIEW v_customer_timeline IS
    'Convenience view: chronological event timeline per customer with name and email joined in.
     The visual representation of the campaign + sales chronological overlay.';

CREATE VIEW v_attribution_summary AS
SELECT
    attribution_segment,
    COUNT(DISTINCT customer_id)                                     AS customer_count,
    ROUND(COUNT(DISTINCT customer_id) * 100.0 / SUM(COUNT(DISTINCT customer_id)) OVER (), 2) AS pct_of_customers,
    ROUND(AVG(avg_campaigns_before_purchase), 2)                    AS avg_campaigns_before_purchase,
    ROUND(AVG(total_revenue), 2)                                    AS avg_lifetime_revenue,
    ROUND(AVG(average_order_value), 2)                              AS avg_order_value,
    ROUND(AVG(total_orders), 2)                                     AS avg_orders,
    ROUND(AVG(email_open_rate), 4)                                  AS avg_open_rate,
    ROUND(AVG(email_click_rate), 4)                                 AS avg_click_rate
FROM customer_derivatives
GROUP BY attribution_segment
ORDER BY avg_lifetime_revenue DESC;

COMMENT ON VIEW v_attribution_summary IS
    'High-level attribution segment summary — the key business output of this use case.
     Answers: which customer segment (by campaigns before purchase) generates the most revenue?';

