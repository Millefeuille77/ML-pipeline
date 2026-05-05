-- FMCG Demand Forecasting & Product Intelligence Platform
-- PostgreSQL schema. Idempotent: re-runnable via init_db.

-- Drop in reverse-FK order so re-creation is safe.
DROP TABLE IF EXISTS batch_predictions CASCADE;
DROP TABLE IF EXISTS demand_forecasts CASCADE;
DROP TABLE IF EXISTS enrichment_features CASCADE;
DROP TABLE IF EXISTS weekly_features CASCADE;
DROP TABLE IF EXISTS daily_sales CASCADE;
DROP TABLE IF EXISTS products CASCADE;

CREATE TABLE products (
    sku        VARCHAR(8) PRIMARY KEY,
    brand      VARCHAR(64) NOT NULL,
    segment    VARCHAR(64) NOT NULL,
    category   VARCHAR(32) NOT NULL,
    pack_type  VARCHAR(32) NOT NULL,
    CONSTRAINT products_sku_format CHECK (sku ~ '^[A-Z]{2}-[0-9]{3}$'),
    CONSTRAINT products_category_valid CHECK (
        category IN ('Milk', 'Yogurt', 'ReadyMeal', 'Juice', 'SnackBar')
    ),
    CONSTRAINT products_pack_type_valid CHECK (
        pack_type IN ('Multipack', 'Single', 'Carton')
    )
);
COMMENT ON TABLE products IS 'Product master: 30 SKUs across 5 categories.';
CREATE INDEX IF NOT EXISTS idx_products_category ON products (category);

CREATE TABLE daily_sales (
    id               BIGSERIAL PRIMARY KEY,
    sku              VARCHAR(8) NOT NULL REFERENCES products (sku) ON DELETE CASCADE,
    sale_date        DATE NOT NULL,
    channel          VARCHAR(16) NOT NULL,
    region           VARCHAR(16) NOT NULL,
    price_unit       NUMERIC(10, 4) NOT NULL,
    promotion_flag   SMALLINT NOT NULL,
    delivery_days    SMALLINT NOT NULL,
    stock_available  NUMERIC(14, 4) NOT NULL,
    delivered_qty    NUMERIC(14, 4) NOT NULL,
    units_sold       NUMERIC(14, 4) NOT NULL,
    CONSTRAINT daily_sales_unique UNIQUE (sku, sale_date, channel, region),
    CONSTRAINT daily_sales_price_positive CHECK (price_unit > 0),
    CONSTRAINT daily_sales_delivery_range CHECK (delivery_days BETWEEN 1 AND 5),
    CONSTRAINT daily_sales_promo_flag CHECK (promotion_flag IN (0, 1)),
    CONSTRAINT daily_sales_channel_valid CHECK (
        channel IN ('Retail', 'Discount', 'E-commerce')
    ),
    CONSTRAINT daily_sales_region_valid CHECK (
        region IN ('PL-Central', 'PL-North', 'PL-South')
    )
    -- units_sold and delivered_qty intentionally allow negatives (returns).
);
COMMENT ON TABLE daily_sales IS
    'Daily fact table from FMCG_2022_2024.csv. units_sold/delivered_qty MAY be negative (returns).';
CREATE INDEX IF NOT EXISTS idx_daily_sales_sku ON daily_sales (sku);
CREATE INDEX IF NOT EXISTS idx_daily_sales_date ON daily_sales (sale_date);
CREATE INDEX IF NOT EXISTS idx_daily_sales_channel_region ON daily_sales (channel, region);
CREATE INDEX IF NOT EXISTS idx_daily_sales_sku_date ON daily_sales (sku, sale_date);

CREATE TABLE weekly_features (
    id                BIGSERIAL PRIMARY KEY,
    sku               VARCHAR(8) NOT NULL REFERENCES products (sku) ON DELETE CASCADE,
    week              DATE NOT NULL,
    channel           VARCHAR(16) NOT NULL,
    region            VARCHAR(16) NOT NULL,
    units_sold        NUMERIC(14, 4) NOT NULL,
    stock_available   NUMERIC(14, 4) NOT NULL,
    promotion_flag    SMALLINT NOT NULL,
    price_unit        NUMERIC(10, 4) NOT NULL,
    delivery_days     NUMERIC(6, 3) NOT NULL,
    is_holiday_peak   SMALLINT NOT NULL,
    week_number       SMALLINT NOT NULL,
    month             SMALLINT NOT NULL,
    year              SMALLINT NOT NULL,
    is_holiday_week   SMALLINT NOT NULL,
    is_summer         SMALLINT NOT NULL,
    is_winter         SMALLINT NOT NULL,
    sku_age           INTEGER NOT NULL,
    lifecycle_stage   VARCHAR(16) NOT NULL,
    lag_1             NUMERIC(14, 4),
    lag_2             NUMERIC(14, 4),
    rolling_mean_4    NUMERIC(14, 4),
    rolling_std_4     NUMERIC(14, 4),
    momentum          NUMERIC(14, 4),
    target_next_week  NUMERIC(14, 4),
    CONSTRAINT weekly_features_unique UNIQUE (sku, week, channel, region),
    CONSTRAINT weekly_features_lifecycle_valid CHECK (
        lifecycle_stage IN ('Growth', 'Mature', 'Decline')
    ),
    CONSTRAINT weekly_features_price_positive CHECK (price_unit > 0),
    CONSTRAINT weekly_features_promo_flag CHECK (promotion_flag IN (0, 1)),
    CONSTRAINT weekly_features_channel_valid CHECK (
        channel IN ('Retail', 'Discount', 'E-commerce')
    ),
    CONSTRAINT weekly_features_region_valid CHECK (
        region IN ('PL-Central', 'PL-North', 'PL-South')
    )
);
COMMENT ON TABLE weekly_features IS
    'Weekly modeling table from weekly_df_final_for_modeling.csv. Includes lag/rolling features.';
CREATE INDEX IF NOT EXISTS idx_weekly_features_sku ON weekly_features (sku);
CREATE INDEX IF NOT EXISTS idx_weekly_features_week ON weekly_features (week);
CREATE INDEX IF NOT EXISTS idx_weekly_features_channel_region
    ON weekly_features (channel, region);
CREATE INDEX IF NOT EXISTS idx_weekly_features_sku_week ON weekly_features (sku, week);

CREATE TABLE enrichment_features (
    id                 BIGSERIAL PRIMARY KEY,
    sku                VARCHAR(8) NOT NULL REFERENCES products (sku) ON DELETE CASCADE,
    week               DATE NOT NULL,
    channel            VARCHAR(16) NOT NULL,
    region             VARCHAR(16) NOT NULL,
    price_avg          NUMERIC(10, 4) NOT NULL,
    promo_rate         NUMERIC(6, 4) NOT NULL,
    stock_avg          NUMERIC(14, 4) NOT NULL,
    deliveries         INTEGER NOT NULL,
    avg_temp           NUMERIC(6, 2) NOT NULL,
    inflation_index    NUMERIC(8, 4) NOT NULL,
    school_in_session  SMALLINT NOT NULL,
    category_trend     NUMERIC(10, 4) NOT NULL,
    event_score        NUMERIC(8, 4) NOT NULL,
    CONSTRAINT enrichment_features_unique UNIQUE (sku, week, channel, region),
    CONSTRAINT enrichment_features_promo_rate CHECK (promo_rate BETWEEN 0 AND 1),
    CONSTRAINT enrichment_features_price_avg_positive CHECK (price_avg >= 0)
);
COMMENT ON TABLE enrichment_features IS
    'Generalized enrichment columns (template: df_weekly_MI-006_enriched.csv).';
CREATE INDEX IF NOT EXISTS idx_enrichment_sku_week ON enrichment_features (sku, week);

CREATE TABLE demand_forecasts (
    id                BIGSERIAL PRIMARY KEY,
    sku               VARCHAR(8) NOT NULL REFERENCES products (sku) ON DELETE CASCADE,
    channel           VARCHAR(16) NOT NULL,
    region            VARCHAR(16) NOT NULL,
    forecast_week     DATE NOT NULL,
    predicted_units   NUMERIC(14, 4) NOT NULL,
    confidence_lower  NUMERIC(14, 4) NOT NULL,
    confidence_upper  NUMERIC(14, 4) NOT NULL,
    model_version     VARCHAR(64) NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT demand_forecasts_unique
        UNIQUE (sku, channel, region, forecast_week, model_version),
    CONSTRAINT demand_forecasts_interval_valid
        CHECK (confidence_lower <= predicted_units AND predicted_units <= confidence_upper)
);
COMMENT ON TABLE demand_forecasts IS
    'Persisted weekly forecasts with quantile prediction intervals.';
CREATE INDEX IF NOT EXISTS idx_forecasts_sku_week ON demand_forecasts (sku, forecast_week);

CREATE TABLE batch_predictions (
    id                BIGSERIAL PRIMARY KEY,
    batch_id          VARCHAR(128) NOT NULL,
    sku               VARCHAR(8) NOT NULL REFERENCES products (sku) ON DELETE CASCADE,
    week              DATE NOT NULL,
    channel           VARCHAR(16) NOT NULL,
    region            VARCHAR(16) NOT NULL,
    predicted_units   NUMERIC(14, 4) NOT NULL,
    confidence_lower  NUMERIC(14, 4) NOT NULL,
    confidence_upper  NUMERIC(14, 4) NOT NULL,
    model_version     VARCHAR(64) NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT batch_predictions_unique
        UNIQUE (batch_id, sku, week, channel, region)
);
COMMENT ON TABLE batch_predictions IS
    'Predictions produced by run_batch_pipeline (parquet-driven weekly batches).';
CREATE INDEX IF NOT EXISTS idx_batch_predictions_batch ON batch_predictions (batch_id);
CREATE INDEX IF NOT EXISTS idx_batch_predictions_sku ON batch_predictions (sku);
