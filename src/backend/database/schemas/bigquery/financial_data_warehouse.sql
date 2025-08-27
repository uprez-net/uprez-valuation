-- =====================================================
-- BigQuery Data Warehouse Schema for IPO Valuation Platform
-- Optimized for analytical workloads and historical data
-- =====================================================

-- =====================================================
-- COMPANY DIMENSION TABLES
-- =====================================================

-- Companies dimension table with SCD Type 2 for historical tracking
CREATE OR REPLACE TABLE `uprez-valuation.analytics.dim_companies` (
    company_sk STRING NOT NULL, -- Surrogate key
    company_id STRING NOT NULL, -- Natural key from PostgreSQL
    
    -- Company information
    company_name STRING NOT NULL,
    legal_name STRING,
    ticker_symbol STRING,
    exchange STRING,
    isin STRING,
    cusip STRING,
    lei STRING,
    
    -- Classification
    sector STRING,
    industry STRING,
    sub_industry STRING,
    gics_sector_code STRING,
    gics_industry_code STRING,
    
    -- Geographic
    country_of_incorporation STRING,
    country_of_domicile STRING,
    headquarters_country STRING,
    headquarters_city STRING,
    
    -- IPO information
    ipo_date DATE,
    ipo_price NUMERIC,
    ipo_shares_offered INT64,
    ipo_proceeds NUMERIC,
    underwriters ARRAY<STRING>,
    
    -- Business details
    description STRING,
    business_model STRING,
    
    -- Status
    status STRING,
    is_public BOOL,
    is_ipo_candidate BOOL,
    
    -- External identifiers
    bloomberg_ticker STRING,
    reuters_ric STRING,
    factset_entity_id STRING,
    
    -- SCD Type 2 fields
    effective_date DATE NOT NULL,
    expiration_date DATE,
    is_current BOOL NOT NULL,
    
    -- Data quality
    data_quality_score NUMERIC,
    last_updated TIMESTAMP,
    
    -- ETL metadata
    etl_batch_id STRING,
    etl_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(_PARTITIONTIME)
CLUSTER BY company_id, sector, country_of_incorporation;

-- Date dimension for time-based analysis
CREATE OR REPLACE TABLE `uprez-valuation.analytics.dim_date` (
    date_key INT64 NOT NULL, -- YYYYMMDD format
    full_date DATE NOT NULL,
    
    -- Date components
    year INT64,
    quarter INT64,
    month INT64,
    month_name STRING,
    day_of_month INT64,
    day_of_week INT64,
    day_name STRING,
    week_of_year INT64,
    
    -- Business calendar
    is_weekend BOOL,
    is_holiday BOOL,
    holiday_name STRING,
    is_business_day BOOL,
    
    -- Financial calendar
    fiscal_year INT64,
    fiscal_quarter INT64,
    fiscal_month INT64,
    
    -- Relative periods
    is_current_year BOOL,
    is_current_quarter BOOL,
    is_current_month BOOL,
    
    -- Market calendar
    is_trading_day BOOL,
    market_open_time TIME,
    market_close_time TIME
);

-- User dimension for analytics
CREATE OR REPLACE TABLE `uprez-valuation.analytics.dim_users` (
    user_sk STRING NOT NULL,
    user_id STRING NOT NULL,
    
    -- User information
    email STRING,
    full_name STRING,
    title STRING,
    organization STRING,
    department STRING,
    
    -- Account details
    subscription_tier STRING,
    user_status STRING,
    
    -- Geographic
    country STRING,
    timezone STRING,
    locale STRING,
    
    -- SCD Type 2 fields
    effective_date DATE NOT NULL,
    expiration_date DATE,
    is_current BOOL NOT NULL,
    
    -- ETL metadata
    etl_batch_id STRING,
    etl_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(_PARTITIONTIME)
CLUSTER BY organization, subscription_tier;

-- =====================================================
-- FACT TABLES
-- =====================================================

-- Financial statements fact table (main analytical table)
CREATE OR REPLACE TABLE `uprez-valuation.analytics.fact_financial_statements` (
    -- Keys
    company_sk STRING NOT NULL,
    date_key INT64 NOT NULL,
    statement_sk STRING NOT NULL, -- Unique identifier for this statement
    
    -- Dimensions
    company_id STRING NOT NULL,
    statement_type STRING NOT NULL, -- income_statement, balance_sheet, cash_flow
    period_type STRING NOT NULL, -- annual, quarterly, ttm
    period_end_date DATE NOT NULL,
    fiscal_year INT64 NOT NULL,
    fiscal_quarter INT64,
    
    -- Financial metrics (Income Statement)
    revenue NUMERIC,
    gross_profit NUMERIC,
    operating_income NUMERIC,
    ebitda NUMERIC,
    ebit NUMERIC,
    net_income NUMERIC,
    eps_basic NUMERIC,
    eps_diluted NUMERIC,
    shares_outstanding NUMERIC,
    shares_outstanding_diluted NUMERIC,
    
    -- Cost structure
    cost_of_revenue NUMERIC,
    research_development NUMERIC,
    sales_marketing NUMERIC,
    general_administrative NUMERIC,
    total_operating_expenses NUMERIC,
    
    -- Non-operating items
    interest_income NUMERIC,
    interest_expense NUMERIC,
    other_income NUMERIC,
    tax_expense NUMERIC,
    
    -- Balance Sheet Assets
    cash_and_equivalents NUMERIC,
    short_term_investments NUMERIC,
    accounts_receivable NUMERIC,
    inventory NUMERIC,
    total_current_assets NUMERIC,
    property_plant_equipment NUMERIC,
    intangible_assets NUMERIC,
    goodwill NUMERIC,
    total_assets NUMERIC,
    
    -- Balance Sheet Liabilities
    accounts_payable NUMERIC,
    short_term_debt NUMERIC,
    total_current_liabilities NUMERIC,
    long_term_debt NUMERIC,
    total_liabilities NUMERIC,
    
    -- Equity
    shareholders_equity NUMERIC,
    retained_earnings NUMERIC,
    
    -- Cash Flow Statement
    operating_cash_flow NUMERIC,
    investing_cash_flow NUMERIC,
    financing_cash_flow NUMERIC,
    free_cash_flow NUMERIC,
    capex NUMERIC,
    
    -- Calculated ratios and metrics
    gross_margin NUMERIC,
    operating_margin NUMERIC,
    net_margin NUMERIC,
    roe NUMERIC, -- Return on Equity
    roa NUMERIC, -- Return on Assets
    current_ratio NUMERIC,
    quick_ratio NUMERIC,
    debt_to_equity NUMERIC,
    debt_to_assets NUMERIC,
    interest_coverage NUMERIC,
    
    -- Growth metrics (YoY)
    revenue_growth NUMERIC,
    gross_profit_growth NUMERIC,
    operating_income_growth NUMERIC,
    net_income_growth NUMERIC,
    
    -- Data quality and source
    currency STRING DEFAULT 'USD',
    scale_factor INT64 DEFAULT 1,
    data_quality_score NUMERIC,
    source_system STRING,
    is_restated BOOL DEFAULT FALSE,
    
    -- ETL metadata
    etl_batch_id STRING,
    etl_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    source_updated_at TIMESTAMP
)
PARTITION BY period_end_date
CLUSTER BY company_id, statement_type, fiscal_year DESC;

-- Valuations fact table
CREATE OR REPLACE TABLE `uprez-valuation.analytics.fact_valuations` (
    -- Keys
    valuation_sk STRING NOT NULL,
    company_sk STRING NOT NULL,
    user_sk STRING NOT NULL,
    date_key INT64 NOT NULL,
    
    -- Dimensions
    valuation_id STRING NOT NULL,
    company_id STRING NOT NULL,
    user_id STRING NOT NULL,
    model_type STRING NOT NULL,
    model_name STRING,
    
    -- Valuation details
    valuation_date DATE NOT NULL,
    base_date DATE NOT NULL,
    scenario STRING, -- bear, base, bull
    
    -- Valuation results
    enterprise_value NUMERIC,
    equity_value NUMERIC,
    per_share_value NUMERIC,
    
    -- Value ranges
    value_range_low NUMERIC,
    value_range_high NUMERIC,
    confidence_level NUMERIC,
    
    -- Key assumptions (flattened from JSONB)
    discount_rate NUMERIC,
    terminal_growth_rate NUMERIC,
    terminal_multiple NUMERIC,
    forecast_years INT64,
    
    -- Market data at valuation date
    market_cap NUMERIC,
    trading_multiple_pe NUMERIC,
    trading_multiple_ev_ebitda NUMERIC,
    trading_multiple_ev_revenue NUMERIC,
    
    -- Peer comparison metrics
    peer_median_pe NUMERIC,
    peer_median_ev_ebitda NUMERIC,
    peer_median_ev_revenue NUMERIC,
    peer_premium_discount NUMERIC,
    
    -- Model performance
    accuracy_score NUMERIC, -- When compared to actual market price
    model_version STRING,
    
    -- Status and workflow
    status STRING,
    is_approved BOOL,
    
    -- Usage metrics
    view_count INT64 DEFAULT 0,
    share_count INT64 DEFAULT 0,
    
    -- ETL metadata
    etl_batch_id STRING,
    etl_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    source_updated_at TIMESTAMP
)
PARTITION BY valuation_date
CLUSTER BY company_id, model_type, valuation_date DESC;

-- Market data fact table for benchmarking
CREATE OR REPLACE TABLE `uprez-valuation.analytics.fact_market_data` (
    -- Keys
    market_data_sk STRING NOT NULL,
    company_sk STRING NOT NULL,
    date_key INT64 NOT NULL,
    
    -- Dimensions
    company_id STRING NOT NULL,
    ticker_symbol STRING NOT NULL,
    exchange STRING,
    data_date DATE NOT NULL,
    
    -- Price data
    open_price NUMERIC,
    high_price NUMERIC,
    low_price NUMERIC,
    close_price NUMERIC,
    adjusted_close NUMERIC,
    volume INT64,
    
    -- Market metrics
    market_cap NUMERIC,
    shares_outstanding NUMERIC,
    
    -- Trading multiples
    pe_ratio NUMERIC,
    pb_ratio NUMERIC,
    ps_ratio NUMERIC,
    ev_ebitda NUMERIC,
    ev_revenue NUMERIC,
    
    -- Technical indicators
    moving_avg_50d NUMERIC,
    moving_avg_200d NUMERIC,
    rsi NUMERIC,
    volatility_30d NUMERIC,
    
    -- Data source and quality
    data_source STRING,
    data_quality_score NUMERIC,
    
    -- ETL metadata
    etl_batch_id STRING,
    etl_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY data_date
CLUSTER BY company_id, data_date DESC;

-- IPO performance tracking
CREATE OR REPLACE TABLE `uprez-valuation.analytics.fact_ipo_performance` (
    -- Keys
    ipo_sk STRING NOT NULL,
    company_sk STRING NOT NULL,
    date_key INT64 NOT NULL,
    
    -- Company and IPO details
    company_id STRING NOT NULL,
    company_name STRING NOT NULL,
    ipo_date DATE NOT NULL,
    
    -- IPO pricing and sizing
    ipo_price NUMERIC NOT NULL,
    shares_offered INT64,
    shares_outstanding_post_ipo INT64,
    gross_proceeds NUMERIC,
    net_proceeds NUMERIC,
    
    -- Valuation metrics at IPO
    market_cap_at_ipo NUMERIC,
    enterprise_value_at_ipo NUMERIC,
    
    -- IPO multiples (based on last 12 months)
    ipo_pe_multiple NUMERIC,
    ipo_ev_revenue_multiple NUMERIC,
    ipo_ev_ebitda_multiple NUMERIC,
    ipo_price_book_multiple NUMERIC,
    
    -- Performance tracking (various time periods)
    first_day_close NUMERIC,
    first_day_return NUMERIC,
    first_week_return NUMERIC,
    first_month_return NUMERIC,
    first_quarter_return NUMERIC,
    first_year_return NUMERIC,
    
    -- Volume and trading data
    first_day_volume INT64,
    avg_daily_volume_30d INT64,
    
    -- Underwriting details
    lead_underwriter STRING,
    underwriter_spread NUMERIC,
    greenshoe_exercised BOOL,
    greenshoe_amount NUMERIC,
    
    -- Market conditions at IPO
    sp500_level NUMERIC,
    sector_index_level NUMERIC,
    vix_level NUMERIC,
    
    -- Company fundamentals at IPO (TTM)
    revenue_ttm NUMERIC,
    net_income_ttm NUMERIC,
    ebitda_ttm NUMERIC,
    free_cash_flow_ttm NUMERIC,
    
    -- Growth metrics (pre-IPO)
    revenue_cagr_3y NUMERIC,
    revenue_cagr_5y NUMERIC,
    
    -- Classification
    sector STRING,
    industry STRING,
    ipo_size_category STRING, -- small, medium, large, mega
    
    -- ETL metadata
    etl_batch_id STRING,
    etl_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY ipo_date
CLUSTER BY ipo_date DESC, sector, ipo_size_category;

-- =====================================================
-- AGGREGATE TABLES FOR PERFORMANCE
-- =====================================================

-- Monthly company financial summaries
CREATE OR REPLACE TABLE `uprez-valuation.analytics.agg_company_financials_monthly` (
    company_id STRING NOT NULL,
    year_month STRING NOT NULL, -- YYYY-MM format
    
    -- Latest available metrics for the month
    latest_revenue NUMERIC,
    latest_net_income NUMERIC,
    latest_ebitda NUMERIC,
    latest_free_cash_flow NUMERIC,
    
    -- Growth calculations
    revenue_mom_growth NUMERIC, -- Month over month
    revenue_yoy_growth NUMERIC, -- Year over year
    
    -- Ratios
    latest_gross_margin NUMERIC,
    latest_operating_margin NUMERIC,
    latest_net_margin NUMERIC,
    
    -- Data quality
    statements_count INT64,
    avg_data_quality_score NUMERIC,
    
    -- ETL metadata
    etl_batch_id STRING,
    etl_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY PARSE_DATE('%Y-%m', year_month)
CLUSTER BY company_id;

-- Sector performance aggregates
CREATE OR REPLACE TABLE `uprez-valuation.analytics.agg_sector_performance` (
    sector STRING NOT NULL,
    period_date DATE NOT NULL,
    period_type STRING NOT NULL, -- monthly, quarterly, annual
    
    -- Company counts
    total_companies INT64,
    public_companies INT64,
    ipo_candidates INT64,
    
    -- Valuation metrics
    median_pe_ratio NUMERIC,
    median_ev_revenue NUMERIC,
    median_ev_ebitda NUMERIC,
    
    -- Performance metrics
    median_revenue_growth NUMERIC,
    median_margin NUMERIC,
    median_roe NUMERIC,
    
    -- IPO activity
    ipo_count_ytd INT64,
    total_ipo_proceeds_ytd NUMERIC,
    avg_ipo_return_first_day NUMERIC,
    
    -- ETL metadata
    etl_batch_id STRING,
    etl_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY period_date
CLUSTER BY sector, period_type;

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- Latest financial statements view
CREATE OR REPLACE VIEW `uprez-valuation.analytics.v_latest_financials` AS
SELECT 
    fs.*,
    c.company_name,
    c.sector,
    c.industry,
    c.ticker_symbol
FROM `uprez-valuation.analytics.fact_financial_statements` fs
JOIN `uprez-valuation.analytics.dim_companies` c
    ON fs.company_id = c.company_id AND c.is_current = TRUE
WHERE fs.period_end_date = (
    SELECT MAX(period_end_date)
    FROM `uprez-valuation.analytics.fact_financial_statements` fs2
    WHERE fs2.company_id = fs.company_id 
    AND fs2.statement_type = fs.statement_type
);

-- IPO pipeline analysis view
CREATE OR REPLACE VIEW `uprez-valuation.analytics.v_ipo_pipeline` AS
SELECT 
    c.company_id,
    c.company_name,
    c.sector,
    c.industry,
    c.headquarters_country,
    f.revenue,
    f.net_income,
    f.revenue_growth,
    v.per_share_value as estimated_ipo_price,
    v.confidence_level,
    CASE 
        WHEN f.revenue > 1000000000 THEN 'Large'
        WHEN f.revenue > 100000000 THEN 'Medium'
        ELSE 'Small'
    END as company_size
FROM `uprez-valuation.analytics.dim_companies` c
LEFT JOIN `uprez-valuation.analytics.v_latest_financials` f
    ON c.company_id = f.company_id
LEFT JOIN (
    SELECT DISTINCT
        company_id,
        per_share_value,
        confidence_level,
        ROW_NUMBER() OVER (PARTITION BY company_id ORDER BY valuation_date DESC) as rn
    FROM `uprez-valuation.analytics.fact_valuations`
    WHERE status = 'approved'
) v ON c.company_id = v.company_id AND v.rn = 1
WHERE c.is_current = TRUE 
AND c.is_ipo_candidate = TRUE
AND c.status = 'active';

-- =====================================================
-- OPTIMIZATION CONFIGURATIONS
-- =====================================================

-- Create materialized views for better performance on common queries
CREATE MATERIALIZED VIEW `uprez-valuation.analytics.mv_sector_benchmarks`
PARTITION BY period_date
CLUSTER BY sector
AS
SELECT 
    sector,
    DATE_TRUNC(period_end_date, QUARTER) as period_date,
    APPROX_QUANTILES(pe_ratio, 100)[OFFSET(50)] as median_pe,
    APPROX_QUANTILES(ev_revenue, 100)[OFFSET(50)] as median_ev_revenue,
    APPROX_QUANTILES(ev_ebitda, 100)[OFFSET(50)] as median_ev_ebitda,
    APPROX_QUANTILES(revenue_growth, 100)[OFFSET(50)] as median_revenue_growth,
    COUNT(*) as company_count
FROM (
    SELECT 
        c.sector,
        fs.period_end_date,
        md.pe_ratio,
        md.ev_revenue,
        md.ev_ebitda,
        fs.revenue_growth
    FROM `uprez-valuation.analytics.fact_financial_statements` fs
    JOIN `uprez-valuation.analytics.dim_companies` c
        ON fs.company_id = c.company_id AND c.is_current = TRUE
    LEFT JOIN `uprez-valuation.analytics.fact_market_data` md
        ON fs.company_id = md.company_id 
        AND md.data_date = fs.period_end_date
    WHERE fs.statement_type = 'income_statement'
    AND fs.period_type = 'quarterly'
    AND c.is_public = TRUE
)
GROUP BY sector, period_date;

-- =====================================================
-- DATA QUALITY MONITORING
-- =====================================================

-- Create table for data quality metrics
CREATE OR REPLACE TABLE `uprez-valuation.analytics.data_quality_metrics` (
    table_name STRING NOT NULL,
    metric_name STRING NOT NULL,
    metric_value NUMERIC,
    measurement_date DATE NOT NULL,
    
    -- Thresholds
    warning_threshold NUMERIC,
    critical_threshold NUMERIC,
    
    -- Status
    status STRING, -- ok, warning, critical
    
    -- ETL metadata
    etl_batch_id STRING,
    etl_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY measurement_date
CLUSTER BY table_name, metric_name;

-- =====================================================
-- STORED PROCEDURES FOR COMMON ANALYTICS
-- =====================================================

-- Procedure to calculate company valuation benchmarks
CREATE OR REPLACE PROCEDURE `uprez-valuation.analytics.calculate_valuation_benchmarks`(
    target_company_id STRING,
    benchmark_date DATE
)
BEGIN
    DECLARE sector_name STRING;
    DECLARE industry_name STRING;
    
    -- Get company sector and industry
    SELECT sector, industry INTO sector_name, industry_name
    FROM `uprez-valuation.analytics.dim_companies`
    WHERE company_id = target_company_id AND is_current = TRUE;
    
    -- Create temporary table with peer companies
    CREATE TEMP TABLE peer_metrics AS
    SELECT 
        c.company_id,
        c.company_name,
        md.pe_ratio,
        md.ev_ebitda,
        md.ev_revenue,
        md.market_cap
    FROM `uprez-valuation.analytics.dim_companies` c
    JOIN `uprez-valuation.analytics.fact_market_data` md
        ON c.company_id = md.company_id
    WHERE c.sector = sector_name
    AND c.is_current = TRUE
    AND c.is_public = TRUE
    AND md.data_date = benchmark_date
    AND c.company_id != target_company_id;
    
    -- Return benchmark statistics
    SELECT 
        'sector_benchmark' as benchmark_type,
        COUNT(*) as peer_count,
        APPROX_QUANTILES(pe_ratio, 4)[OFFSET(2)] as median_pe,
        APPROX_QUANTILES(ev_ebitda, 4)[OFFSET(2)] as median_ev_ebitda,
        APPROX_QUANTILES(ev_revenue, 4)[OFFSET(2)] as median_ev_revenue,
        AVG(market_cap) as avg_market_cap
    FROM peer_metrics;
END;

-- =====================================================
-- COMMENTS
-- =====================================================

-- Add table descriptions
ALTER TABLE `uprez-valuation.analytics.dim_companies`
SET OPTIONS (
    description = "Company dimension table with SCD Type 2 for historical tracking of company changes"
);

ALTER TABLE `uprez-valuation.analytics.fact_financial_statements`
SET OPTIONS (
    description = "Comprehensive financial statements fact table optimized for analytical queries"
);

ALTER TABLE `uprez-valuation.analytics.fact_valuations`
SET OPTIONS (
    description = "Valuation results and analysis with peer comparison metrics"
);