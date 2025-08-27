-- =====================================================
-- Performance Tuning and Indexing Strategies
-- IPO Valuation Platform Database Optimization
-- =====================================================

-- =====================================================
-- ADVANCED INDEXING STRATEGIES
-- =====================================================

-- Partial indexes for active records only
CREATE INDEX CONCURRENTLY idx_companies_active_ipo_candidates 
ON core.companies (sector, ipo_date DESC, market_cap DESC) 
WHERE status = 'active' AND is_ipo_candidate = true;

-- Expression indexes for computed values
CREATE INDEX CONCURRENTLY idx_companies_name_lower 
ON core.companies (lower(name));

CREATE INDEX CONCURRENTLY idx_companies_name_trigram 
ON core.companies USING gin (name gin_trgm_ops);

-- Covering indexes to avoid table lookups
CREATE INDEX CONCURRENTLY idx_financial_statements_covering 
ON core.financial_statements (company_id, period_end_date DESC, statement_type) 
INCLUDE (revenue, net_income, ebitda);

-- Multi-column indexes for complex queries
CREATE INDEX CONCURRENTLY idx_valuations_complex_search 
ON core.valuations (organization_id, status, model_type, valuation_date DESC)
WHERE status IN ('approved', 'final');

-- BRIN indexes for time-series data (memory efficient)
CREATE INDEX CONCURRENTLY idx_financial_statements_brin_date 
ON core.financial_statements USING brin (period_end_date);

CREATE INDEX CONCURRENTLY idx_audit_log_brin_timestamp 
ON security.audit_log USING brin (timestamp);

-- Hash indexes for exact equality searches
CREATE INDEX CONCURRENTLY idx_companies_ticker_hash 
ON core.companies USING hash (ticker_symbol)
WHERE ticker_symbol IS NOT NULL;

-- =====================================================
-- PARTITIONING STRATEGIES
-- =====================================================

-- Partition audit log by month for better performance
CREATE TABLE security.audit_log_partitioned (
    LIKE security.audit_log INCLUDING ALL
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions for 2024
CREATE TABLE security.audit_log_2024_01 PARTITION OF security.audit_log_partitioned
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE security.audit_log_2024_02 PARTITION OF security.audit_log_partitioned
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Continue creating partitions...
-- Add a script to automatically create future partitions

-- Partition financial statements by year for large datasets
CREATE TABLE core.financial_statements_partitioned (
    LIKE core.financial_statements INCLUDING ALL
) PARTITION BY RANGE (period_end_date);

-- Create yearly partitions
CREATE TABLE core.financial_statements_2023 PARTITION OF core.financial_statements_partitioned
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

CREATE TABLE core.financial_statements_2024 PARTITION OF core.financial_statements_partitioned
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

-- =====================================================
-- MATERIALIZED VIEWS FOR PERFORMANCE
-- =====================================================

-- Materialized view for latest company financials
CREATE MATERIALIZED VIEW mv_latest_company_financials AS
SELECT DISTINCT ON (company_id, statement_type)
    company_id,
    statement_type,
    period_end_date,
    fiscal_year,
    revenue,
    gross_profit,
    operating_income,
    net_income,
    ebitda,
    total_assets,
    shareholders_equity,
    operating_cash_flow,
    free_cash_flow,
    -- Calculated ratios
    CASE WHEN revenue > 0 THEN (gross_profit / revenue * 100) END as gross_margin_pct,
    CASE WHEN revenue > 0 THEN (operating_income / revenue * 100) END as operating_margin_pct,
    CASE WHEN revenue > 0 THEN (net_income / revenue * 100) END as net_margin_pct,
    CASE WHEN shareholders_equity > 0 THEN (net_income / shareholders_equity * 100) END as roe_pct,
    updated_at
FROM core.financial_statements
ORDER BY company_id, statement_type, period_end_date DESC;

-- Index on materialized view
CREATE UNIQUE INDEX idx_mv_latest_financials_pk 
ON mv_latest_company_financials (company_id, statement_type);

-- Materialized view for sector benchmarks
CREATE MATERIALIZED VIEW mv_sector_benchmarks AS
SELECT 
    c.sector,
    c.industry,
    DATE_TRUNC('quarter', fs.period_end_date) as quarter,
    COUNT(*) as company_count,
    
    -- Financial metrics aggregates
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY fs.revenue) as median_revenue,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY fs.net_income) as median_net_income,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY fs.ebitda) as median_ebitda,
    
    -- Ratio aggregates
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY 
        CASE WHEN fs.revenue > 0 THEN (fs.gross_profit / fs.revenue * 100) END
    ) as median_gross_margin,
    
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY 
        CASE WHEN fs.revenue > 0 THEN (fs.operating_income / fs.revenue * 100) END  
    ) as median_operating_margin,
    
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY 
        CASE WHEN fs.shareholders_equity > 0 THEN (fs.net_income / fs.shareholders_equity * 100) END
    ) as median_roe,
    
    -- Valuation multiples from recent valuations
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY 
        CASE WHEN fs.revenue > 0 THEN (v.enterprise_value / fs.revenue) END
    ) as median_ev_revenue_multiple,
    
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY 
        CASE WHEN fs.ebitda > 0 THEN (v.enterprise_value / fs.ebitda) END
    ) as median_ev_ebitda_multiple,
    
    MAX(fs.updated_at) as last_updated
FROM core.companies c
JOIN core.financial_statements fs ON c.id = fs.company_id
LEFT JOIN LATERAL (
    SELECT enterprise_value, equity_value
    FROM core.valuations val
    WHERE val.company_id = c.id 
    AND val.status = 'approved'
    ORDER BY val.valuation_date DESC
    LIMIT 1
) v ON true
WHERE c.status = 'active'
AND fs.statement_type = 'income_statement'
AND fs.period_type = 'quarterly'
AND fs.period_end_date >= CURRENT_DATE - INTERVAL '2 years'
GROUP BY c.sector, c.industry, DATE_TRUNC('quarter', fs.period_end_date);

-- Index on sector benchmarks materialized view
CREATE INDEX idx_mv_sector_benchmarks_sector_quarter 
ON mv_sector_benchmarks (sector, quarter DESC);

-- =====================================================
-- QUERY OPTIMIZATION FUNCTIONS
-- =====================================================

-- Function to get company peer group efficiently
CREATE OR REPLACE FUNCTION get_company_peers(
    target_company_id UUID,
    max_peers INTEGER DEFAULT 10
)
RETURNS TABLE (
    peer_company_id UUID,
    similarity_score NUMERIC,
    company_name VARCHAR,
    ticker_symbol VARCHAR,
    market_cap NUMERIC
)
LANGUAGE SQL
STABLE
AS $$
    WITH target_company AS (
        SELECT sector, industry, market_cap, country_of_incorporation
        FROM core.companies 
        WHERE id = target_company_id
    ),
    peer_candidates AS (
        SELECT 
            c.id,
            c.name,
            c.ticker_symbol,
            c.market_cap,
            -- Similarity scoring
            CASE 
                WHEN c.sector = tc.sector AND c.industry = tc.industry THEN 1.0
                WHEN c.sector = tc.sector THEN 0.7
                ELSE 0.3
            END +
            CASE 
                WHEN c.country_of_incorporation = tc.country_of_incorporation THEN 0.2
                ELSE 0.0
            END +
            -- Size similarity (inverse of ratio difference)
            CASE 
                WHEN tc.market_cap > 0 AND c.market_cap > 0 THEN
                    1.0 - ABS(LOG(c.market_cap / tc.market_cap)) / 3.0
                ELSE 0.0
            END as similarity_score
        FROM core.companies c, target_company tc
        WHERE c.id != target_company_id
        AND c.status = 'active'
        AND c.is_public = true
        AND c.market_cap IS NOT NULL
    )
    SELECT 
        id,
        similarity_score,
        name,
        ticker_symbol,
        market_cap
    FROM peer_candidates
    WHERE similarity_score > 0.5
    ORDER BY similarity_score DESC
    LIMIT max_peers;
$$;

-- Function for fast valuation model calculations
CREATE OR REPLACE FUNCTION calculate_dcf_summary(
    target_company_id UUID,
    discount_rate NUMERIC DEFAULT 0.10,
    terminal_growth_rate NUMERIC DEFAULT 0.03,
    forecast_years INTEGER DEFAULT 5
)
RETURNS TABLE (
    enterprise_value NUMERIC,
    equity_value NUMERIC,
    per_share_value NUMERIC,
    calculation_date TIMESTAMP
)
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    latest_financials RECORD;
    shares_outstanding NUMERIC;
    net_debt NUMERIC;
BEGIN
    -- Get latest financial data
    SELECT INTO latest_financials
        revenue, operating_income, free_cash_flow, total_assets, 
        total_liabilities, cash_and_equivalents, total_debt
    FROM mv_latest_company_financials 
    WHERE company_id = target_company_id 
    AND statement_type = 'income_statement';
    
    -- Get shares outstanding (simplified - would normally be more complex)
    SELECT INTO shares_outstanding
        COALESCE(shares_outstanding, 100000000)  -- Default if not available
    FROM core.companies 
    WHERE id = target_company_id;
    
    -- Calculate net debt
    net_debt := COALESCE(latest_financials.total_debt, 0) - 
                COALESCE(latest_financials.cash_and_equivalents, 0);
    
    -- Simplified DCF calculation (real implementation would be much more complex)
    enterprise_value := latest_financials.free_cash_flow * 
                       ((1 - POWER(1 + discount_rate, -forecast_years)) / discount_rate) +
                       (latest_financials.free_cash_flow * (1 + terminal_growth_rate)) /
                       ((discount_rate - terminal_growth_rate) * POWER(1 + discount_rate, forecast_years));
    
    equity_value := enterprise_value - net_debt;
    per_share_value := equity_value / shares_outstanding;
    calculation_date := NOW();
    
    RETURN QUERY SELECT 
        enterprise_value, 
        equity_value, 
        per_share_value, 
        calculation_date;
END;
$$;

-- =====================================================
-- PERFORMANCE MONITORING VIEWS
-- =====================================================

-- View for monitoring slow queries
CREATE VIEW v_performance_slow_queries AS
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    stddev_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements
WHERE mean_time > 100  -- Queries with mean time > 100ms
ORDER BY mean_time DESC;

-- View for monitoring table and index usage
CREATE VIEW v_performance_table_stats AS
SELECT 
    schemaname,
    tablename,
    seq_scan,
    seq_tup_read,
    idx_scan,
    idx_tup_fetch,
    n_tup_ins,
    n_tup_upd,
    n_tup_del,
    n_tup_hot_upd,
    n_live_tup,
    n_dead_tup,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
ORDER BY seq_scan DESC;

-- View for unused indexes
CREATE VIEW v_performance_unused_indexes AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexrelname::regclass)) as index_size
FROM pg_stat_user_indexes
WHERE idx_scan < 50  -- Indexes with less than 50 scans
ORDER BY pg_relation_size(indexrelname::regclass) DESC;

-- View for cache hit ratios
CREATE VIEW v_performance_cache_stats AS
SELECT 
    'database' as cache_type,
    sum(blks_hit) as hits,
    sum(blks_read) as reads,
    sum(blks_hit) / (sum(blks_hit) + sum(blks_read)) * 100 as hit_ratio
FROM pg_stat_database
UNION ALL
SELECT 
    'index' as cache_type,
    sum(idx_blks_hit) as hits,
    sum(idx_blks_read) as reads,
    sum(idx_blks_hit) / (sum(idx_blks_hit) + sum(idx_blks_read)) * 100 as hit_ratio
FROM pg_statio_user_indexes;

-- =====================================================
-- AUTOMATED MAINTENANCE PROCEDURES
-- =====================================================

-- Procedure to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_performance_views()
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    -- Refresh materialized views concurrently to avoid blocking
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_latest_company_financials;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_sector_benchmarks;
    
    -- Update statistics for better query planning
    ANALYZE core.companies;
    ANALYZE core.financial_statements;
    ANALYZE core.valuations;
    
    RAISE NOTICE 'Performance views refreshed successfully';
END;
$$;

-- Schedule materialized view refresh (to be called by cron or scheduler)
-- This would typically be handled by pg_cron extension or external scheduler

-- Procedure to create monthly audit log partitions
CREATE OR REPLACE FUNCTION create_audit_log_partition(partition_date DATE)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    start_date := DATE_TRUNC('month', partition_date);
    end_date := start_date + INTERVAL '1 month';
    partition_name := 'audit_log_' || TO_CHAR(start_date, 'YYYY_MM');
    
    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS security.%I PARTITION OF security.audit_log_partitioned
         FOR VALUES FROM (%L) TO (%L)',
        partition_name, start_date, end_date
    );
    
    -- Create index on new partition
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS idx_%I_timestamp 
         ON security.%I (timestamp DESC)',
        partition_name, partition_name
    );
    
    RAISE NOTICE 'Created partition % for dates % to %', partition_name, start_date, end_date;
END;
$$;

-- =====================================================
-- QUERY OPTIMIZATION HINTS AND PATTERNS
-- =====================================================

-- Example of optimized company search query
/*
-- GOOD: Uses covering index and proper filtering
EXPLAIN (ANALYZE, BUFFERS)
SELECT c.id, c.name, c.ticker_symbol, c.sector, c.market_cap
FROM core.companies c
WHERE c.status = 'active' 
AND c.is_ipo_candidate = true
AND c.sector = 'Technology'
AND c.market_cap > 1000000000
ORDER BY c.market_cap DESC
LIMIT 20;
*/

-- Example of optimized financial data query
/*
-- GOOD: Uses materialized view for latest data
EXPLAIN (ANALYZE, BUFFERS)
SELECT 
    c.name,
    c.ticker_symbol,
    f.revenue,
    f.net_income,
    f.gross_margin_pct,
    f.operating_margin_pct
FROM core.companies c
JOIN mv_latest_company_financials f ON c.id = f.company_id
WHERE c.sector = 'Technology'
AND f.statement_type = 'income_statement'
AND f.revenue > 100000000
ORDER BY f.revenue DESC;
*/

-- Example of optimized valuation comparison query
/*
-- GOOD: Uses CTEs and proper joins
WITH recent_valuations AS (
    SELECT DISTINCT ON (company_id)
        company_id,
        per_share_value,
        enterprise_value,
        valuation_date,
        model_type
    FROM core.valuations
    WHERE status = 'approved'
    AND valuation_date >= CURRENT_DATE - INTERVAL '6 months'
    ORDER BY company_id, valuation_date DESC
),
sector_stats AS (
    SELECT 
        c.sector,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY v.per_share_value) as median_share_price
    FROM recent_valuations v
    JOIN core.companies c ON v.company_id = c.id
    GROUP BY c.sector
)
SELECT 
    c.name,
    c.sector,
    v.per_share_value,
    s.median_share_price,
    (v.per_share_value / s.median_share_price - 1) * 100 as premium_discount_pct
FROM recent_valuations v
JOIN core.companies c ON v.company_id = c.id
JOIN sector_stats s ON c.sector = s.sector
ORDER BY premium_discount_pct DESC;
*/

-- =====================================================
-- PERFORMANCE TESTING PROCEDURES
-- =====================================================

-- Function to generate test data for performance testing
CREATE OR REPLACE FUNCTION generate_test_financial_data(
    company_count INTEGER DEFAULT 1000,
    years_of_data INTEGER DEFAULT 5
)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    company_rec RECORD;
    test_date DATE;
    quarter INTEGER;
BEGIN
    -- This would generate synthetic test data for performance testing
    -- Implementation would create realistic financial data patterns
    RAISE NOTICE 'Would generate test data for % companies over % years', company_count, years_of_data;
    
    -- In a real implementation, this would:
    -- 1. Create test companies with realistic attributes
    -- 2. Generate quarterly financial statements
    -- 3. Create realistic valuation data
    -- 4. Ensure referential integrity
END;
$$;

-- Function to run performance benchmarks
CREATE OR REPLACE FUNCTION run_performance_benchmark()
RETURNS TABLE (
    test_name TEXT,
    execution_time_ms NUMERIC,
    rows_processed INTEGER,
    rows_per_second NUMERIC
)
LANGUAGE plpgsql
AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    row_count INTEGER;
BEGIN
    -- Benchmark 1: Company search performance
    start_time := clock_timestamp();
    SELECT COUNT(*) INTO row_count 
    FROM core.companies 
    WHERE status = 'active' AND sector = 'Technology';
    end_time := clock_timestamp();
    
    RETURN QUERY SELECT 
        'Company Search'::TEXT,
        EXTRACT(EPOCH FROM (end_time - start_time)) * 1000,
        row_count,
        row_count / EXTRACT(EPOCH FROM (end_time - start_time));
    
    -- Benchmark 2: Financial data aggregation
    start_time := clock_timestamp();
    SELECT COUNT(*) INTO row_count 
    FROM mv_latest_company_financials 
    WHERE revenue > 100000000;
    end_time := clock_timestamp();
    
    RETURN QUERY SELECT 
        'Financial Aggregation'::TEXT,
        EXTRACT(EPOCH FROM (end_time - start_time)) * 1000,
        row_count,
        row_count / EXTRACT(EPOCH FROM (end_time - start_time));
    
    -- Add more benchmarks as needed
END;
$$;

-- =====================================================
-- COMMENTS AND DOCUMENTATION
-- =====================================================

COMMENT ON FUNCTION get_company_peers IS 'Efficiently finds peer companies using similarity scoring algorithm';
COMMENT ON FUNCTION calculate_dcf_summary IS 'Performs simplified DCF calculation for quick valuation estimates';
COMMENT ON FUNCTION refresh_performance_views IS 'Refreshes all materialized views for optimal query performance';
COMMENT ON VIEW v_performance_slow_queries IS 'Monitors queries that may need optimization';
COMMENT ON VIEW v_performance_unused_indexes IS 'Identifies indexes that may be candidates for removal';

-- Best practices reminder comments
/*
PERFORMANCE OPTIMIZATION BEST PRACTICES:

1. INDEXING:
   - Use covering indexes for frequently accessed columns
   - Create partial indexes for filtered queries
   - Use expression indexes for computed values
   - Consider BRIN indexes for time-series data

2. QUERIES:
   - Use LIMIT for pagination
   - Avoid SELECT * in application code
   - Use EXISTS instead of IN with subqueries
   - Use CTEs for complex queries readability

3. MAINTENANCE:
   - Regularly ANALYZE tables for updated statistics
   - Monitor and refresh materialized views
   - Use VACUUM and REINDEX as needed
   - Monitor slow query log

4. PARTITIONING:
   - Partition large tables by date ranges
   - Create indexes on partition keys
   - Use constraint exclusion

5. MONITORING:
   - Track query performance over time
   - Monitor index usage statistics
   - Check cache hit ratios regularly
   - Set up alerts for performance degradation
*/