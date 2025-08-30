# BigQuery ML Integration Guide for IPO Valuation Platform

## Overview

BigQuery ML enables the creation and execution of machine learning models using SQL queries directly in BigQuery. This guide covers implementing financial valuation models, time series forecasting, and analytical queries for the IPO valuation platform.

## Architecture Overview

### Core Components
- **SQL-based ML Models**: Financial prediction models using BQML syntax
- **Time Series Models**: Market trend and volatility forecasting
- **Classification Models**: Risk category prediction and document classification
- **Feature Engineering**: SQL-based feature creation and transformation
- **Model Deployment**: Serving predictions directly from BigQuery

### Integration Flow

```sql
-- High-level BQML workflow
-- 1. Data Preparation
CREATE OR REPLACE TABLE `project.ipo_valuation.training_data` AS
SELECT * FROM `project.ipo_valuation.financial_data`
WHERE training_split = TRUE;

-- 2. Model Creation
CREATE OR REPLACE MODEL `project.ipo_valuation.valuation_model`
OPTIONS(model_type='linear_reg', input_label_cols=['ipo_valuation'])
AS SELECT * FROM `project.ipo_valuation.training_data`;

-- 3. Model Evaluation
SELECT * FROM ML.EVALUATE(MODEL `project.ipo_valuation.valuation_model`);

-- 4. Predictions
SELECT * FROM ML.PREDICT(MODEL `project.ipo_valuation.valuation_model`, 
  TABLE `project.ipo_valuation.prediction_data`);
```

## Financial Valuation Models

### 1. IPO Valuation Prediction Model

```sql
-- Create IPO valuation prediction model
CREATE OR REPLACE MODEL `ipo_valuation.dcf_valuation_model`
OPTIONS(
  model_type='linear_reg',
  input_label_cols=['enterprise_value'],
  data_split_method='seq',
  data_split_col='fiscal_quarter',
  data_split_eval_fraction=0.2
) AS
SELECT
  -- Financial metrics
  revenue,
  revenue_growth_rate,
  gross_profit_margin,
  operating_margin,
  net_margin,
  
  -- Cash flow metrics
  operating_cash_flow,
  free_cash_flow,
  cash_flow_growth_rate,
  
  -- Balance sheet metrics
  total_assets,
  total_debt,
  debt_to_equity_ratio,
  current_ratio,
  
  -- Market metrics
  market_sector,
  company_age,
  employee_count,
  
  -- Economic indicators
  interest_rate,
  market_volatility_index,
  sector_pe_ratio,
  
  -- Target variable
  enterprise_value
  
FROM `ipo_valuation.financial_training_data`
WHERE fiscal_quarter BETWEEN '2020-Q1' AND '2024-Q2';
```

### 2. Revenue Forecasting Model

```sql
-- Time series model for revenue forecasting
CREATE OR REPLACE MODEL `ipo_valuation.revenue_forecast_model`
OPTIONS(
  model_type='arima_plus',
  time_series_timestamp_col='quarter_date',
  time_series_data_col='quarterly_revenue',
  time_series_id_col='company_id',
  auto_arima=TRUE,
  data_frequency='quarterly'
) AS
SELECT
  company_id,
  quarter_date,
  quarterly_revenue,
  
  -- External regressors
  gdp_growth_rate,
  sector_growth_rate,
  consumer_sentiment_index
  
FROM `ipo_valuation.quarterly_financial_data`
WHERE quarter_date BETWEEN '2018-01-01' AND '2024-06-30'
  AND quarterly_revenue IS NOT NULL;
```

### 3. Risk Assessment Model

```sql
-- Classification model for risk assessment
CREATE OR REPLACE MODEL `ipo_valuation.risk_classification_model`
OPTIONS(
  model_type='logistic_reg',
  input_label_cols=['risk_category'],
  class_weights=[('low_risk', 0.3), ('medium_risk', 0.4), ('high_risk', 0.3)]
) AS
SELECT
  -- Financial health indicators
  debt_to_asset_ratio,
  interest_coverage_ratio,
  current_ratio,
  working_capital_ratio,
  
  -- Profitability metrics
  return_on_equity,
  return_on_assets,
  gross_margin_stability,
  
  -- Cash flow indicators
  operating_cash_flow_ratio,
  cash_conversion_cycle,
  
  -- Market factors
  beta,
  market_cap_volatility,
  trading_volume_stability,
  
  -- Industry factors
  industry_concentration,
  regulatory_intensity,
  
  -- Management quality (derived metrics)
  management_tenure,
  board_independence,
  
  -- Target classification
  CASE 
    WHEN risk_score <= 0.3 THEN 'low_risk'
    WHEN risk_score <= 0.7 THEN 'medium_risk'
    ELSE 'high_risk'
  END AS risk_category
  
FROM `ipo_valuation.risk_training_data`;
```

### 4. Comparable Company Analysis Model

```sql
-- Model for comparable company valuation multiples
CREATE OR REPLACE MODEL `ipo_valuation.comparable_analysis_model`
OPTIONS(
  model_type='boosted_tree_regressor',
  input_label_cols=['price_to_sales_ratio'],
  max_iterations=100,
  learn_rate=0.1,
  min_tree_child_weight=10
) AS
SELECT
  -- Company characteristics
  market_sector,
  company_size_category,
  business_model,
  geographic_presence,
  
  -- Financial metrics
  revenue_growth_3yr,
  profit_margin,
  asset_turnover,
  return_on_invested_capital,
  
  -- Market conditions
  market_phase, -- growth, mature, declining
  competitive_intensity,
  regulatory_environment,
  
  -- Innovation metrics
  rd_intensity,
  patent_count,
  technology_adoption_score,
  
  -- Target multiple
  price_to_sales_ratio
  
FROM `ipo_valuation.comparable_companies_data`
WHERE price_to_sales_ratio IS NOT NULL
  AND price_to_sales_ratio BETWEEN 0.1 AND 50; -- Remove outliers
```

## Advanced Time Series Models

### 1. Market Volatility Forecasting

```sql
-- ARIMA model for market volatility prediction
CREATE OR REPLACE MODEL `ipo_valuation.volatility_forecast_model`
OPTIONS(
  model_type='arima_plus',
  time_series_timestamp_col='trading_date',
  time_series_data_col='volatility_index',
  holiday_region='US',
  auto_arima=TRUE,
  data_frequency='daily'
) AS
SELECT
  trading_date,
  volatility_index,
  
  -- Market indicators
  sp500_return,
  nasdaq_return,
  treasury_10yr_yield,
  
  -- Economic indicators
  vix_index,
  economic_uncertainty_index,
  
  -- Sentiment indicators
  fear_greed_index,
  put_call_ratio
  
FROM `ipo_valuation.market_daily_data`
WHERE trading_date BETWEEN '2019-01-01' AND CURRENT_DATE();
```

### 2. IPO Performance Prediction

```sql
-- Predict IPO first-day performance
CREATE OR REPLACE MODEL `ipo_valuation.ipo_performance_model`
OPTIONS(
  model_type='dnn_regressor',
  input_label_cols=['first_day_return'],
  hidden_units=[128, 64, 32],
  dropout=0.2,
  batch_size=32,
  max_iterations=1000
) AS
SELECT
  -- Offering characteristics
  offer_size_millions,
  price_range_midpoint,
  shares_offered_millions,
  overallotment_option_size,
  
  -- Company fundamentals
  revenue_trailing_12m,
  net_income_trailing_12m,
  revenue_growth_rate,
  company_age_years,
  
  -- Market conditions
  market_return_30d,
  sector_return_30d,
  recent_ipo_performance,
  market_volatility_30d,
  
  -- Underwriter quality
  lead_underwriter_rank,
  syndicate_size,
  underwriter_market_share,
  
  -- Investor demand indicators
  subscription_ratio,
  institutional_allocation_pct,
  roadshow_cities_count,
  
  -- Media and sentiment
  media_mentions_count,
  analyst_coverage_count,
  sentiment_score,
  
  -- Target variable
  first_day_return
  
FROM `ipo_valuation.ipo_historical_performance`
WHERE ipo_date BETWEEN '2015-01-01' AND '2023-12-31'
  AND first_day_return IS NOT NULL;
```

## Feature Engineering with SQL

### 1. Financial Ratios and Metrics

```sql
-- Create comprehensive financial features
CREATE OR REPLACE TABLE `ipo_valuation.engineered_features` AS
WITH financial_ratios AS (
  SELECT
    company_id,
    fiscal_quarter,
    
    -- Profitability ratios
    SAFE_DIVIDE(net_income, revenue) AS net_margin,
    SAFE_DIVIDE(gross_profit, revenue) AS gross_margin,
    SAFE_DIVIDE(operating_income, revenue) AS operating_margin,
    SAFE_DIVIDE(ebitda, revenue) AS ebitda_margin,
    
    -- Liquidity ratios
    SAFE_DIVIDE(current_assets, current_liabilities) AS current_ratio,
    SAFE_DIVIDE((current_assets - inventory), current_liabilities) AS quick_ratio,
    SAFE_DIVIDE(cash_and_equivalents, current_liabilities) AS cash_ratio,
    
    -- Leverage ratios
    SAFE_DIVIDE(total_debt, total_equity) AS debt_to_equity,
    SAFE_DIVIDE(total_debt, total_assets) AS debt_to_assets,
    SAFE_DIVIDE(ebitda, interest_expense) AS interest_coverage,
    
    -- Efficiency ratios
    SAFE_DIVIDE(revenue, total_assets) AS asset_turnover,
    SAFE_DIVIDE(revenue, inventory) AS inventory_turnover,
    SAFE_DIVIDE(cost_of_goods_sold, average_inventory) AS inventory_turnover_cogs,
    
    -- Growth metrics (using window functions)
    SAFE_DIVIDE(
      (revenue - LAG(revenue, 4) OVER (PARTITION BY company_id ORDER BY fiscal_quarter)), 
      LAG(revenue, 4) OVER (PARTITION BY company_id ORDER BY fiscal_quarter)
    ) AS revenue_growth_yoy,
    
    SAFE_DIVIDE(
      (net_income - LAG(net_income, 4) OVER (PARTITION BY company_id ORDER BY fiscal_quarter)), 
      LAG(net_income, 4) OVER (PARTITION BY company_id ORDER BY fiscal_quarter)
    ) AS net_income_growth_yoy
    
  FROM `ipo_valuation.quarterly_financials`
),

market_metrics AS (
  SELECT
    company_id,
    fiscal_quarter,
    
    -- Market valuation metrics
    market_cap,
    SAFE_DIVIDE(market_cap, revenue) AS price_to_sales,
    SAFE_DIVIDE(market_cap, net_income) AS price_to_earnings,
    SAFE_DIVIDE(market_cap, book_value) AS price_to_book,
    SAFE_DIVIDE(enterprise_value, ebitda) AS ev_to_ebitda,
    
    -- Volatility metrics
    STDDEV(daily_return) OVER (
      PARTITION BY company_id 
      ORDER BY fiscal_quarter 
      ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
    ) AS volatility_90d,
    
    -- Beta calculation (simplified)
    CORR(daily_return, market_return) OVER (
      PARTITION BY company_id 
      ORDER BY fiscal_quarter 
      ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
    ) AS beta_1yr
    
  FROM `ipo_valuation.market_data`
),

trend_features AS (
  SELECT
    company_id,
    fiscal_quarter,
    
    -- Moving averages
    AVG(revenue) OVER (
      PARTITION BY company_id 
      ORDER BY fiscal_quarter 
      ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) AS revenue_ma_4q,
    
    AVG(net_margin) OVER (
      PARTITION BY company_id 
      ORDER BY fiscal_quarter 
      ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) AS net_margin_ma_8q,
    
    -- Trend indicators
    CASE 
      WHEN revenue > LAG(revenue, 1) OVER (PARTITION BY company_id ORDER BY fiscal_quarter) 
      THEN 1 ELSE 0 
    END AS revenue_increasing_qoq,
    
    -- Stability metrics
    STDDEV(net_margin) OVER (
      PARTITION BY company_id 
      ORDER BY fiscal_quarter 
      ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) AS margin_stability_8q
    
  FROM `ipo_valuation.quarterly_financials`
)

SELECT
  fr.*,
  mm.* EXCEPT (company_id, fiscal_quarter),
  tf.* EXCEPT (company_id, fiscal_quarter)
FROM financial_ratios fr
JOIN market_metrics mm USING (company_id, fiscal_quarter)
JOIN trend_features tf USING (company_id, fiscal_quarter);
```

### 2. Industry and Sector Analysis

```sql
-- Create sector-relative metrics
CREATE OR REPLACE TABLE `ipo_valuation.sector_relative_metrics` AS
WITH sector_benchmarks AS (
  SELECT
    sector,
    fiscal_quarter,
    
    -- Sector averages
    AVG(net_margin) AS sector_avg_net_margin,
    AVG(revenue_growth_yoy) AS sector_avg_growth,
    AVG(debt_to_equity) AS sector_avg_leverage,
    AVG(price_to_sales) AS sector_avg_ps_ratio,
    
    -- Sector percentiles
    PERCENTILE_CONT(net_margin, 0.25) OVER (PARTITION BY sector, fiscal_quarter) AS sector_p25_margin,
    PERCENTILE_CONT(net_margin, 0.75) OVER (PARTITION BY sector, fiscal_quarter) AS sector_p75_margin,
    
    -- Sector volatility
    STDDEV(revenue_growth_yoy) AS sector_growth_volatility
    
  FROM `ipo_valuation.engineered_features` ef
  JOIN `ipo_valuation.company_master` cm ON ef.company_id = cm.company_id
  GROUP BY sector, fiscal_quarter
)

SELECT
  ef.*,
  
  -- Relative performance metrics
  SAFE_DIVIDE(ef.net_margin, sb.sector_avg_net_margin) AS margin_vs_sector,
  SAFE_DIVIDE(ef.revenue_growth_yoy, sb.sector_avg_growth) AS growth_vs_sector,
  SAFE_DIVIDE(ef.price_to_sales, sb.sector_avg_ps_ratio) AS valuation_vs_sector,
  
  -- Sector ranking
  PERCENT_RANK() OVER (
    PARTITION BY cm.sector, ef.fiscal_quarter 
    ORDER BY ef.net_margin
  ) AS margin_sector_percentile,
  
  PERCENT_RANK() OVER (
    PARTITION BY cm.sector, ef.fiscal_quarter 
    ORDER BY ef.revenue_growth_yoy
  ) AS growth_sector_percentile
  
FROM `ipo_valuation.engineered_features` ef
JOIN `ipo_valuation.company_master` cm ON ef.company_id = cm.company_id
JOIN sector_benchmarks sb ON cm.sector = sb.sector AND ef.fiscal_quarter = sb.fiscal_quarter;
```

## Model Evaluation and Validation

### 1. Model Performance Metrics

```sql
-- Evaluate valuation model performance
WITH model_evaluation AS (
  SELECT
    predicted_enterprise_value,
    enterprise_value AS actual_enterprise_value,
    company_id,
    fiscal_quarter
  FROM ML.PREDICT(
    MODEL `ipo_valuation.dcf_valuation_model`,
    (SELECT * FROM `ipo_valuation.validation_data`)
  )
)

SELECT
  -- Basic metrics
  COUNT(*) AS total_predictions,
  CORR(predicted_enterprise_value, actual_enterprise_value) AS correlation,
  
  -- Error metrics
  AVG(ABS(predicted_enterprise_value - actual_enterprise_value)) AS mae,
  SQRT(AVG(POW(predicted_enterprise_value - actual_enterprise_value, 2))) AS rmse,
  
  -- Percentage errors
  AVG(ABS((predicted_enterprise_value - actual_enterprise_value) / actual_enterprise_value)) * 100 AS mape,
  
  -- Directional accuracy
  AVG(
    CASE 
      WHEN SIGN(predicted_enterprise_value - LAG(predicted_enterprise_value) OVER (PARTITION BY company_id ORDER BY fiscal_quarter)) = 
           SIGN(actual_enterprise_value - LAG(actual_enterprise_value) OVER (PARTITION BY company_id ORDER BY fiscal_quarter))
      THEN 1.0 ELSE 0.0 
    END
  ) AS directional_accuracy,
  
  -- Performance by quartile
  AVG(
    CASE 
      WHEN NTILE(4) OVER (ORDER BY actual_enterprise_value) = 1 
      THEN ABS((predicted_enterprise_value - actual_enterprise_value) / actual_enterprise_value)
    END
  ) * 100 AS mape_q1,
  
  AVG(
    CASE 
      WHEN NTILE(4) OVER (ORDER BY actual_enterprise_value) = 4 
      THEN ABS((predicted_enterprise_value - actual_enterprise_value) / actual_enterprise_value)
    END
  ) * 100 AS mape_q4

FROM model_evaluation;
```

### 2. Feature Importance Analysis

```sql
-- Analyze feature importance for boosted tree models
SELECT
  feature,
  importance,
  RANK() OVER (ORDER BY importance DESC) AS importance_rank
FROM ML.FEATURE_IMPORTANCE(MODEL `ipo_valuation.comparable_analysis_model`)
ORDER BY importance DESC;
```

### 3. Model Explainability

```sql
-- Generate explanations for individual predictions
SELECT
  company_id,
  predicted_price_to_sales_ratio,
  
  -- Feature attributions
  feature_attributions.feature AS contributing_feature,
  feature_attributions.attribution AS feature_contribution
  
FROM ML.EXPLAIN_PREDICT(
  MODEL `ipo_valuation.comparable_analysis_model`,
  (SELECT * FROM `ipo_valuation.explanation_data` WHERE company_id = 'TARGET_COMPANY'),
  STRUCT(3 AS top_k_features)
)
ORDER BY ABS(feature_attributions.attribution) DESC;
```

## Advanced Analytics Queries

### 1. Market Timing Analysis

```sql
-- Analyze optimal IPO timing based on market conditions
WITH market_performance AS (
  SELECT
    ipo_quarter,
    COUNT(*) AS ipo_count,
    AVG(first_day_return) AS avg_first_day_return,
    AVG(six_month_return) AS avg_six_month_return,
    STDDEV(first_day_return) AS first_day_volatility,
    
    -- Market condition indicators
    AVG(market_volatility_30d) AS avg_market_volatility,
    AVG(sector_performance_30d) AS avg_sector_performance,
    AVG(treasury_10yr_yield) AS avg_interest_rate
    
  FROM `ipo_valuation.ipo_historical_performance`
  WHERE ipo_date BETWEEN '2015-01-01' AND '2023-12-31'
  GROUP BY ipo_quarter
),

optimal_timing AS (
  SELECT
    *,
    -- Score based on returns and volatility
    (avg_six_month_return / first_day_volatility) AS risk_adjusted_return,
    
    -- Market condition score
    CASE 
      WHEN avg_market_volatility < 20 AND avg_sector_performance > 0.05 THEN 'Favorable'
      WHEN avg_market_volatility > 30 OR avg_sector_performance < -0.05 THEN 'Unfavorable'
      ELSE 'Neutral'
    END AS market_condition
    
  FROM market_performance
)

SELECT
  ipo_quarter,
  ipo_count,
  ROUND(avg_first_day_return * 100, 2) AS avg_first_day_return_pct,
  ROUND(avg_six_month_return * 100, 2) AS avg_six_month_return_pct,
  ROUND(risk_adjusted_return, 3) AS risk_adjusted_return,
  market_condition,
  
  -- Ranking
  RANK() OVER (ORDER BY risk_adjusted_return DESC) AS timing_rank
  
FROM optimal_timing
ORDER BY risk_adjusted_return DESC;
```

### 2. Sector Analysis Dashboard Query

```sql
-- Comprehensive sector analysis for IPO readiness
WITH sector_metrics AS (
  SELECT
    sector,
    COUNT(DISTINCT company_id) AS companies_tracked,
    
    -- Financial health metrics
    AVG(current_ratio) AS avg_liquidity,
    AVG(debt_to_equity) AS avg_leverage,
    AVG(net_margin) AS avg_profitability,
    AVG(revenue_growth_yoy) AS avg_growth,
    
    -- Market metrics
    AVG(price_to_sales) AS avg_valuation_multiple,
    AVG(volatility_90d) AS avg_volatility,
    
    -- IPO activity
    COUNT(
      CASE WHEN ipo_status = 'completed' 
           AND ipo_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH)
      THEN 1 END
    ) AS recent_ipo_count,
    
    AVG(
      CASE WHEN ipo_status = 'completed'
           AND ipo_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH)
      THEN first_day_return END
    ) AS avg_ipo_performance
    
  FROM `ipo_valuation.sector_relative_metrics` srm
  JOIN `ipo_valuation.company_master` cm ON srm.company_id = cm.company_id
  LEFT JOIN `ipo_valuation.ipo_activity` ia ON cm.company_id = ia.company_id
  WHERE srm.fiscal_quarter = (SELECT MAX(fiscal_quarter) FROM `ipo_valuation.sector_relative_metrics`)
  GROUP BY sector
),

sector_rankings AS (
  SELECT
    *,
    -- Composite scores
    (avg_growth + avg_profitability - avg_volatility) AS growth_stability_score,
    (avg_ipo_performance + (1/avg_volatility)) AS ipo_attractiveness_score,
    
    -- Rankings
    RANK() OVER (ORDER BY avg_growth DESC) AS growth_rank,
    RANK() OVER (ORDER BY avg_profitability DESC) AS profitability_rank,
    RANK() OVER (ORDER BY avg_ipo_performance DESC) AS ipo_performance_rank
    
  FROM sector_metrics
)

SELECT
  sector,
  companies_tracked,
  
  -- Formatted metrics
  ROUND(avg_liquidity, 2) AS liquidity_ratio,
  ROUND(avg_leverage, 2) AS debt_equity_ratio,
  ROUND(avg_profitability * 100, 1) AS profit_margin_pct,
  ROUND(avg_growth * 100, 1) AS revenue_growth_pct,
  ROUND(avg_valuation_multiple, 1) AS ps_multiple,
  
  -- IPO metrics
  recent_ipo_count,
  ROUND(COALESCE(avg_ipo_performance * 100, 0), 1) AS avg_ipo_return_pct,
  
  -- Scores and rankings
  ROUND(growth_stability_score, 3) AS growth_stability_score,
  growth_rank,
  profitability_rank,
  ipo_performance_rank,
  
  -- Overall assessment
  CASE 
    WHEN growth_rank <= 3 AND profitability_rank <= 3 THEN 'High Potential'
    WHEN growth_rank <= 6 AND profitability_rank <= 6 THEN 'Moderate Potential'
    ELSE 'Lower Potential'
  END AS ipo_readiness_assessment
  
FROM sector_rankings
ORDER BY growth_stability_score DESC;
```

## Model Deployment and Serving

### 1. Real-time Prediction API

```sql
-- Create view for real-time valuation predictions
CREATE OR REPLACE VIEW `ipo_valuation.realtime_valuations` AS
SELECT
  company_id,
  predicted_enterprise_value,
  
  -- Confidence intervals (if model supports)
  predicted_enterprise_value_lower_bound,
  predicted_enterprise_value_upper_bound,
  
  -- Contributing factors
  feature_attributions,
  
  -- Metadata
  model_version,
  prediction_timestamp
  
FROM ML.PREDICT(
  MODEL `ipo_valuation.dcf_valuation_model`,
  (
    SELECT * FROM `ipo_valuation.current_financial_data`
    WHERE data_freshness_hours <= 24
  )
);
```

### 2. Batch Prediction Pipeline

```sql
-- Scheduled batch predictions
CREATE OR REPLACE TABLE `ipo_valuation.monthly_valuations_$(FORMAT_DATE('%Y%m', CURRENT_DATE()))` AS
SELECT
  mp.*,
  cd.company_name,
  cd.sector,
  cd.ipo_status,
  
  -- Prediction metadata
  CURRENT_TIMESTAMP() AS prediction_run_timestamp,
  'monthly_batch_v1.0' AS prediction_job_id
  
FROM ML.PREDICT(
  MODEL `ipo_valuation.dcf_valuation_model`,
  (
    SELECT ef.*, cm.sector 
    FROM `ipo_valuation.engineered_features` ef
    JOIN `ipo_valuation.company_master` cm ON ef.company_id = cm.company_id
    WHERE ef.fiscal_quarter = (SELECT MAX(fiscal_quarter) FROM `ipo_valuation.engineered_features`)
      AND cm.ipo_status IN ('pre_ipo', 'filed_s1', 'roadshow')
  )
) mp
JOIN `ipo_valuation.company_master` cd ON mp.company_id = cd.company_id;
```

### 3. Model Monitoring

```sql
-- Model drift detection
WITH recent_predictions AS (
  SELECT
    prediction_date,
    AVG(predicted_enterprise_value) AS avg_prediction,
    STDDEV(predicted_enterprise_value) AS prediction_stddev,
    COUNT(*) AS prediction_count
    
  FROM `ipo_valuation.prediction_history`
  WHERE prediction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
  GROUP BY prediction_date
),

historical_baseline AS (
  SELECT
    AVG(predicted_enterprise_value) AS baseline_avg_prediction,
    STDDEV(predicted_enterprise_value) AS baseline_stddev
    
  FROM `ipo_valuation.prediction_history`
  WHERE prediction_date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 365 DAY) 
                            AND DATE_SUB(CURRENT_DATE(), INTERVAL 60 DAY)
),

drift_analysis AS (
  SELECT
    rp.prediction_date,
    rp.avg_prediction,
    hb.baseline_avg_prediction,
    
    -- Drift metrics
    ABS(rp.avg_prediction - hb.baseline_avg_prediction) / hb.baseline_avg_prediction AS mean_drift_ratio,
    ABS(rp.prediction_stddev - hb.baseline_stddev) / hb.baseline_stddev AS variance_drift_ratio,
    
    -- Drift flags
    CASE 
      WHEN ABS(rp.avg_prediction - hb.baseline_avg_prediction) / hb.baseline_avg_prediction > 0.2 
      THEN TRUE ELSE FALSE 
    END AS significant_mean_drift,
    
    CASE 
      WHEN ABS(rp.prediction_stddev - hb.baseline_stddev) / hb.baseline_stddev > 0.3 
      THEN TRUE ELSE FALSE 
    END AS significant_variance_drift
    
  FROM recent_predictions rp
  CROSS JOIN historical_baseline hb
)

SELECT
  prediction_date,
  ROUND(mean_drift_ratio * 100, 2) AS mean_drift_pct,
  ROUND(variance_drift_ratio * 100, 2) AS variance_drift_pct,
  significant_mean_drift,
  significant_variance_drift,
  
  -- Alert level
  CASE 
    WHEN significant_mean_drift OR significant_variance_drift THEN 'HIGH'
    WHEN mean_drift_ratio > 0.1 OR variance_drift_ratio > 0.15 THEN 'MEDIUM'
    ELSE 'LOW'
  END AS drift_alert_level
  
FROM drift_analysis
ORDER BY prediction_date DESC;
```

## Performance Optimization

### 1. Query Optimization

```sql
-- Optimized feature engineering with partitioning
CREATE OR REPLACE TABLE `ipo_valuation.optimized_features`
PARTITION BY DATE_TRUNC(fiscal_quarter, QUARTER)
CLUSTER BY company_id, sector AS

SELECT
  company_id,
  fiscal_quarter,
  sector,
  
  -- Use APPROX functions for large datasets
  APPROX_QUANTILES(revenue, 100)[OFFSET(50)] OVER (
    PARTITION BY sector 
    ORDER BY fiscal_quarter 
    ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
  ) AS sector_median_revenue_12q,
  
  -- Optimized window functions
  revenue / NULLIF(LAG(revenue, 4) OVER (
    PARTITION BY company_id 
    ORDER BY fiscal_quarter
  ), 0) - 1 AS revenue_growth_yoy_optimized

FROM `ipo_valuation.quarterly_financials`
WHERE fiscal_quarter >= '2015-Q1';
```

### 2. Materialized Views for Performance

```sql
-- Create materialized view for frequent queries
CREATE MATERIALIZED VIEW `ipo_valuation.company_latest_metrics`
PARTITION BY sector
CLUSTER BY market_cap_category AS

SELECT
  company_id,
  company_name,
  sector,
  
  -- Latest financial metrics
  revenue,
  net_margin,
  debt_to_equity,
  revenue_growth_yoy,
  
  -- Market metrics
  market_cap,
  CASE 
    WHEN market_cap < 2000000000 THEN 'Small Cap'
    WHEN market_cap < 10000000000 THEN 'Mid Cap'
    ELSE 'Large Cap'
  END AS market_cap_category,
  
  -- Rankings within sector
  RANK() OVER (PARTITION BY sector ORDER BY revenue_growth_yoy DESC) AS growth_rank_in_sector,
  
  fiscal_quarter
  
FROM `ipo_valuation.sector_relative_metrics` 
WHERE fiscal_quarter = (SELECT MAX(fiscal_quarter) FROM `ipo_valuation.sector_relative_metrics`);
```

## Cost Optimization

### 1. Query Cost Analysis

```sql
-- Analyze query costs and optimize
SELECT
  query,
  total_slot_ms,
  total_bytes_processed,
  total_bytes_billed,
  
  -- Cost estimation (approximate)
  (total_bytes_billed / 1024 / 1024 / 1024 / 1024) * 5 AS estimated_cost_usd,
  
  creation_time
  
FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
WHERE statement_type = 'SELECT'
  AND job_type = 'QUERY'
  AND creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
ORDER BY total_bytes_billed DESC
LIMIT 20;
```

### 2. Storage Optimization

```sql
-- Implement table lifecycle management
ALTER TABLE `ipo_valuation.prediction_history`
SET OPTIONS (
  partition_expiration_days = 365,
  description = 'Historical predictions with 1-year retention'
);

-- Create summary tables for historical data
CREATE OR REPLACE TABLE `ipo_valuation.monthly_summary`
PARTITION BY DATE_TRUNC(month_date, MONTH) AS

SELECT
  DATE_TRUNC(fiscal_quarter, MONTH) AS month_date,
  sector,
  
  -- Aggregated metrics
  COUNT(DISTINCT company_id) AS company_count,
  AVG(revenue) AS avg_sector_revenue,
  AVG(net_margin) AS avg_sector_margin,
  STDDEV(revenue_growth_yoy) AS growth_volatility
  
FROM `ipo_valuation.quarterly_financials`
GROUP BY month_date, sector;
```

## Next Steps

1. **Model Enhancement**: Implement ensemble methods combining multiple BQML models
2. **Real-time Integration**: Set up streaming data pipelines for real-time features
3. **AutoML Integration**: Combine BQML with Vertex AI AutoML for hybrid approaches
4. **Advanced Analytics**: Implement more sophisticated time series and forecasting models
5. **Monitoring Dashboard**: Create comprehensive monitoring and alerting system

## Related Documentation

- [Vertex AI Integration](../vertex-ai/README.md)
- [Natural Language AI](../natural-language/README.md)
- [Cost Optimization](../cost-optimization/README.md)
- [Authentication Setup](../authentication/README.md)