-- BigQuery ML Models for IPO Valuation Platform

-- Create dataset for ML models
CREATE SCHEMA IF NOT EXISTS `ipo_valuation.ml_models`
OPTIONS (
  description = 'Machine Learning models for IPO valuation',
  location = 'US'
);

-- 1. DCF Valuation Prediction Model
CREATE OR REPLACE MODEL `ipo_valuation.ml_models.dcf_prediction_model`
OPTIONS (
  model_type = 'BOOSTED_TREE_REGRESSOR',
  input_label_cols = ['dcf_value'],
  auto_class_weights = TRUE,
  max_iterations = 50,
  early_stop = TRUE,
  min_rel_progress = 0.005,
  data_split_method = 'AUTO_SPLIT',
  data_split_eval_fraction = 0.2,
  data_split_col = 'data_split'
) AS
SELECT
  -- Financial Metrics
  revenue_growth_rate,
  ebitda_margin,
  capex_percentage,
  working_capital_change,
  tax_rate,
  
  -- Market Metrics
  wacc,
  terminal_growth_rate,
  market_risk_premium,
  beta,
  
  -- Company Characteristics
  company_age,
  employee_count,
  market_sector,
  geographic_focus,
  
  -- Economic Indicators
  interest_rate_10y,
  gdp_growth_rate,
  inflation_rate,
  
  -- Target Variable
  dcf_value,
  
  -- Data split column for training/validation
  CASE 
    WHEN RAND() < 0.8 THEN 'TRAIN'
    ELSE 'EVAL'
  END as data_split
  
FROM `ipo_valuation.financial_data.company_financials`
WHERE dcf_value IS NOT NULL
  AND revenue_growth_rate BETWEEN -1.0 AND 5.0
  AND ebitda_margin BETWEEN -0.5 AND 1.0;

-- 2. Market Multiple Prediction Model
CREATE OR REPLACE MODEL `ipo_valuation.ml_models.market_multiple_model`
OPTIONS (
  model_type = 'LINEAR_REG',
  input_label_cols = ['pe_multiple'],
  l1_reg = 0.1,
  l2_reg = 0.1,
  optimize_strategy = 'BATCH_GRADIENT_DESCENT',
  learn_rate_strategy = 'CONSTANT',
  learn_rate = 0.01,
  max_iterations = 100
) AS
SELECT
  -- Financial Ratios
  revenue_growth_rate,
  profit_margin,
  roe,
  roa,
  debt_to_equity,
  current_ratio,
  
  -- Industry Metrics
  industry_avg_pe,
  industry_avg_pb,
  industry_growth_rate,
  
  -- Market Conditions
  market_volatility,
  sector_performance,
  
  -- Company Size
  LOG(market_cap) as log_market_cap,
  LOG(revenue) as log_revenue,
  
  -- Categorical Features (one-hot encoded)
  CASE WHEN market_sector = 'Technology' THEN 1 ELSE 0 END as tech_sector,
  CASE WHEN market_sector = 'Healthcare' THEN 1 ELSE 0 END as healthcare_sector,
  CASE WHEN market_sector = 'Finance' THEN 1 ELSE 0 END as finance_sector,
  CASE WHEN market_sector = 'Consumer' THEN 1 ELSE 0 END as consumer_sector,
  CASE WHEN market_sector = 'Industrial' THEN 1 ELSE 0 END as industrial_sector,
  
  -- Target Variable
  pe_multiple
  
FROM `ipo_valuation.market_data.sector_multiples`
WHERE pe_multiple IS NOT NULL
  AND pe_multiple BETWEEN 5 AND 100;

-- 3. Risk Assessment Classification Model
CREATE OR REPLACE MODEL `ipo_valuation.ml_models.risk_classification_model`
OPTIONS (
  model_type = 'LOGISTIC_REG',
  input_label_cols = ['risk_category'],
  auto_class_weights = TRUE,
  class_weights = [('LOW', 1.0), ('MEDIUM', 1.2), ('HIGH', 1.5), ('VERY_HIGH', 2.0)]
) AS
SELECT
  -- Financial Health Indicators
  debt_to_equity,
  current_ratio,
  interest_coverage_ratio,
  cash_burn_rate,
  
  -- Profitability Metrics
  gross_margin,
  operating_margin,
  net_margin,
  
  -- Growth Metrics
  revenue_growth_volatility,
  customer_concentration,
  market_share,
  
  -- Operational Risk Factors
  regulatory_risk_score,
  competitive_risk_score,
  management_experience_years,
  
  -- Market Risk Factors
  market_beta,
  correlation_with_market,
  
  -- ESG Risk Factors
  esg_score,
  governance_score,
  
  -- External Risk Factors
  economic_sensitivity,
  political_risk_score,
  
  -- Target Classification
  risk_category
  
FROM `ipo_valuation.risk_data.company_risk_profiles`
WHERE risk_category IN ('LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH');

-- 4. Time Series Forecasting for Market Trends
CREATE OR REPLACE MODEL `ipo_valuation.ml_models.market_trend_forecast`
OPTIONS (
  model_type = 'ARIMA_PLUS',
  time_series_timestamp_col = 'date',
  time_series_data_col = 'market_index_value',
  time_series_id_col = 'market_sector',
  auto_arima = TRUE,
  data_frequency = 'DAILY',
  horizon = 30,
  holiday_region = 'US'
) AS
SELECT
  date,
  market_sector,
  market_index_value,
  -- External regressors
  interest_rate_10y,
  vix_index,
  gdp_growth_rate
FROM `ipo_valuation.market_data.daily_indices`
WHERE date >= '2018-01-01';

-- 5. Sentiment Analysis for IPO Documents
CREATE OR REPLACE MODEL `ipo_valuation.ml_models.document_sentiment_model`
OPTIONS (
  model_type = 'DNN_CLASSIFIER',
  input_label_cols = ['sentiment_category'],
  hidden_units = [128, 64, 32],
  activation_fn = 'RELU',
  dropout = 0.3,
  batch_size = 32,
  max_iterations = 100,
  early_stop = TRUE,
  learn_rate = 0.001
) AS
SELECT
  -- Text features (TF-IDF vectors)
  ML.FEATURE_CROSS(
    STRUCT(
      ML.BUCKETIZE(word_count, [10, 50, 100, 500]) as word_count_bucket,
      ML.BUCKETIZE(sentence_count, [5, 15, 30, 60]) as sentence_count_bucket
    )
  ) as text_features,
  
  -- Financial keyword density
  risk_keyword_density,
  growth_keyword_density,
  competitive_keyword_density,
  
  -- Document metadata
  document_section,
  document_length,
  
  -- Target sentiment
  sentiment_category
  
FROM `ipo_valuation.document_data.processed_documents`
WHERE sentiment_category IS NOT NULL;

-- 6. Anomaly Detection for Financial Data
CREATE OR REPLACE MODEL `ipo_valuation.ml_models.financial_anomaly_detector`
OPTIONS (
  model_type = 'AUTOENCODER',
  hidden_units = [10, 5, 10],
  activation_fn = 'RELU',
  l1_reg = 0.001,
  l2_reg = 0.001
) AS
SELECT
  -- Normalize financial metrics for anomaly detection
  (revenue - avg_revenue) / stddev_revenue as norm_revenue,
  (profit - avg_profit) / stddev_profit as norm_profit,
  (cash_flow - avg_cash_flow) / stddev_cash_flow as norm_cash_flow,
  (assets - avg_assets) / stddev_assets as norm_assets,
  (liabilities - avg_liabilities) / stddev_liabilities as norm_liabilities
  
FROM (
  SELECT
    revenue,
    profit,
    cash_flow,
    assets,
    liabilities,
    AVG(revenue) OVER() as avg_revenue,
    STDDEV(revenue) OVER() as stddev_revenue,
    AVG(profit) OVER() as avg_profit,
    STDDEV(profit) OVER() as stddev_profit,
    AVG(cash_flow) OVER() as avg_cash_flow,
    STDDEV(cash_flow) OVER() as stddev_cash_flow,
    AVG(assets) OVER() as avg_assets,
    STDDEV(assets) OVER() as stddev_assets,
    AVG(liabilities) OVER() as avg_liabilities,
    STDDEV(liabilities) OVER() as stddev_liabilities
  FROM `ipo_valuation.financial_data.quarterly_reports`
);

-- Model Evaluation and Performance Monitoring
-- 1. Evaluate DCF Prediction Model
SELECT
  mean_absolute_error,
  mean_squared_error,
  mean_squared_log_error,
  median_absolute_error,
  r2_score,
  explained_variance
FROM ML.EVALUATE(
  MODEL `ipo_valuation.ml_models.dcf_prediction_model`
);

-- 2. Feature Importance for Risk Classification
SELECT
  feature,
  importance
FROM ML.GLOBAL_EXPLAIN(
  MODEL `ipo_valuation.ml_models.risk_classification_model`
)
ORDER BY importance DESC;

-- 3. Hyperparameter Tuning Results
SELECT
  trial_id,
  hyperparameters,
  objective_value
FROM ML.TRIAL_INFO(
  MODEL `ipo_valuation.ml_models.dcf_prediction_model`
)
ORDER BY objective_value ASC;

-- Batch Prediction Functions
-- 1. Batch DCF Predictions
CREATE OR REPLACE TABLE `ipo_valuation.predictions.dcf_batch_predictions` AS
SELECT
  company_id,
  prediction as predicted_dcf_value,
  prediction_confidence_interval_lower_bound,
  prediction_confidence_interval_upper_bound
FROM ML.PREDICT(
  MODEL `ipo_valuation.ml_models.dcf_prediction_model`,
  (
    SELECT
      company_id,
      revenue_growth_rate,
      ebitda_margin,
      capex_percentage,
      working_capital_change,
      tax_rate,
      wacc,
      terminal_growth_rate,
      market_risk_premium,
      beta,
      company_age,
      employee_count,
      market_sector,
      geographic_focus,
      interest_rate_10y,
      gdp_growth_rate,
      inflation_rate
    FROM `ipo_valuation.financial_data.company_financials`
    WHERE prediction_date = CURRENT_DATE()
  )
);

-- 2. Market Trend Forecast
CREATE OR REPLACE TABLE `ipo_valuation.predictions.market_forecasts` AS
SELECT
  forecast_timestamp,
  market_sector,
  forecast_value,
  confidence_interval_lower_bound,
  confidence_interval_upper_bound,
  confidence_level
FROM ML.FORECAST(
  MODEL `ipo_valuation.ml_models.market_trend_forecast`,
  STRUCT(30 AS horizon, 0.95 AS confidence_level)
);

-- Model Refresh and Retraining Schedule
-- These would typically be scheduled using Cloud Composer or Cloud Scheduler

-- Weekly model retraining
CREATE OR REPLACE PROCEDURE `ipo_valuation.ml_models.retrain_models`()
BEGIN
  -- Retrain DCF model with latest data
  CREATE OR REPLACE MODEL `ipo_valuation.ml_models.dcf_prediction_model_new`
  OPTIONS (
    model_type = 'BOOSTED_TREE_REGRESSOR',
    input_label_cols = ['dcf_value'],
    auto_class_weights = TRUE,
    max_iterations = 50
  ) AS
  SELECT * FROM `ipo_valuation.financial_data.company_financials`
  WHERE update_timestamp >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY);
  
  -- Evaluate new model performance
  DECLARE model_performance FLOAT64;
  SET model_performance = (
    SELECT r2_score 
    FROM ML.EVALUATE(MODEL `ipo_valuation.ml_models.dcf_prediction_model_new`)
  );
  
  -- Replace model if performance is acceptable
  IF model_performance > 0.8 THEN
    DROP MODEL `ipo_valuation.ml_models.dcf_prediction_model`;
    ALTER MODEL `ipo_valuation.ml_models.dcf_prediction_model_new`
    RENAME TO `ipo_valuation.ml_models.dcf_prediction_model`;
  END IF;
END;