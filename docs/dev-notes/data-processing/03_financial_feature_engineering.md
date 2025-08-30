# Financial Feature Engineering for ML Models

This document provides comprehensive guidance for creating and transforming financial features specifically designed for IPO valuation machine learning models.

## 📊 Overview

Financial feature engineering transforms raw financial data into meaningful features that machine learning models can effectively use for valuation predictions. This includes financial ratios, technical indicators, time-based features, and interaction terms.

## 🎯 Key Objectives

- **ML-Ready Features**: Create features optimized for machine learning algorithms
- **Domain Knowledge**: Incorporate financial domain expertise into feature design
- **Temporal Patterns**: Capture time-based relationships and trends
- **Cross-Sectional Comparisons**: Enable peer company analysis
- **Risk Indicators**: Quantify various types of financial risk

## 1. Financial Ratio Calculations and Transformations

### 1.1 Core Financial Ratios

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

@dataclass
class FinancialStatement:
    """Structure for financial statement data."""
    revenue: float
    cost_of_goods_sold: float
    gross_profit: float
    operating_expenses: float
    operating_income: float
    interest_expense: float
    net_income: float
    total_assets: float
    current_assets: float
    total_liabilities: float
    current_liabilities: float
    shareholders_equity: float
    cash_and_equivalents: float
    inventory: float
    accounts_receivable: float
    long_term_debt: float
    shares_outstanding: float

class FinancialRatioCalculator:
    """
    Calculate comprehensive financial ratios for IPO valuation analysis.
    
    Includes profitability, liquidity, leverage, efficiency, and market ratios.
    """
    
    def __init__(self, handle_missing: str = 'interpolate'):
        self.handle_missing = handle_missing
        self.logger = logging.getLogger(__name__)
        
    def calculate_all_ratios(self, financial_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all financial ratios from financial statement data.
        
        Args:
            financial_data: DataFrame with financial statement items
            
        Returns:
            DataFrame with calculated financial ratios
        """
        ratios_df = financial_data.copy()
        
        # Profitability Ratios
        ratios_df = self._calculate_profitability_ratios(ratios_df)
        
        # Liquidity Ratios
        ratios_df = self._calculate_liquidity_ratios(ratios_df)
        
        # Leverage Ratios
        ratios_df = self._calculate_leverage_ratios(ratios_df)
        
        # Efficiency Ratios
        ratios_df = self._calculate_efficiency_ratios(ratios_df)
        
        # Market Ratios (if market data available)
        if 'market_price' in ratios_df.columns:
            ratios_df = self._calculate_market_ratios(ratios_df)
        
        # Growth Ratios
        ratios_df = self._calculate_growth_ratios(ratios_df)
        
        # Handle missing values
        if self.handle_missing != 'keep':
            ratios_df = self._handle_missing_ratios(ratios_df)
        
        return ratios_df
    
    def _calculate_profitability_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate profitability ratios."""
        # Gross Profit Margin
        df['gross_profit_margin'] = self._safe_divide(
            df['gross_profit'], df['revenue']
        )
        
        # Operating Profit Margin
        df['operating_profit_margin'] = self._safe_divide(
            df['operating_income'], df['revenue']
        )
        
        # Net Profit Margin
        df['net_profit_margin'] = self._safe_divide(
            df['net_income'], df['revenue']
        )
        
        # Return on Assets (ROA)
        df['roa'] = self._safe_divide(
            df['net_income'], df['total_assets']
        )
        
        # Return on Equity (ROE)
        df['roe'] = self._safe_divide(
            df['net_income'], df['shareholders_equity']
        )
        
        # Return on Invested Capital (ROIC)
        invested_capital = df['shareholders_equity'] + df.get('long_term_debt', 0)
        df['roic'] = self._safe_divide(
            df['operating_income'] * (1 - 0.25),  # Assume 25% tax rate
            invested_capital
        )
        
        # EBITDA Margin (approximation)
        # Note: This is simplified - actual EBITDA would need depreciation/amortization
        ebitda_approx = df['operating_income'] + df.get('depreciation', 0)
        df['ebitda_margin'] = self._safe_divide(ebitda_approx, df['revenue'])
        
        return df
    
    def _calculate_liquidity_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity ratios."""
        # Current Ratio
        df['current_ratio'] = self._safe_divide(
            df['current_assets'], df['current_liabilities']
        )
        
        # Quick Ratio (Acid Test)
        quick_assets = df['current_assets'] - df.get('inventory', 0)
        df['quick_ratio'] = self._safe_divide(
            quick_assets, df['current_liabilities']
        )
        
        # Cash Ratio
        df['cash_ratio'] = self._safe_divide(
            df['cash_and_equivalents'], df['current_liabilities']
        )
        
        # Operating Cash Flow Ratio (if cash flow data available)
        if 'operating_cash_flow' in df.columns:
            df['operating_cash_flow_ratio'] = self._safe_divide(
                df['operating_cash_flow'], df['current_liabilities']
            )
        
        return df
    
    def _calculate_leverage_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate leverage/solvency ratios."""
        # Debt-to-Equity Ratio
        total_debt = df.get('long_term_debt', 0) + df.get('short_term_debt', 0)
        df['debt_to_equity'] = self._safe_divide(
            total_debt, df['shareholders_equity']
        )
        
        # Debt-to-Assets Ratio
        df['debt_to_assets'] = self._safe_divide(
            total_debt, df['total_assets']
        )
        
        # Equity Ratio
        df['equity_ratio'] = self._safe_divide(
            df['shareholders_equity'], df['total_assets']
        )
        
        # Interest Coverage Ratio
        df['interest_coverage'] = self._safe_divide(
            df['operating_income'], df.get('interest_expense', 1)
        )
        
        # Times Interest Earned
        ebit = df['operating_income']  # Approximation
        df['times_interest_earned'] = self._safe_divide(
            ebit, df.get('interest_expense', 1)
        )
        
        return df
    
    def _calculate_efficiency_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate efficiency/activity ratios."""
        # Asset Turnover
        df['asset_turnover'] = self._safe_divide(
            df['revenue'], df['total_assets']
        )
        
        # Inventory Turnover
        if 'inventory' in df.columns:
            df['inventory_turnover'] = self._safe_divide(
                df['cost_of_goods_sold'], df['inventory']
            )
            
            # Days Inventory Outstanding
            df['days_inventory_outstanding'] = self._safe_divide(
                365, df['inventory_turnover']
            )
        
        # Receivables Turnover
        if 'accounts_receivable' in df.columns:
            df['receivables_turnover'] = self._safe_divide(
                df['revenue'], df['accounts_receivable']
            )
            
            # Days Sales Outstanding
            df['days_sales_outstanding'] = self._safe_divide(
                365, df['receivables_turnover']
            )
        
        # Working Capital Turnover
        working_capital = df['current_assets'] - df['current_liabilities']
        df['working_capital_turnover'] = self._safe_divide(
            df['revenue'], working_capital
        )
        
        return df
    
    def _calculate_market_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market-based ratios."""
        if 'market_price' not in df.columns or 'shares_outstanding' not in df.columns:
            return df
        
        # Market Capitalization
        df['market_cap'] = df['market_price'] * df['shares_outstanding']
        
        # Price-to-Earnings Ratio
        earnings_per_share = self._safe_divide(df['net_income'], df['shares_outstanding'])
        df['pe_ratio'] = self._safe_divide(df['market_price'], earnings_per_share)
        
        # Price-to-Book Ratio
        book_value_per_share = self._safe_divide(
            df['shareholders_equity'], df['shares_outstanding']
        )
        df['pb_ratio'] = self._safe_divide(df['market_price'], book_value_per_share)
        
        # Price-to-Sales Ratio
        sales_per_share = self._safe_divide(df['revenue'], df['shares_outstanding'])
        df['ps_ratio'] = self._safe_divide(df['market_price'], sales_per_share)
        
        # Enterprise Value ratios (if debt data available)
        if 'long_term_debt' in df.columns:
            enterprise_value = (df['market_cap'] + 
                              df.get('long_term_debt', 0) - 
                              df['cash_and_equivalents'])
            
            df['ev_revenue'] = self._safe_divide(enterprise_value, df['revenue'])
            df['ev_ebitda'] = self._safe_divide(
                enterprise_value, 
                df['operating_income'] + df.get('depreciation', 0)
            )
        
        return df
    
    def _calculate_growth_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate growth rates (requires time series data)."""
        if df.index.dtype != 'datetime64[ns]':
            # Try to convert index to datetime
            try:
                df.index = pd.to_datetime(df.index)
            except:
                self.logger.warning("Cannot calculate growth ratios without datetime index")
                return df
        
        # Revenue Growth Rate
        df['revenue_growth'] = df['revenue'].pct_change(periods=4)  # YoY growth
        
        # Net Income Growth Rate
        df['net_income_growth'] = df['net_income'].pct_change(periods=4)
        
        # Asset Growth Rate
        df['asset_growth'] = df['total_assets'].pct_change(periods=4)
        
        # Equity Growth Rate
        df['equity_growth'] = df['shareholders_equity'].pct_change(periods=4)
        
        # Compound Annual Growth Rate (CAGR) over multiple periods
        periods = min(len(df), 20)  # Use up to 5 years of quarterly data
        if periods >= 8:  # Need at least 2 years
            years = periods / 4
            df['revenue_cagr'] = (
                (df['revenue'] / df['revenue'].shift(periods)) ** (1/years) - 1
            )
            df['net_income_cagr'] = (
                (df['net_income'] / df['net_income'].shift(periods)) ** (1/years) - 1
            )
        
        return df
    
    def _safe_divide(self, numerator: Union[pd.Series, float], 
                    denominator: Union[pd.Series, float]) -> pd.Series:
        """Safely divide two series/values, handling division by zero."""
        if isinstance(numerator, (int, float)) and isinstance(denominator, (int, float)):
            return numerator / denominator if denominator != 0 else np.nan
        
        # Handle Series division
        result = numerator / denominator
        
        # Replace infinite values with NaN
        if isinstance(result, pd.Series):
            result = result.replace([np.inf, -np.inf], np.nan)
        
        return result
    
    def _handle_missing_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in calculated ratios."""
        ratio_columns = [col for col in df.columns if col not in [
            'revenue', 'cost_of_goods_sold', 'gross_profit', 'operating_expenses',
            'operating_income', 'interest_expense', 'net_income', 'total_assets',
            'current_assets', 'total_liabilities', 'current_liabilities',
            'shareholders_equity', 'cash_and_equivalents', 'inventory',
            'accounts_receivable', 'long_term_debt', 'shares_outstanding'
        ]]
        
        if self.handle_missing == 'interpolate':
            # Forward fill then backward fill
            df[ratio_columns] = df[ratio_columns].fillna(method='ffill').fillna(method='bfill')
        elif self.handle_missing == 'median':
            # Fill with median values
            df[ratio_columns] = df[ratio_columns].fillna(df[ratio_columns].median())
        elif self.handle_missing == 'drop':
            # Drop rows with any missing ratios
            df = df.dropna(subset=ratio_columns)
        
        return df
```

### 1.2 Advanced Financial Transformations

```python
from scipy import stats
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
import warnings

class AdvancedFinancialTransforms:
    """
    Advanced transformations for financial features to improve ML model performance.
    
    Includes normalization, winsorization, log transforms, and industry adjustments.
    """
    
    def __init__(self):
        self.transformers = {}
        self.industry_stats = {}
    
    def apply_financial_transforms(self, df: pd.DataFrame,
                                 industry_column: str = None,
                                 transform_methods: List[str] = None) -> pd.DataFrame:
        """
        Apply comprehensive financial transformations.
        
        Args:
            df: DataFrame with financial ratios
            industry_column: Column containing industry classifications
            transform_methods: List of transformation methods to apply
            
        Returns:
            Transformed DataFrame
        """
        if transform_methods is None:
            transform_methods = ['winsorize', 'log_transform', 'normalize', 'industry_adjust']
        
        df_transformed = df.copy()
        
        # Identify financial ratio columns
        ratio_columns = self._identify_ratio_columns(df_transformed)
        
        for method in transform_methods:
            if method == 'winsorize':
                df_transformed = self._winsorize_outliers(df_transformed, ratio_columns)
            elif method == 'log_transform':
                df_transformed = self._apply_log_transforms(df_transformed, ratio_columns)
            elif method == 'normalize':
                df_transformed = self._normalize_ratios(df_transformed, ratio_columns)
            elif method == 'industry_adjust' and industry_column:
                df_transformed = self._industry_adjust_ratios(
                    df_transformed, ratio_columns, industry_column
                )
            elif method == 'rank_transform':
                df_transformed = self._apply_rank_transforms(df_transformed, ratio_columns)
        
        return df_transformed
    
    def _identify_ratio_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify columns that contain financial ratios."""
        ratio_indicators = [
            'ratio', 'margin', 'coverage', 'turnover', 'return', 'roa', 'roe', 'roic',
            'pe_', 'pb_', 'ps_', 'ev_', 'debt_to_', 'current_', 'quick_', 'cash_',
            'growth', 'cagr', 'outstanding'
        ]
        
        ratio_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in ratio_indicators):
                if df[col].dtype in ['float64', 'int64']:
                    ratio_columns.append(col)
        
        return ratio_columns
    
    def _winsorize_outliers(self, df: pd.DataFrame, columns: List[str],
                          limits: Tuple[float, float] = (0.01, 0.01)) -> pd.DataFrame:
        """Winsorize extreme values in financial ratios."""
        from scipy.stats.mstats import winsorize
        
        df_winsorized = df.copy()
        
        for col in columns:
            if col in df_winsorized.columns:
                # Only winsorize if there are enough non-null values
                valid_data = df_winsorized[col].dropna()
                if len(valid_data) > 10:
                    winsorized_values = winsorize(valid_data, limits=limits)
                    df_winsorized.loc[valid_data.index, col] = winsorized_values
        
        return df_winsorized
    
    def _apply_log_transforms(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply logarithmic transformations where appropriate."""
        df_transformed = df.copy()
        
        for col in columns:
            if col not in df_transformed.columns:
                continue
            
            # Check if log transformation is appropriate
            if self._should_log_transform(df_transformed[col]):
                # Handle negative values by shifting
                min_val = df_transformed[col].min()
                if min_val <= 0:
                    shift_value = abs(min_val) + 1e-6
                    df_transformed[f'{col}_log'] = np.log(df_transformed[col] + shift_value)
                else:
                    df_transformed[f'{col}_log'] = np.log(df_transformed[col])
                
                # Store transformation info
                self.transformers[f'{col}_log'] = {
                    'type': 'log',
                    'shift_value': shift_value if min_val <= 0 else 0
                }
        
        return df_transformed
    
    def _should_log_transform(self, series: pd.Series) -> bool:
        """Determine if a series should be log-transformed based on skewness."""
        # Remove null values
        clean_series = series.dropna()
        
        if len(clean_series) < 10:
            return False
        
        # Check skewness
        skewness = stats.skew(clean_series)
        
        # Log transform if highly right-skewed and values are positive
        return skewness > 2 and clean_series.min() > 0
    
    def _normalize_ratios(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Normalize financial ratios using robust scaling."""
        df_normalized = df.copy()
        
        scaler = RobustScaler()
        
        for col in columns:
            if col in df_normalized.columns:
                valid_mask = df_normalized[col].notna()
                if valid_mask.sum() > 5:  # Need at least 5 valid values
                    values = df_normalized.loc[valid_mask, col].values.reshape(-1, 1)
                    normalized_values = scaler.fit_transform(values).flatten()
                    df_normalized.loc[valid_mask, f'{col}_normalized'] = normalized_values
                    
                    # Store scaler for inverse transformation
                    self.transformers[f'{col}_normalized'] = scaler
        
        return df_normalized
    
    def _industry_adjust_ratios(self, df: pd.DataFrame, columns: List[str],
                              industry_column: str) -> pd.DataFrame:
        """Adjust ratios relative to industry medians."""
        df_adjusted = df.copy()
        
        if industry_column not in df.columns:
            return df_adjusted
        
        # Calculate industry statistics
        industry_stats = df_adjusted.groupby(industry_column)[columns].agg(['median', 'std'])
        
        for col in columns:
            if col in df_adjusted.columns:
                # Create industry-adjusted ratio
                adjusted_col = f'{col}_industry_adj'
                df_adjusted[adjusted_col] = np.nan
                
                for industry in df_adjusted[industry_column].unique():
                    if pd.isna(industry):
                        continue
                    
                    industry_mask = df_adjusted[industry_column] == industry
                    industry_median = industry_stats.loc[industry, (col, 'median')]
                    industry_std = industry_stats.loc[industry, (col, 'std')]
                    
                    if not pd.isna(industry_median) and industry_std > 0:
                        # Z-score relative to industry
                        industry_values = df_adjusted.loc[industry_mask, col]
                        adjusted_values = (industry_values - industry_median) / industry_std
                        df_adjusted.loc[industry_mask, adjusted_col] = adjusted_values
        
        # Store industry stats
        self.industry_stats = industry_stats
        
        return df_adjusted
    
    def _apply_rank_transforms(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply rank-based transformations."""
        df_ranked = df.copy()
        
        for col in columns:
            if col in df_ranked.columns:
                # Percentile ranks
                df_ranked[f'{col}_rank'] = df_ranked[col].rank(pct=True)
                
                # Quantile transformation to normal distribution
                valid_mask = df_ranked[col].notna()
                if valid_mask.sum() > 10:
                    qt = QuantileTransformer(output_distribution='normal', random_state=42)
                    values = df_ranked.loc[valid_mask, col].values.reshape(-1, 1)
                    transformed_values = qt.fit_transform(values).flatten()
                    df_ranked.loc[valid_mask, f'{col}_quantile_normal'] = transformed_values
                    
                    # Store transformer
                    self.transformers[f'{col}_quantile_normal'] = qt
        
        return df_ranked
```

## 2. Technical Indicators and Market Sentiment

### 2.1 Technical Analysis Indicators

```python
import talib
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

class TechnicalIndicatorCalculator:
    """
    Calculate technical indicators for market sentiment analysis in IPO valuation.
    
    Includes trend, momentum, volatility, and volume indicators.
    """
    
    def __init__(self):
        self.indicators = {}
        
    def calculate_all_indicators(self, price_data: pd.DataFrame,
                               volume_data: pd.Series = None) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators.
        
        Args:
            price_data: DataFrame with OHLCV data
            volume_data: Series with volume data (optional if in price_data)
            
        Returns:
            DataFrame with technical indicators
        """
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in price_data.columns for col in required_cols):
            raise ValueError(f"Price data must contain columns: {required_cols}")
        
        indicators_df = price_data.copy()
        
        # Extract price series
        open_prices = indicators_df['open'].values
        high_prices = indicators_df['high'].values
        low_prices = indicators_df['low'].values
        close_prices = indicators_df['close'].values
        
        if 'volume' in indicators_df.columns:
            volume = indicators_df['volume'].values
        elif volume_data is not None:
            volume = volume_data.values
        else:
            volume = None
        
        # Trend Indicators
        indicators_df = self._calculate_trend_indicators(
            indicators_df, open_prices, high_prices, low_prices, close_prices
        )
        
        # Momentum Indicators
        indicators_df = self._calculate_momentum_indicators(
            indicators_df, open_prices, high_prices, low_prices, close_prices
        )
        
        # Volatility Indicators
        indicators_df = self._calculate_volatility_indicators(
            indicators_df, open_prices, high_prices, low_prices, close_prices
        )
        
        # Volume Indicators (if volume data available)
        if volume is not None:
            indicators_df = self._calculate_volume_indicators(
                indicators_df, high_prices, low_prices, close_prices, volume
            )
        
        # Market Sentiment Indicators
        indicators_df = self._calculate_sentiment_indicators(indicators_df)
        
        return indicators_df
    
    def _calculate_trend_indicators(self, df: pd.DataFrame,
                                  open_prices: np.array, high_prices: np.array,
                                  low_prices: np.array, close_prices: np.array) -> pd.DataFrame:
        """Calculate trend-following indicators."""
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = talib.SMA(close_prices, timeperiod=period)
        
        # Exponential Moving Averages
        for period in [12, 26, 50]:
            df[f'ema_{period}'] = talib.EMA(close_prices, timeperiod=period)
        
        # MACD (Moving Average Convergence Divergence)
        macd, macd_signal, macd_hist = talib.MACD(close_prices)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (close_prices - bb_lower) / (bb_upper - bb_lower)
        
        # Parabolic SAR
        df['sar'] = talib.SAR(high_prices, low_prices)
        
        # Average Directional Index (ADX)
        df['adx'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
        df['plus_di'] = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
        df['minus_di'] = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
        
        return df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame,
                                     open_prices: np.array, high_prices: np.array,
                                     low_prices: np.array, close_prices: np.array) -> pd.DataFrame:
        """Calculate momentum oscillators."""
        # Relative Strength Index (RSI)
        for period in [14, 21]:
            df[f'rsi_{period}'] = talib.RSI(close_prices, timeperiod=period)
        
        # Stochastic Oscillator
        stoch_k, stoch_d = talib.STOCH(high_prices, low_prices, close_prices)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # Williams %R
        df['williams_r'] = talib.WILLR(high_prices, low_prices, close_prices)
        
        # Rate of Change (ROC)
        for period in [10, 20]:
            df[f'roc_{period}'] = talib.ROC(close_prices, timeperiod=period)
        
        # Commodity Channel Index (CCI)
        df['cci'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
        
        # Money Flow Index (if volume available)
        if 'volume' in df.columns:
            df['mfi'] = talib.MFI(high_prices, low_prices, close_prices, 
                                df['volume'].values, timeperiod=14)
        
        return df
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame,
                                       open_prices: np.array, high_prices: np.array,
                                       low_prices: np.array, close_prices: np.array) -> pd.DataFrame:
        """Calculate volatility indicators."""
        # Average True Range (ATR)
        df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        df['atr_ratio'] = df['atr'] / close_prices
        
        # Historical Volatility
        for period in [10, 20, 30]:
            returns = np.log(close_prices[1:] / close_prices[:-1])
            returns_padded = np.concatenate([[np.nan], returns])
            
            rolling_std = pd.Series(returns_padded).rolling(window=period).std()
            df[f'volatility_{period}'] = rolling_std * np.sqrt(252)  # Annualized
        
        # Bollinger Band Width (volatility measure)
        if 'bb_width' in df.columns:
            df['bb_width_normalized'] = df['bb_width'] / df['bb_width'].rolling(20).mean()
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame,
                                   high_prices: np.array, low_prices: np.array,
                                   close_prices: np.array, volume: np.array) -> pd.DataFrame:
        """Calculate volume-based indicators."""
        # On-Balance Volume (OBV)
        df['obv'] = talib.OBV(close_prices, volume)
        
        # Accumulation/Distribution Line
        df['ad'] = talib.AD(high_prices, low_prices, close_prices, volume)
        
        # Chaikin Money Flow
        df['cmf'] = talib.ADOSC(high_prices, low_prices, close_prices, volume,
                               fastperiod=3, slowperiod=10)
        
        # Volume-Weighted Average Price (VWAP)
        typical_price = (high_prices + low_prices + close_prices) / 3
        df['vwap'] = np.cumsum(typical_price * volume) / np.cumsum(volume)
        
        # Volume Moving Averages
        for period in [10, 20]:
            df[f'volume_sma_{period}'] = talib.SMA(volume.astype(float), timeperiod=period)
        
        # Volume Rate of Change
        df['volume_roc'] = talib.ROC(volume.astype(float), timeperiod=10)
        
        return df
    
    def _calculate_sentiment_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market sentiment indicators."""
        # Price vs Moving Average Signals
        if 'sma_20' in df.columns:
            df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        
        if 'sma_50' in df.columns:
            df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
        
        # MACD Signal
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # RSI Overbought/Oversold
        if 'rsi_14' in df.columns:
            df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
            df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        
        # Bollinger Band Position
        if 'bb_position' in df.columns:
            df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).quantile(0.2)).astype(int)
        
        # Volume Trend
        if 'volume' in df.columns:
            df['volume_trend'] = (
                df['volume'] > df['volume'].rolling(10).mean()
            ).astype(int)
        
        return df
```

## 3. Time-based Features and Temporal Patterns

### 3.1 Time Series Feature Engineering

```python
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import holidays

class TimeBasedFeatureEngineer:
    """
    Create time-based features for financial time series analysis.
    
    Includes cyclical features, lag variables, and temporal aggregations.
    """
    
    def __init__(self, country: str = 'Australia'):
        self.country = country
        self.holidays = holidays.country_holidays(country)
        
    def create_temporal_features(self, df: pd.DataFrame,
                               datetime_column: str = None,
                               target_columns: List[str] = None) -> pd.DataFrame:
        """
        Create comprehensive temporal features.
        
        Args:
            df: DataFrame with time series data
            datetime_column: Name of datetime column (uses index if None)
            target_columns: Columns to create lag features for
            
        Returns:
            DataFrame with temporal features
        """
        df_temporal = df.copy()
        
        # Ensure datetime index
        if datetime_column:
            df_temporal.index = pd.to_datetime(df_temporal[datetime_column])
        elif not isinstance(df_temporal.index, pd.DatetimeIndex):
            df_temporal.index = pd.to_datetime(df_temporal.index)
        
        # Calendar-based features
        df_temporal = self._create_calendar_features(df_temporal)
        
        # Cyclical features
        df_temporal = self._create_cyclical_features(df_temporal)
        
        # Business calendar features
        df_temporal = self._create_business_features(df_temporal)
        
        # Lag features
        if target_columns:
            df_temporal = self._create_lag_features(df_temporal, target_columns)
        
        # Rolling window features
        if target_columns:
            df_temporal = self._create_rolling_features(df_temporal, target_columns)
        
        # Seasonality features
        df_temporal = self._create_seasonality_features(df_temporal)
        
        return df_temporal
    
    def _create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic calendar features."""
        # Basic time components
        df['year'] = df.index.year
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['week'] = df.index.isocalendar().week
        df['day_of_year'] = df.index.dayofyear
        df['day_of_month'] = df.index.day
        df['day_of_week'] = df.index.dayofweek
        df['hour'] = df.index.hour
        
        # Weekend indicator
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Month-end indicator
        df['is_month_end'] = (df.index == df.index.to_period('M').end_time).astype(int)
        
        # Quarter-end indicator
        df['is_quarter_end'] = (df.index.month % 3 == 0) & df['is_month_end']
        df['is_quarter_end'] = df['is_quarter_end'].astype(int)
        
        # Year-end indicator
        df['is_year_end'] = (df.index.month == 12) & df['is_month_end']
        df['is_year_end'] = df['is_year_end'].astype(int)
        
        return df
    
    def _create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical encoding of time features."""
        # Month cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Day of week cyclical encoding
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Day of year cyclical encoding
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Hour cyclical encoding (if hourly data)
        if df['hour'].max() > 0:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        return df
    
    def _create_business_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create business calendar features."""
        # Holiday indicator
        df['is_holiday'] = df.index.to_series().apply(
            lambda x: x.date() in self.holidays
        ).astype(int)
        
        # Business day indicator
        from pandas.tseries.offsets import BDay
        df['is_business_day'] = df.index.to_series().apply(
            lambda x: len(pd.bdate_range(x, x)) > 0
        ).astype(int)
        
        # Trading day features for ASX
        # ASX is closed on weekends and Australian public holidays
        df['is_trading_day'] = (
            (df['is_business_day'] == 1) & 
            (df['is_holiday'] == 0)
        ).astype(int)
        
        # Days until/since important dates
        financial_year_end = pd.Timestamp(f"{df.index.year[0]}-06-30")
        df['days_to_fy_end'] = (financial_year_end - df.index).days
        
        # Reporting season indicator (Australian companies typically report in Feb/Aug)
        df['is_reporting_season'] = (
            (df['month'].isin([2, 8]))
        ).astype(int)
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Create lagged versions of specified columns."""
        lag_periods = [1, 2, 3, 5, 10, 20, 60, 120, 252]  # Various lags including 1 year
        
        for col in columns:
            if col in df.columns:
                for lag in lag_periods:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Create rolling window statistics."""
        windows = [5, 10, 20, 60, 120, 252]  # Various windows including 1 year
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    # Rolling mean
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                    
                    # Rolling standard deviation
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
                    
                    # Rolling min/max
                    df[f'{col}_rolling_min_{window}'] = df[col].rolling(window).min()
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window).max()
                    
                    # Rolling quantiles
                    df[f'{col}_rolling_q25_{window}'] = df[col].rolling(window).quantile(0.25)
                    df[f'{col}_rolling_q75_{window}'] = df[col].rolling(window).quantile(0.75)
                    
                    # Rolling rank (percentile)
                    df[f'{col}_rolling_rank_{window}'] = (
                        df[col].rolling(window).rank(pct=True)
                    )
        
        return df
    
    def _create_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create seasonality indicators."""
        # Australian financial year (July 1 - June 30)
        df['financial_year'] = df.index.to_series().apply(
            lambda x: x.year if x.month >= 7 else x.year - 1
        )
        
        # Financial quarter (Australian)
        df['financial_quarter'] = df.index.to_series().apply(
            lambda x: ((x.month - 7) % 12) // 3 + 1 if x.month >= 7 else ((x.month + 5) % 12) // 3 + 1
        )
        
        # Seasonal patterns for different industries
        # Technology sector patterns
        df['tech_earnings_season'] = (df['month'].isin([2, 5, 8, 11])).astype(int)
        
        # Retail sector patterns (Christmas, Easter)
        df['retail_peak_season'] = (df['month'].isin([11, 12, 1, 3, 4])).astype(int)
        
        # Resources sector patterns (commodity cycles)
        df['resources_reporting'] = (df['month'].isin([2, 8])).astype(int)
        
        return df
```

This comprehensive financial feature engineering documentation provides the foundation for creating ML-ready features from raw financial data. The next sections will cover text feature engineering and data validation frameworks.