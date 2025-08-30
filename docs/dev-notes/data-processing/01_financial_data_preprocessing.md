# Financial Data Preprocessing for IPO Valuation

This document provides comprehensive guidance for preprocessing financial data in the IPO valuation platform, with specific focus on Australian market requirements and ASX/ASIC data sources.

## 📊 Overview

Financial data preprocessing is the foundation of accurate IPO valuations. This module handles time series data cleaning, outlier detection, missing value imputation, and normalization techniques specifically tailored for Australian financial markets.

## 🎯 Key Objectives

- **Data Quality**: Ensure high-quality, consistent financial data
- **Regulatory Compliance**: Meet ASIC and ASX reporting standards
- **ML Readiness**: Prepare data for machine learning models
- **Real-time Processing**: Support both batch and streaming data
- **Scalability**: Handle large volumes of historical and real-time data

## 1. Time Series Data Cleaning

### 1.1 Data Source Integration

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import warnings
from dataclasses import dataclass

@dataclass
class DataSourceConfig:
    """Configuration for financial data sources."""
    source_name: str
    frequency: str  # 'daily', 'weekly', 'monthly', 'quarterly'
    timezone: str = 'Australia/Sydney'
    currency: str = 'AUD'
    trading_calendar: str = 'ASX'

class FinancialDataCleaner:
    """
    Comprehensive financial data cleaning for IPO valuation.
    
    Handles ASX market data, company financials, and regulatory filings.
    """
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.trading_calendar = self._load_trading_calendar()
        
    def clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean stock price and trading volume data.
        
        Args:
            df: DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume']
            
        Returns:
            Cleaned DataFrame with validated price data
        """
        df = df.copy()
        
        # 1. Basic validation
        df = self._validate_price_constraints(df)
        
        # 2. Corporate action adjustments
        df = self._adjust_for_splits_and_dividends(df)
        
        # 3. Handle missing trading days
        df = self._fill_non_trading_days(df)
        
        # 4. Outlier detection and correction
        df = self._detect_price_outliers(df)
        
        return df
    
    def _validate_price_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate basic price constraints (open, high, low, close relationships)."""
        # Ensure high >= max(open, close) and low <= min(open, close)
        df.loc[df['high'] < df[['open', 'close']].max(axis=1), 'high'] = \
            df[['open', 'close']].max(axis=1)
            
        df.loc[df['low'] > df[['open', 'close']].min(axis=1), 'low'] = \
            df[['open', 'close']].min(axis=1)
            
        # Remove negative prices and volumes
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].clip(lower=0.001)  # Minimum 0.1 cent
        df['volume'] = df['volume'].clip(lower=0)
        
        return df
    
    def _adjust_for_splits_and_dividends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adjust prices for stock splits and dividend payments."""
        # This would integrate with ASX corporate actions API
        # Placeholder for corporate action adjustments
        
        # Example adjustment factor calculation
        adjustment_factors = self._get_adjustment_factors(df.index)
        
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df[col] = df[col] * adjustment_factors
            
        df['volume'] = df['volume'] / adjustment_factors
        
        return df
    
    def _get_adjustment_factors(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Calculate cumulative adjustment factors for splits and dividends."""
        # Placeholder - would integrate with ASX data
        return pd.Series(1.0, index=dates)
```

### 1.2 Outlier Detection Methods

```python
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class OutlierDetector:
    """Advanced outlier detection for financial time series."""
    
    def __init__(self, methods: List[str] = None):
        self.methods = methods or ['zscore', 'iqr', 'isolation_forest', 'macd_divergence']
        self.scalers = {}
    
    def detect_outliers(self, df: pd.DataFrame, 
                       columns: List[str] = None) -> Dict[str, pd.Series]:
        """
        Detect outliers using multiple methods.
        
        Returns:
            Dictionary of boolean Series indicating outliers for each method
        """
        if columns is None:
            columns = ['open', 'high', 'low', 'close', 'volume']
            
        outliers = {}
        
        for method in self.methods:
            if method == 'zscore':
                outliers[method] = self._zscore_outliers(df, columns)
            elif method == 'iqr':
                outliers[method] = self._iqr_outliers(df, columns)
            elif method == 'isolation_forest':
                outliers[method] = self._isolation_forest_outliers(df, columns)
            elif method == 'macd_divergence':
                outliers[method] = self._macd_divergence_outliers(df)
                
        return outliers
    
    def _zscore_outliers(self, df: pd.DataFrame, columns: List[str], 
                        threshold: float = 3.0) -> pd.Series:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(df[columns], nan_policy='omit'))
        return (z_scores > threshold).any(axis=1)
    
    def _iqr_outliers(self, df: pd.DataFrame, columns: List[str], 
                     multiplier: float = 1.5) -> pd.Series:
        """Detect outliers using Interquartile Range method."""
        outliers = pd.Series(False, index=df.index)
        
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers = outliers | col_outliers
            
        return outliers
    
    def _isolation_forest_outliers(self, df: pd.DataFrame, columns: List[str],
                                 contamination: float = 0.1) -> pd.Series:
        """Detect outliers using Isolation Forest."""
        # Normalize data first
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[columns].fillna(df[columns].mean()))
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(scaled_data) == -1
        
        return pd.Series(outliers, index=df.index)
    
    def _macd_divergence_outliers(self, df: pd.DataFrame) -> pd.Series:
        """Detect price-volume divergence outliers."""
        # Calculate MACD and volume indicators
        close_prices = df['close']
        volume = df['volume']
        
        # MACD calculation
        ema12 = close_prices.ewm(span=12).mean()
        ema26 = close_prices.ewm(span=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9).mean()
        
        # Volume moving average
        volume_ma = volume.rolling(window=20).mean()
        
        # Detect divergences
        price_direction = macd > macd_signal
        volume_direction = volume > volume_ma
        
        # Outliers are significant price-volume divergences
        divergence_strength = abs(macd - macd_signal) * abs(volume - volume_ma)
        threshold = divergence_strength.quantile(0.95)
        
        return divergence_strength > threshold
```

## 2. Missing Value Imputation Strategies

### 2.1 Financial-Specific Imputation

```python
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class FinancialImputer:
    """Specialized imputation for financial data."""
    
    def __init__(self):
        self.imputers = {
            'forward_fill': self._forward_fill_imputer,
            'business_day_interpolation': self._business_day_interpolation,
            'sector_mean': self._sector_mean_imputation,
            'knn': self._knn_imputation,
            'mice': self._mice_imputation
        }
    
    def impute_missing_values(self, df: pd.DataFrame, 
                            method: str = 'business_day_interpolation',
                            **kwargs) -> pd.DataFrame:
        """
        Impute missing values using specified method.
        
        Args:
            df: DataFrame with missing values
            method: Imputation method to use
            **kwargs: Additional parameters for imputation method
            
        Returns:
            DataFrame with imputed values
        """
        if method not in self.imputers:
            raise ValueError(f"Unknown imputation method: {method}")
            
        return self.imputers[method](df, **kwargs)
    
    def _forward_fill_imputer(self, df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
        """Forward fill with limit for financial data."""
        df_filled = df.copy()
        
        # Different strategies for different data types
        price_cols = ['open', 'high', 'low', 'close']
        volume_cols = ['volume']
        
        # Prices: forward fill with limit
        df_filled[price_cols] = df_filled[price_cols].fillna(method='ffill', limit=limit)
        
        # Volume: use median of surrounding days
        for col in volume_cols:
            df_filled[col] = df_filled[col].fillna(
                df_filled[col].rolling(window=5, center=True).median()
            )
        
        return df_filled
    
    def _business_day_interpolation(self, df: pd.DataFrame, 
                                  trading_calendar: pd.DatetimeIndex = None) -> pd.DataFrame:
        """Interpolate missing values considering trading days only."""
        df_filled = df.copy()
        
        if trading_calendar is None:
            # Use ASX trading calendar
            trading_calendar = self._get_asx_trading_calendar(df.index)
        
        # Reindex to trading days only
        df_trading = df_filled.reindex(trading_calendar)
        
        # Interpolate missing values
        numeric_cols = df_trading.select_dtypes(include=[np.number]).columns
        df_trading[numeric_cols] = df_trading[numeric_cols].interpolate(
            method='time', limit_direction='both'
        )
        
        return df_trading.reindex(df.index)
    
    def _sector_mean_imputation(self, df: pd.DataFrame, 
                              sector_data: pd.DataFrame) -> pd.DataFrame:
        """Impute using sector averages for fundamental ratios."""
        df_filled = df.copy()
        
        # Get sector information (would come from company metadata)
        sector_means = sector_data.groupby('sector').mean()
        
        for idx, row in df_filled.iterrows():
            if pd.isna(row).any():
                company_sector = self._get_company_sector(idx)
                if company_sector in sector_means.index:
                    # Fill missing values with sector means
                    missing_mask = pd.isna(row)
                    df_filled.loc[idx, missing_mask] = \
                        sector_means.loc[company_sector, missing_mask]
        
        return df_filled
    
    def _knn_imputation(self, df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
        """KNN-based imputation for similar companies."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_filled = df.copy()
        df_filled[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        return df_filled
    
    def _mice_imputation(self, df: pd.DataFrame, max_iter: int = 10) -> pd.DataFrame:
        """Multiple Imputation by Chained Equations."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        imputer = IterativeImputer(max_iter=max_iter, random_state=42)
        df_filled = df.copy()
        df_filled[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        return df_filled
```

## 3. Normalization and Standardization

### 3.1 Financial Data Normalization

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer

class FinancialNormalizer:
    """Specialized normalization for financial data."""
    
    def __init__(self):
        self.scalers = {}
        self.normalization_stats = {}
    
    def normalize_financial_data(self, df: pd.DataFrame, 
                                method: str = 'robust',
                                by_sector: bool = True) -> pd.DataFrame:
        """
        Normalize financial data using appropriate method.
        
        Args:
            df: Financial data DataFrame
            method: Normalization method ('standard', 'minmax', 'robust', 'quantile')
            by_sector: Whether to normalize within sectors
            
        Returns:
            Normalized DataFrame
        """
        df_normalized = df.copy()
        
        if by_sector:
            # Normalize within each sector
            sectors = df.get('sector', pd.Series('default', index=df.index))
            for sector in sectors.unique():
                sector_mask = sectors == sector
                sector_data = df.loc[sector_mask]
                
                normalized_sector = self._apply_normalization(
                    sector_data, method, sector_key=sector
                )
                df_normalized.loc[sector_mask] = normalized_sector
        else:
            df_normalized = self._apply_normalization(df, method)
            
        return df_normalized
    
    def _apply_normalization(self, df: pd.DataFrame, method: str, 
                           sector_key: str = None) -> pd.DataFrame:
        """Apply specific normalization method."""
        scaler_key = f"{method}_{sector_key}" if sector_key else method
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'quantile':
            scaler = QuantileTransformer(output_distribution='normal')
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_normalized = df.copy()
        
        # Fit and transform
        df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        # Store scaler for inverse transformation
        self.scalers[scaler_key] = scaler
        
        # Store normalization statistics
        self.normalization_stats[scaler_key] = {
            'mean': df[numeric_cols].mean().to_dict(),
            'std': df[numeric_cols].std().to_dict(),
            'min': df[numeric_cols].min().to_dict(),
            'max': df[numeric_cols].max().to_dict()
        }
        
        return df_normalized
    
    def inverse_transform(self, df: pd.DataFrame, 
                         method: str = 'robust', 
                         sector_key: str = None) -> pd.DataFrame:
        """Inverse normalize the data back to original scale."""
        scaler_key = f"{method}_{sector_key}" if sector_key else method
        
        if scaler_key not in self.scalers:
            raise ValueError(f"No scaler found for key: {scaler_key}")
        
        scaler = self.scalers[scaler_key]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        df_original = df.copy()
        df_original[numeric_cols] = scaler.inverse_transform(df[numeric_cols])
        
        return df_original
```

## 4. Currency Conversion and Inflation Adjustment

### 4.1 Multi-Currency Handling

```python
import yfinance as yf
from forex_python.converter import CurrencyRates
import requests
from typing import Dict
import sqlite3

class CurrencyConverter:
    """Handle multi-currency financial data with inflation adjustments."""
    
    def __init__(self, base_currency: str = 'AUD'):
        self.base_currency = base_currency
        self.exchange_rates = {}
        self.inflation_data = {}
        self._setup_cache()
    
    def _setup_cache(self):
        """Setup local cache for exchange rates and inflation data."""
        self.cache_db = sqlite3.connect('cache/financial_data.db')
        self._create_cache_tables()
    
    def _create_cache_tables(self):
        """Create tables for caching financial data."""
        cursor = self.cache_db.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exchange_rates (
                date DATE,
                from_currency TEXT,
                to_currency TEXT,
                rate REAL,
                PRIMARY KEY (date, from_currency, to_currency)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inflation_data (
                date DATE,
                country TEXT,
                cpi REAL,
                inflation_rate REAL,
                PRIMARY KEY (date, country)
            )
        ''')
        
        self.cache_db.commit()
    
    def convert_currency(self, df: pd.DataFrame, 
                        from_currency: str,
                        to_currency: str = None,
                        date_column: str = 'date') -> pd.DataFrame:
        """
        Convert financial data from one currency to another.
        
        Args:
            df: DataFrame with financial data
            from_currency: Source currency code
            to_currency: Target currency code (default: base_currency)
            date_column: Column containing dates
            
        Returns:
            DataFrame with converted currency values
        """
        if to_currency is None:
            to_currency = self.base_currency
            
        if from_currency == to_currency:
            return df
        
        df_converted = df.copy()
        
        # Get exchange rates for all dates
        exchange_rates = self._get_exchange_rates(
            df[date_column].unique(),
            from_currency,
            to_currency
        )
        
        # Apply conversion to monetary columns
        monetary_columns = self._identify_monetary_columns(df)
        
        for col in monetary_columns:
            df_converted[col] = df_converted.apply(
                lambda row: row[col] * exchange_rates.get(row[date_column], 1.0),
                axis=1
            )
        
        # Update currency metadata
        df_converted.attrs['currency'] = to_currency
        df_converted.attrs['original_currency'] = from_currency
        
        return df_converted
    
    def _get_exchange_rates(self, dates: np.array, 
                           from_curr: str, to_curr: str) -> Dict:
        """Get historical exchange rates for specified dates."""
        rates = {}
        
        for date in dates:
            # Check cache first
            cached_rate = self._get_cached_exchange_rate(date, from_curr, to_curr)
            
            if cached_rate is not None:
                rates[date] = cached_rate
            else:
                # Fetch from API
                try:
                    rate = self._fetch_exchange_rate(date, from_curr, to_curr)
                    rates[date] = rate
                    
                    # Cache the result
                    self._cache_exchange_rate(date, from_curr, to_curr, rate)
                    
                except Exception as e:
                    warnings.warn(f"Could not fetch exchange rate for {date}: {e}")
                    rates[date] = 1.0  # Fallback to no conversion
        
        return rates
    
    def adjust_for_inflation(self, df: pd.DataFrame,
                           base_date: str = None,
                           country: str = 'Australia') -> pd.DataFrame:
        """
        Adjust financial values for inflation to real terms.
        
        Args:
            df: DataFrame with financial data
            base_date: Reference date for inflation adjustment
            country: Country for inflation data
            
        Returns:
            DataFrame with inflation-adjusted values
        """
        if base_date is None:
            base_date = df.index.max().strftime('%Y-%m-%d')
        
        df_adjusted = df.copy()
        
        # Get inflation data
        inflation_data = self._get_inflation_data(country, df.index)
        base_cpi = inflation_data.get(base_date, 100)  # Default CPI = 100
        
        # Identify monetary columns
        monetary_columns = self._identify_monetary_columns(df)
        
        # Apply inflation adjustment
        for col in monetary_columns:
            df_adjusted[col] = df_adjusted.apply(
                lambda row: row[col] * base_cpi / inflation_data.get(
                    row.name.strftime('%Y-%m-%d'), base_cpi
                ),
                axis=1
            )
        
        # Update metadata
        df_adjusted.attrs['inflation_adjusted'] = True
        df_adjusted.attrs['base_date'] = base_date
        df_adjusted.attrs['country'] = country
        
        return df_adjusted
    
    def _identify_monetary_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify columns containing monetary values."""
        monetary_indicators = [
            'price', 'revenue', 'profit', 'cost', 'asset', 'liability',
            'equity', 'cash', 'debt', 'value', 'amount', 'fee',
            'open', 'high', 'low', 'close', 'dividend'
        ]
        
        monetary_columns = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in monetary_indicators):
                monetary_columns.append(col)
        
        return monetary_columns
```

## 5. Seasonal Adjustment and Detrending

### 5.1 Financial Time Series Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.stattools import adfuller

class SeasonalAdjuster:
    """Handle seasonal adjustment and detrending for financial time series."""
    
    def __init__(self):
        self.decomposition_results = {}
        self.trend_filters = {}
    
    def seasonal_adjustment(self, df: pd.DataFrame,
                          columns: List[str] = None,
                          method: str = 'multiplicative',
                          period: int = None) -> pd.DataFrame:
        """
        Apply seasonal adjustment to financial time series.
        
        Args:
            df: Time series DataFrame
            columns: Columns to adjust (default: all numeric)
            method: 'additive' or 'multiplicative'
            period: Seasonal period (auto-detect if None)
            
        Returns:
            DataFrame with seasonally adjusted data
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        df_adjusted = df.copy()
        
        for col in columns:
            try:
                # Auto-detect period if not specified
                if period is None:
                    detected_period = self._detect_seasonality(df[col])
                else:
                    detected_period = period
                
                if detected_period > 1:
                    # Perform seasonal decomposition
                    decomposition = seasonal_decompose(
                        df[col].dropna(),
                        model=method,
                        period=detected_period
                    )
                    
                    # Store decomposition results
                    self.decomposition_results[col] = decomposition
                    
                    # Apply seasonal adjustment
                    if method == 'multiplicative':
                        df_adjusted[col] = df[col] / decomposition.seasonal
                    else:  # additive
                        df_adjusted[col] = df[col] - decomposition.seasonal
                        
            except Exception as e:
                warnings.warn(f"Could not apply seasonal adjustment to {col}: {e}")
                continue
        
        return df_adjusted
    
    def detrend_series(self, df: pd.DataFrame,
                      columns: List[str] = None,
                      method: str = 'hp_filter') -> pd.DataFrame:
        """
        Remove trends from financial time series.
        
        Args:
            df: Time series DataFrame
            columns: Columns to detrend
            method: Detrending method ('hp_filter', 'linear', 'polynomial')
            
        Returns:
            Detrended DataFrame
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        df_detrended = df.copy()
        
        for col in columns:
            series = df[col].dropna()
            
            if len(series) < 10:  # Need sufficient data points
                continue
                
            try:
                if method == 'hp_filter':
                    # Hodrick-Prescott filter
                    cycle, trend = hpfilter(series, lamb=1600)  # Standard for quarterly data
                    df_detrended.loc[series.index, col] = cycle
                    self.trend_filters[col] = trend
                    
                elif method == 'linear':
                    # Linear detrending
                    from scipy.signal import detrend
                    detrended = detrend(series.values)
                    df_detrended.loc[series.index, col] = detrended
                    
                elif method == 'polynomial':
                    # Polynomial detrending (degree 2)
                    from numpy.polynomial import polynomial as P
                    x = np.arange(len(series))
                    coeffs = P.polyfit(x, series.values, 2)
                    trend = P.polyval(x, coeffs)
                    df_detrended.loc[series.index, col] = series.values - trend
                    
            except Exception as e:
                warnings.warn(f"Could not detrend {col}: {e}")
                continue
        
        return df_detrended
    
    def _detect_seasonality(self, series: pd.Series, 
                          max_period: int = None) -> int:
        """Detect seasonal period in time series."""
        from scipy.fft import fft, fftfreq
        
        if max_period is None:
            max_period = len(series) // 3
        
        # Remove NaN values
        clean_series = series.dropna()
        
        if len(clean_series) < 20:  # Need sufficient data
            return 1
        
        # Apply FFT to detect dominant frequencies
        fft_values = fft(clean_series.values)
        freqs = fftfreq(len(clean_series))
        
        # Find dominant frequency (excluding DC component)
        power_spectrum = np.abs(fft_values[1:len(fft_values)//2])
        dominant_freq_idx = np.argmax(power_spectrum) + 1
        
        # Convert frequency to period
        if freqs[dominant_freq_idx] != 0:
            period = int(1 / abs(freqs[dominant_freq_idx]))
            return min(period, max_period)
        
        return 1  # No seasonality detected
    
    def stationarity_test(self, df: pd.DataFrame, 
                         columns: List[str] = None) -> Dict[str, Dict]:
        """
        Test for stationarity using Augmented Dickey-Fuller test.
        
        Returns:
            Dictionary with test results for each column
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        results = {}
        
        for col in columns:
            series = df[col].dropna()
            
            if len(series) < 10:
                continue
            
            try:
                adf_result = adfuller(series)
                
                results[col] = {
                    'adf_statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'critical_values': adf_result[4],
                    'is_stationary': adf_result[1] < 0.05,
                    'confidence_level': '95%' if adf_result[1] < 0.05 else 'Not significant'
                }
                
            except Exception as e:
                results[col] = {'error': str(e)}
        
        return results
```

## 6. Configuration and Usage Examples

### 6.1 Configuration Templates

```yaml
# financial_preprocessing.yaml
financial_preprocessing:
  data_sources:
    asx_data:
      frequency: daily
      timezone: Australia/Sydney
      currency: AUD
      trading_calendar: ASX
    
    asic_filings:
      frequency: quarterly
      timezone: Australia/Sydney
      currency: AUD
      
  cleaning:
    outlier_detection:
      methods: [zscore, iqr, isolation_forest]
      zscore_threshold: 3.0
      iqr_multiplier: 1.5
      isolation_contamination: 0.1
      
    missing_values:
      default_method: business_day_interpolation
      max_consecutive_missing: 5
      
    price_validation:
      min_price: 0.001
      max_daily_change: 0.5  # 50% max daily change
      
  normalization:
    default_method: robust
    by_sector: true
    
  currency:
    base_currency: AUD
    supported_currencies: [USD, EUR, GBP, JPY, NZD]
    
  seasonality:
    auto_detect: true
    default_method: multiplicative
    detrend_method: hp_filter
```

### 6.2 Usage Examples

```python
# Example: Complete financial data preprocessing pipeline
from src.data_processing.financial_preprocessor import FinancialPreprocessor

# Initialize preprocessor with configuration
config_path = "config/financial_preprocessing.yaml"
preprocessor = FinancialPreprocessor(config_path)

# Load raw financial data
raw_data = pd.read_csv("data/raw/asx_stock_data.csv")

# Apply complete preprocessing pipeline
processed_data = preprocessor.preprocess_pipeline(
    data=raw_data,
    steps=[
        'clean_data',
        'detect_outliers',
        'impute_missing',
        'normalize_data',
        'seasonal_adjustment',
        'currency_conversion'
    ]
)

# Save processed data
processed_data.to_parquet("data/processed/asx_cleaned.parquet")

# Generate preprocessing report
report = preprocessor.generate_report()
print(report)
```

## 7. Performance Optimization

### 7.1 Parallel Processing

```python
import multiprocessing as mp
from functools import partial
import dask.dataframe as dd

class ParallelFinancialProcessor:
    """Parallel processing for large financial datasets."""
    
    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or mp.cpu_count()
        
    def parallel_preprocess(self, df: pd.DataFrame,
                          chunk_size: int = 10000) -> pd.DataFrame:
        """Process large DataFrame in parallel chunks."""
        # Split DataFrame into chunks
        chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        
        # Create partial function with preprocessing parameters
        process_func = partial(self._preprocess_chunk)
        
        # Process chunks in parallel
        with mp.Pool(processes=self.n_workers) as pool:
            processed_chunks = pool.map(process_func, chunks)
        
        # Combine results
        return pd.concat(processed_chunks, ignore_index=True)
    
    def _preprocess_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process individual chunk."""
        # Apply preprocessing steps to chunk
        processor = FinancialDataCleaner()
        return processor.clean_price_data(chunk)
```

## 8. Testing and Validation

### 8.1 Data Quality Tests

```python
import pytest
from src.data_processing.financial_preprocessor import FinancialPreprocessor

class TestFinancialPreprocessing:
    """Test suite for financial data preprocessing."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample financial data for testing."""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'date': dates,
            'open': np.random.normal(100, 10, len(dates)),
            'high': np.random.normal(105, 10, len(dates)),
            'low': np.random.normal(95, 10, len(dates)),
            'close': np.random.normal(100, 10, len(dates)),
            'volume': np.random.poisson(1000000, len(dates))
        })
        
        # Introduce some data quality issues
        data.loc[100:105, 'close'] = np.nan  # Missing values
        data.loc[200, 'high'] = data.loc[200, 'low'] - 1  # Invalid high < low
        
        return data
    
    def test_price_validation(self, sample_data):
        """Test price constraint validation."""
        processor = FinancialDataCleaner()
        cleaned_data = processor.clean_price_data(sample_data)
        
        # Check that high >= max(open, close)
        assert (cleaned_data['high'] >= 
                cleaned_data[['open', 'close']].max(axis=1)).all()
        
        # Check that low <= min(open, close)
        assert (cleaned_data['low'] <= 
                cleaned_data[['open', 'close']].min(axis=1)).all()
    
    def test_outlier_detection(self, sample_data):
        """Test outlier detection methods."""
        detector = OutlierDetector()
        outliers = detector.detect_outliers(sample_data)
        
        # Check that outliers are detected
        assert len(outliers) > 0
        
        # Check that outlier ratios are reasonable
        for method, outlier_mask in outliers.items():
            outlier_ratio = outlier_mask.sum() / len(sample_data)
            assert 0 < outlier_ratio < 0.2  # Reasonable outlier ratio
    
    def test_missing_value_imputation(self, sample_data):
        """Test missing value imputation."""
        imputer = FinancialImputer()
        
        # Count missing values before
        missing_before = sample_data.isna().sum().sum()
        
        # Apply imputation
        imputed_data = imputer.impute_missing_values(sample_data)
        
        # Count missing values after
        missing_after = imputed_data.isna().sum().sum()
        
        # Check that missing values were reduced
        assert missing_after < missing_before
```

This comprehensive documentation provides a solid foundation for financial data preprocessing in the IPO valuation platform. The next sections will cover document preprocessing, feature engineering, and integration patterns.