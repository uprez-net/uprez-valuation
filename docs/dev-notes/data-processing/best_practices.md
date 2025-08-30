# Data Processing Best Practices for IPO Valuation

This document provides comprehensive best practices, industry standards, and recommendations for implementing robust data processing pipelines in the IPO valuation platform.

## 📋 Overview

Best practices ensure data processing pipelines are reliable, scalable, maintainable, and compliant with financial industry standards. These guidelines cover data quality, performance optimization, error handling, and regulatory compliance.

## 🎯 Core Principles

1. **Data Integrity First**: Never compromise on data accuracy and completeness
2. **Fail Fast**: Detect and report issues as early as possible in the pipeline
3. **Auditability**: Maintain complete lineage and audit trails for all data transformations
4. **Scalability**: Design for growth in data volume and complexity
5. **Compliance**: Adhere to financial regulations and data protection laws

## 1. Data Quality Management

### 1.1 Data Quality Framework

```python
"""
Data Quality Framework Implementation
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
from datetime import datetime
import logging

class DataQualityDimension(Enum):
    """Six dimensions of data quality."""
    ACCURACY = "accuracy"           # Data represents real-world values correctly
    COMPLETENESS = "completeness"   # All required data is present
    CONSISTENCY = "consistency"     # Data is uniform across systems/time
    TIMELINESS = "timeliness"      # Data is available when needed
    VALIDITY = "validity"          # Data conforms to defined formats
    UNIQUENESS = "uniqueness"      # No unwanted duplicates exist

@dataclass
class QualityRule:
    """Definition of a data quality rule."""
    name: str
    dimension: DataQualityDimension
    description: str
    rule_function: Callable
    severity: str = "ERROR"        # ERROR, WARNING, INFO
    business_impact: str = "HIGH"  # HIGH, MEDIUM, LOW
    
class DataQualityFramework:
    """Comprehensive data quality management framework."""
    
    def __init__(self):
        self.rules = []
        self.quality_metrics = {}
        self.logger = logging.getLogger(__name__)
    
    def add_rule(self, rule: QualityRule):
        """Add a quality rule to the framework."""
        self.rules.append(rule)
    
    def assess_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """Assess data quality across all dimensions."""
        quality_scores = {}
        
        for dimension in DataQualityDimension:
            dimension_rules = [r for r in self.rules if r.dimension == dimension]
            if dimension_rules:
                dimension_score = self._calculate_dimension_score(df, dimension_rules)
                quality_scores[dimension.value] = dimension_score
        
        # Calculate overall quality score
        quality_scores['overall'] = np.mean(list(quality_scores.values()))
        
        return quality_scores
    
    def _calculate_dimension_score(self, df: pd.DataFrame, rules: List[QualityRule]) -> float:
        """Calculate quality score for a specific dimension."""
        scores = []
        
        for rule in rules:
            try:
                score = rule.rule_function(df)
                scores.append(score)
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule.name}: {e}")
                scores.append(0.0)  # Failed rule gets 0 score
        
        return np.mean(scores) if scores else 0.0

# Example quality rules
def completeness_check(df: pd.DataFrame) -> float:
    """Check data completeness (percentage of non-null values)."""
    total_cells = df.size
    non_null_cells = df.count().sum()
    return non_null_cells / total_cells if total_cells > 0 else 0.0

def accuracy_check_prices(df: pd.DataFrame) -> float:
    """Check price data accuracy (positive prices, OHLC relationships)."""
    if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        return 1.0  # Skip if price columns not present
    
    # Check positive prices
    price_cols = ['open', 'high', 'low', 'close']
    positive_prices = (df[price_cols] > 0).all(axis=1)
    
    # Check OHLC relationships
    high_valid = df['high'] >= df[['open', 'close']].max(axis=1)
    low_valid = df['low'] <= df[['open', 'close']].min(axis=1)
    
    valid_records = positive_prices & high_valid & low_valid
    return valid_records.mean() if len(df) > 0 else 0.0

def consistency_check_dates(df: pd.DataFrame) -> float:
    """Check date consistency (chronological order, business days)."""
    if 'date' not in df.columns:
        return 1.0
    
    # Check chronological order
    dates_sorted = df['date'].is_monotonic_increasing
    
    # Check for future dates
    future_dates = df['date'] > datetime.now()
    no_future_dates = not future_dates.any()
    
    return float(dates_sorted and no_future_dates)
```

### 1.2 Data Quality Metrics and KPIs

| Metric | Formula | Target | Critical Threshold |
|--------|---------|---------|-------------------|
| **Completeness Rate** | (Non-null values / Total values) × 100 | >95% | <90% |
| **Accuracy Rate** | (Valid values / Total values) × 100 | >98% | <95% |
| **Consistency Score** | (Consistent records / Total records) × 100 | >99% | <97% |
| **Timeliness Score** | (On-time deliveries / Total deliveries) × 100 | >95% | <90% |
| **Uniqueness Rate** | (Unique records / Total records) × 100 | >99.5% | <99% |
| **Validity Rate** | (Format-compliant values / Total values) × 100 | >99% | <97% |

### 1.3 Data Quality Implementation Checklist

#### ✅ Data Ingestion
- [ ] Implement schema validation at ingestion point
- [ ] Set up data type validation and automatic conversion
- [ ] Configure real-time data quality monitoring
- [ ] Establish data lineage tracking from source
- [ ] Implement duplicate detection and handling
- [ ] Set up automated data profiling for new sources

#### ✅ Data Processing
- [ ] Validate business rules at each transformation step
- [ ] Implement statistical outlier detection
- [ ] Configure cross-field validation rules
- [ ] Set up temporal consistency checks
- [ ] Implement referential integrity validation
- [ ] Configure automated anomaly detection

#### ✅ Data Storage
- [ ] Implement data versioning and change tracking
- [ ] Set up backup and recovery procedures
- [ ] Configure data retention policies
- [ ] Implement access controls and audit logs
- [ ] Set up data encryption at rest and in transit
- [ ] Configure automated backup verification

## 2. Performance Optimization

### 2.1 Efficient Data Processing Patterns

```python
"""
Performance optimization patterns for financial data processing.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import dask.dataframe as dd
from functools import lru_cache
import cProfile
import time

class PerformanceOptimizer:
    """Optimize data processing performance."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.memory_usage_threshold = 0.8  # 80% memory threshold
    
    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by downcasting numeric types."""
        optimized_df = df.copy()
        
        # Optimize integer columns
        int_columns = optimized_df.select_dtypes(include=['int']).columns
        for col in int_columns:
            col_min = optimized_df[col].min()
            col_max = optimized_df[col].max()
            
            if col_min >= 0:  # Unsigned integers
                if col_max < np.iinfo(np.uint8).max:
                    optimized_df[col] = optimized_df[col].astype(np.uint8)
                elif col_max < np.iinfo(np.uint16).max:
                    optimized_df[col] = optimized_df[col].astype(np.uint16)
                elif col_max < np.iinfo(np.uint32).max:
                    optimized_df[col] = optimized_df[col].astype(np.uint32)
            else:  # Signed integers
                if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                    optimized_df[col] = optimized_df[col].astype(np.int8)
                elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                    optimized_df[col] = optimized_df[col].astype(np.int16)
                elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                    optimized_df[col] = optimized_df[col].astype(np.int32)
        
        # Optimize float columns
        float_columns = optimized_df.select_dtypes(include=['float']).columns
        for col in float_columns:
            if optimized_df[col].dtype == 'float64':
                # Check if values fit in float32
                if (optimized_df[col].min() >= np.finfo(np.float32).min and
                    optimized_df[col].max() <= np.finfo(np.float32).max):
                    optimized_df[col] = optimized_df[col].astype(np.float32)
        
        # Convert string columns to categorical if beneficial
        string_columns = optimized_df.select_dtypes(include=['object']).columns
        for col in string_columns:
            unique_ratio = optimized_df[col].nunique() / len(optimized_df)
            if unique_ratio < 0.5:  # If less than 50% unique values
                optimized_df[col] = optimized_df[col].astype('category')
        
        return optimized_df
    
    def process_large_dataset_in_chunks(self, df: pd.DataFrame, 
                                       processing_func: callable,
                                       chunk_size: int = 10000) -> pd.DataFrame:
        """Process large dataset in chunks to manage memory."""
        results = []
        
        for start_idx in range(0, len(df), chunk_size):
            end_idx = min(start_idx + chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx]
            
            # Process chunk
            processed_chunk = processing_func(chunk)
            results.append(processed_chunk)
            
            # Optional: Clear memory
            del chunk
        
        return pd.concat(results, ignore_index=True)
    
    def parallel_group_processing(self, df: pd.DataFrame, 
                                 group_column: str,
                                 processing_func: callable) -> pd.DataFrame:
        """Process groups in parallel."""
        groups = [group for name, group in df.groupby(group_column)]
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(processing_func, groups))
        
        return pd.concat(results, ignore_index=True)
    
    @staticmethod
    def vectorized_operations_example(df: pd.DataFrame) -> pd.DataFrame:
        """Example of vectorized operations for better performance."""
        # ❌ Slow: Using iterrows
        # for idx, row in df.iterrows():
        #     df.loc[idx, 'result'] = expensive_calculation(row['value'])
        
        # ✅ Fast: Vectorized operation
        df['result'] = df['value'].apply(lambda x: x ** 2 + np.log(x + 1))
        
        # ✅ Even faster: Pure numpy operations
        df['result_numpy'] = np.power(df['value'].values, 2) + np.log(df['value'].values + 1)
        
        return df
    
    @lru_cache(maxsize=1000)
    def cached_expensive_calculation(self, value: float) -> float:
        """Example of caching expensive calculations."""
        # Simulate expensive calculation
        time.sleep(0.001)
        return value ** 3 + np.sin(value)
    
    def use_dask_for_big_data(self, file_path: str) -> dd.DataFrame:
        """Use Dask for out-of-core processing of large datasets."""
        # Read large CSV with Dask
        df = dd.read_csv(file_path)
        
        # Perform operations lazily
        df_processed = df.groupby('symbol').agg({
            'close': 'mean',
            'volume': 'sum',
            'high': 'max',
            'low': 'min'
        })
        
        return df_processed
    
    def profile_performance(self, func: callable, *args, **kwargs):
        """Profile function performance."""
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        profiler.print_stats(sort='cumulative')
        
        return result

# Performance benchmarking
class PerformanceBenchmark:
    """Benchmark different implementation approaches."""
    
    @staticmethod
    def benchmark_dataframe_operations():
        """Benchmark different DataFrame operations."""
        df = pd.DataFrame({
            'A': np.random.randn(100000),
            'B': np.random.randn(100000),
            'C': np.random.choice(['X', 'Y', 'Z'], 100000)
        })
        
        # Benchmark different groupby methods
        results = {}
        
        # Method 1: Standard groupby
        start_time = time.time()
        result1 = df.groupby('C')['A'].mean()
        results['standard_groupby'] = time.time() - start_time
        
        # Method 2: Using transform
        start_time = time.time()
        result2 = df['A'] / df.groupby('C')['A'].transform('mean')
        results['groupby_transform'] = time.time() - start_time
        
        # Method 3: Using numpy operations
        start_time = time.time()
        df_sorted = df.sort_values('C')
        result3 = df_sorted.groupby('C')['A'].mean()
        results['sorted_groupby'] = time.time() - start_time
        
        return results
```

### 2.2 Memory Management Best Practices

#### Memory Optimization Techniques

1. **Data Type Optimization**
   ```python
   # Use appropriate data types
   df['small_int'] = df['large_int'].astype('int32')  # vs int64
   df['category_col'] = df['string_col'].astype('category')  # vs object
   df['float_col'] = df['double_col'].astype('float32')  # vs float64
   ```

2. **Chunked Processing**
   ```python
   def process_large_file(file_path, chunk_size=10000):
       for chunk in pd.read_csv(file_path, chunksize=chunk_size):
           processed_chunk = process_chunk(chunk)
           yield processed_chunk
   ```

3. **Memory Monitoring**
   ```python
   import psutil
   
   def monitor_memory():
       process = psutil.Process()
       memory_info = process.memory_info()
       return memory_info.rss / 1024 / 1024  # MB
   ```

### 2.3 Performance Benchmarks and Targets

| Operation | Target Time | Acceptable Limit | Critical Threshold |
|-----------|-------------|------------------|-------------------|
| **Data Ingestion (1M records)** | <30 seconds | <60 seconds | >120 seconds |
| **Feature Engineering** | <5 minutes | <10 minutes | >20 minutes |
| **Data Validation (full suite)** | <2 minutes | <5 minutes | >10 minutes |
| **Model Training Data Prep** | <10 minutes | <20 minutes | >30 minutes |
| **Real-time Processing (per record)** | <50ms | <100ms | >500ms |
| **Report Generation** | <30 seconds | <60 seconds | >120 seconds |

## 3. Error Handling and Recovery

### 3.1 Robust Error Handling Framework

```python
"""
Comprehensive error handling and recovery framework.
"""

import logging
import traceback
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time
from functools import wraps
import pickle
from datetime import datetime, timedelta

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    HALT = "halt"
    MANUAL = "manual"

@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_type: str
    error_message: str
    stack_trace: str
    timestamp: datetime
    component: str
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    retry_count: int = 0
    metadata: Dict[str, Any] = None

class DataProcessingError(Exception):
    """Base exception for data processing errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
                 metadata: Dict[str, Any] = None):
        super().__init__(message)
        self.severity = severity
        self.recovery_strategy = recovery_strategy
        self.metadata = metadata or {}

class DataQualityError(DataProcessingError):
    """Exception for data quality issues."""
    pass

class IntegrationError(DataProcessingError):
    """Exception for data integration issues."""
    pass

class ErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_history = []
        self.recovery_strategies = {}
        self.circuit_breakers = {}
        
        # Default recovery strategies
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """Setup default recovery strategies."""
        self.recovery_strategies = {
            RecoveryStrategy.RETRY: self._retry_strategy,
            RecoveryStrategy.SKIP: self._skip_strategy,
            RecoveryStrategy.FALLBACK: self._fallback_strategy,
            RecoveryStrategy.HALT: self._halt_strategy,
            RecoveryStrategy.MANUAL: self._manual_strategy
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorContext:
        """Handle error with appropriate recovery strategy."""
        error_context = ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            timestamp=datetime.now(),
            component=context.get('component', 'unknown') if context else 'unknown',
            severity=getattr(error, 'severity', ErrorSeverity.MEDIUM),
            recovery_strategy=getattr(error, 'recovery_strategy', RecoveryStrategy.RETRY),
            metadata=getattr(error, 'metadata', {})
        )
        
        # Log error
        self._log_error(error_context)
        
        # Store in history
        self.error_history.append(error_context)
        
        # Apply recovery strategy
        self._apply_recovery_strategy(error_context)
        
        return error_context
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate severity level."""
        log_message = (
            f"Error in {error_context.component}: {error_context.error_message}"
        )
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _apply_recovery_strategy(self, error_context: ErrorContext):
        """Apply appropriate recovery strategy."""
        strategy_func = self.recovery_strategies.get(error_context.recovery_strategy)
        if strategy_func:
            strategy_func(error_context)
    
    def _retry_strategy(self, error_context: ErrorContext, max_retries: int = 3):
        """Implement retry strategy with exponential backoff."""
        if error_context.retry_count < max_retries:
            backoff_time = 2 ** error_context.retry_count
            self.logger.info(f"Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
            error_context.retry_count += 1
        else:
            self.logger.error(f"Max retries ({max_retries}) exceeded")
            self._escalate_error(error_context)
    
    def _skip_strategy(self, error_context: ErrorContext):
        """Skip the failed operation and continue."""
        self.logger.warning(f"Skipping failed operation: {error_context.error_message}")
    
    def _fallback_strategy(self, error_context: ErrorContext):
        """Use fallback method or default values."""
        self.logger.info("Using fallback strategy")
        # Implementation would use alternative data sources or methods
    
    def _halt_strategy(self, error_context: ErrorContext):
        """Halt processing due to critical error."""
        self.logger.critical("Halting processing due to critical error")
        raise DataProcessingError(
            f"Critical error: {error_context.error_message}",
            severity=ErrorSeverity.CRITICAL
        )
    
    def _manual_strategy(self, error_context: ErrorContext):
        """Require manual intervention."""
        self.logger.warning("Manual intervention required")
        # Would trigger alerts to operations team
    
    def _escalate_error(self, error_context: ErrorContext):
        """Escalate error to higher severity level."""
        error_context.severity = ErrorSeverity.HIGH
        self.logger.error(f"Escalating error: {error_context.error_message}")
        # Would trigger additional alerting

# Decorators for error handling
def handle_data_processing_errors(recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
                                 severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Decorator for handling data processing errors."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = ErrorHandler()
            max_attempts = 3 if recovery_strategy == RecoveryStrategy.RETRY else 1
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        # Last attempt, handle error
                        context = {
                            'component': func.__name__,
                            'attempt': attempt + 1
                        }
                        error_handler.handle_error(e, context)
                        raise
                    else:
                        # Retry with backoff
                        time.sleep(2 ** attempt)
            
            return wrapper
    return decorator

# Circuit breaker pattern
class CircuitBreaker:
    """Implement circuit breaker pattern for external service calls."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise IntegrationError(
                        "Circuit breaker is OPEN",
                        severity=ErrorSeverity.HIGH,
                        recovery_strategy=RecoveryStrategy.FALLBACK
                    )
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        )
    
    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
```

### 3.2 Recovery Strategies by Error Type

| Error Type | Recovery Strategy | Max Retries | Backoff | Escalation |
|------------|------------------|-------------|---------|------------|
| **Network Timeout** | Retry with exponential backoff | 5 | 2^n seconds | After 5 failures |
| **Data Quality Issue** | Skip record, log for review | 0 | N/A | After 100 skipped |
| **Authentication Error** | Retry with token refresh | 3 | 5 seconds | After 3 failures |
| **Rate Limit Exceeded** | Wait and retry | 10 | Rate limit window | After 10 failures |
| **Memory Error** | Chunk processing fallback | 1 | N/A | Immediate |
| **Critical System Error** | Halt processing | 0 | N/A | Immediate |

## 4. Monitoring and Alerting

### 4.1 Comprehensive Monitoring Framework

```python
"""
Monitoring and alerting framework for data processing pipelines.
"""

import time
import psutil
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Metric:
    """Represents a monitoring metric."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class Alert:
    """Represents an alert condition."""
    name: str
    condition: str
    threshold: float
    severity: AlertSeverity
    message: str
    recipients: List[str] = field(default_factory=list)

class MetricsCollector:
    """Collect and manage system and application metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.timeseries_data = []
        self.lock = threading.Lock()
        
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None,
                     metric_type: MetricType = MetricType.GAUGE):
        """Record a metric value."""
        with self.lock:
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {},
                metric_type=metric_type
            )
            
            self.metrics[name] = metric
            self.timeseries_data.append(metric)
    
    def increment_counter(self, name: str, increment: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        current_value = self.metrics.get(name, Metric(name, 0, datetime.now())).value
        self.record_metric(name, current_value + increment, tags, MetricType.COUNTER)
    
    def record_timer(self, name: str, start_time: float, tags: Dict[str, str] = None):
        """Record timing information."""
        duration = time.time() - start_time
        self.record_metric(name, duration, tags, MetricType.TIMER)
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Collect system performance metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_metrics = {
            'system.cpu.percent': cpu_percent,
            'system.memory.percent': memory.percent,
            'system.memory.available_gb': memory.available / (1024**3),
            'system.disk.percent': disk.percent,
            'system.disk.free_gb': disk.free / (1024**3)
        }
        
        for metric_name, value in system_metrics.items():
            self.record_metric(metric_name, value)
        
        return system_metrics
    
    def get_process_metrics(self) -> Dict[str, float]:
        """Collect process-specific metrics."""
        process = psutil.Process()
        
        process_metrics = {
            'process.cpu.percent': process.cpu_percent(),
            'process.memory.mb': process.memory_info().rss / (1024**2),
            'process.threads': process.num_threads(),
            'process.open_files': len(process.open_files())
        }
        
        for metric_name, value in process_metrics.items():
            self.record_metric(metric_name, value)
        
        return process_metrics

class DataQualityMonitor:
    """Monitor data quality metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
    
    def monitor_data_quality(self, df: pd.DataFrame, dataset_name: str):
        """Monitor data quality for a dataset."""
        tags = {'dataset': dataset_name}
        
        # Completeness metrics
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness_rate = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0
        
        self.metrics_collector.record_metric(
            'data_quality.completeness_rate',
            completeness_rate,
            tags
        )
        
        # Uniqueness metrics (for each column)
        for column in df.columns:
            if df[column].dtype in ['object', 'category']:
                unique_rate = df[column].nunique() / len(df) if len(df) > 0 else 0
                column_tags = {**tags, 'column': column}
                self.metrics_collector.record_metric(
                    'data_quality.uniqueness_rate',
                    unique_rate,
                    column_tags
                )
        
        # Validity metrics (for numeric columns)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            # Check for extreme outliers (beyond 3 standard deviations)
            if len(df[column].dropna()) > 0:
                mean = df[column].mean()
                std = df[column].std()
                outliers = np.abs(df[column] - mean) > 3 * std
                validity_rate = 1 - (outliers.sum() / len(df))
                
                column_tags = {**tags, 'column': column}
                self.metrics_collector.record_metric(
                    'data_quality.validity_rate',
                    validity_rate,
                    column_tags
                )

class PerformanceMonitor:
    """Monitor pipeline performance metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.operation_timers = {}
    
    def start_timer(self, operation_name: str) -> str:
        """Start timing an operation."""
        timer_id = f"{operation_name}_{int(time.time() * 1000)}"
        self.operation_timers[timer_id] = time.time()
        return timer_id
    
    def end_timer(self, timer_id: str, operation_name: str, tags: Dict[str, str] = None):
        """End timing an operation."""
        if timer_id in self.operation_timers:
            start_time = self.operation_timers.pop(timer_id)
            self.metrics_collector.record_timer(
                f'performance.{operation_name}.duration',
                start_time,
                tags
            )
    
    def monitor_throughput(self, records_processed: int, operation_name: str, 
                          duration_seconds: float, tags: Dict[str, str] = None):
        """Monitor processing throughput."""
        throughput = records_processed / duration_seconds if duration_seconds > 0 else 0
        self.metrics_collector.record_metric(
            f'performance.{operation_name}.throughput',
            throughput,
            tags
        )

class AlertManager:
    """Manage alerts based on metric thresholds."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts = []
        self.alert_history = []
        self.notification_queue = queue.Queue()
        
        # Start notification worker thread
        self.notification_worker = threading.Thread(
            target=self._process_notifications,
            daemon=True
        )
        self.notification_worker.start()
    
    def add_alert(self, alert: Alert):
        """Add an alert rule."""
        self.alerts.append(alert)
    
    def check_alerts(self):
        """Check all alert conditions."""
        triggered_alerts = []
        
        for alert in self.alerts:
            metric = self.metrics_collector.metrics.get(alert.name)
            if metric and self._evaluate_condition(metric.value, alert):
                triggered_alert = {
                    'alert': alert,
                    'metric_value': metric.value,
                    'timestamp': datetime.now()
                }
                triggered_alerts.append(triggered_alert)
                self.alert_history.append(triggered_alert)
                
                # Queue notification
                self.notification_queue.put(triggered_alert)
        
        return triggered_alerts
    
    def _evaluate_condition(self, value: float, alert: Alert) -> bool:
        """Evaluate alert condition."""
        if alert.condition == 'greater_than':
            return value > alert.threshold
        elif alert.condition == 'less_than':
            return value < alert.threshold
        elif alert.condition == 'equals':
            return value == alert.threshold
        else:
            return False
    
    def _process_notifications(self):
        """Process notification queue."""
        while True:
            try:
                triggered_alert = self.notification_queue.get(timeout=1)
                self._send_notification(triggered_alert)
                self.notification_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing notification: {e}")
    
    def _send_notification(self, triggered_alert: Dict[str, Any]):
        """Send alert notification."""
        alert = triggered_alert['alert']
        
        # Email notification (simplified)
        if alert.recipients:
            subject = f"ALERT: {alert.name} - {alert.severity.value.upper()}"
            message = f"""
            Alert: {alert.name}
            Severity: {alert.severity.value}
            Current Value: {triggered_alert['metric_value']}
            Threshold: {alert.threshold}
            Time: {triggered_alert['timestamp']}
            
            {alert.message}
            """
            
            # Would implement actual email sending here
            logging.warning(f"ALERT TRIGGERED: {subject}")

# Example usage and setup
def setup_monitoring():
    """Setup comprehensive monitoring for data processing pipeline."""
    
    # Initialize components
    metrics_collector = MetricsCollector()
    data_quality_monitor = DataQualityMonitor(metrics_collector)
    performance_monitor = PerformanceMonitor(metrics_collector)
    alert_manager = AlertManager(metrics_collector)
    
    # Setup alerts
    alerts = [
        Alert(
            name="data_quality.completeness_rate",
            condition="less_than",
            threshold=0.95,
            severity=AlertSeverity.WARNING,
            message="Data completeness below 95%",
            recipients=["data-team@company.com"]
        ),
        Alert(
            name="system.memory.percent",
            condition="greater_than",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            message="System memory usage above 80%",
            recipients=["ops-team@company.com"]
        ),
        Alert(
            name="performance.data_processing.duration",
            condition="greater_than",
            threshold=300.0,  # 5 minutes
            severity=AlertSeverity.CRITICAL,
            message="Data processing taking longer than 5 minutes",
            recipients=["data-team@company.com", "ops-team@company.com"]
        )
    ]
    
    for alert in alerts:
        alert_manager.add_alert(alert)
    
    return {
        'metrics_collector': metrics_collector,
        'data_quality_monitor': data_quality_monitor,
        'performance_monitor': performance_monitor,
        'alert_manager': alert_manager
    }
```

### 4.2 Key Performance Indicators (KPIs)

#### System Health KPIs
- **Uptime**: >99.9% monthly availability
- **Response Time**: <2 seconds for API calls
- **Error Rate**: <0.1% of total operations
- **Memory Usage**: <80% peak utilization
- **CPU Usage**: <70% average utilization

#### Data Quality KPIs
- **Completeness**: >95% non-null values
- **Accuracy**: >98% valid records
- **Timeliness**: <1 hour data freshness
- **Consistency**: >99% cross-system agreement
- **Uniqueness**: <0.1% duplicate records

#### Processing Performance KPIs
- **Throughput**: >10,000 records/minute
- **Latency**: <100ms per record processing
- **Pipeline Success Rate**: >99.5%
- **Resource Efficiency**: <$0.01 per 1000 records
- **Scalability**: Linear scaling up to 10x load

## 5. Security and Compliance

### 5.1 Data Security Best Practices

#### Access Controls
```python
# Role-based access control example
RBAC_PERMISSIONS = {
    'data_analyst': ['read_financial_data', 'read_reports'],
    'data_engineer': ['read_financial_data', 'write_processed_data', 'run_pipelines'],
    'data_scientist': ['read_financial_data', 'read_processed_data', 'train_models'],
    'admin': ['*']  # All permissions
}

def check_permission(user_role: str, required_permission: str) -> bool:
    user_permissions = RBAC_PERMISSIONS.get(user_role, [])
    return '*' in user_permissions or required_permission in user_permissions
```

#### Data Encryption
- **At Rest**: AES-256 encryption for all stored data
- **In Transit**: TLS 1.3 for all data transfers
- **Key Management**: AWS KMS or Azure Key Vault integration
- **Field-Level Encryption**: Sensitive fields encrypted separately

#### Audit Logging
```python
class AuditLogger:
    """Comprehensive audit logging for compliance."""
    
    def log_data_access(self, user_id: str, dataset: str, action: str, 
                       timestamp: datetime = None):
        audit_record = {
            'user_id': user_id,
            'dataset': dataset,
            'action': action,
            'timestamp': timestamp or datetime.now(),
            'ip_address': self.get_client_ip(),
            'session_id': self.get_session_id()
        }
        # Write to secure audit log
        self.write_audit_record(audit_record)
```

### 5.2 Regulatory Compliance

#### Australian Financial Regulations
- **ASIC Compliance**: Maintain data lineage for regulatory reporting
- **Privacy Act 1988**: Implement privacy by design principles
- **Banking Act**: Ensure data security for banking relationships
- **Corporations Act**: Maintain audit trails for financial analysis

#### Data Governance Framework
1. **Data Classification**: Public, Internal, Confidential, Restricted
2. **Retention Policies**: Automated data lifecycle management
3. **Data Lineage**: Complete traceability of data transformations
4. **Quality Controls**: Automated validation and approval workflows

## 6. Documentation and Knowledge Management

### 6.1 Documentation Standards

#### Code Documentation
```python
def calculate_financial_ratios(financial_data: pd.DataFrame, 
                             config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Calculate comprehensive financial ratios for company analysis.
    
    This function computes profitability, liquidity, leverage, and efficiency
    ratios based on financial statement data. All calculations follow
    Australian accounting standards (AASB).
    
    Args:
        financial_data (pd.DataFrame): DataFrame containing financial statement data
            Required columns: ['revenue', 'net_income', 'total_assets', 
            'current_assets', 'current_liabilities', 'total_liabilities', 
            'shareholders_equity']
        config (Dict[str, Any], optional): Configuration for ratio calculations
            - 'handle_missing': Method for handling missing values ('drop', 'interpolate')
            - 'industry_adjust': Boolean to adjust for industry benchmarks
            - 'include_growth_ratios': Boolean to include growth rate calculations
    
    Returns:
        pd.DataFrame: Original data with additional ratio columns
            New columns include: ['roa', 'roe', 'current_ratio', 'debt_to_equity', 
            'gross_margin', 'net_margin', 'asset_turnover']
    
    Raises:
        ValueError: If required columns are missing from financial_data
        DataQualityError: If data quality checks fail
    
    Example:
        >>> import pandas as pd
        >>> financial_data = pd.DataFrame({
        ...     'revenue': [1000000, 1100000],
        ...     'net_income': [50000, 60000],
        ...     'total_assets': [500000, 520000],
        ...     'shareholders_equity': [300000, 320000]
        ... })
        >>> ratios = calculate_financial_ratios(financial_data)
        >>> print(ratios[['roa', 'roe']].round(3))
               roa    roe
        0    0.100  0.167
        1    0.115  0.188
    
    Note:
        - All ratios are calculated as decimal values (not percentages)
        - Missing values in source data may result in NaN ratios
        - Industry adjustment requires additional industry classification data
    
    References:
        - AASB 101: Presentation of Financial Statements
        - AASB 107: Statement of Cash Flows
        - CPA Australia Financial Ratio Analysis Guide
    """
    # Implementation here...
```

#### Process Documentation Template
```markdown
# Process: [Process Name]

## Purpose
Brief description of what this process does and why it's important.

## Scope
What data/systems this process covers and any exclusions.

## Prerequisites
- Required data sources
- System dependencies  
- Access permissions needed

## Step-by-Step Process
1. **Step 1 Name**
   - Detailed instructions
   - Expected inputs/outputs
   - Quality checks to perform

## Configuration
- Configuration parameters
- Environment-specific settings
- Default values and ranges

## Error Handling
- Common error scenarios
- Troubleshooting steps
- Escalation procedures

## Quality Assurance
- Validation steps
- Quality metrics to monitor
- Acceptance criteria

## Dependencies
- Upstream processes
- Downstream consumers
- External dependencies

## Change History
| Date | Change | Author | Version |
|------|--------|---------|---------|
| 2023-12-01 | Initial version | J. Smith | 1.0 |
```

### 6.2 Knowledge Management System

#### Decision Documentation
- **Architecture Decision Records (ADRs)**: Document technical decisions
- **Data Model Documentation**: Schema definitions and relationships  
- **API Documentation**: Comprehensive endpoint documentation
- **Operational Runbooks**: Step-by-step operational procedures

#### Training and Onboarding
- **Developer Onboarding Guide**: 2-week structured program
- **Data Processing Certification**: Internal certification program
- **Best Practices Workshops**: Monthly knowledge sharing sessions
- **Code Review Guidelines**: Standards for code quality and reviews

This comprehensive best practices guide ensures that the IPO valuation platform maintains the highest standards of data quality, performance, security, and reliability while remaining compliant with Australian financial regulations.