"""
Model Performance Monitoring and Drift Detection
Comprehensive monitoring for ML models in production
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta
import json
import time
from pathlib import Path
from collections import defaultdict, deque
import sqlite3

# Statistical libraries
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KolmogorovSmirnovTest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox

# Alerting and notifications
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformanceMetrics:
    """Model performance metrics tracking"""
    model_name: str
    model_version: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Prediction accuracy metrics
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None  # Mean Absolute Percentage Error
    r2: Optional[float] = None
    
    # Business metrics
    prediction_count: int = 0
    average_prediction_time_ms: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    
    # Quality metrics
    average_confidence_score: float = 0.0
    low_confidence_predictions: int = 0
    outlier_predictions: int = 0
    
    # Data quality indicators
    missing_features_rate: float = 0.0
    out_of_range_features: int = 0
    data_quality_score: float = 1.0

@dataclass
class DataDriftMetrics:
    """Data drift detection metrics"""
    feature_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Distribution comparison
    ks_statistic: float = 0.0
    ks_pvalue: float = 1.0
    js_divergence: float = 0.0  # Jensen-Shannon divergence
    wasserstein_distance: float = 0.0
    
    # Statistical properties
    mean_shift: float = 0.0
    std_shift: float = 0.0
    quantile_shifts: Dict[str, float] = field(default_factory=dict)
    
    # Drift indicators
    drift_detected: bool = False
    drift_severity: str = "none"  # none, low, medium, high, critical
    drift_confidence: float = 0.0

@dataclass
class ModelDriftMetrics:
    """Model drift detection metrics"""
    model_name: str
    model_version: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Performance drift
    performance_degradation: float = 0.0
    accuracy_drift: float = 0.0
    prediction_distribution_shift: float = 0.0
    
    # Feature importance drift
    feature_importance_changes: Dict[str, float] = field(default_factory=dict)
    
    # Concept drift indicators
    concept_drift_detected: bool = False
    drift_detection_method: str = ""
    drift_magnitude: float = 0.0
    
    # Model stability
    prediction_variance_change: float = 0.0
    confidence_calibration_drift: float = 0.0

@dataclass
class Alert:
    """Monitoring alert"""
    alert_id: str
    alert_type: str  # performance, data_drift, model_drift, system
    severity: str  # low, medium, high, critical
    model_name: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Alert details
    metric_name: str = ""
    current_value: float = 0.0
    threshold_value: float = 0.0
    
    # Alert status
    status: str = "active"  # active, acknowledged, resolved
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Notification status
    notifications_sent: List[str] = field(default_factory=list)

class MetricsCollector:
    """Collects and stores model performance metrics"""
    
    def __init__(self, storage_path: str = "monitoring/metrics.db"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # In-memory buffers for real-time metrics
        self.prediction_buffer = deque(maxlen=10000)
        self.performance_buffer = deque(maxlen=1000)
        
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    mae REAL,
                    rmse REAL,
                    mape REAL,
                    r2 REAL,
                    prediction_count INTEGER,
                    avg_prediction_time_ms REAL,
                    error_rate REAL,
                    cache_hit_rate REAL,
                    avg_confidence_score REAL,
                    low_confidence_predictions INTEGER,
                    outlier_predictions INTEGER,
                    missing_features_rate REAL,
                    out_of_range_features INTEGER,
                    data_quality_score REAL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS data_drift_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_name TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    ks_statistic REAL,
                    ks_pvalue REAL,
                    js_divergence REAL,
                    wasserstein_distance REAL,
                    mean_shift REAL,
                    std_shift REAL,
                    drift_detected BOOLEAN,
                    drift_severity TEXT,
                    drift_confidence REAL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_drift_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    performance_degradation REAL,
                    accuracy_drift REAL,
                    prediction_distribution_shift REAL,
                    concept_drift_detected BOOLEAN,
                    drift_detection_method TEXT,
                    drift_magnitude REAL,
                    prediction_variance_change REAL,
                    confidence_calibration_drift REAL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    prediction_value REAL,
                    confidence_score REAL,
                    prediction_time_ms REAL,
                    features TEXT,  -- JSON string of features
                    true_value REAL  -- For performance calculation when available
                )
            ''')
    
    async def record_prediction(
        self,
        model_name: str,
        prediction_value: float,
        confidence_score: float,
        prediction_time_ms: float,
        features: Dict[str, Any],
        true_value: Optional[float] = None
    ):
        """Record individual prediction for monitoring"""
        
        prediction_record = {
            'model_name': model_name,
            'timestamp': datetime.now(),
            'prediction_value': prediction_value,
            'confidence_score': confidence_score,
            'prediction_time_ms': prediction_time_ms,
            'features': json.dumps(features),
            'true_value': true_value
        }
        
        # Add to buffer
        self.prediction_buffer.append(prediction_record)
        
        # Store in database
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute('''
                INSERT INTO predictions_log 
                (model_name, timestamp, prediction_value, confidence_score, 
                 prediction_time_ms, features, true_value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_record['model_name'],
                prediction_record['timestamp'],
                prediction_record['prediction_value'],
                prediction_record['confidence_score'],
                prediction_record['prediction_time_ms'],
                prediction_record['features'],
                prediction_record['true_value']
            ))
    
    async def record_performance_metrics(self, metrics: ModelPerformanceMetrics):
        """Record model performance metrics"""
        
        self.performance_buffer.append(metrics)
        
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute('''
                INSERT INTO performance_metrics
                (model_name, model_version, timestamp, mae, rmse, mape, r2,
                 prediction_count, avg_prediction_time_ms, error_rate, cache_hit_rate,
                 avg_confidence_score, low_confidence_predictions, outlier_predictions,
                 missing_features_rate, out_of_range_features, data_quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.model_name, metrics.model_version, metrics.timestamp,
                metrics.mae, metrics.rmse, metrics.mape, metrics.r2,
                metrics.prediction_count, metrics.average_prediction_time_ms,
                metrics.error_rate, metrics.cache_hit_rate,
                metrics.average_confidence_score, metrics.low_confidence_predictions,
                metrics.outlier_predictions, metrics.missing_features_rate,
                metrics.out_of_range_features, metrics.data_quality_score
            ))
    
    async def get_performance_history(
        self,
        model_name: str,
        hours: int = 24
    ) -> List[ModelPerformanceMetrics]:
        """Get performance metrics history"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM performance_metrics 
                WHERE model_name = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            ''', (model_name, cutoff_time))
            
            results = []
            for row in cursor.fetchall():
                metrics = ModelPerformanceMetrics(
                    model_name=row[1],
                    model_version=row[2],
                    timestamp=datetime.fromisoformat(row[3]),
                    mae=row[4],
                    rmse=row[5],
                    mape=row[6],
                    r2=row[7],
                    prediction_count=row[8],
                    average_prediction_time_ms=row[9],
                    error_rate=row[10],
                    cache_hit_rate=row[11],
                    average_confidence_score=row[12],
                    low_confidence_predictions=row[13],
                    outlier_predictions=row[14],
                    missing_features_rate=row[15],
                    out_of_range_features=row[16],
                    data_quality_score=row[17]
                )
                results.append(metrics)
            
            return results
    
    async def get_recent_predictions(
        self,
        model_name: str,
        hours: int = 1,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get recent predictions for analysis"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.storage_path) as conn:
            query = '''
                SELECT * FROM predictions_log 
                WHERE model_name = ? AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
            '''
            
            df = pd.read_sql_query(query, conn, params=(model_name, cutoff_time, limit))
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Parse features JSON
                df['features_dict'] = df['features'].apply(
                    lambda x: json.loads(x) if x else {}
                )
            
            return df

class DataDriftDetector:
    """Detects data drift in model features"""
    
    def __init__(self, reference_window_days: int = 30):
        self.reference_window_days = reference_window_days
        self.reference_distributions = {}
        self.drift_thresholds = {
            'ks_pvalue': 0.05,
            'js_divergence': 0.1,
            'wasserstein_distance': 0.2
        }
    
    async def update_reference_distribution(
        self,
        feature_name: str,
        data: np.ndarray
    ):
        """Update reference distribution for feature"""
        
        self.reference_distributions[feature_name] = {
            'data': data.copy(),
            'mean': np.mean(data),
            'std': np.std(data),
            'quantiles': np.percentile(data, [25, 50, 75]),
            'updated_at': datetime.now()
        }
    
    async def detect_drift(
        self,
        feature_name: str,
        current_data: np.ndarray,
        min_samples: int = 100
    ) -> DataDriftMetrics:
        """Detect drift for a specific feature"""
        
        if feature_name not in self.reference_distributions:
            raise ValueError(f"No reference distribution for feature: {feature_name}")
        
        if len(current_data) < min_samples:
            logger.warning(f"Insufficient samples for drift detection: {len(current_data)}")
            return DataDriftMetrics(
                feature_name=feature_name,
                drift_detected=False,
                drift_severity="insufficient_data"
            )
        
        reference = self.reference_distributions[feature_name]
        reference_data = reference['data']
        
        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.ks_2samp(reference_data, current_data)
        
        # Jensen-Shannon divergence
        js_divergence = self._calculate_js_divergence(reference_data, current_data)
        
        # Wasserstein distance
        wasserstein_distance = stats.wasserstein_distance(reference_data, current_data)
        
        # Statistical property shifts
        current_mean = np.mean(current_data)
        current_std = np.std(current_data)
        
        mean_shift = abs(current_mean - reference['mean']) / reference['std']
        std_shift = abs(current_std - reference['std']) / reference['std']
        
        # Quantile shifts
        current_quantiles = np.percentile(current_data, [25, 50, 75])
        quantile_shifts = {
            'q25': abs(current_quantiles[0] - reference['quantiles'][0]),
            'q50': abs(current_quantiles[1] - reference['quantiles'][1]),
            'q75': abs(current_quantiles[2] - reference['quantiles'][2])
        }
        
        # Drift detection
        drift_detected = (
            ks_pvalue < self.drift_thresholds['ks_pvalue'] or
            js_divergence > self.drift_thresholds['js_divergence'] or
            wasserstein_distance > self.drift_thresholds['wasserstein_distance']
        )
        
        # Drift severity
        if not drift_detected:
            drift_severity = "none"
        elif js_divergence > 0.3 or wasserstein_distance > 0.5:
            drift_severity = "critical"
        elif js_divergence > 0.2 or wasserstein_distance > 0.3:
            drift_severity = "high"
        elif js_divergence > 0.15 or wasserstein_distance > 0.25:
            drift_severity = "medium"
        else:
            drift_severity = "low"
        
        # Confidence in drift detection
        drift_confidence = 1 - ks_pvalue if drift_detected else ks_pvalue
        
        return DataDriftMetrics(
            feature_name=feature_name,
            ks_statistic=ks_statistic,
            ks_pvalue=ks_pvalue,
            js_divergence=js_divergence,
            wasserstein_distance=wasserstein_distance,
            mean_shift=mean_shift,
            std_shift=std_shift,
            quantile_shifts=quantile_shifts,
            drift_detected=drift_detected,
            drift_severity=drift_severity,
            drift_confidence=drift_confidence
        )
    
    def _calculate_js_divergence(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence between two distributions"""
        
        # Create histograms
        min_val = min(np.min(data1), np.min(data2))
        max_val = max(np.max(data1), np.max(data2))
        bins = np.linspace(min_val, max_val, 50)
        
        hist1, _ = np.histogram(data1, bins=bins, density=True)
        hist2, _ = np.histogram(data2, bins=bins, density=True)
        
        # Normalize
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        hist1 = hist1 + epsilon
        hist2 = hist2 + epsilon
        
        # Jensen-Shannon divergence
        m = 0.5 * (hist1 + hist2)
        js_div = 0.5 * stats.entropy(hist1, m) + 0.5 * stats.entropy(hist2, m)
        
        return js_div

class ModelDriftDetector:
    """Detects model performance and concept drift"""
    
    def __init__(self, performance_window_size: int = 100):
        self.performance_window_size = performance_window_size
        self.baseline_performance = {}
        
    async def set_baseline_performance(
        self,
        model_name: str,
        baseline_metrics: ModelPerformanceMetrics
    ):
        """Set baseline performance metrics"""
        
        self.baseline_performance[model_name] = baseline_metrics
    
    async def detect_performance_drift(
        self,
        model_name: str,
        current_metrics: ModelPerformanceMetrics,
        performance_history: List[ModelPerformanceMetrics]
    ) -> ModelDriftMetrics:
        """Detect model performance drift"""
        
        if model_name not in self.baseline_performance:
            logger.warning(f"No baseline performance for model: {model_name}")
            return ModelDriftMetrics(
                model_name=model_name,
                model_version=current_metrics.model_version,
                concept_drift_detected=False,
                drift_detection_method="no_baseline"
            )
        
        baseline = self.baseline_performance[model_name]
        
        # Performance degradation
        performance_degradation = 0.0
        accuracy_drift = 0.0
        
        if baseline.mae and current_metrics.mae:
            performance_degradation = (current_metrics.mae - baseline.mae) / baseline.mae
        
        if baseline.r2 and current_metrics.r2:
            accuracy_drift = (baseline.r2 - current_metrics.r2) / baseline.r2
        
        # Analyze performance trend
        trend_analysis = await self._analyze_performance_trend(performance_history)
        
        # Prediction distribution analysis
        prediction_distribution_shift = await self._analyze_prediction_distribution_shift(
            model_name, performance_history
        )
        
        # Confidence calibration drift
        confidence_calibration_drift = await self._analyze_confidence_calibration(
            baseline, current_metrics
        )
        
        # Concept drift detection
        concept_drift_detected = (
            abs(performance_degradation) > 0.1 or  # 10% degradation
            abs(accuracy_drift) > 0.05 or         # 5% accuracy drop
            prediction_distribution_shift > 0.2    # Significant distribution shift
        )
        
        drift_magnitude = max(
            abs(performance_degradation),
            abs(accuracy_drift),
            prediction_distribution_shift
        )
        
        return ModelDriftMetrics(
            model_name=model_name,
            model_version=current_metrics.model_version,
            performance_degradation=performance_degradation,
            accuracy_drift=accuracy_drift,
            prediction_distribution_shift=prediction_distribution_shift,
            concept_drift_detected=concept_drift_detected,
            drift_detection_method="performance_analysis",
            drift_magnitude=drift_magnitude,
            prediction_variance_change=trend_analysis.get('variance_change', 0.0),
            confidence_calibration_drift=confidence_calibration_drift
        )
    
    async def _analyze_performance_trend(
        self,
        performance_history: List[ModelPerformanceMetrics]
    ) -> Dict[str, float]:
        """Analyze performance trends over time"""
        
        if len(performance_history) < 10:
            return {'variance_change': 0.0, 'trend_slope': 0.0}
        
        # Extract MAE values over time
        mae_values = [m.mae for m in performance_history if m.mae is not None]
        
        if len(mae_values) < 5:
            return {'variance_change': 0.0, 'trend_slope': 0.0}
        
        # Calculate variance change
        recent_variance = np.var(mae_values[:len(mae_values)//2])
        old_variance = np.var(mae_values[len(mae_values)//2:])
        
        variance_change = (recent_variance - old_variance) / old_variance if old_variance > 0 else 0
        
        # Calculate trend slope
        x = np.arange(len(mae_values))
        trend_slope, _, _, _, _ = stats.linregress(x, mae_values)
        
        return {
            'variance_change': variance_change,
            'trend_slope': trend_slope
        }
    
    async def _analyze_prediction_distribution_shift(
        self,
        model_name: str,
        performance_history: List[ModelPerformanceMetrics]
    ) -> float:
        """Analyze shift in prediction value distributions"""
        
        # This would analyze actual prediction values from the database
        # For now, return a simplified calculation based on confidence scores
        
        if len(performance_history) < 10:
            return 0.0
        
        recent_confidence = [m.average_confidence_score for m in performance_history[:5]]
        older_confidence = [m.average_confidence_score for m in performance_history[-5:]]
        
        if len(recent_confidence) == 0 or len(older_confidence) == 0:
            return 0.0
        
        # Simple distribution shift based on confidence score changes
        recent_mean = np.mean(recent_confidence)
        older_mean = np.mean(older_confidence)
        
        shift = abs(recent_mean - older_mean) / older_mean if older_mean > 0 else 0
        
        return shift
    
    async def _analyze_confidence_calibration(
        self,
        baseline: ModelPerformanceMetrics,
        current: ModelPerformanceMetrics
    ) -> float:
        """Analyze confidence calibration drift"""
        
        # Simplified calibration drift based on confidence score changes
        if baseline.average_confidence_score == 0 or current.average_confidence_score == 0:
            return 0.0
        
        calibration_drift = abs(
            current.average_confidence_score - baseline.average_confidence_score
        ) / baseline.average_confidence_score
        
        return calibration_drift

class AlertManager:
    """Manages monitoring alerts and notifications"""
    
    def __init__(self, storage_path: str = "monitoring/alerts.db"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Alert rules configuration
        self.alert_rules = self._load_default_alert_rules()
        
        # Active alerts tracking
        self.active_alerts = {}
        
    def _init_database(self):
        """Initialize alerts database"""
        
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    metric_name TEXT,
                    current_value REAL,
                    threshold_value REAL,
                    status TEXT DEFAULT 'active',
                    acknowledged_by TEXT,
                    acknowledged_at DATETIME,
                    resolved_at DATETIME,
                    notifications_sent TEXT  -- JSON array
                )
            ''')
    
    def _load_default_alert_rules(self) -> Dict[str, Dict]:
        """Load default alert rules"""
        
        return {
            'performance_degradation': {
                'metric': 'mae',
                'threshold': 0.1,  # 10% increase in MAE
                'severity': 'medium',
                'comparison': 'increase'
            },
            'accuracy_drop': {
                'metric': 'r2',
                'threshold': 0.05,  # 5% decrease in R2
                'severity': 'high',
                'comparison': 'decrease'
            },
            'low_confidence': {
                'metric': 'average_confidence_score',
                'threshold': 0.7,  # Below 70% confidence
                'severity': 'medium',
                'comparison': 'below'
            },
            'high_error_rate': {
                'metric': 'error_rate',
                'threshold': 0.05,  # Above 5% error rate
                'severity': 'high',
                'comparison': 'above'
            },
            'data_drift_detected': {
                'metric': 'drift_detected',
                'threshold': True,
                'severity': 'medium',
                'comparison': 'equals'
            },
            'concept_drift_detected': {
                'metric': 'concept_drift_detected',
                'threshold': True,
                'severity': 'high',
                'comparison': 'equals'
            }
        }
    
    async def check_alerts(
        self,
        model_name: str,
        performance_metrics: ModelPerformanceMetrics,
        data_drift_metrics: List[DataDriftMetrics] = None,
        model_drift_metrics: ModelDriftMetrics = None
    ) -> List[Alert]:
        """Check for alert conditions and create alerts"""
        
        new_alerts = []
        
        # Check performance alerts
        for rule_name, rule in self.alert_rules.items():
            if rule_name.startswith('performance') or rule_name.startswith('accuracy') or rule_name.startswith('low_confidence') or rule_name.startswith('high_error'):
                alert = await self._check_performance_alert(
                    model_name, performance_metrics, rule_name, rule
                )
                if alert:
                    new_alerts.append(alert)
        
        # Check data drift alerts
        if data_drift_metrics:
            for drift_metric in data_drift_metrics:
                if drift_metric.drift_detected:
                    alert = await self._create_data_drift_alert(model_name, drift_metric)
                    if alert:
                        new_alerts.append(alert)
        
        # Check model drift alerts
        if model_drift_metrics and model_drift_metrics.concept_drift_detected:
            alert = await self._create_model_drift_alert(model_name, model_drift_metrics)
            if alert:
                new_alerts.append(alert)
        
        # Store new alerts
        for alert in new_alerts:
            await self._store_alert(alert)
            self.active_alerts[alert.alert_id] = alert
        
        return new_alerts
    
    async def _check_performance_alert(
        self,
        model_name: str,
        metrics: ModelPerformanceMetrics,
        rule_name: str,
        rule: Dict
    ) -> Optional[Alert]:
        """Check individual performance alert rule"""
        
        metric_name = rule['metric']
        threshold = rule['threshold']
        comparison = rule['comparison']
        severity = rule['severity']
        
        # Get current metric value
        current_value = getattr(metrics, metric_name, None)
        if current_value is None:
            return None
        
        # Check threshold
        alert_triggered = False
        if comparison == 'above' and current_value > threshold:
            alert_triggered = True
        elif comparison == 'below' and current_value < threshold:
            alert_triggered = True
        elif comparison == 'increase':
            # Need baseline comparison - simplified for now
            alert_triggered = current_value > threshold
        elif comparison == 'decrease':
            # Need baseline comparison - simplified for now
            alert_triggered = current_value < (1 - threshold)
        
        if not alert_triggered:
            return None
        
        # Create alert
        alert_id = f"{model_name}_{rule_name}_{int(time.time())}"
        
        message = f"Model {model_name}: {metric_name} is {current_value:.3f}, threshold: {threshold}"
        
        return Alert(
            alert_id=alert_id,
            alert_type="performance",
            severity=severity,
            model_name=model_name,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold
        )
    
    async def _create_data_drift_alert(
        self,
        model_name: str,
        drift_metrics: DataDriftMetrics
    ) -> Optional[Alert]:
        """Create data drift alert"""
        
        alert_id = f"{model_name}_data_drift_{drift_metrics.feature_name}_{int(time.time())}"
        
        message = (f"Data drift detected for feature '{drift_metrics.feature_name}' "
                  f"in model {model_name}. Severity: {drift_metrics.drift_severity}, "
                  f"Confidence: {drift_metrics.drift_confidence:.2f}")
        
        return Alert(
            alert_id=alert_id,
            alert_type="data_drift",
            severity=drift_metrics.drift_severity,
            model_name=model_name,
            message=message,
            metric_name="data_drift",
            current_value=drift_metrics.drift_confidence,
            threshold_value=0.8
        )
    
    async def _create_model_drift_alert(
        self,
        model_name: str,
        drift_metrics: ModelDriftMetrics
    ) -> Optional[Alert]:
        """Create model drift alert"""
        
        alert_id = f"{model_name}_model_drift_{int(time.time())}"
        
        message = (f"Concept drift detected in model {model_name}. "
                  f"Magnitude: {drift_metrics.drift_magnitude:.3f}, "
                  f"Performance degradation: {drift_metrics.performance_degradation:.3f}")
        
        return Alert(
            alert_id=alert_id,
            alert_type="model_drift",
            severity="high",
            model_name=model_name,
            message=message,
            metric_name="concept_drift",
            current_value=drift_metrics.drift_magnitude,
            threshold_value=0.1
        )
    
    async def _store_alert(self, alert: Alert):
        """Store alert in database"""
        
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO alerts
                (alert_id, alert_type, severity, model_name, message, timestamp,
                 metric_name, current_value, threshold_value, status,
                 acknowledged_by, acknowledged_at, resolved_at, notifications_sent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id, alert.alert_type, alert.severity, alert.model_name,
                alert.message, alert.timestamp, alert.metric_name, alert.current_value,
                alert.threshold_value, alert.status, alert.acknowledged_by,
                alert.acknowledged_at, alert.resolved_at,
                json.dumps(alert.notifications_sent)
            ))
    
    async def get_active_alerts(self, model_name: Optional[str] = None) -> List[Alert]:
        """Get active alerts"""
        
        with sqlite3.connect(self.storage_path) as conn:
            if model_name:
                cursor = conn.execute(
                    "SELECT * FROM alerts WHERE model_name = ? AND status = 'active' ORDER BY timestamp DESC",
                    (model_name,)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM alerts WHERE status = 'active' ORDER BY timestamp DESC"
                )
            
            alerts = []
            for row in cursor.fetchall():
                alert = Alert(
                    alert_id=row[1],
                    alert_type=row[2],
                    severity=row[3],
                    model_name=row[4],
                    message=row[5],
                    timestamp=datetime.fromisoformat(row[6]),
                    metric_name=row[7],
                    current_value=row[8],
                    threshold_value=row[9],
                    status=row[10],
                    acknowledged_by=row[11],
                    acknowledged_at=datetime.fromisoformat(row[12]) if row[12] else None,
                    resolved_at=datetime.fromisoformat(row[13]) if row[13] else None,
                    notifications_sent=json.loads(row[14]) if row[14] else []
                )
                alerts.append(alert)
            
            return alerts
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert"""
        
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute('''
                UPDATE alerts 
                SET status = 'acknowledged', acknowledged_by = ?, acknowledged_at = ?
                WHERE alert_id = ?
            ''', (acknowledged_by, datetime.now(), alert_id))
        
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].status = 'acknowledged'
            self.active_alerts[alert_id].acknowledged_by = acknowledged_by
            self.active_alerts[alert_id].acknowledged_at = datetime.now()

class ModelPerformanceMonitor:
    """
    Comprehensive Model Performance Monitor
    
    Features:
    - Real-time performance tracking
    - Data drift detection
    - Model drift detection
    - Automated alerting
    - Performance visualization
    - Historical analysis
    """
    
    def __init__(
        self,
        storage_path: str = "monitoring/",
        check_interval_minutes: int = 15
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.metrics_collector = MetricsCollector(self.storage_path / "metrics.db")
        self.data_drift_detector = DataDriftDetector()
        self.model_drift_detector = ModelDriftDetector()
        self.alert_manager = AlertManager(self.storage_path / "alerts.db")
        
        self.check_interval_minutes = check_interval_minutes
        self.monitoring_active = False
        
        # Background monitoring task
        self._monitoring_task = None
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Model monitoring started")
    
    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        
        self.monitoring_active = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Model monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Get list of models to monitor
                models_to_monitor = await self._get_monitored_models()
                
                for model_name in models_to_monitor:
                    await self._check_model_health(model_name)
                
                # Wait for next check
                await asyncio.sleep(self.check_interval_minutes * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _get_monitored_models(self) -> List[str]:
        """Get list of models to monitor"""
        # This would typically query a model registry or configuration
        # For now, return common model names
        return ['ensemble', 'dcf', 'cca', 'risk', 'sentiment']
    
    async def _check_model_health(self, model_name: str):
        """Check health of a specific model"""
        
        try:
            # Get recent performance data
            performance_history = await self.metrics_collector.get_performance_history(
                model_name, hours=24
            )
            
            if not performance_history:
                logger.warning(f"No performance data for model: {model_name}")
                return
            
            current_performance = performance_history[0]
            
            # Data drift detection
            data_drift_metrics = await self._check_data_drift(model_name)
            
            # Model drift detection
            model_drift_metrics = await self.model_drift_detector.detect_performance_drift(
                model_name, current_performance, performance_history
            )
            
            # Check for alerts
            alerts = await self.alert_manager.check_alerts(
                model_name, current_performance, data_drift_metrics, model_drift_metrics
            )
            
            if alerts:
                logger.info(f"Generated {len(alerts)} alerts for model: {model_name}")
            
        except Exception as e:
            logger.error(f"Health check failed for model {model_name}: {e}")
    
    async def _check_data_drift(self, model_name: str) -> List[DataDriftMetrics]:
        """Check for data drift in model features"""
        
        try:
            # Get recent predictions with features
            recent_data = await self.metrics_collector.get_recent_predictions(
                model_name, hours=24
            )
            
            if recent_data.empty:
                return []
            
            drift_metrics = []
            
            # Analyze each feature (simplified - would need actual feature extraction)
            # For now, analyze prediction values as a proxy
            if 'prediction_value' in recent_data.columns:
                prediction_values = recent_data['prediction_value'].dropna().values
                
                if len(prediction_values) > 100:
                    # Update reference distribution periodically
                    if model_name not in self.data_drift_detector.reference_distributions:
                        await self.data_drift_detector.update_reference_distribution(
                            'prediction_values', prediction_values[:1000]  # Use first 1000 as reference
                        )
                    
                    # Check for drift
                    drift_metric = await self.data_drift_detector.detect_drift(
                        'prediction_values', prediction_values[-500:]  # Recent 500 predictions
                    )
                    drift_metrics.append(drift_metric)
            
            return drift_metrics
            
        except Exception as e:
            logger.error(f"Data drift check failed for model {model_name}: {e}")
            return []
    
    async def record_prediction(
        self,
        model_name: str,
        prediction_value: float,
        confidence_score: float,
        prediction_time_ms: float,
        features: Dict[str, Any],
        true_value: Optional[float] = None
    ):
        """Record a prediction for monitoring"""
        
        await self.metrics_collector.record_prediction(
            model_name, prediction_value, confidence_score,
            prediction_time_ms, features, true_value
        )
    
    async def calculate_and_record_performance_metrics(
        self,
        model_name: str,
        model_version: str = "latest",
        window_hours: int = 1
    ) -> ModelPerformanceMetrics:
        """Calculate and record performance metrics"""
        
        # Get recent predictions
        recent_data = await self.metrics_collector.get_recent_predictions(
            model_name, hours=window_hours
        )
        
        if recent_data.empty:
            logger.warning(f"No recent data for model: {model_name}")
            return ModelPerformanceMetrics(
                model_name=model_name,
                model_version=model_version
            )
        
        # Calculate metrics
        metrics = ModelPerformanceMetrics(
            model_name=model_name,
            model_version=model_version,
            prediction_count=len(recent_data),
            average_prediction_time_ms=recent_data['prediction_time_ms'].mean(),
            average_confidence_score=recent_data['confidence_score'].mean()
        )
        
        # Calculate accuracy metrics if true values are available
        true_values = recent_data['true_value'].dropna()
        predictions = recent_data.loc[true_values.index, 'prediction_value']
        
        if len(true_values) > 0:
            metrics.mae = mean_absolute_error(true_values, predictions)
            metrics.rmse = np.sqrt(mean_squared_error(true_values, predictions))
            metrics.r2 = r2_score(true_values, predictions)
            
            # MAPE (Mean Absolute Percentage Error)
            non_zero_true = true_values[true_values != 0]
            non_zero_pred = predictions[true_values != 0]
            if len(non_zero_true) > 0:
                metrics.mape = np.mean(np.abs((non_zero_true - non_zero_pred) / non_zero_true)) * 100
        
        # Quality metrics
        low_confidence_threshold = 0.7
        metrics.low_confidence_predictions = len(
            recent_data[recent_data['confidence_score'] < low_confidence_threshold]
        )
        
        # Record metrics
        await self.metrics_collector.record_performance_metrics(metrics)
        
        return metrics
    
    async def get_monitoring_dashboard_data(self, model_name: str) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        
        # Performance history
        performance_history = await self.metrics_collector.get_performance_history(
            model_name, hours=168  # 1 week
        )
        
        # Recent predictions
        recent_predictions = await self.metrics_collector.get_recent_predictions(
            model_name, hours=24
        )
        
        # Active alerts
        active_alerts = await self.alert_manager.get_active_alerts(model_name)
        
        # Summary statistics
        summary_stats = {}
        if performance_history:
            latest = performance_history[0]
            summary_stats = {
                'latest_mae': latest.mae,
                'latest_r2': latest.r2,
                'prediction_count_24h': sum(p.prediction_count for p in performance_history[:24]),
                'average_confidence': latest.average_confidence_score,
                'error_rate': latest.error_rate
            }
        
        return {
            'model_name': model_name,
            'summary_stats': summary_stats,
            'performance_history': performance_history,
            'recent_predictions': recent_predictions.to_dict('records') if not recent_predictions.empty else [],
            'active_alerts': active_alerts,
            'monitoring_status': 'active' if self.monitoring_active else 'stopped'
        }

# Factory function
def create_model_performance_monitor(**kwargs) -> ModelPerformanceMonitor:
    """Factory function for creating model performance monitor"""
    return ModelPerformanceMonitor(**kwargs)

# Utility functions
def calculate_model_stability_score(performance_history: List[ModelPerformanceMetrics]) -> float:
    """Calculate model stability score based on performance variance"""
    
    if len(performance_history) < 5:
        return 1.0  # Insufficient data
    
    mae_values = [p.mae for p in performance_history if p.mae is not None]
    r2_values = [p.r2 for p in performance_history if p.r2 is not None]
    
    if len(mae_values) < 3 or len(r2_values) < 3:
        return 1.0
    
    # Calculate coefficient of variation for stability
    mae_cv = np.std(mae_values) / np.mean(mae_values) if np.mean(mae_values) > 0 else 0
    r2_cv = np.std(r2_values) / np.mean(r2_values) if np.mean(r2_values) > 0 else 0
    
    # Stability score (lower CV = higher stability)
    stability_score = 1 / (1 + mae_cv + r2_cv)
    
    return min(1.0, max(0.0, stability_score))

def detect_anomalous_predictions(
    predictions: np.ndarray,
    confidence_scores: np.ndarray,
    method: str = 'iqr'
) -> List[int]:
    """Detect anomalous predictions"""
    
    anomaly_indices = []
    
    if method == 'iqr':
        Q1 = np.percentile(predictions, 25)
        Q3 = np.percentile(predictions, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        anomaly_indices = np.where(
            (predictions < lower_bound) | (predictions > upper_bound)
        )[0].tolist()
    
    elif method == 'z_score':
        z_scores = np.abs(stats.zscore(predictions))
        anomaly_indices = np.where(z_scores > 3)[0].tolist()
    
    elif method == 'confidence':
        # Low confidence predictions as anomalies
        low_confidence_threshold = 0.5
        anomaly_indices = np.where(confidence_scores < low_confidence_threshold)[0].tolist()
    
    return anomaly_indices