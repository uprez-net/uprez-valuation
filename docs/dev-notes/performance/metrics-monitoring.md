# Performance Metrics and Monitoring for IPO Valuation Platform

## Overview

This document outlines comprehensive performance monitoring, metrics collection, and observability strategies for the Uprez IPO valuation platform, including financial model-specific evaluation metrics, real-time monitoring dashboards, and business impact measurement.

## Architecture Overview

```
Monitoring Stack Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                     Data Collection Layer                       │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Application   │   Infrastructure │      Business Metrics      │
│   Metrics       │   Metrics       │      (Custom)               │
│   (Prometheus)  │   (Node Exporter)│      (Application)         │
└─────────────────┴─────────────────┴─────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                     Processing Layer                            │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Prometheus    │   Grafana       │      AlertManager          │
│   (TSDB)        │   (Visualization)│      (Alerts)              │
└─────────────────┴─────────────────┴─────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      Alerting & Actions                        │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   PagerDuty     │   Slack         │      Auto-scaling           │
│   (Incidents)   │   (Notifications)│      (Kubernetes HPA)      │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## 1. Financial Model-Specific Evaluation Metrics

### Model Performance Metrics

```python
# monitoring/financial_metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ModelPerformanceMetrics:
    """Financial model performance metrics"""
    rmse: float
    mae: float
    mape: float  # Mean Absolute Percentage Error
    r2: float
    directional_accuracy: float
    sharpe_ratio: float
    information_ratio: float
    maximum_drawdown: float
    hit_rate: float
    profit_loss_ratio: float

class FinancialMetricsCalculator:
    """
    Calculate financial model-specific performance metrics
    """
    
    def __init__(self):
        self.metrics_history = []
        
    def calculate_regression_metrics(self, y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic regression metrics for valuation models
        """
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Financial-specific metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Directional accuracy (for price movements)
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            directional_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            directional_accuracy = 0.0
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        }
    
    def calculate_trading_metrics(self, returns: np.ndarray, 
                                benchmark_returns: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate trading/investment performance metrics
        """
        # Risk-free rate (approximate)
        risk_free_rate = 0.02 / 252  # 2% annual, daily
        
        # Sharpe Ratio
        excess_returns = returns - risk_free_rate
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
        
        # Information Ratio (vs benchmark)
        if benchmark_returns is not None:
            active_returns = returns - benchmark_returns
            information_ratio = np.mean(active_returns) / np.std(active_returns) * np.sqrt(252)
        else:
            information_ratio = 0.0
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        maximum_drawdown = np.min(drawdowns) * 100
        
        # Hit Rate (percentage of positive returns)
        hit_rate = np.mean(returns > 0) * 100
        
        # Profit/Loss Ratio
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) > 0:
            profit_loss_ratio = np.mean(positive_returns) / abs(np.mean(negative_returns))
        else:
            profit_loss_ratio = float('inf')
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'information_ratio': information_ratio,
            'maximum_drawdown': maximum_drawdown,
            'hit_rate': hit_rate,
            'profit_loss_ratio': profit_loss_ratio
        }
    
    def calculate_ipo_specific_metrics(self, predicted_returns: np.ndarray,
                                     actual_returns: np.ndarray,
                                     offer_prices: np.ndarray) -> Dict[str, float]:
        """
        Calculate IPO-specific performance metrics
        """
        # First-day pop prediction accuracy
        first_day_accuracy = np.mean(
            np.sign(predicted_returns) == np.sign(actual_returns)
        ) * 100
        
        # Underpricing prediction error
        underpricing_error = np.mean(np.abs(predicted_returns - actual_returns))
        
        # Value at Risk (VaR) for predictions
        var_95 = np.percentile(predicted_returns - actual_returns, 5)
        
        # Expected Shortfall (Conditional VaR)
        errors = predicted_returns - actual_returns
        es_95 = np.mean(errors[errors <= var_95])
        
        # Overallocation risk (predicting too high returns)
        overallocation_rate = np.mean(predicted_returns > actual_returns) * 100
        
        return {
            'first_day_accuracy': first_day_accuracy,
            'underpricing_error': underpricing_error,
            'var_95': var_95,
            'expected_shortfall_95': es_95,
            'overallocation_rate': overallocation_rate
        }
    
    def generate_model_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                            returns: np.ndarray = None,
                            benchmark_returns: np.ndarray = None) -> ModelPerformanceMetrics:
        """
        Generate comprehensive model performance report
        """
        # Basic regression metrics
        regression_metrics = self.calculate_regression_metrics(y_true, y_pred)
        
        # Trading metrics if returns provided
        trading_metrics = {}
        if returns is not None:
            trading_metrics = self.calculate_trading_metrics(returns, benchmark_returns)
        
        # IPO-specific metrics
        if len(y_true) > 0:
            offer_prices = np.ones_like(y_true)  # Placeholder
            ipo_metrics = self.calculate_ipo_specific_metrics(y_pred, y_true, offer_prices)
        else:
            ipo_metrics = {}
        
        # Combine all metrics
        all_metrics = {**regression_metrics, **trading_metrics, **ipo_metrics}
        
        return ModelPerformanceMetrics(
            rmse=all_metrics.get('rmse', 0),
            mae=all_metrics.get('mae', 0),
            mape=all_metrics.get('mape', 0),
            r2=all_metrics.get('r2', 0),
            directional_accuracy=all_metrics.get('directional_accuracy', 0),
            sharpe_ratio=all_metrics.get('sharpe_ratio', 0),
            information_ratio=all_metrics.get('information_ratio', 0),
            maximum_drawdown=all_metrics.get('maximum_drawdown', 0),
            hit_rate=all_metrics.get('hit_rate', 0),
            profit_loss_ratio=all_metrics.get('profit_loss_ratio', 0)
        )

class RealTimeMetricsCollector:
    """
    Real-time metrics collection and publishing
    """
    
    def __init__(self, prometheus_gateway: str = "localhost:9091"):
        self.prometheus_gateway = prometheus_gateway
        self.metrics_buffer = []
        
    def collect_prediction_metrics(self, model_name: str, prediction_time: float,
                                 batch_size: int, model_version: str) -> None:
        """
        Collect metrics from model prediction
        """
        from prometheus_client import CollectorRegistry, Gauge, Counter, push_to_gateway
        
        registry = CollectorRegistry()
        
        # Prediction latency
        prediction_latency = Gauge(
            'model_prediction_latency_seconds',
            'Time taken for model prediction',
            ['model_name', 'model_version'],
            registry=registry
        )
        prediction_latency.labels(model_name=model_name, model_version=model_version).set(prediction_time)
        
        # Batch size
        batch_size_gauge = Gauge(
            'model_prediction_batch_size',
            'Batch size for predictions',
            ['model_name', 'model_version'],
            registry=registry
        )
        batch_size_gauge.labels(model_name=model_name, model_version=model_version).set(batch_size)
        
        # Prediction counter
        prediction_counter = Counter(
            'model_predictions_total',
            'Total number of predictions',
            ['model_name', 'model_version'],
            registry=registry
        )
        prediction_counter.labels(model_name=model_name, model_version=model_version).inc(batch_size)
        
        # Push to Prometheus
        push_to_gateway(self.prometheus_gateway, job='ipo_valuation', registry=registry)
    
    def collect_business_metrics(self, active_users: int, valuations_requested: int,
                               api_calls: int, revenue_generated: float) -> None:
        """
        Collect business-specific metrics
        """
        from prometheus_client import CollectorRegistry, Gauge, Counter, push_to_gateway
        
        registry = CollectorRegistry()
        
        # Active users
        active_users_gauge = Gauge(
            'active_users_total',
            'Number of active users',
            registry=registry
        )
        active_users_gauge.set(active_users)
        
        # Valuations requested
        valuations_counter = Counter(
            'valuations_requested_total',
            'Total valuations requested',
            registry=registry
        )
        valuations_counter.inc(valuations_requested)
        
        # API calls
        api_calls_counter = Counter(
            'api_calls_total',
            'Total API calls',
            registry=registry
        )
        api_calls_counter.inc(api_calls)
        
        # Revenue
        revenue_gauge = Gauge(
            'revenue_generated_dollars',
            'Revenue generated in dollars',
            registry=registry
        )
        revenue_gauge.set(revenue_generated)
        
        push_to_gateway(self.prometheus_gateway, job='business_metrics', registry=registry)

# Custom Prometheus metrics for FastAPI
from prometheus_client import Counter, Histogram, Gauge, Info
import time

# Application metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections'
)

MODEL_INFERENCE_TIME = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration',
    ['model_name', 'model_version']
)

MODEL_QUEUE_LENGTH = Gauge(
    'model_queue_length',
    'Number of requests in model queue',
    ['model_name']
)

APP_INFO = Info(
    'app_info',
    'Application information'
)
```

### Backtesting Framework

```python
# monitoring/backtesting.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import logging

@dataclass
class BacktestResults:
    """Backtesting results container"""
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    total_trades: int
    profitable_trades: int
    average_trade_return: float
    best_trade: float
    worst_trade: float

class IPOBacktester:
    """
    Backtesting framework for IPO valuation models
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.results_history = []
        
    def backtest_strategy(self, predictions: pd.DataFrame, 
                         actual_returns: pd.DataFrame,
                         strategy_func: Callable,
                         transaction_cost: float = 0.001) -> BacktestResults:
        """
        Run backtest for IPO investment strategy
        """
        # Initialize portfolio
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'portfolio_value': self.initial_capital,
            'returns': []
        }
        
        trades = []
        
        # Simulate trading over time
        for date in predictions.index:
            if date in actual_returns.index:
                # Get predictions and actual returns for this date
                pred = predictions.loc[date]
                actual = actual_returns.loc[date]
                
                # Apply strategy
                signals = strategy_func(pred, portfolio)
                
                # Execute trades
                for symbol, signal in signals.items():
                    if signal['action'] == 'buy' and signal['amount'] > 0:
                        # Buy position
                        cost = signal['amount'] * (1 + transaction_cost)
                        if portfolio['cash'] >= cost:
                            portfolio['cash'] -= cost
                            portfolio['positions'][symbol] = portfolio['positions'].get(symbol, 0) + signal['amount']
                            
                            trades.append({
                                'date': date,
                                'symbol': symbol,
                                'action': 'buy',
                                'amount': signal['amount'],
                                'price': 1.0,  # Normalized
                                'cost': cost
                            })
                    
                    elif signal['action'] == 'sell' and symbol in portfolio['positions']:
                        # Sell position
                        if portfolio['positions'][symbol] >= signal['amount']:
                            proceeds = signal['amount'] * (1 - transaction_cost)
                            portfolio['cash'] += proceeds
                            portfolio['positions'][symbol] -= signal['amount']
                            
                            if portfolio['positions'][symbol] == 0:
                                del portfolio['positions'][symbol]
                            
                            trades.append({
                                'date': date,
                                'symbol': symbol,
                                'action': 'sell',
                                'amount': signal['amount'],
                                'price': 1.0 + actual[symbol] if symbol in actual else 1.0,
                                'proceeds': proceeds
                            })
                
                # Update portfolio value
                position_value = sum(
                    amount * (1 + actual.get(symbol, 0)) 
                    for symbol, amount in portfolio['positions'].items()
                )
                
                portfolio['portfolio_value'] = portfolio['cash'] + position_value
                
                # Calculate returns
                if len(portfolio['returns']) == 0:
                    daily_return = (portfolio['portfolio_value'] - self.initial_capital) / self.initial_capital
                else:
                    prev_value = portfolio['returns'][-1]['portfolio_value']
                    daily_return = (portfolio['portfolio_value'] - prev_value) / prev_value
                
                portfolio['returns'].append({
                    'date': date,
                    'portfolio_value': portfolio['portfolio_value'],
                    'daily_return': daily_return,
                    'cash': portfolio['cash'],
                    'position_value': position_value
                })
        
        # Calculate performance metrics
        returns_df = pd.DataFrame(portfolio['returns'])
        returns_array = returns_df['daily_return'].values
        
        total_return = (portfolio['portfolio_value'] - self.initial_capital) / self.initial_capital
        annual_return = (1 + total_return) ** (252 / len(returns_array)) - 1
        volatility = np.std(returns_array) * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative_returns = (1 + returns_df['daily_return']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Trade analysis
        profitable_trades = len([t for t in trades if t.get('proceeds', 0) > t.get('cost', float('inf'))])
        total_trades = len([t for t in trades if t['action'] == 'sell'])
        hit_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Trade returns
        trade_returns = []
        buy_trades = {t['symbol']: t for t in trades if t['action'] == 'buy'}
        
        for trade in trades:
            if trade['action'] == 'sell' and trade['symbol'] in buy_trades:
                buy_trade = buy_trades[trade['symbol']]
                trade_return = (trade['proceeds'] - buy_trade['cost']) / buy_trade['cost']
                trade_returns.append(trade_return)
        
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0
        best_trade = max(trade_returns) if trade_returns else 0
        worst_trade = min(trade_returns) if trade_returns else 0
        
        return BacktestResults(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            hit_rate=hit_rate,
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            average_trade_return=avg_trade_return,
            best_trade=best_trade,
            worst_trade=worst_trade
        )
    
    def compare_strategies(self, strategies: Dict[str, Callable],
                          predictions: pd.DataFrame,
                          actual_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compare multiple investment strategies
        """
        results = []
        
        for strategy_name, strategy_func in strategies.items():
            logging.info(f"Backtesting strategy: {strategy_name}")
            result = self.backtest_strategy(predictions, actual_returns, strategy_func)
            
            results.append({
                'Strategy': strategy_name,
                'Total Return': result.total_return * 100,
                'Annual Return': result.annual_return * 100,
                'Volatility': result.volatility * 100,
                'Sharpe Ratio': result.sharpe_ratio,
                'Max Drawdown': result.max_drawdown * 100,
                'Hit Rate': result.hit_rate * 100,
                'Total Trades': result.total_trades,
                'Avg Trade Return': result.average_trade_return * 100
            })
        
        return pd.DataFrame(results)
    
    def plot_backtest_results(self, results_df: pd.DataFrame) -> None:
        """
        Plot backtest comparison results
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Strategy Comparison Results', fontsize=16)
        
        # Total Return
        axes[0, 0].bar(results_df['Strategy'], results_df['Total Return'])
        axes[0, 0].set_title('Total Return (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Sharpe Ratio
        axes[0, 1].bar(results_df['Strategy'], results_df['Sharpe Ratio'])
        axes[0, 1].set_title('Sharpe Ratio')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Max Drawdown
        axes[0, 2].bar(results_df['Strategy'], results_df['Max Drawdown'])
        axes[0, 2].set_title('Max Drawdown (%)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Hit Rate
        axes[1, 0].bar(results_df['Strategy'], results_df['Hit Rate'])
        axes[1, 0].set_title('Hit Rate (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Volatility
        axes[1, 1].bar(results_df['Strategy'], results_df['Volatility'])
        axes[1, 1].set_title('Volatility (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Total Trades
        axes[1, 2].bar(results_df['Strategy'], results_df['Total Trades'])
        axes[1, 2].set_title('Total Trades')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

# Example strategies
def conservative_strategy(predictions: pd.DataFrame, portfolio: Dict) -> Dict[str, Dict]:
    """
    Conservative investment strategy - only invest in high-confidence predictions
    """
    signals = {}
    
    for symbol in predictions.index:
        pred_return = predictions[symbol]
        
        # Only invest if predicted return > 10% and confidence high
        if pred_return > 0.1:
            # Allocate 5% of portfolio value
            allocation = portfolio['portfolio_value'] * 0.05
            signals[symbol] = {'action': 'buy', 'amount': allocation}
        elif symbol in portfolio['positions']:
            # Sell if we have position and prediction is negative
            if pred_return < -0.05:
                signals[symbol] = {'action': 'sell', 'amount': portfolio['positions'][symbol]}
    
    return signals

def aggressive_strategy(predictions: pd.DataFrame, portfolio: Dict) -> Dict[str, Dict]:
    """
    Aggressive investment strategy - larger positions, more trades
    """
    signals = {}
    
    for symbol in predictions.index:
        pred_return = predictions[symbol]
        
        # Invest if predicted return > 5%
        if pred_return > 0.05:
            # Allocate 10% of portfolio value
            allocation = portfolio['portfolio_value'] * 0.10
            signals[symbol] = {'action': 'buy', 'amount': allocation}
        elif symbol in portfolio['positions']:
            # Sell if prediction is negative
            if pred_return < 0:
                signals[symbol] = {'action': 'sell', 'amount': portfolio['positions'][symbol]}
    
    return signals
```

## 2. Real-time Performance Monitoring Dashboards

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "id": null,
    "title": "Uprez IPO Valuation Platform - Performance Dashboard",
    "tags": ["uprez", "ipo", "valuation", "performance"],
    "timezone": "UTC",
    "panels": [
      {
        "id": 1,
        "title": "API Response Time",
        "type": "stat",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 0.5},
                {"color": "red", "value": 1.0}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "Model Inference Time",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(model_inference_duration_seconds_sum[5m]) / rate(model_inference_duration_seconds_count[5m])",
            "legendFormat": "{{model_name}}"
          }
        ]
      },
      {
        "id": 3,
        "title": "Request Rate",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "id": 4,
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(http_requests_total{status_code=~\"5..\"}[5m]) / rate(http_requests_total[5m]) * 100",
            "legendFormat": "Error Rate %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 1},
                {"color": "red", "value": 5}
              ]
            }
          }
        }
      },
      {
        "id": 5,
        "title": "Active Users",
        "type": "stat",
        "targets": [
          {
            "expr": "active_users_total",
            "legendFormat": "Active Users"
          }
        ]
      },
      {
        "id": 6,
        "title": "Model Queue Length",
        "type": "timeseries",
        "targets": [
          {
            "expr": "model_queue_length",
            "legendFormat": "{{model_name}}"
          }
        ]
      },
      {
        "id": 7,
        "title": "System Resources",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(container_cpu_usage_seconds_total[5m]) * 100",
            "legendFormat": "CPU Usage - {{pod}}"
          },
          {
            "expr": "container_memory_usage_bytes / container_spec_memory_limit_bytes * 100",
            "legendFormat": "Memory Usage - {{pod}}"
          }
        ]
      },
      {
        "id": 8,
        "title": "Database Performance",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(postgresql_queries_total[5m])",
            "legendFormat": "Queries/sec"
          },
          {
            "expr": "postgresql_connections_active",
            "legendFormat": "Active Connections"
          }
        ]
      },
      {
        "id": 9,
        "title": "Business Metrics",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(valuations_requested_total[1h])",
            "legendFormat": "Valuations/hour"
          },
          {
            "expr": "revenue_generated_dollars",
            "legendFormat": "Revenue Generated"
          }
        ]
      },
      {
        "id": 10,
        "title": "Model Accuracy Metrics",
        "type": "table",
        "targets": [
          {
            "expr": "model_rmse",
            "legendFormat": "RMSE"
          },
          {
            "expr": "model_mae",
            "legendFormat": "MAE"
          },
          {
            "expr": "model_r2_score",
            "legendFormat": "R² Score"
          },
          {
            "expr": "model_directional_accuracy",
            "legendFormat": "Directional Accuracy"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

### Custom Monitoring Dashboard

```python
# monitoring/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Any

class PerformanceDashboard:
    """
    Real-time performance monitoring dashboard
    """
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url
        
    def query_prometheus(self, query: str, start_time: datetime = None, 
                        end_time: datetime = None) -> Dict[str, Any]:
        """
        Query Prometheus for metrics data
        """
        if start_time and end_time:
            # Range query
            params = {
                'query': query,
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'step': '1m'
            }
            endpoint = f"{self.prometheus_url}/api/v1/query_range"
        else:
            # Instant query
            params = {'query': query}
            endpoint = f"{self.prometheus_url}/api/v1/query"
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"Error querying Prometheus: {e}")
            return {}
    
    def get_api_metrics(self) -> Dict[str, float]:
        """
        Get current API performance metrics
        """
        metrics = {}
        
        # Response time (95th percentile)
        query = 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))'
        result = self.query_prometheus(query)
        if result.get('data', {}).get('result'):
            metrics['response_time_95p'] = float(result['data']['result'][0]['value'][1])
        
        # Request rate
        query = 'rate(http_requests_total[5m])'
        result = self.query_prometheus(query)
        if result.get('data', {}).get('result'):
            metrics['request_rate'] = sum(float(r['value'][1]) for r in result['data']['result'])
        
        # Error rate
        query = 'rate(http_requests_total{status_code=~"5.."}[5m]) / rate(http_requests_total[5m]) * 100'
        result = self.query_prometheus(query)
        if result.get('data', {}).get('result'):
            metrics['error_rate'] = float(result['data']['result'][0]['value'][1])
        
        return metrics
    
    def get_model_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get ML model performance metrics
        """
        models_metrics = {}
        
        # Model inference time
        query = 'rate(model_inference_duration_seconds_sum[5m]) / rate(model_inference_duration_seconds_count[5m])'
        result = self.query_prometheus(query)
        
        if result.get('data', {}).get('result'):
            for metric in result['data']['result']:
                model_name = metric['metric'].get('model_name', 'unknown')
                if model_name not in models_metrics:
                    models_metrics[model_name] = {}
                models_metrics[model_name]['inference_time'] = float(metric['value'][1])
        
        # Model queue length
        query = 'model_queue_length'
        result = self.query_prometheus(query)
        
        if result.get('data', {}).get('result'):
            for metric in result['data']['result']:
                model_name = metric['metric'].get('model_name', 'unknown')
                if model_name not in models_metrics:
                    models_metrics[model_name] = {}
                models_metrics[model_name]['queue_length'] = float(metric['value'][1])
        
        return models_metrics
    
    def get_business_metrics(self) -> Dict[str, float]:
        """
        Get business performance metrics
        """
        metrics = {}
        
        # Active users
        query = 'active_users_total'
        result = self.query_prometheus(query)
        if result.get('data', {}).get('result'):
            metrics['active_users'] = float(result['data']['result'][0]['value'][1])
        
        # Valuations per hour
        query = 'rate(valuations_requested_total[1h])'
        result = self.query_prometheus(query)
        if result.get('data', {}).get('result'):
            metrics['valuations_per_hour'] = float(result['data']['result'][0]['value'][1])
        
        # Revenue
        query = 'revenue_generated_dollars'
        result = self.query_prometheus(query)
        if result.get('data', {}).get('result'):
            metrics['revenue'] = float(result['data']['result'][0]['value'][1])
        
        return metrics
    
    def render_dashboard(self):
        """
        Render Streamlit dashboard
        """
        st.set_page_config(
            page_title="Uprez IPO Valuation - Performance Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📈 Uprez IPO Valuation Platform - Performance Dashboard")
        st.markdown("Real-time monitoring of system performance and business metrics")
        
        # Sidebar controls
        st.sidebar.header("Dashboard Controls")
        auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
        
        if auto_refresh:
            # Auto refresh every 30 seconds
            time.sleep(30)
            st.experimental_rerun()
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Get current metrics
        api_metrics = self.get_api_metrics()
        model_metrics = self.get_model_metrics()
        business_metrics = self.get_business_metrics()
        
        with col1:
            st.metric(
                label="API Response Time (95p)",
                value=f"{api_metrics.get('response_time_95p', 0):.3f}s",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Request Rate",
                value=f"{api_metrics.get('request_rate', 0):.1f}/sec",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Error Rate",
                value=f"{api_metrics.get('error_rate', 0):.2f}%",
                delta=None
            )
        
        with col4:
            st.metric(
                label="Active Users",
                value=f"{business_metrics.get('active_users', 0):.0f}",
                delta=None
            )
        
        # Charts section
        st.header("📊 Performance Charts")
        
        # API Performance over time
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("API Response Time Trend")
            
            # Get historical data
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
            
            query = 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))'
            result = self.query_prometheus(query, start_time, end_time)
            
            if result.get('data', {}).get('result'):
                values = result['data']['result'][0]['values']
                df = pd.DataFrame(values, columns=['timestamp', 'value'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df['value'] = df['value'].astype(float)
                
                fig = px.line(df, x='timestamp', y='value', title='API Response Time (95p)')
                fig.update_layout(yaxis_title='Response Time (seconds)')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Request Rate Trend")
            
            query = 'rate(http_requests_total[5m])'
            result = self.query_prometheus(query, start_time, end_time)
            
            if result.get('data', {}).get('result'):
                # Aggregate all endpoints
                all_values = []
                timestamps = set()
                
                for series in result['data']['result']:
                    for timestamp, value in series['values']:
                        timestamps.add(timestamp)
                        all_values.append((timestamp, float(value)))
                
                # Group by timestamp and sum
                timestamp_sums = {}
                for timestamp, value in all_values:
                    timestamp_sums[timestamp] = timestamp_sums.get(timestamp, 0) + value
                
                df = pd.DataFrame(list(timestamp_sums.items()), columns=['timestamp', 'value'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                
                fig = px.line(df, x='timestamp', y='value', title='Request Rate')
                fig.update_layout(yaxis_title='Requests/second')
                st.plotly_chart(fig, use_container_width=True)
        
        # Model Performance
        st.header("🤖 Model Performance")
        
        if model_metrics:
            model_names = list(model_metrics.keys())
            selected_model = st.selectbox("Select Model", model_names)
            
            if selected_model:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    inference_time = model_metrics[selected_model].get('inference_time', 0)
                    st.metric("Inference Time", f"{inference_time:.3f}s")
                
                with col2:
                    queue_length = model_metrics[selected_model].get('queue_length', 0)
                    st.metric("Queue Length", f"{queue_length:.0f}")
                
                with col3:
                    # Calculate throughput
                    throughput = 1 / inference_time if inference_time > 0 else 0
                    st.metric("Throughput", f"{throughput:.1f}/sec")
        
        # Business Metrics
        st.header("💼 Business Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Valuations/Hour",
                f"{business_metrics.get('valuations_per_hour', 0):.1f}"
            )
        
        with col2:
            st.metric(
                "Revenue Generated",
                f"${business_metrics.get('revenue', 0):,.2f}"
            )
        
        with col3:
            # Calculate conversion rate (example)
            conversion_rate = (business_metrics.get('valuations_per_hour', 0) / 
                             max(business_metrics.get('active_users', 1), 1)) * 100
            st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
        
        # System Resources
        st.header("🖥️ System Resources")
        
        # CPU and Memory usage charts would go here
        # Similar pattern to above charts

if __name__ == "__main__":
    dashboard = PerformanceDashboard()
    dashboard.render_dashboard()
```

## 3. Model Drift Detection and Alert Systems

### Drift Detection Framework

```python
# monitoring/drift_detection.py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

@dataclass
class DriftAlert:
    """Drift detection alert"""
    alert_type: str
    severity: str
    metric_name: str
    current_value: float
    baseline_value: float
    threshold: float
    timestamp: datetime
    description: str

class ModelDriftDetector:
    """
    Detect model performance drift and data distribution changes
    """
    
    def __init__(self, baseline_window: int = 30, alert_threshold: float = 0.1):
        self.baseline_window = baseline_window
        self.alert_threshold = alert_threshold
        self.baseline_metrics = {}
        self.performance_history = []
        self.alerts = []
        
    def update_baseline(self, y_true: np.ndarray, y_pred: np.ndarray,
                       feature_data: np.ndarray = None) -> None:
        """
        Update baseline metrics for drift detection
        """
        # Performance metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Directional accuracy
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            directional_accuracy = np.mean(true_direction == pred_direction)
        else:
            directional_accuracy = 0.5
        
        # Residual statistics
        residuals = y_true - y_pred
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        residual_skewness = stats.skew(residuals)
        residual_kurtosis = stats.kurtosis(residuals)
        
        self.baseline_metrics.update({
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'residual_skewness': residual_skewness,
            'residual_kurtosis': residual_kurtosis
        })
        
        # Feature distribution statistics (if provided)
        if feature_data is not None:
            for i in range(feature_data.shape[1]):
                feature_values = feature_data[:, i]
                self.baseline_metrics[f'feature_{i}_mean'] = np.mean(feature_values)
                self.baseline_metrics[f'feature_{i}_std'] = np.std(feature_values)
                self.baseline_metrics[f'feature_{i}_skew'] = stats.skew(feature_values)
        
        logging.info("Baseline metrics updated")
    
    def detect_performance_drift(self, y_true: np.ndarray, y_pred: np.ndarray) -> List[DriftAlert]:
        """
        Detect performance drift in model predictions
        """
        alerts = []
        
        if not self.baseline_metrics:
            logging.warning("No baseline metrics available for drift detection")
            return alerts
        
        # Calculate current metrics
        current_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        current_mae = np.mean(np.abs(y_true - y_pred))
        current_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Directional accuracy
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            current_directional_accuracy = np.mean(true_direction == pred_direction)
        else:
            current_directional_accuracy = 0.5
        
        # Check for significant changes
        metrics_to_check = {
            'rmse': (current_rmse, 'higher'),
            'mae': (current_mae, 'higher'),
            'mape': (current_mape, 'higher'),
            'directional_accuracy': (current_directional_accuracy, 'lower')
        }
        
        for metric_name, (current_value, direction) in metrics_to_check.items():
            baseline_value = self.baseline_metrics.get(metric_name, current_value)
            
            # Calculate relative change
            if baseline_value != 0:
                relative_change = abs(current_value - baseline_value) / baseline_value
            else:
                relative_change = 0
            
            # Check if change exceeds threshold
            if relative_change > self.alert_threshold:
                # Determine severity
                if relative_change > self.alert_threshold * 2:
                    severity = 'CRITICAL'
                elif relative_change > self.alert_threshold * 1.5:
                    severity = 'HIGH'
                else:
                    severity = 'MEDIUM'
                
                # Check direction of change
                is_degradation = (
                    (direction == 'higher' and current_value > baseline_value) or
                    (direction == 'lower' and current_value < baseline_value)
                )
                
                if is_degradation:
                    alert = DriftAlert(
                        alert_type='PERFORMANCE_DRIFT',
                        severity=severity,
                        metric_name=metric_name,
                        current_value=current_value,
                        baseline_value=baseline_value,
                        threshold=self.alert_threshold,
                        timestamp=datetime.now(),
                        description=f"{metric_name} has degraded by {relative_change:.2%}"
                    )
                    alerts.append(alert)
        
        return alerts
    
    def detect_data_drift(self, current_data: np.ndarray) -> List[DriftAlert]:
        """
        Detect data distribution drift using statistical tests
        """
        alerts = []
        
        if not self.baseline_metrics:
            logging.warning("No baseline metrics available for drift detection")
            return alerts
        
        # Check feature distribution changes
        for i in range(current_data.shape[1]):
            feature_values = current_data[:, i]
            
            baseline_mean = self.baseline_metrics.get(f'feature_{i}_mean')
            baseline_std = self.baseline_metrics.get(f'feature_{i}_std')
            
            if baseline_mean is not None and baseline_std is not None:
                current_mean = np.mean(feature_values)
                current_std = np.std(feature_values)
                
                # Statistical tests for distribution changes
                
                # 1. Z-test for mean change
                if baseline_std > 0:
                    z_score = abs(current_mean - baseline_mean) / (baseline_std / np.sqrt(len(feature_values)))
                    p_value_mean = 2 * (1 - stats.norm.cdf(abs(z_score)))
                    
                    if p_value_mean < 0.01:  # 99% confidence
                        alert = DriftAlert(
                            alert_type='DATA_DRIFT',
                            severity='HIGH' if p_value_mean < 0.001 else 'MEDIUM',
                            metric_name=f'feature_{i}_mean',
                            current_value=current_mean,
                            baseline_value=baseline_mean,
                            threshold=0.01,
                            timestamp=datetime.now(),
                            description=f"Feature {i} mean has shifted significantly (p={p_value_mean:.4f})"
                        )
                        alerts.append(alert)
                
                # 2. F-test for variance change
                if baseline_std > 0 and current_std > 0:
                    f_statistic = (current_std ** 2) / (baseline_std ** 2)
                    
                    if f_statistic > 2 or f_statistic < 0.5:  # Significant variance change
                        alert = DriftAlert(
                            alert_type='DATA_DRIFT',
                            severity='MEDIUM',
                            metric_name=f'feature_{i}_std',
                            current_value=current_std,
                            baseline_value=baseline_std,
                            threshold=2.0,
                            timestamp=datetime.now(),
                            description=f"Feature {i} variance has changed significantly (F={f_statistic:.2f})"
                        )
                        alerts.append(alert)
        
        return alerts
    
    def detect_concept_drift(self, y_true: np.ndarray, y_pred: np.ndarray,
                           window_size: int = 50) -> List[DriftAlert]:
        """
        Detect concept drift using sliding window approach
        """
        alerts = []
        
        if len(y_true) < window_size * 2:
            return alerts
        
        # Split data into windows
        n_windows = len(y_true) // window_size
        window_performances = []
        
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size
            
            window_true = y_true[start_idx:end_idx]
            window_pred = y_pred[start_idx:end_idx]
            
            window_rmse = np.sqrt(mean_squared_error(window_true, window_pred))
            window_performances.append(window_rmse)
        
        # Check for trend in performance degradation
        if len(window_performances) >= 3:
            # Linear regression to detect trend
            x = np.arange(len(window_performances))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, window_performances)
            
            # If there's a significant upward trend in RMSE (performance degradation)
            if slope > 0 and p_value < 0.05 and r_value ** 2 > 0.5:
                alert = DriftAlert(
                    alert_type='CONCEPT_DRIFT',
                    severity='HIGH' if slope > np.mean(window_performances) * 0.1 else 'MEDIUM',
                    metric_name='rmse_trend',
                    current_value=window_performances[-1],
                    baseline_value=window_performances[0],
                    threshold=0.05,
                    timestamp=datetime.now(),
                    description=f"Concept drift detected: performance degrading with slope {slope:.4f}"
                )
                alerts.append(alert)
        
        return alerts
    
    def generate_drift_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                            feature_data: np.ndarray = None) -> Dict[str, Any]:
        """
        Generate comprehensive drift detection report
        """
        all_alerts = []
        
        # Performance drift
        performance_alerts = self.detect_performance_drift(y_true, y_pred)
        all_alerts.extend(performance_alerts)
        
        # Data drift
        if feature_data is not None:
            data_alerts = self.detect_data_drift(feature_data)
            all_alerts.extend(data_alerts)
        
        # Concept drift
        concept_alerts = self.detect_concept_drift(y_true, y_pred)
        all_alerts.extend(concept_alerts)
        
        # Store alerts
        self.alerts.extend(all_alerts)
        
        # Summary statistics
        alert_summary = {
            'total_alerts': len(all_alerts),
            'critical_alerts': len([a for a in all_alerts if a.severity == 'CRITICAL']),
            'high_alerts': len([a for a in all_alerts if a.severity == 'HIGH']),
            'medium_alerts': len([a for a in all_alerts if a.severity == 'MEDIUM']),
            'alert_types': {
                'performance_drift': len([a for a in all_alerts if a.alert_type == 'PERFORMANCE_DRIFT']),
                'data_drift': len([a for a in all_alerts if a.alert_type == 'DATA_DRIFT']),
                'concept_drift': len([a for a in all_alerts if a.alert_type == 'CONCEPT_DRIFT'])
            }
        }
        
        return {
            'alerts': all_alerts,
            'summary': alert_summary,
            'timestamp': datetime.now(),
            'baseline_metrics': self.baseline_metrics.copy()
        }

class AlertManager:
    """
    Manage drift alerts and notifications
    """
    
    def __init__(self, notification_config: Dict[str, Any]):
        self.notification_config = notification_config
        self.alert_history = []
        
    def send_alert(self, alert: DriftAlert) -> bool:
        """
        Send alert notification
        """
        try:
            # Email notification
            if 'email' in self.notification_config:
                self._send_email_alert(alert)
            
            # Slack notification
            if 'slack' in self.notification_config:
                self._send_slack_alert(alert)
            
            # PagerDuty for critical alerts
            if alert.severity == 'CRITICAL' and 'pagerduty' in self.notification_config:
                self._send_pagerduty_alert(alert)
            
            self.alert_history.append(alert)
            logging.info(f"Alert sent: {alert.alert_type} - {alert.severity}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to send alert: {e}")
            return False
    
    def _send_email_alert(self, alert: DriftAlert) -> None:
        """Send email alert"""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        config = self.notification_config['email']
        
        msg = MIMEMultipart()
        msg['From'] = config['from']
        msg['To'] = ', '.join(config['to'])
        msg['Subject'] = f"[UPREZ] Model Drift Alert - {alert.severity}"
        
        body = f"""
        Model Drift Alert Detected
        
        Alert Type: {alert.alert_type}
        Severity: {alert.severity}
        Metric: {alert.metric_name}
        Current Value: {alert.current_value:.4f}
        Baseline Value: {alert.baseline_value:.4f}
        Threshold: {alert.threshold:.4f}
        
        Description: {alert.description}
        
        Timestamp: {alert.timestamp}
        
        Please investigate and take appropriate action.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        server.starttls()
        server.login(config['username'], config['password'])
        text = msg.as_string()
        server.sendmail(config['from'], config['to'], text)
        server.quit()
    
    def _send_slack_alert(self, alert: DriftAlert) -> None:
        """Send Slack alert"""
        import requests
        
        config = self.notification_config['slack']
        
        color = {
            'CRITICAL': '#FF0000',
            'HIGH': '#FF8000',
            'MEDIUM': '#FFFF00',
            'LOW': '#00FF00'
        }.get(alert.severity, '#808080')
        
        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"Model Drift Alert - {alert.severity}",
                    "fields": [
                        {
                            "title": "Alert Type",
                            "value": alert.alert_type,
                            "short": True
                        },
                        {
                            "title": "Metric",
                            "value": alert.metric_name,
                            "short": True
                        },
                        {
                            "title": "Current Value",
                            "value": f"{alert.current_value:.4f}",
                            "short": True
                        },
                        {
                            "title": "Baseline Value",
                            "value": f"{alert.baseline_value:.4f}",
                            "short": True
                        }
                    ],
                    "text": alert.description,
                    "ts": int(alert.timestamp.timestamp())
                }
            ]
        }
        
        response = requests.post(config['webhook_url'], json=payload)
        response.raise_for_status()
    
    def _send_pagerduty_alert(self, alert: DriftAlert) -> None:
        """Send PagerDuty alert for critical issues"""
        import requests
        
        config = self.notification_config['pagerduty']
        
        payload = {
            "routing_key": config['integration_key'],
            "event_action": "trigger",
            "payload": {
                "summary": f"CRITICAL: Model Drift Detected - {alert.metric_name}",
                "source": "uprez-model-monitoring",
                "severity": "critical",
                "component": "ml-model",
                "group": "model-performance",
                "class": alert.alert_type,
                "custom_details": {
                    "metric": alert.metric_name,
                    "current_value": alert.current_value,
                    "baseline_value": alert.baseline_value,
                    "description": alert.description
                }
            }
        }
        
        response = requests.post(
            "https://events.pagerduty.com/v2/enqueue",
            json=payload
        )
        response.raise_for_status()

# Example usage
if __name__ == "__main__":
    # Initialize drift detector
    drift_detector = ModelDriftDetector(baseline_window=30, alert_threshold=0.15)
    
    # Initialize alert manager
    alert_config = {
        'email': {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'alerts@uprez.com',
            'password': 'app_password',
            'from': 'alerts@uprez.com',
            'to': ['devops@uprez.com', 'ml-team@uprez.com']
        },
        'slack': {
            'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        }
    }
    
    alert_manager = AlertManager(alert_config)
    
    # Example: Monitor model performance
    # In practice, this would be called periodically with new data
    y_true = np.random.normal(0, 1, 100)
    y_pred = np.random.normal(0.1, 1.2, 100)  # Slightly biased and more variance
    
    # Update baseline (would be done with initial good performance data)
    drift_detector.update_baseline(y_true[:50], y_pred[:50])
    
    # Detect drift on new data
    drift_report = drift_detector.generate_drift_report(y_true[50:], y_pred[50:])
    
    # Send alerts if any detected
    for alert in drift_report['alerts']:
        alert_manager.send_alert(alert)
    
    print(f"Drift detection complete. {drift_report['summary']['total_alerts']} alerts generated.")
```

This performance metrics and monitoring documentation provides:

1. **Financial Model-Specific Metrics**: Comprehensive evaluation metrics tailored for IPO valuation models including trading metrics, risk measures, and IPO-specific performance indicators
2. **Real-time Monitoring**: Grafana dashboard configurations and custom Streamlit dashboards for live performance monitoring
3. **Drift Detection**: Advanced model drift detection using statistical tests and automated alerting systems
4. **Backtesting Framework**: Tools for validating model performance against historical data
5. **Alert Management**: Multi-channel notification system for performance issues

The system provides production-ready monitoring capabilities specifically designed for financial ML applications.