# Model Training Pipeline for IPO Valuation Platform

## Overview

This document outlines the comprehensive model training pipeline for the Uprez IPO valuation platform, including automated training, cross-validation strategies, hyperparameter optimization, and MLOps best practices.

## Architecture Overview

```
Data Sources → Feature Engineering → Model Training → Validation → Deployment
     ↓              ↓                    ↓             ↓           ↓
  ASX Data    Feature Store    Training Pipeline  Evaluation   Production
  Company     Preprocessing      Hyperparameter    Metrics      Serving
  Financials  Transformations    Optimization      Backtesting  Monitoring
```

## 1. Automated Training Pipeline Implementation

### Pipeline Components

```python
# training/pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
from google.cloud import aiplatform
import mlflow
import joblib
from typing import Dict, List, Any
import logging
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model_type: str
    target_variable: str
    feature_columns: List[str]
    validation_strategy: str
    n_splits: int = 5
    test_size: float = 0.2
    random_state: int = 42
    hyperparameter_trials: int = 100
    early_stopping_rounds: int = 50

class IPOValuationTrainer:
    """
    Automated training pipeline for IPO valuation models
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.models = {
            'random_forest': RandomForestRegressor,
            'xgboost': XGBRegressor,
            'lightgbm': LGBMRegressor
        }
        self.best_model = None
        self.best_params = None
        self.training_metrics = {}
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and preprocess training data"""
        df = pd.read_csv(data_path)
        df['listing_date'] = pd.to_datetime(df['listing_date'])
        df = df.sort_values('listing_date')
        
        # Financial feature engineering
        df['revenue_growth'] = df['revenue'].pct_change()
        df['profit_margin'] = df['net_income'] / df['revenue']
        df['debt_to_equity'] = df['total_debt'] / df['shareholders_equity']
        df['current_ratio'] = df['current_assets'] / df['current_liabilities']
        df['roe'] = df['net_income'] / df['shareholders_equity']
        df['roa'] = df['net_income'] / df['total_assets']
        
        # Market condition features
        df['market_volatility'] = df['asx_200_volatility']
        df['sector_performance'] = df['sector_index_return']
        df['ipo_volume'] = df['monthly_ipo_count']
        
        return df.dropna()
    
    def create_features(self, df: pd.DataFrame) -> tuple:
        """Create feature matrix and target variable"""
        X = df[self.config.feature_columns].copy()
        y = df[self.config.target_variable].copy()
        
        # Log transform for price targets
        if 'price' in self.config.target_variable:
            y = np.log1p(y)
            
        return X, y
    
    def time_series_split(self, X: pd.DataFrame, y: pd.Series) -> List[tuple]:
        """Create time-aware train/validation splits"""
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        splits = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            splits.append((X_train, X_val, y_train, y_val))
            
        return splits
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            params = self._suggest_params(trial, self.config.model_type)
            model = self.models[self.config.model_type](**params)
            
            scores = []
            splits = self.time_series_split(X, y)
            
            for X_train, X_val, y_train, y_val in splits:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = self._calculate_score(y_val, y_pred)
                scores.append(score)
                
            return np.mean(scores)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.config.hyperparameter_trials)
        
        return study.best_params
    
    def _suggest_params(self, trial, model_type: str) -> Dict[str, Any]:
        """Suggest hyperparameters for different model types"""
        if model_type == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': self.config.random_state
            }
        elif model_type == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': self.config.random_state
            }
        elif model_type == 'lightgbm':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': self.config.random_state
            }
    
    def _calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate evaluation score (RMSE for regression)"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def train_model(self, data_path: str) -> None:
        """Main training pipeline"""
        logging.info("Starting model training pipeline...")
        
        # Load and prepare data
        df = self.load_data(data_path)
        X, y = self.create_features(df)
        
        # Optimize hyperparameters
        logging.info("Optimizing hyperparameters...")
        best_params = self.optimize_hyperparameters(X, y)
        self.best_params = best_params
        
        # Train final model
        logging.info("Training final model...")
        self.best_model = self.models[self.config.model_type](**best_params)
        
        # Time series validation
        scores = []
        splits = self.time_series_split(X, y)
        
        for i, (X_train, X_val, y_train, y_val) in enumerate(splits):
            self.best_model.fit(X_train, y_train)
            y_pred = self.best_model.predict(X_val)
            score = self._calculate_score(y_val, y_pred)
            scores.append(score)
            
            logging.info(f"Fold {i+1} RMSE: {score:.4f}")
        
        self.training_metrics = {
            'mean_rmse': np.mean(scores),
            'std_rmse': np.std(scores),
            'best_params': best_params,
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
        
        # Train on full dataset
        self.best_model.fit(X, y)
        
        logging.info(f"Training completed. Mean RMSE: {self.training_metrics['mean_rmse']:.4f}")
    
    def save_model(self, model_path: str) -> None:
        """Save trained model and metadata"""
        joblib.dump({
            'model': self.best_model,
            'config': self.config,
            'metrics': self.training_metrics,
            'feature_columns': self.config.feature_columns
        }, model_path)
        
        logging.info(f"Model saved to {model_path}")

# Example usage
if __name__ == "__main__":
    config = TrainingConfig(
        model_type='xgboost',
        target_variable='first_day_return',
        feature_columns=[
            'revenue', 'revenue_growth', 'profit_margin', 'debt_to_equity',
            'current_ratio', 'roe', 'roa', 'market_volatility',
            'sector_performance', 'ipo_volume', 'offer_size'
        ]
    )
    
    trainer = IPOValuationTrainer(config)
    trainer.train_model('data/ipo_training_data.csv')
    trainer.save_model('models/ipo_valuation_model.joblib')
```

## 2. Cross-Validation Strategies for Financial Time Series

### Time Series Cross-Validation

```python
# training/time_series_validation.py
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

class FinancialTimeSeriesCV:
    """
    Advanced cross-validation strategies for financial time series data
    """
    
    def __init__(self, data: pd.DataFrame, date_column: str = 'listing_date'):
        self.data = data.sort_values(date_column)
        self.date_column = date_column
        
    def expanding_window_cv(self, model, X: pd.DataFrame, y: pd.Series, 
                           min_train_size: int = 100) -> Dict[str, List[float]]:
        """
        Expanding window cross-validation for time series
        Training set grows, test set is fixed size
        """
        results = {'rmse': [], 'mae': [], 'r2': [], 'predictions': []}
        
        for i in range(min_train_size, len(X) - 50, 50):  # Step by 50 samples
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]
            X_test = X.iloc[i:i+50]
            y_test = y.iloc[i:i+50]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            results['mae'].append(mean_absolute_error(y_test, y_pred))
            results['r2'].append(r2_score(y_test, y_pred))
            results['predictions'].append(y_pred)
            
        return results
    
    def rolling_window_cv(self, model, X: pd.DataFrame, y: pd.Series,
                         window_size: int = 200, test_size: int = 50) -> Dict[str, List[float]]:
        """
        Rolling window cross-validation
        Both training and test sets are fixed size and roll forward
        """
        results = {'rmse': [], 'mae': [], 'r2': [], 'predictions': []}
        
        for i in range(window_size, len(X) - test_size, test_size):
            X_train = X.iloc[i-window_size:i]
            y_train = y.iloc[i-window_size:i]
            X_test = X.iloc[i:i+test_size]
            y_test = y.iloc[i:i+test_size]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            results['mae'].append(mean_absolute_error(y_test, y_pred))
            results['r2'].append(r2_score(y_test, y_pred))
            results['predictions'].append(y_pred)
            
        return results
    
    def blocked_time_series_cv(self, model, X: pd.DataFrame, y: pd.Series,
                              n_splits: int = 5, gap: int = 30) -> Dict[str, List[float]]:
        """
        Blocked time series cross-validation with gap to prevent data leakage
        """
        results = {'rmse': [], 'mae': [], 'r2': [], 'predictions': []}
        
        total_samples = len(X)
        fold_size = total_samples // (n_splits + 1)
        
        for i in range(n_splits):
            # Training set: everything before the current fold (with gap)
            train_end = (i + 1) * fold_size - gap
            test_start = (i + 1) * fold_size
            test_end = (i + 2) * fold_size
            
            if train_end <= 0 or test_end > total_samples:
                continue
                
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            results['mae'].append(mean_absolute_error(y_test, y_pred))
            results['r2'].append(r2_score(y_test, y_pred))
            results['predictions'].append(y_pred)
            
        return results
    
    def plot_cv_results(self, results: Dict[str, List[float]], title: str = "CV Results"):
        """Plot cross-validation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # RMSE over folds
        axes[0, 0].plot(results['rmse'], marker='o')
        axes[0, 0].set_title('RMSE across folds')
        axes[0, 0].set_xlabel('Fold')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].grid(True)
        
        # MAE over folds
        axes[0, 1].plot(results['mae'], marker='s', color='orange')
        axes[0, 1].set_title('MAE across folds')
        axes[0, 1].set_xlabel('Fold')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].grid(True)
        
        # R2 over folds
        axes[1, 0].plot(results['r2'], marker='^', color='green')
        axes[1, 0].set_title('R² across folds')
        axes[1, 0].set_xlabel('Fold')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].grid(True)
        
        # Distribution of metrics
        metrics_df = pd.DataFrame({
            'RMSE': results['rmse'],
            'MAE': results['mae'],
            'R²': results['r2']
        })
        
        sns.boxplot(data=metrics_df, ax=axes[1, 1])
        axes[1, 1].set_title('Distribution of metrics')
        
        plt.tight_layout()
        plt.show()
        
        return fig

class WalkForwardAnalysis:
    """
    Walk-forward analysis for model validation in trading/financial contexts
    """
    
    def __init__(self, retrain_frequency: int = 30):
        self.retrain_frequency = retrain_frequency
        
    def walk_forward_validation(self, model_class, params: Dict[str, Any],
                               X: pd.DataFrame, y: pd.Series,
                               initial_train_size: int = 500) -> Dict[str, Any]:
        """
        Perform walk-forward analysis with periodic model retraining
        """
        results = {
            'predictions': [],
            'actuals': [],
            'dates': [],
            'metrics': {
                'rmse': [],
                'mae': [],
                'directional_accuracy': []
            }
        }
        
        current_pos = initial_train_size
        model = None
        
        while current_pos < len(X):
            # Retrain model if needed
            if model is None or current_pos % self.retrain_frequency == 0:
                X_train = X.iloc[:current_pos]
                y_train = y.iloc[:current_pos]
                
                model = model_class(**params)
                model.fit(X_train, y_train)
            
            # Make prediction for next period
            X_test = X.iloc[current_pos:current_pos+1]
            y_test = y.iloc[current_pos]
            
            prediction = model.predict(X_test)[0]
            
            results['predictions'].append(prediction)
            results['actuals'].append(y_test)
            results['dates'].append(X.index[current_pos])
            
            current_pos += 1
        
        # Calculate metrics
        predictions = np.array(results['predictions'])
        actuals = np.array(results['actuals'])
        
        results['metrics']['rmse'] = np.sqrt(mean_squared_error(actuals, predictions))
        results['metrics']['mae'] = mean_absolute_error(actuals, predictions)
        
        # Directional accuracy for returns
        direction_correct = np.sum(np.sign(predictions) == np.sign(actuals))
        results['metrics']['directional_accuracy'] = direction_correct / len(predictions)
        
        return results
```

## 3. Hyperparameter Optimization

### Optuna Integration

```python
# training/hyperparameter_optimization.py
import optuna
from optuna.integration import MLflowCallback
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import joblib
from datetime import datetime
import logging

class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization for IPO valuation models
    """
    
    def __init__(self, model_registry: Dict[str, Any], study_name: str = None):
        self.model_registry = model_registry
        self.study_name = study_name or f"ipo_valuation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.best_models = {}
        
    def objective_function(self, trial, model_name: str, X: pd.DataFrame, y: pd.Series,
                          cv_strategy, scorer) -> float:
        """
        Objective function for hyperparameter optimization
        """
        try:
            # Get model class and parameter suggestions
            model_class = self.model_registry[model_name]['class']
            param_suggestions = self.model_registry[model_name]['params']
            
            # Suggest parameters
            params = {}
            for param_name, param_config in param_suggestions.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
            
            # Create and evaluate model
            model = model_class(**params)
            scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scorer, n_jobs=-1)
            
            return -np.mean(scores)  # Minimize negative score
            
        except Exception as e:
            logging.error(f"Error in trial: {e}")
            return float('inf')
    
    def optimize_model(self, model_name: str, X: pd.DataFrame, y: pd.Series,
                      cv_strategy, n_trials: int = 200, 
                      scorer_name: str = 'neg_root_mean_squared_error') -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific model
        """
        scorer = make_scorer(lambda y_true, y_pred: -np.sqrt(np.mean((y_true - y_pred) ** 2)),
                           greater_is_better=True)
        
        # Create MLflow callback for experiment tracking
        mlflc = MLflowCallback(
            tracking_uri="sqlite:///mlflow.db",
            metric_name="rmse",
        )
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            study_name=f"{self.study_name}_{model_name}",
            storage=f"sqlite:///optuna_{self.study_name}.db",
            load_if_exists=True,
            callbacks=[mlflc]
        )
        
        # Optimize
        study.optimize(
            lambda trial: self.objective_function(trial, model_name, X, y, cv_strategy, scorer),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        # Store best model
        best_params = study.best_params
        model_class = self.model_registry[model_name]['class']
        best_model = model_class(**best_params)
        best_model.fit(X, y)
        
        self.best_models[model_name] = {
            'model': best_model,
            'params': best_params,
            'score': study.best_value,
            'study': study
        }
        
        return {
            'best_params': best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'study': study
        }
    
    def multi_model_optimization(self, model_names: List[str], X: pd.DataFrame, y: pd.Series,
                               cv_strategy, n_trials_per_model: int = 100) -> Dict[str, Any]:
        """
        Optimize multiple models and compare results
        """
        results = {}
        
        for model_name in model_names:
            logging.info(f"Optimizing {model_name}...")
            result = self.optimize_model(
                model_name, X, y, cv_strategy, n_trials_per_model
            )
            results[model_name] = result
            
        # Find best overall model
        best_model_name = min(results.keys(), 
                             key=lambda k: results[k]['best_score'])
        
        results['best_overall'] = {
            'model_name': best_model_name,
            'score': results[best_model_name]['best_score'],
            'params': results[best_model_name]['best_params']
        }
        
        return results
    
    def save_optimization_results(self, results: Dict[str, Any], 
                                 output_dir: str = "optimization_results/") -> None:
        """
        Save optimization results and best models
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save results summary
        summary_path = f"{output_dir}/optimization_summary_{timestamp}.json"
        
        summary = {}
        for model_name, result in results.items():
            if model_name != 'best_overall':
                summary[model_name] = {
                    'best_score': result['best_score'],
                    'best_params': result['best_params'],
                    'n_trials': result['n_trials']
                }
        
        if 'best_overall' in results:
            summary['best_overall'] = results['best_overall']
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save best models
        for model_name, model_info in self.best_models.items():
            model_path = f"{output_dir}/best_{model_name}_{timestamp}.joblib"
            joblib.dump(model_info, model_path)
            
        logging.info(f"Optimization results saved to {output_dir}")

# Model registry configuration
MODEL_REGISTRY = {
    'random_forest': {
        'class': RandomForestRegressor,
        'params': {
            'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
            'max_depth': {'type': 'int', 'low': 3, 'high': 20},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
            'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]}
        }
    },
    'xgboost': {
        'class': XGBRegressor,
        'params': {
            'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
            'max_depth': {'type': 'int', 'low': 3, 'high': 12},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'reg_alpha': {'type': 'float', 'low': 0, 'high': 100, 'log': True},
            'reg_lambda': {'type': 'float', 'low': 1, 'high': 100, 'log': True}
        }
    },
    'lightgbm': {
        'class': LGBMRegressor,
        'params': {
            'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
            'max_depth': {'type': 'int', 'low': 3, 'high': 12},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'reg_alpha': {'type': 'float', 'low': 0, 'high': 100, 'log': True},
            'reg_lambda': {'type': 'float', 'low': 0, 'high': 100, 'log': True},
            'num_leaves': {'type': 'int', 'low': 10, 'high': 300}
        }
    }
}
```

## 4. Distributed Training on Google Cloud

### Vertex AI Training Configuration

```python
# training/distributed_training.py
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os
import logging
from typing import Dict, List, Any

class DistributedTrainer:
    """
    Distributed training on Google Cloud Vertex AI
    """
    
    def __init__(self, project_id: str, region: str = "us-central1", 
                 staging_bucket: str = None):
        self.project_id = project_id
        self.region = region
        self.staging_bucket = staging_bucket or f"{project_id}-ml-staging"
        
        aiplatform.init(
            project=project_id,
            location=region,
            staging_bucket=f"gs://{self.staging_bucket}"
        )
        
    def create_custom_training_job(self, 
                                  display_name: str,
                                  script_path: str,
                                  container_uri: str,
                                  machine_type: str = "n1-standard-4",
                                  replica_count: int = 1,
                                  args: List[str] = None) -> Any:
        """
        Create custom training job on Vertex AI
        """
        job = aiplatform.CustomTrainingJob(
            display_name=display_name,
            script_path=script_path,
            container_uri=container_uri,
            requirements=["pandas", "numpy", "scikit-learn", "xgboost", "lightgbm", 
                         "optuna", "mlflow", "google-cloud-storage"],
            model_serving_container_image_uri=container_uri,
        )
        
        model = job.run(
            dataset=None,
            replica_count=replica_count,
            machine_type=machine_type,
            args=args,
            environment_variables={
                "PROJECT_ID": self.project_id,
                "REGION": self.region,
                "STAGING_BUCKET": self.staging_bucket
            }
        )
        
        return model
    
    def create_hyperparameter_tuning_job(self,
                                       display_name: str,
                                       script_path: str,
                                       container_uri: str,
                                       parameter_spec: Dict[str, Any],
                                       metric_spec: Dict[str, str],
                                       max_trial_count: int = 100,
                                       parallel_trial_count: int = 5) -> Any:
        """
        Create hyperparameter tuning job
        """
        job = aiplatform.HyperparameterTuningJob(
            display_name=display_name,
            custom_job=aiplatform.CustomJob(
                display_name=f"{display_name}_custom_job",
                worker_pool_specs=[{
                    "machine_spec": {
                        "machine_type": "n1-standard-4",
                    },
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": container_uri,
                        "command": ["python", script_path],
                    },
                }],
            ),
            metric_spec=metric_spec,
            parameter_spec=parameter_spec,
            max_trial_count=max_trial_count,
            parallel_trial_count=parallel_trial_count,
        )
        
        job.run(sync=False)
        return job
    
    def deploy_model_endpoint(self, model, endpoint_display_name: str,
                             machine_type: str = "n1-standard-2",
                             min_replica_count: int = 1,
                             max_replica_count: int = 5) -> Any:
        """
        Deploy model to Vertex AI endpoint
        """
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name,
            project=self.project_id,
            location=self.region,
        )
        
        deployed_model = model.deploy(
            endpoint=endpoint,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            deployed_model_display_name=f"{endpoint_display_name}_deployed",
        )
        
        return endpoint, deployed_model

# Training script for distributed execution
def distributed_training_script():
    """
    Main training script for distributed execution
    This script will be run on Vertex AI training instances
    """
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="xgboost")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--cv-folds", type=int, default=5)
    
    args = parser.parse_args()
    
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data from Cloud Storage
    from google.cloud import storage
    client = storage.Client()
    
    # Download training data
    bucket_name = args.data_path.split("/")[2]
    blob_path = "/".join(args.data_path.split("/")[3:])
    
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename("training_data.csv")
    
    # Load and prepare data
    df = pd.read_csv("training_data.csv")
    
    # Feature engineering (same as in local training)
    # ... (feature engineering code)
    
    # Create trainer and run training
    config = TrainingConfig(
        model_type=args.model_type,
        target_variable='first_day_return',
        feature_columns=['revenue', 'profit_margin', 'debt_to_equity'],  # Add all features
        n_splits=args.cv_folds,
        hyperparameter_trials=args.n_trials
    )
    
    trainer = IPOValuationTrainer(config)
    trainer.train_model("training_data.csv")
    
    # Save model to Cloud Storage
    trainer.save_model("trained_model.joblib")
    
    output_bucket = args.output_path.split("/")[2]
    output_blob_path = "/".join(args.output_path.split("/")[3:])
    
    output_bucket_obj = client.bucket(output_bucket)
    output_blob = output_bucket_obj.blob(output_blob_path)
    output_blob.upload_from_filename("trained_model.joblib")
    
    logging.info(f"Model saved to {args.output_path}")

if __name__ == "__main__":
    distributed_training_script()
```

## 5. MLOps Best Practices

### Model Versioning and Registry

```python
# training/model_registry.py
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import hashlib

class ModelRegistry:
    """
    MLOps model registry for version control and lifecycle management
    """
    
    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db", 
                 registry_uri: str = None):
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri or tracking_uri
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(registry_uri)
        
        self.client = MlflowClient(tracking_uri)
        
    def register_model(self, model, model_name: str, experiment_name: str,
                      metrics: Dict[str, float], parameters: Dict[str, Any],
                      tags: Dict[str, str] = None,
                      description: str = None) -> str:
        """
        Register model with versioning and metadata
        """
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(parameters)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            model_info = mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=model_name,
                signature=self._infer_signature(model),
                input_example=self._get_input_example(model)
            )
            
            # Add tags
            if tags:
                mlflow.set_tags(tags)
                
            # Add description
            if description:
                mlflow.set_tag("description", description)
                
            # Add data signature
            data_hash = self._calculate_data_hash(parameters.get('training_data_path'))
            mlflow.set_tag("data_hash", data_hash)
            
            run_id = run.info.run_id
            
        logging.info(f"Model registered: {model_name}, Run ID: {run_id}")
        return run_id
    
    def promote_model(self, model_name: str, version: int, stage: str) -> None:
        """
        Promote model to different stage (Staging, Production, Archived)
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        
        logging.info(f"Model {model_name} version {version} promoted to {stage}")
    
    def get_latest_model(self, model_name: str, stage: str = "Production"):
        """
        Get latest model version from specified stage
        """
        latest_version = self.client.get_latest_versions(
            model_name, stages=[stage]
        )[0]
        
        model_uri = f"models:/{model_name}/{latest_version.version}"
        model = mlflow.sklearn.load_model(model_uri)
        
        return model, latest_version
    
    def compare_models(self, model_name: str, version1: int, version2: int) -> Dict[str, Any]:
        """
        Compare two model versions
        """
        # Get model versions
        v1_info = self.client.get_model_version(model_name, version1)
        v2_info = self.client.get_model_version(model_name, version2)
        
        # Get metrics for both versions
        v1_run = self.client.get_run(v1_info.run_id)
        v2_run = self.client.get_run(v2_info.run_id)
        
        comparison = {
            'version1': {
                'version': version1,
                'metrics': v1_run.data.metrics,
                'parameters': v1_run.data.params,
                'tags': v1_run.data.tags
            },
            'version2': {
                'version': version2,
                'metrics': v2_run.data.metrics,
                'parameters': v2_run.data.params,
                'tags': v2_run.data.tags
            }
        }
        
        # Calculate metric differences
        comparison['metric_differences'] = {}
        for metric in v1_run.data.metrics:
            if metric in v2_run.data.metrics:
                diff = v2_run.data.metrics[metric] - v1_run.data.metrics[metric]
                comparison['metric_differences'][metric] = diff
                
        return comparison
    
    def _infer_signature(self, model):
        """Infer MLflow signature from model"""
        # This would need actual training data to infer properly
        # For now, return None - implement based on your specific models
        return None
    
    def _get_input_example(self, model):
        """Get input example for model"""
        # Return sample input - implement based on your feature schema
        return None
    
    def _calculate_data_hash(self, data_path: Optional[str]) -> str:
        """Calculate hash of training data for reproducibility"""
        if not data_path:
            return "unknown"
            
        try:
            with open(data_path, 'rb') as f:
                data_hash = hashlib.md5(f.read()).hexdigest()
            return data_hash
        except:
            return "unknown"

class ModelLifecycleManager:
    """
    Manage model lifecycle stages and automated promotion
    """
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        
    def automated_model_promotion(self, model_name: str, 
                                 promotion_criteria: Dict[str, Any]) -> bool:
        """
        Automatically promote model based on performance criteria
        """
        try:
            # Get current staging model
            staging_model, staging_version = self.registry.get_latest_model(
                model_name, "Staging"
            )
            
            # Get current production model (if exists)
            try:
                prod_model, prod_version = self.registry.get_latest_model(
                    model_name, "Production"
                )
                
                # Compare models
                comparison = self.registry.compare_models(
                    model_name, int(prod_version.version), int(staging_version.version)
                )
                
                # Check promotion criteria
                should_promote = self._check_promotion_criteria(
                    comparison, promotion_criteria
                )
                
            except:
                # No production model exists, promote staging if it meets criteria
                should_promote = self._check_absolute_criteria(
                    staging_version, promotion_criteria
                )
            
            if should_promote:
                self.registry.promote_model(
                    model_name, int(staging_version.version), "Production"
                )
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Error in automated promotion: {e}")
            return False
    
    def _check_promotion_criteria(self, comparison: Dict[str, Any], 
                                 criteria: Dict[str, Any]) -> bool:
        """
        Check if staging model meets promotion criteria vs production
        """
        for metric, threshold in criteria.items():
            if metric in comparison['metric_differences']:
                improvement = comparison['metric_differences'][metric]
                
                # For metrics where lower is better (like RMSE)
                if metric.lower() in ['rmse', 'mae', 'mse']:
                    if improvement > -threshold:  # Negative improvement needed
                        return False
                # For metrics where higher is better (like R2)
                else:
                    if improvement < threshold:
                        return False
                        
        return True
    
    def _check_absolute_criteria(self, version_info, criteria: Dict[str, Any]) -> bool:
        """
        Check if model meets absolute performance criteria
        """
        # Get run metrics
        run = self.registry.client.get_run(version_info.run_id)
        metrics = run.data.metrics
        
        for metric, threshold in criteria.items():
            if metric not in metrics:
                return False
                
            value = metrics[metric]
            
            # Check threshold based on metric type
            if metric.lower() in ['rmse', 'mae', 'mse']:
                if value > threshold:
                    return False
            else:
                if value < threshold:
                    return False
                    
        return True

# Example usage
if __name__ == "__main__":
    # Initialize registry
    registry = ModelRegistry()
    
    # Register a model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100)
    
    # Example: Register model after training
    run_id = registry.register_model(
        model=model,
        model_name="ipo_valuation_rf",
        experiment_name="ipo_valuation",
        metrics={'rmse': 0.15, 'mae': 0.12, 'r2': 0.85},
        parameters={'n_estimators': 100, 'max_depth': 10},
        tags={'version': '1.0.0', 'model_type': 'random_forest'},
        description="Initial Random Forest model for IPO valuation"
    )
    
    # Automated promotion example
    lifecycle_manager = ModelLifecycleManager(registry)
    promotion_criteria = {'rmse': 0.02}  # Must improve RMSE by at least 0.02
    
    promoted = lifecycle_manager.automated_model_promotion(
        "ipo_valuation_rf", promotion_criteria
    )
    
    print(f"Model promoted: {promoted}")
```

This comprehensive training documentation covers:

1. **Automated Training Pipeline**: Complete pipeline implementation with feature engineering, model training, and validation
2. **Cross-validation Strategies**: Time series-specific validation techniques for financial data
3. **Hyperparameter Optimization**: Advanced optimization with Optuna and MLflow integration
4. **Distributed Training**: Google Cloud Vertex AI integration for scalable training
5. **MLOps Best Practices**: Model registry, versioning, and lifecycle management

The documentation includes executable code, configuration examples, and best practices specifically tailored for IPO valuation models.