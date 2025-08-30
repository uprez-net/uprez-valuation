# Custom Model Training on Vertex AI

## Overview

This guide covers training custom machine learning models on Vertex AI for IPO valuation tasks. We'll focus on financial-specific models that require domain expertise and custom architectures.

## Training Scripts Structure

### Base Training Script Template

```python
# training/base_financial_model.py
import os
import json
import argparse
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import storage
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class FinancialModelTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load training data from GCS or local path"""
        if data_path.startswith('gs://'):
            # Load from GCS
            df = pd.read_csv(data_path)
        else:
            # Load from local path
            df = pd.read_csv(data_path)
        
        self.logger.info(f"Loaded {len(df)} training samples")
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess financial data for training"""
        # Financial preprocessing
        df = self._calculate_financial_ratios(df)
        df = self._handle_missing_values(df)
        df = self._remove_outliers(df)
        
        # Feature selection
        feature_columns = self.config.get('feature_columns', [])
        target_column = self.config.get('target_column')
        
        X = df[feature_columns].values
        y = df[target_column].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def _calculate_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate financial ratios and derived features"""
        # Revenue growth
        if 'revenue' in df.columns:
            df['revenue_growth'] = df.groupby('company_id')['revenue'].pct_change(periods=4)
        
        # Profitability ratios
        if 'net_income' in df.columns and 'revenue' in df.columns:
            df['profit_margin'] = df['net_income'] / df['revenue']
        
        if 'net_income' in df.columns and 'total_equity' in df.columns:
            df['roe'] = df['net_income'] / df['total_equity']
        
        # Leverage ratios
        if 'total_debt' in df.columns and 'total_assets' in df.columns:
            df['debt_ratio'] = df['total_debt'] / df['total_assets']
        
        if 'total_debt' in df.columns and 'total_equity' in df.columns:
            df['debt_to_equity'] = df['total_debt'] / df['total_equity']
        
        # Efficiency ratios
        if 'revenue' in df.columns and 'total_assets' in df.columns:
            df['asset_turnover'] = df['revenue'] / df['total_assets']
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in financial data"""
        # Forward fill for time series data
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df.groupby('company_id')[numeric_columns].fillna(method='ffill')
        
        # Fill remaining with industry medians
        for col in numeric_columns:
            if 'sector' in df.columns:
                sector_median = df.groupby('sector')[col].transform('median')
                df[col] = df[col].fillna(sector_median)
            else:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers using IQR method"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def build_model(self) -> tf.keras.Model:
        """Build the neural network model"""
        raise NotImplementedError("Subclasses must implement build_model")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train the model"""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Build model
        self.model = self.build_model()
        
        # Setup callbacks
        callbacks = self._setup_callbacks()
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=self.config.get('epochs', 100),
            batch_size=self.config.get('batch_size', 32),
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        metrics = self._evaluate_model(X_val, y_val)
        
        return {
            'history': history.history,
            'metrics': metrics
        }
    
    def _setup_callbacks(self) -> list:
        """Setup training callbacks"""
        callbacks = []
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
        callbacks.append(checkpoint)
        
        # Reduce learning rate
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
        callbacks.append(reduce_lr)
        
        return callbacks
    
    def _evaluate_model(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.model.predict(X_val)
        
        # Calculate metrics
        mse = tf.keras.metrics.mean_squared_error(y_val, predictions).numpy().mean()
        mae = tf.keras.metrics.mean_absolute_error(y_val, predictions).numpy().mean()
        
        # Financial-specific metrics
        mape = np.mean(np.abs((y_val - predictions.flatten()) / y_val)) * 100
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(np.sqrt(mse)),
            'mape': float(mape)
        }
    
    def save_model(self, model_path: str):
        """Save trained model and artifacts"""
        # Save model
        self.model.save(os.path.join(model_path, 'model'))
        
        # Save scaler
        import joblib
        joblib.dump(self.scaler, os.path.join(model_path, 'scaler.joblib'))
        
        # Save config
        with open(os.path.join(model_path, 'config.json'), 'w') as f:
            json.dump(self.config, f)
```

### DCF Model Implementation

```python
# training/dcf_model.py
class DCFModelTrainer(FinancialModelTrainer):
    """Discounted Cash Flow valuation model trainer"""
    
    def build_model(self) -> tf.keras.Model:
        """Build DCF-specific neural network"""
        input_dim = len(self.config.get('feature_columns', []))
        
        inputs = tf.keras.layers.Input(shape=(input_dim,))
        
        # Feature embedding layers
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # DCF-specific layers
        # Cash flow prediction branch
        cash_flow = tf.keras.layers.Dense(32, activation='relu', name='cash_flow_branch')(x)
        cash_flow = tf.keras.layers.Dense(1, activation='linear', name='cash_flow_output')(cash_flow)
        
        # Growth rate prediction branch
        growth_rate = tf.keras.layers.Dense(32, activation='relu', name='growth_rate_branch')(x)
        growth_rate = tf.keras.layers.Dense(1, activation='sigmoid', name='growth_rate_output')(growth_rate)
        
        # Discount rate prediction branch
        discount_rate = tf.keras.layers.Dense(32, activation='relu', name='discount_rate_branch')(x)
        discount_rate = tf.keras.layers.Dense(1, activation='sigmoid', name='discount_rate_output')(discount_rate)
        
        # DCF calculation layer
        dcf_value = tf.keras.layers.Lambda(
            self._dcf_calculation,
            name='dcf_calculation'
        )([cash_flow, growth_rate, discount_rate])
        
        model = tf.keras.Model(
            inputs=inputs,
            outputs=[dcf_value, cash_flow, growth_rate, discount_rate]
        )
        
        # Custom loss function for DCF
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'dcf_calculation': 'mse',
                'cash_flow_output': 'mse',
                'growth_rate_output': 'mse',
                'discount_rate_output': 'mse'
            },
            loss_weights={
                'dcf_calculation': 1.0,
                'cash_flow_output': 0.3,
                'growth_rate_output': 0.3,
                'discount_rate_output': 0.3
            },
            metrics=['mae']
        )
        
        return model
    
    def _dcf_calculation(self, inputs):
        """Custom DCF calculation layer"""
        cash_flow, growth_rate, discount_rate = inputs
        
        # Simplified DCF calculation (5-year projection + terminal value)
        years = tf.range(1, 6, dtype=tf.float32)
        
        # Project cash flows
        projected_cf = cash_flow * tf.pow(1 + growth_rate, years)
        
        # Discount cash flows
        discount_factors = tf.pow(1 + discount_rate, years)
        pv_cash_flows = projected_cf / discount_factors
        
        # Terminal value (simplified)
        terminal_growth = growth_rate * 0.5  # Assume terminal growth is half of projection period
        terminal_cf = cash_flow * tf.pow(1 + growth_rate, 5.0) * (1 + terminal_growth)
        terminal_value = terminal_cf / (discount_rate - terminal_growth)
        pv_terminal = terminal_value / tf.pow(1 + discount_rate, 5.0)
        
        # Total DCF value
        dcf_value = tf.reduce_sum(pv_cash_flows, axis=-1, keepdims=True) + pv_terminal
        
        return dcf_value
```

### Training Script Entry Point

```python
# training/train_dcf_model.py
import argparse
import json
from dcf_model import DCFModelTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data', type=str, required=True)
    parser.add_argument('--model_output', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Initialize trainer
    trainer = DCFModelTrainer(config)
    
    # Load and preprocess data
    df = trainer.load_data(args.training_data)
    X, y = trainer.preprocess_data(df)
    
    # Train model
    results = trainer.train(X, y)
    
    # Save model
    trainer.save_model(args.model_output)
    
    print(f"Training completed. Model saved to {args.model_output}")
    print(f"Final metrics: {results['metrics']}")

if __name__ == '__main__':
    main()
```

## Container Configuration

### Dockerfile for Custom Training

```dockerfile
# training/Dockerfile
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-11

# Install additional dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy training code
COPY . /app
WORKDIR /app

# Set entry point
ENTRYPOINT ["python", "train_dcf_model.py"]
```

### Requirements File

```txt
# training/requirements.txt
tensorflow==2.11.0
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.2.2
google-cloud-storage==2.8.0
joblib==1.2.0
```

## Hyperparameter Tuning

```python
# training/hyperparameter_tuning.py
from google.cloud import aiplatform

def run_hyperparameter_tuning(
    project_id: str,
    region: str,
    training_data: str,
    container_uri: str
):
    """Run hyperparameter tuning job"""
    
    # Define hyperparameter search space
    parameter_spec = {
        'learning_rate': aiplatform.hyperparameter_tuning.DoubleParameterSpec(
            min=0.0001, max=0.01, scale='log'
        ),
        'batch_size': aiplatform.hyperparameter_tuning.DiscreteParameterSpec(
            values=[16, 32, 64, 128]
        ),
        'hidden_units': aiplatform.hyperparameter_tuning.IntegerParameterSpec(
            min=64, max=512
        ),
        'dropout_rate': aiplatform.hyperparameter_tuning.DoubleParameterSpec(
            min=0.1, max=0.5
        )
    }
    
    # Define metric to optimize
    metric_spec = aiplatform.hyperparameter_tuning.MetricSpec(
        metric_id='val_loss',
        goal='MINIMIZE'
    )
    
    # Create hyperparameter tuning job
    job = aiplatform.HyperparameterTuningJob(
        display_name='dcf-model-tuning',
        custom_job=aiplatform.CustomJob.from_local_script(
            display_name='dcf-training-job',
            script_path='train_dcf_model.py',
            container_uri=container_uri,
            args=[
                '--training_data', training_data,
                '--model_output', '/tmp/model'
            ]
        ),
        metric_spec=metric_spec,
        parameter_spec=parameter_spec,
        max_trial_count=50,
        parallel_trial_count=5
    )
    
    # Run tuning job
    job.run(sync=True)
    
    return job
```

## Model Validation

### Financial Model Validation

```python
# validation/financial_model_validator.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class FinancialModelValidator:
    """Validate financial models for accuracy and business logic"""
    
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
    
    def validate_predictions(
        self,
        test_data: pd.DataFrame,
        prediction_column: str = 'predicted_value'
    ) -> Dict[str, Any]:
        """Comprehensive validation of model predictions"""
        
        results = {}
        
        # Statistical validation
        results['statistical'] = self._statistical_validation(test_data, prediction_column)
        
        # Business logic validation
        results['business_logic'] = self._business_logic_validation(test_data, prediction_column)
        
        # Sector-specific validation
        results['sector_analysis'] = self._sector_validation(test_data, prediction_column)
        
        # Time series validation (if applicable)
        results['temporal'] = self._temporal_validation(test_data, prediction_column)
        
        return results
    
    def _statistical_validation(
        self,
        df: pd.DataFrame,
        pred_col: str
    ) -> Dict[str, float]:
        """Statistical accuracy metrics"""
        
        actual = df['actual_value']
        predicted = df[pred_col]
        
        # Calculate metrics
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Correlation
        correlation = np.corrcoef(actual, predicted)[0, 1]
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r_squared': r_squared,
            'correlation': correlation
        }
    
    def _business_logic_validation(
        self,
        df: pd.DataFrame,
        pred_col: str
    ) -> Dict[str, Any]:
        """Validate business logic constraints"""
        
        violations = []
        
        # Check for negative valuations (usually not valid)
        negative_valuations = (df[pred_col] < 0).sum()
        if negative_valuations > 0:
            violations.append(f"Found {negative_valuations} negative valuations")
        
        # Check for unrealistic valuation multiples
        if 'revenue' in df.columns:
            p_s_ratio = df[pred_col] / df['revenue']
            extreme_ps = ((p_s_ratio > 50) | (p_s_ratio < 0.1)).sum()
            if extreme_ps > 0:
                violations.append(f"Found {extreme_ps} extreme P/S ratios")
        
        # Check for valuation consistency with financial health
        if 'debt_ratio' in df.columns:
            high_debt_high_val = ((df['debt_ratio'] > 0.8) & (df[pred_col] > df[pred_col].quantile(0.9))).sum()
            if high_debt_high_val > 0:
                violations.append(f"Found {high_debt_high_val} high valuations for high-debt companies")
        
        return {
            'violations': violations,
            'violation_count': len(violations)
        }
    
    def _sector_validation(
        self,
        df: pd.DataFrame,
        pred_col: str
    ) -> Dict[str, Dict[str, float]]:
        """Validate predictions by sector"""
        
        if 'sector' not in df.columns:
            return {'error': 'No sector information available'}
        
        sector_results = {}
        
        for sector in df['sector'].unique():
            sector_df = df[df['sector'] == sector]
            
            if len(sector_df) > 5:  # Minimum samples for meaningful analysis
                sector_results[sector] = self._statistical_validation(sector_df, pred_col)
        
        return sector_results
    
    def _temporal_validation(
        self,
        df: pd.DataFrame,
        pred_col: str
    ) -> Dict[str, Any]:
        """Validate temporal consistency"""
        
        if 'date' not in df.columns:
            return {'error': 'No temporal information available'}
        
        df_sorted = df.sort_values('date')
        
        # Check for temporal consistency
        predictions = df_sorted[pred_col].values
        
        # Calculate prediction volatility
        prediction_changes = np.diff(predictions)
        volatility = np.std(prediction_changes)
        
        # Check for sudden jumps (potential model instability)
        sudden_jumps = (np.abs(prediction_changes) > 3 * np.std(prediction_changes)).sum()
        
        return {
            'prediction_volatility': volatility,
            'sudden_jumps': sudden_jumps,
            'mean_prediction_change': np.mean(np.abs(prediction_changes))
        }
```

## Deployment Configuration

### Model Export for Serving

```python
# deployment/model_exporter.py
import tensorflow as tf
import joblib
import json
import os

def export_model_for_serving(
    model_path: str,
    export_path: str,
    signature_name: str = 'serving_default'
):
    """Export trained model for Vertex AI serving"""
    
    # Load model
    model = tf.keras.models.load_model(os.path.join(model_path, 'model'))
    
    # Create serving signature
    @tf.function
    def serve_fn(inputs):
        return model(inputs)
    
    # Export model
    tf.saved_model.save(
        model,
        export_path,
        signatures={signature_name: serve_fn}
    )
    
    # Copy preprocessing artifacts
    scaler_path = os.path.join(model_path, 'scaler.joblib')
    if os.path.exists(scaler_path):
        import shutil
        shutil.copy(scaler_path, os.path.join(export_path, 'scaler.joblib'))
    
    # Copy config
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        import shutil
        shutil.copy(config_path, os.path.join(export_path, 'config.json'))

    print(f"Model exported to {export_path}")
```

## Next Steps

1. **Implement Specific Models**: Create trainers for each valuation model type
2. **Set up CI/CD**: Automate training pipeline deployment
3. **Model Registry**: Integrate with Vertex AI Model Registry
4. **Monitoring**: Set up training job monitoring and alerting
5. **Experimentation**: Implement experiment tracking with Vertex AI Experiments

## Related Files

- [Model Serving Guide](./model-serving.md)
- [Hyperparameter Tuning](./hyperparameter-tuning.md)
- [Model Validation](./model-validation.md)