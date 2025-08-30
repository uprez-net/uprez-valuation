# Quality Assurance Framework for IPO Valuation Platform

## Overview

This document outlines comprehensive quality assurance procedures, automated testing pipelines, data validation frameworks, and regulatory compliance checking for the Uprez IPO valuation platform.

## QA Framework Architecture

```
Quality Assurance Framework:
┌─────────────────────────────────────────────────────────────────┐
│                    Code Quality Gates                           │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Static        │   Security      │     Performance             │
│   Analysis      │   Scanning      │     Testing                 │
│   (SonarQube)   │   (SAST/DAST)   │     (Load Tests)           │
└─────────────────┴─────────────────┴─────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                  Automated Testing Pipeline                     │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Unit Tests    │   Integration   │     E2E Testing             │
│   (95% coverage)│   Tests         │     (Cypress/Playwright)    │
└─────────────────┴─────────────────┴─────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                   Data Quality & Compliance                     │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Data          │   Model         │     Regulatory              │
│   Validation    │   Validation    │     Compliance              │
│   (Great Exp.)  │   (ML Tests)    │     (ASIC/ASX)             │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## 1. Automated Testing Pipelines

### Test Pipeline Configuration

```yaml
# .github/workflows/qa-pipeline.yml
name: Quality Assurance Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  NODE_VERSION: '18'
  PYTHON_VERSION: '3.11'
  DATABASE_URL: 'postgresql://postgres:postgres@localhost:5432/uprez_test'

jobs:
  # Code Quality Analysis
  code-quality:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
      with:
        fetch-depth: 0  # Shallow clones should be disabled for better analysis
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
    
    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Type checking with mypy
      run: mypy src/ --ignore-missing-imports
    
    - name: Security linting with bandit
      run: bandit -r src/ -f json -o bandit-report.json
    
    - name: SonarCloud Scan
      uses: SonarSource/sonarcloud-github-action@master
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}

  # Unit Tests
  unit-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: uprez_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run unit tests with coverage
      run: |
        pytest tests/unit/ -v \
          --cov=src/ \
          --cov-report=xml \
          --cov-report=html \
          --cov-fail-under=95 \
          --junitxml=test-results.xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  # Integration Tests
  integration-tests:
    runs-on: ubuntu-latest
    needs: [unit-tests]
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: uprez_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run database migrations
      run: |
        alembic upgrade head
      env:
        DATABASE_URL: ${{ env.DATABASE_URL }}
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v \
          --junitxml=integration-test-results.xml
      env:
        DATABASE_URL: ${{ env.DATABASE_URL }}
        REDIS_URL: redis://localhost:6379

  # ML Model Tests
  model-tests:
    runs-on: ubuntu-latest
    needs: [unit-tests]
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install ML dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-ml.txt
    
    - name: Download test models and data
      run: |
        mkdir -p models/test_models
        mkdir -p data/test_data
        # Download test models from artifact store
        python scripts/download_test_assets.py
    
    - name: Run model validation tests
      run: |
        pytest tests/models/ -v \
          --junitxml=model-test-results.xml
    
    - name: Model performance tests
      run: |
        python tests/performance/model_performance_tests.py

  # Frontend Tests
  frontend-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: ${{ env.NODE_VERSION }}
        cache: 'yarn'
    
    - name: Install dependencies
      run: yarn install --frozen-lockfile
      working-directory: ./frontend
    
    - name: Run lint
      run: yarn lint
      working-directory: ./frontend
    
    - name: Run type check
      run: yarn type-check
      working-directory: ./frontend
    
    - name: Run unit tests
      run: yarn test --coverage --watchAll=false
      working-directory: ./frontend
    
    - name: Build application
      run: yarn build
      working-directory: ./frontend
      env:
        NEXT_PUBLIC_API_URL: http://localhost:8000

  # End-to-End Tests
  e2e-tests:
    runs-on: ubuntu-latest
    needs: [integration-tests, frontend-tests]
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: ${{ env.NODE_VERSION }}
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Start services with Docker Compose
      run: |
        docker-compose -f docker-compose.test.yml up -d
        sleep 30  # Wait for services to start
    
    - name: Install E2E test dependencies
      run: |
        yarn install
        npx playwright install --with-deps
      working-directory: ./e2e-tests
    
    - name: Run E2E tests
      run: npx playwright test
      working-directory: ./e2e-tests
    
    - name: Upload E2E test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: playwright-report
        path: e2e-tests/playwright-report/

  # Security Tests
  security-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run Snyk to check for vulnerabilities
      uses: snyk/actions/python@master
      env:
        SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
      with:
        args: --severity-threshold=high
    
    - name: Run OWASP ZAP security scan
      uses: zaproxy/action-full-scan@v0.4.0
      with:
        target: 'http://localhost:8000'

  # Performance Tests
  performance-tests:
    runs-on: ubuntu-latest
    needs: [integration-tests]
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup k6 for load testing
      run: |
        sudo gpg -k
        sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
        echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
        sudo apt-get update
        sudo apt-get install k6
    
    - name: Start test environment
      run: |
        docker-compose -f docker-compose.test.yml up -d
        sleep 30
    
    - name: Run load tests
      run: |
        k6 run tests/performance/load_test.js \
          --out json=load-test-results.json

  # Build and Test Docker Images
  docker-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Build test images
      run: |
        docker build -t uprez/api:test -f docker/api/Dockerfile .
        docker build -t uprez/ml-service:test -f docker/ml-service/Dockerfile .
        docker build -t uprez/frontend:test -f docker/frontend/Dockerfile ./frontend
    
    - name: Test Docker images
      run: |
        # Test API image
        docker run --rm -d --name api-test -p 8000:8000 uprez/api:test
        sleep 10
        curl -f http://localhost:8000/health || exit 1
        docker stop api-test
        
        # Test ML service image
        docker run --rm -d --name ml-test -p 8001:8000 uprez/ml-service:test
        sleep 20  # ML service takes longer to start
        curl -f http://localhost:8001/health || exit 1
        docker stop ml-test
    
    - name: Scan images for vulnerabilities
      run: |
        # Install Trivy
        sudo apt-get update
        sudo apt-get install wget apt-transport-https gnupg lsb-release
        wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
        echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
        sudo apt-get update
        sudo apt-get install trivy
        
        # Scan images
        trivy image --exit-code 1 --severity HIGH,CRITICAL uprez/api:test
        trivy image --exit-code 1 --severity HIGH,CRITICAL uprez/ml-service:test
        trivy image --exit-code 1 --severity HIGH,CRITICAL uprez/frontend:test
```

### Comprehensive Test Suite Implementation

```python
# tests/conftest.py
import pytest
import asyncio
from typing import AsyncGenerator, Generator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock

from src.main import app
from src.database import get_db_session, Base
from src.config import Settings

# Test database configuration
TEST_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/uprez_test"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()

@pytest.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session

@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """Create test client."""
    with TestClient(app) as client:
        yield client

@pytest.fixture
def sample_ipo_data() -> pd.DataFrame:
    """Create sample IPO data for testing."""
    np.random.seed(42)
    
    data = {
        'company_name': [f'Company_{i}' for i in range(100)],
        'listing_date': pd.date_range('2020-01-01', periods=100, freq='D'),
        'offer_price': np.random.uniform(10, 100, 100),
        'first_day_close': np.random.uniform(8, 150, 100),
        'revenue': np.random.uniform(1e6, 1e9, 100),
        'net_income': np.random.uniform(-1e6, 1e8, 100),
        'total_assets': np.random.uniform(1e6, 1e10, 100),
        'sector': np.random.choice(['Technology', 'Healthcare', 'Finance', 'Energy'], 100)
    }
    
    df = pd.DataFrame(data)
    df['first_day_return'] = (df['first_day_close'] - df['offer_price']) / df['offer_price']
    
    return df

@pytest.fixture
def mock_asx_api():
    """Mock ASX API responses."""
    mock = Mock()
    mock.get_company_data.return_value = {
        'company_code': 'TEST',
        'company_name': 'Test Company',
        'sector': 'Technology',
        'market_cap': 1000000000,
        'shares_outstanding': 10000000
    }
    mock.get_market_data.return_value = {
        'date': '2023-01-01',
        'asx_200_close': 7000.0,
        'volatility': 0.15,
        'volume': 1000000
    }
    return mock

@pytest.fixture
def mock_ml_model():
    """Mock ML model for testing."""
    mock = Mock()
    mock.predict.return_value = np.array([0.1, 0.05, -0.02, 0.08, 0.12])
    mock.predict_proba.return_value = np.array([[0.7, 0.3], [0.8, 0.2]])
    return mock

# Test utilities
class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_company_data(**kwargs):
        """Create test company data."""
        defaults = {
            'company_name': 'Test Company',
            'company_code': 'TEST',
            'sector': 'Technology',
            'revenue': 100000000,
            'net_income': 10000000,
            'total_assets': 500000000,
            'current_assets': 200000000,
            'current_liabilities': 100000000,
            'total_debt': 150000000,
            'shareholders_equity': 300000000
        }
        defaults.update(kwargs)
        return defaults
    
    @staticmethod
    def create_ipo_data(**kwargs):
        """Create test IPO data."""
        defaults = {
            'company_name': 'Test Company',
            'listing_date': '2023-01-15',
            'offer_price': 25.0,
            'offer_size': 250000000,
            'shares_offered': 10000000,
            'underwriter': 'Test Investment Bank',
            'sector': 'Technology'
        }
        defaults.update(kwargs)
        return defaults

# Performance test utilities
@pytest.fixture
def performance_monitor():
    """Monitor performance during tests."""
    import time
    import psutil
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            
        def start(self):
            self.start_time = time.time()
            self.start_memory = psutil.virtual_memory().used
            
        def stop(self):
            end_time = time.time()
            end_memory = psutil.virtual_memory().used
            
            return {
                'duration': end_time - self.start_time,
                'memory_delta': end_memory - self.start_memory
            }
    
    return PerformanceMonitor()
```

### Model Validation Tests

```python
# tests/models/test_model_validation.py
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import joblib
from unittest.mock import Mock, patch

from src.models.ipo_valuation_model import IPOValuationModel
from src.models.model_validator import ModelValidator, ValidationResult

class TestModelValidation:
    """Test ML model validation and performance."""
    
    @pytest.fixture
    def trained_model(self, sample_ipo_data):
        """Create a trained model for testing."""
        model = IPOValuationModel()
        
        # Prepare features and target
        features = [
            'revenue', 'net_income', 'total_assets', 'offer_price'
        ]
        X = sample_ipo_data[features]
        y = sample_ipo_data['first_day_return']
        
        # Train model
        model.train(X, y)
        
        return model
    
    def test_model_training_validation(self, trained_model, sample_ipo_data):
        """Test model training produces valid results."""
        assert trained_model.model is not None
        assert hasattr(trained_model, 'feature_columns')
        assert hasattr(trained_model, 'scaler')
        
        # Test predictions
        features = ['revenue', 'net_income', 'total_assets', 'offer_price']
        X_test = sample_ipo_data[features].iloc[:10]
        predictions = trained_model.predict(X_test)
        
        assert len(predictions) == 10
        assert all(isinstance(pred, (int, float)) for pred in predictions)
        assert all(not np.isnan(pred) for pred in predictions)
    
    def test_model_performance_metrics(self, trained_model, sample_ipo_data):
        """Test model performance meets minimum thresholds."""
        features = ['revenue', 'net_income', 'total_assets', 'offer_price']
        X = sample_ipo_data[features]
        y = sample_ipo_data['first_day_return']
        
        predictions = trained_model.predict(X)
        
        # Performance metrics
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        mae = np.mean(np.abs(y - predictions))
        
        # Minimum performance thresholds
        assert rmse < 0.5, f"RMSE {rmse} exceeds threshold"
        assert r2 > 0.1, f"R² {r2} below minimum threshold"
        assert mae < 0.3, f"MAE {mae} exceeds threshold"
    
    def test_model_prediction_bounds(self, trained_model, sample_ipo_data):
        """Test model predictions are within reasonable bounds."""
        features = ['revenue', 'net_income', 'total_assets', 'offer_price']
        X = sample_ipo_data[features]
        
        predictions = trained_model.predict(X)
        
        # IPO returns should generally be between -50% and +200%
        assert all(pred >= -0.5 for pred in predictions), "Predictions below reasonable lower bound"
        assert all(pred <= 2.0 for pred in predictions), "Predictions above reasonable upper bound"
    
    def test_model_robustness(self, trained_model, sample_ipo_data):
        """Test model robustness to input variations."""
        features = ['revenue', 'net_income', 'total_assets', 'offer_price']
        X_base = sample_ipo_data[features].iloc[0:1]
        
        # Test with small perturbations
        noise_levels = [0.01, 0.05, 0.1]
        base_prediction = trained_model.predict(X_base)[0]
        
        for noise_level in noise_levels:
            # Add random noise
            noise = np.random.normal(0, noise_level, X_base.shape)
            X_noisy = X_base + noise * X_base  # Relative noise
            
            prediction_noisy = trained_model.predict(X_noisy)[0]
            
            # Prediction should not change dramatically with small noise
            relative_change = abs(prediction_noisy - base_prediction) / abs(base_prediction)
            assert relative_change < noise_level * 10, f"Model too sensitive to {noise_level} noise"
    
    def test_model_feature_importance(self, trained_model):
        """Test feature importance is reasonable."""
        if hasattr(trained_model.model, 'feature_importances_'):
            importances = trained_model.model.feature_importances_
            
            # All importances should be positive
            assert all(imp >= 0 for imp in importances)
            
            # Importances should sum to 1 (or close to it)
            assert abs(sum(importances) - 1.0) < 0.01
            
            # No single feature should dominate completely
            assert max(importances) < 0.8, "Single feature dominates model"
    
    def test_model_consistency(self, trained_model, sample_ipo_data):
        """Test model produces consistent predictions."""
        features = ['revenue', 'net_income', 'total_assets', 'offer_price']
        X = sample_ipo_data[features].iloc[:5]
        
        # Make predictions multiple times
        predictions1 = trained_model.predict(X)
        predictions2 = trained_model.predict(X)
        
        # Predictions should be identical (no randomness in inference)
        np.testing.assert_array_equal(predictions1, predictions2)
    
    def test_model_serialization(self, trained_model, tmp_path):
        """Test model can be saved and loaded correctly."""
        model_path = tmp_path / "test_model.joblib"
        
        # Save model
        trained_model.save(str(model_path))
        assert model_path.exists()
        
        # Load model
        loaded_model = IPOValuationModel()
        loaded_model.load(str(model_path))
        
        # Test loaded model works
        test_data = pd.DataFrame({
            'revenue': [100000000],
            'net_income': [10000000],
            'total_assets': [500000000],
            'offer_price': [25.0]
        })
        
        original_pred = trained_model.predict(test_data)
        loaded_pred = loaded_model.predict(test_data)
        
        np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=6)

class TestModelValidator:
    """Test model validation framework."""
    
    def test_validation_result_structure(self):
        """Test validation result structure."""
        result = ValidationResult(
            passed=True,
            score=0.85,
            metrics={'rmse': 0.15, 'r2': 0.75},
            errors=[],
            warnings=['Low sample size']
        )
        
        assert result.passed is True
        assert result.score == 0.85
        assert 'rmse' in result.metrics
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
    
    def test_validator_performance_check(self, trained_model, sample_ipo_data):
        """Test performance validation."""
        validator = ModelValidator()
        
        features = ['revenue', 'net_income', 'total_assets', 'offer_price']
        X = sample_ipo_data[features]
        y = sample_ipo_data['first_day_return']
        
        result = validator.validate_performance(trained_model, X, y)
        
        assert isinstance(result, ValidationResult)
        assert 'rmse' in result.metrics
        assert 'r2' in result.metrics
        assert 'mae' in result.metrics
    
    def test_validator_data_quality_check(self, sample_ipo_data):
        """Test data quality validation."""
        validator = ModelValidator()
        
        # Test with good data
        result = validator.validate_data_quality(sample_ipo_data)
        assert result.passed
        
        # Test with bad data
        bad_data = sample_ipo_data.copy()
        bad_data.loc[0:10, 'revenue'] = np.nan  # Add missing values
        bad_data.loc[20:25, 'revenue'] = -1000000  # Add negative revenues
        
        result = validator.validate_data_quality(bad_data)
        assert not result.passed
        assert len(result.errors) > 0
    
    def test_validator_model_stability(self, trained_model, sample_ipo_data):
        """Test model stability validation."""
        validator = ModelValidator()
        
        features = ['revenue', 'net_income', 'total_assets', 'offer_price']
        X = sample_ipo_data[features]
        
        result = validator.validate_model_stability(trained_model, X)
        
        assert isinstance(result, ValidationResult)
        assert 'stability_score' in result.metrics

class TestModelMonitoring:
    """Test model monitoring and drift detection."""
    
    def test_prediction_logging(self, trained_model):
        """Test prediction logging for monitoring."""
        from src.models.model_monitor import ModelMonitor
        
        monitor = ModelMonitor()
        
        # Mock prediction data
        input_data = {
            'revenue': 100000000,
            'net_income': 10000000,
            'total_assets': 500000000,
            'offer_price': 25.0
        }
        
        prediction = 0.15
        
        # Log prediction
        monitor.log_prediction('ipo_valuation', input_data, prediction)
        
        # Check logging worked
        assert len(monitor.prediction_log) == 1
        assert monitor.prediction_log[0]['model'] == 'ipo_valuation'
        assert monitor.prediction_log[0]['prediction'] == prediction
    
    def test_performance_tracking(self):
        """Test model performance tracking over time."""
        from src.models.model_monitor import ModelMonitor
        
        monitor = ModelMonitor()
        
        # Simulate tracking performance over time
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        
        for i, date in enumerate(dates):
            # Simulate degrading performance
            rmse = 0.1 + i * 0.01
            r2 = 0.8 - i * 0.02
            
            monitor.log_performance('ipo_valuation', date, {
                'rmse': rmse,
                'r2': r2
            })
        
        # Check performance tracking
        assert len(monitor.performance_log) == 10
        
        # Check for performance degradation detection
        alerts = monitor.detect_performance_drift('ipo_valuation')
        assert len(alerts) > 0  # Should detect degradation

# Performance benchmarking tests
class TestModelPerformance:
    """Test model performance benchmarks."""
    
    def test_prediction_latency(self, trained_model, performance_monitor):
        """Test prediction latency benchmarks."""
        test_data = pd.DataFrame({
            'revenue': [100000000] * 1000,
            'net_income': [10000000] * 1000,
            'total_assets': [500000000] * 1000,
            'offer_price': [25.0] * 1000
        })
        
        performance_monitor.start()
        predictions = trained_model.predict(test_data)
        metrics = performance_monitor.stop()
        
        # Latency requirements
        avg_latency_per_prediction = metrics['duration'] / len(test_data)
        assert avg_latency_per_prediction < 0.001, f"Prediction latency {avg_latency_per_prediction}s too high"
        
        # Memory usage should be reasonable
        memory_per_prediction = metrics['memory_delta'] / len(test_data)
        assert memory_per_prediction < 1000, f"Memory usage {memory_per_prediction} bytes per prediction too high"
    
    def test_batch_processing_efficiency(self, trained_model):
        """Test batch processing efficiency."""
        # Test different batch sizes
        batch_sizes = [1, 10, 100, 1000]
        times = []
        
        for batch_size in batch_sizes:
            test_data = pd.DataFrame({
                'revenue': [100000000] * batch_size,
                'net_income': [10000000] * batch_size,
                'total_assets': [500000000] * batch_size,
                'offer_price': [25.0] * batch_size
            })
            
            import time
            start_time = time.time()
            predictions = trained_model.predict(test_data)
            end_time = time.time()
            
            times.append((end_time - start_time) / batch_size)
        
        # Larger batches should be more efficient per prediction
        assert times[-1] <= times[0], "Batch processing not efficient"
    
    @pytest.mark.slow
    def test_concurrent_predictions(self, trained_model):
        """Test concurrent prediction performance."""
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor
        
        test_data = pd.DataFrame({
            'revenue': [100000000],
            'net_income': [10000000],
            'total_assets': [500000000],
            'offer_price': [25.0]
        })
        
        def make_predictions():
            return trained_model.predict(test_data)
        
        # Test concurrent requests
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_predictions) for _ in range(100)]
            results = [future.result() for future in futures]
        end_time = time.time()
        
        # All predictions should succeed
        assert len(results) == 100
        assert all(len(result) == 1 for result in results)
        
        # Total time should be reasonable (much less than sequential)
        total_time = end_time - start_time
        assert total_time < 5.0, f"Concurrent predictions took {total_time}s"
```

This quality assurance framework documentation provides:

1. **Automated Testing Pipeline**: Complete GitHub Actions workflow with multiple test stages including code quality, unit tests, integration tests, E2E tests, security scans, and performance tests
2. **Comprehensive Test Configuration**: Test fixtures, utilities, and factory patterns for creating test data and mocking external dependencies
3. **Model Validation Framework**: Extensive ML model testing including performance metrics, robustness testing, feature importance validation, and serialization tests
4. **Performance Benchmarking**: Tests for prediction latency, batch processing efficiency, and concurrent request handling

The framework ensures high-quality, reliable code and models specifically for financial applications with strict performance and accuracy requirements.