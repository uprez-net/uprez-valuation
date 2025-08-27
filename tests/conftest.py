"""
Global pytest configuration and fixtures for the IPO Valuation Platform test suite.

This module provides shared fixtures, configuration, and utilities used across
all test modules in the project.
"""

import asyncio
import os
import tempfile
import warnings
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, Any, Generator
from unittest.mock import Mock, AsyncMock
import uuid

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer
from faker import Faker
import structlog

# Import project modules
from src.backend.api.main import create_application
from src.backend.config.settings import Settings
from src.backend.database.base import Base, get_async_session
from src.backend.database.models import User, Company, Valuation
from tests.utils.factories import (
    UserFactory, CompanyFactory, ValuationFactory,
    ProspectusFactory, MarketDataFactory
)
from tests.utils.mocks import MockGCPServices, MockRedisClient
from tests.utils.test_data import TestDataManager

# Configure pytest-asyncio
pytest_asyncio.fixture_scope = "session"

# Disable warnings for cleaner test output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

# Initialize Faker with a fixed seed for reproducible tests
fake = Faker()
fake.seed_instance(12345)

# Configure structured logging for tests
structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Create test-specific settings configuration."""
    test_env = {
        "ENVIRONMENT": "test",
        "DEBUG": "true",
        "DATABASE_URL": "sqlite:///./test.db",
        "REDIS_URL": "redis://localhost:6379/1",
        "SECRET_KEY": "test-secret-key-for-testing-only",
        "JWT_SECRET": "test-jwt-secret",
        "GOOGLE_CLOUD_PROJECT": "test-project",
        "BIGQUERY_DATASET": "test_dataset",
        "CLOUD_STORAGE_BUCKET": "test-bucket",
        "TESTING": "true"
    }
    
    # Override environment variables for testing
    original_env = os.environ.copy()
    os.environ.update(test_env)
    
    settings = Settings()
    
    yield settings
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="session")
async def postgres_container():
    """Start a PostgreSQL container for testing."""
    with PostgresContainer("postgres:15") as postgres:
        postgres.get_connection_url()  # Ensure container is ready
        yield postgres


@pytest.fixture(scope="session")
async def redis_container():
    """Start a Redis container for testing."""
    with RedisContainer("redis:7") as redis:
        redis.get_connection_url()  # Ensure container is ready
        yield redis


@pytest.fixture
async def test_db_engine(postgres_container):
    """Create a test database engine."""
    connection_url = postgres_container.get_connection_url()
    engine = create_engine(
        connection_url,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    yield engine
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
async def async_db_session(test_db_engine):
    """Create an async database session for testing."""
    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_db_engine
    )
    
    async def get_test_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    session = TestingSessionLocal()
    yield session
    session.close()


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    return MockRedisClient()


@pytest.fixture
def mock_gcp_services():
    """Create mock GCP services."""
    return MockGCPServices()


@pytest.fixture
def test_client(test_settings, async_db_session, mock_redis_client, mock_gcp_services):
    """Create a test client for the FastAPI application."""
    app = create_application()
    
    # Override dependencies
    app.dependency_overrides[get_async_session] = lambda: async_db_session
    
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_test_client(test_settings, async_db_session, mock_redis_client, mock_gcp_services):
    """Create an async test client for the FastAPI application."""
    app = create_application()
    
    # Override dependencies
    app.dependency_overrides[get_async_session] = lambda: async_db_session
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def test_data_manager():
    """Create a test data manager for handling test datasets."""
    return TestDataManager()


@pytest.fixture
def sample_user(async_db_session):
    """Create a sample user for testing."""
    user = UserFactory.create()
    async_db_session.add(user)
    async_db_session.commit()
    async_db_session.refresh(user)
    return user


@pytest.fixture
def sample_company(async_db_session, sample_user):
    """Create a sample company for testing."""
    company = CompanyFactory.create(created_by=sample_user.id)
    async_db_session.add(company)
    async_db_session.commit()
    async_db_session.refresh(company)
    return company


@pytest.fixture
def sample_valuation(async_db_session, sample_company, sample_user):
    """Create a sample valuation for testing."""
    valuation = ValuationFactory.create(
        company_id=sample_company.id,
        created_by=sample_user.id
    )
    async_db_session.add(valuation)
    async_db_session.commit()
    async_db_session.refresh(valuation)
    return valuation


@pytest.fixture
def auth_headers(sample_user):
    """Create authentication headers for testing."""
    # This would normally generate a real JWT token
    # For testing, we'll use a mock token
    token = "test-jwt-token"
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }


@pytest.fixture
def temporary_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_file_upload(temporary_directory):
    """Create a sample file for upload testing."""
    file_path = os.path.join(temporary_directory, "test_prospectus.pdf")
    with open(file_path, "wb") as f:
        f.write(b"Mock PDF content for testing")
    
    return {
        "file": open(file_path, "rb"),
        "document_type": "prospectus",
        "filename": "test_prospectus.pdf"
    }


@pytest.fixture(autouse=True)
async def cleanup_database(async_db_session):
    """Automatically cleanup database after each test."""
    yield
    
    # Clean up all data after each test
    try:
        async_db_session.rollback()
        for table in reversed(Base.metadata.sorted_tables):
            async_db_session.execute(table.delete())
        async_db_session.commit()
    except Exception as e:
        logger.warning(f"Database cleanup failed: {e}")
        async_db_session.rollback()


@pytest.fixture
def mock_market_data():
    """Create mock market data for testing."""
    return {
        "ASX:CBA": {
            "current_price": 105.50,
            "market_cap": 178000000000,
            "pe_ratio": 15.2,
            "dividend_yield": 0.045,
            "volume": 2500000
        },
        "ASX:WBC": {
            "current_price": 25.80,
            "market_cap": 85000000000,
            "pe_ratio": 12.8,
            "dividend_yield": 0.052,
            "volume": 8900000
        }
    }


@pytest.fixture
def mock_ml_models():
    """Create mock ML models for testing."""
    return {
        "dcf_model": Mock(),
        "peer_analysis_model": Mock(),
        "risk_assessment_model": Mock(),
        "sentiment_model": Mock()
    }


@pytest.fixture
def performance_test_data():
    """Create performance test data."""
    return {
        "companies": [CompanyFactory.build() for _ in range(100)],
        "users": [UserFactory.build() for _ in range(50)],
        "valuations": [ValuationFactory.build() for _ in range(200)]
    }


# Pytest markers for test categorization
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as a security test"
    )
    config.addinivalue_line(
        "markers", "ml: mark test as a machine learning test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "external: mark test as requiring external services"
    )


# Skip marks based on environment
def pytest_runtest_setup(item):
    """Skip tests based on markers and environment."""
    # Skip slow tests in CI unless explicitly requested
    if item.get_closest_marker("slow") and os.getenv("SKIP_SLOW_TESTS"):
        pytest.skip("Skipping slow test in CI environment")
    
    # Skip external tests if no external services available
    if item.get_closest_marker("external") and not os.getenv("TEST_EXTERNAL_SERVICES"):
        pytest.skip("Skipping external test - external services not configured")


# Hooks for test reporting
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Generate test reports with additional metadata."""
    outcome = yield
    rep = outcome.get_result()
    
    # Add custom attributes for reporting
    if rep.when == "call":
        setattr(item, f"rep_{rep.outcome}", rep)
        
        # Add test metadata for reporting
        if hasattr(item, "callspec"):
            rep.test_id = f"{item.name}[{item.callspec.id}]"
        else:
            rep.test_id = item.name
            
        rep.test_file = str(item.fspath.relative_to(item.config.rootdir))
        rep.test_category = "unit"  # Default category
        
        # Determine test category from markers
        if item.get_closest_marker("integration"):
            rep.test_category = "integration"
        elif item.get_closest_marker("e2e"):
            rep.test_category = "e2e"
        elif item.get_closest_marker("performance"):
            rep.test_category = "performance"
        elif item.get_closest_marker("security"):
            rep.test_category = "security"
        elif item.get_closest_marker("ml"):
            rep.test_category = "ml"


# Session fixtures for expensive setup
@pytest.fixture(scope="session")
def ml_test_models():
    """Load ML models for testing (expensive operation)."""
    logger.info("Loading ML test models...")
    
    # Mock expensive ML model loading
    models = {
        "valuation_model": Mock(),
        "risk_model": Mock(),
        "peer_analysis_model": Mock()
    }
    
    yield models
    
    logger.info("Cleanup ML test models...")


@pytest.fixture(scope="session")
def test_datasets():
    """Load test datasets (expensive operation)."""
    logger.info("Loading test datasets...")
    
    datasets = {
        "training_data": fake.pydict(nb_elements=1000),
        "validation_data": fake.pydict(nb_elements=200),
        "test_data": fake.pydict(nb_elements=200)
    }
    
    yield datasets
    
    logger.info("Cleanup test datasets...")


# Utility functions for tests
def assert_response_structure(response_json: Dict[str, Any], expected_keys: list):
    """Assert that a response has the expected structure."""
    assert isinstance(response_json, dict), "Response should be a dictionary"
    for key in expected_keys:
        assert key in response_json, f"Missing key '{key}' in response"


def assert_valid_uuid(value: str):
    """Assert that a string is a valid UUID."""
    try:
        uuid.UUID(value)
    except ValueError:
        pytest.fail(f"'{value}' is not a valid UUID")


def assert_valid_timestamp(value: str):
    """Assert that a string is a valid ISO timestamp."""
    try:
        datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        pytest.fail(f"'{value}' is not a valid ISO timestamp")