"""
Mock services and utilities for testing the IPO Valuation Platform.

This module provides mock implementations of external services and dependencies
to enable isolated testing without relying on external systems.
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional, Union
from unittest.mock import Mock, AsyncMock, MagicMock
import uuid


class MockRedisClient:
    """Mock Redis client for testing caching functionality."""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._expiry: Dict[str, datetime] = {}
    
    async def get(self, key: str) -> Optional[bytes]:
        """Get value from mock Redis."""
        if key in self._expiry and datetime.utcnow() > self._expiry[key]:
            del self._data[key]
            del self._expiry[key]
            return None
        
        value = self._data.get(key)
        return value.encode() if value else None
    
    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set value in mock Redis."""
        self._data[key] = value
        if ex:
            self._expiry[key] = datetime.utcnow() + timedelta(seconds=ex)
        return True
    
    async def delete(self, key: str) -> int:
        """Delete key from mock Redis."""
        deleted = 0
        if key in self._data:
            del self._data[key]
            deleted += 1
        if key in self._expiry:
            del self._expiry[key]
        return deleted
    
    async def exists(self, key: str) -> int:
        """Check if key exists in mock Redis."""
        if key in self._expiry and datetime.utcnow() > self._expiry[key]:
            del self._data[key]
            del self._expiry[key]
            return 0
        return 1 if key in self._data else 0
    
    async def flushall(self) -> bool:
        """Clear all data from mock Redis."""
        self._data.clear()
        self._expiry.clear()
        return True
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern."""
        # Simple pattern matching for testing
        if pattern == "*":
            return list(self._data.keys())
        
        # Handle simple wildcard patterns
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return [k for k in self._data.keys() if k.startswith(prefix)]
        
        return [k for k in self._data.keys() if k == pattern]


class MockBigQueryClient:
    """Mock BigQuery client for testing ML and analytics functionality."""
    
    def __init__(self):
        self.datasets = {}
        self.tables = {}
        self.jobs = {}
    
    def dataset(self, dataset_id: str):
        """Get or create mock dataset."""
        return MockBigQueryDataset(dataset_id, self)
    
    def create_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Create mock dataset."""
        self.datasets[dataset_id] = {
            'id': dataset_id,
            'created': datetime.utcnow(),
            'tables': []
        }
        return self.datasets[dataset_id]
    
    def query(self, query: str, job_config: Optional[Dict] = None) -> 'MockQueryJob':
        """Execute mock query."""
        job_id = f"job_{uuid.uuid4().hex[:16]}"
        
        # Generate mock results based on query type
        if "SELECT" in query.upper():
            rows = self._generate_query_results(query)
        else:
            rows = []
        
        job = MockQueryJob(job_id, query, rows)
        self.jobs[job_id] = job
        return job
    
    def _generate_query_results(self, query: str) -> List[Dict[str, Any]]:
        """Generate mock query results."""
        query_lower = query.lower()
        
        if "market_data" in query_lower:
            return [
                {
                    'ticker': f'ASX:{random.choice(["CBA", "WBC", "ANZ", "BHP"])}',
                    'current_price': random.uniform(20.0, 150.0),
                    'market_cap': random.randint(10**9, 10**11),
                    'pe_ratio': random.uniform(8.0, 25.0),
                    'timestamp': datetime.utcnow()
                }
                for _ in range(random.randint(5, 20))
            ]
        
        elif "financial_data" in query_lower:
            return [
                {
                    'company_id': str(uuid.uuid4()),
                    'revenue': random.randint(10**6, 10**9),
                    'profit': random.randint(10**5, 10**8),
                    'year': year
                }
                for year in range(2020, 2024)
            ]
        
        elif "peer_analysis" in query_lower:
            return [
                {
                    'ticker': f'ASX:{random.choice(["XYZ", "ABC", "DEF"])}',
                    'industry_sector': 'Technology',
                    'market_cap': random.randint(10**8, 10**10),
                    'pe_ratio': random.uniform(12.0, 30.0),
                    'ev_ebitda': random.uniform(8.0, 20.0)
                }
                for _ in range(10)
            ]
        
        return [{'result': 'success', 'rows_affected': random.randint(1, 100)}]


class MockBigQueryDataset:
    """Mock BigQuery dataset."""
    
    def __init__(self, dataset_id: str, client: MockBigQueryClient):
        self.dataset_id = dataset_id
        self.client = client
    
    def table(self, table_id: str):
        """Get mock table."""
        return MockBigQueryTable(self.dataset_id, table_id, self.client)


class MockBigQueryTable:
    """Mock BigQuery table."""
    
    def __init__(self, dataset_id: str, table_id: str, client: MockBigQueryClient):
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.client = client
        self.full_table_id = f"{dataset_id}.{table_id}"
    
    def exists(self) -> bool:
        """Check if table exists."""
        return self.full_table_id in self.client.tables


class MockQueryJob:
    """Mock BigQuery query job."""
    
    def __init__(self, job_id: str, query: str, results: List[Dict[str, Any]]):
        self.job_id = job_id
        self.query = query
        self._results = results
        self.state = 'DONE'
        self.errors = None
    
    def result(self):
        """Get query results."""
        return MockQueryResult(self._results)
    
    def to_dataframe(self):
        """Convert results to DataFrame."""
        import pandas as pd
        return pd.DataFrame(self._results)


class MockQueryResult:
    """Mock BigQuery query result."""
    
    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows
    
    def __iter__(self):
        return iter(self._rows)
    
    def to_dataframe(self):
        """Convert to DataFrame."""
        import pandas as pd
        return pd.DataFrame(self._rows)


class MockCloudStorageClient:
    """Mock Google Cloud Storage client."""
    
    def __init__(self):
        self.buckets = {}
    
    def bucket(self, bucket_name: str):
        """Get or create mock bucket."""
        if bucket_name not in self.buckets:
            self.buckets[bucket_name] = MockStorageBucket(bucket_name)
        return self.buckets[bucket_name]


class MockStorageBucket:
    """Mock Cloud Storage bucket."""
    
    def __init__(self, name: str):
        self.name = name
        self.blobs = {}
    
    def blob(self, blob_name: str):
        """Get or create mock blob."""
        if blob_name not in self.blobs:
            self.blobs[blob_name] = MockStorageBlob(blob_name, self)
        return self.blobs[blob_name]
    
    def list_blobs(self, prefix: Optional[str] = None):
        """List blobs in bucket."""
        if prefix:
            return [blob for name, blob in self.blobs.items() if name.startswith(prefix)]
        return list(self.blobs.values())


class MockStorageBlob:
    """Mock Cloud Storage blob."""
    
    def __init__(self, name: str, bucket: MockStorageBucket):
        self.name = name
        self.bucket = bucket
        self._data = b""
        self.size = 0
        self.content_type = "application/octet-stream"
        self.time_created = datetime.utcnow()
        self.updated = datetime.utcnow()
    
    def upload_from_string(self, data: Union[str, bytes], content_type: Optional[str] = None):
        """Upload data to blob."""
        if isinstance(data, str):
            self._data = data.encode()
        else:
            self._data = data
        self.size = len(self._data)
        if content_type:
            self.content_type = content_type
        self.updated = datetime.utcnow()
    
    def upload_from_file(self, file_obj, content_type: Optional[str] = None):
        """Upload from file object."""
        self._data = file_obj.read()
        self.size = len(self._data)
        if content_type:
            self.content_type = content_type
        self.updated = datetime.utcnow()
    
    def download_as_bytes(self) -> bytes:
        """Download blob data."""
        return self._data
    
    def download_as_text(self) -> str:
        """Download blob as text."""
        return self._data.decode()
    
    def exists(self) -> bool:
        """Check if blob exists."""
        return self.name in self.bucket.blobs
    
    def delete(self):
        """Delete blob."""
        if self.name in self.bucket.blobs:
            del self.bucket.blobs[self.name]
    
    @property
    def public_url(self) -> str:
        """Get public URL for blob."""
        return f"https://storage.googleapis.com/{self.bucket.name}/{self.name}"


class MockDocumentAI:
    """Mock Google Document AI client."""
    
    def __init__(self):
        self.processors = {}
    
    def process_document(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process document with mock AI."""
        # Simulate document processing
        document_type = request.get('document', {}).get('mime_type', 'application/pdf')
        
        if 'prospectus' in str(request).lower():
            return self._process_prospectus()
        elif 'financial' in str(request).lower():
            return self._process_financial_statement()
        else:
            return self._process_generic_document()
    
    def _process_prospectus(self) -> Dict[str, Any]:
        """Mock prospectus processing results."""
        return {
            'document': {
                'text': 'Mock extracted text from prospectus...',
                'entities': [
                    {
                        'type': 'COMPANY_NAME',
                        'mention_text': 'TechCorp Pty Ltd',
                        'confidence': 0.98
                    },
                    {
                        'type': 'REVENUE',
                        'mention_text': '$50M',
                        'confidence': 0.95
                    },
                    {
                        'type': 'INDUSTRY',
                        'mention_text': 'Software & Services',
                        'confidence': 0.92
                    }
                ],
                'pages': [
                    {
                        'page_number': 1,
                        'confidence': 0.96
                    }
                ]
            }
        }
    
    def _process_financial_statement(self) -> Dict[str, Any]:
        """Mock financial statement processing results."""
        return {
            'document': {
                'text': 'Mock extracted text from financial statements...',
                'entities': [
                    {
                        'type': 'REVENUE',
                        'mention_text': '$45,000,000',
                        'confidence': 0.99
                    },
                    {
                        'type': 'NET_PROFIT',
                        'mention_text': '$8,500,000',
                        'confidence': 0.97
                    },
                    {
                        'type': 'ASSETS',
                        'mention_text': '$120,000,000',
                        'confidence': 0.94
                    }
                ]
            }
        }
    
    def _process_generic_document(self) -> Dict[str, Any]:
        """Mock generic document processing."""
        return {
            'document': {
                'text': 'Mock extracted text...',
                'entities': [],
                'pages': [{'page_number': 1, 'confidence': 0.90}]
            }
        }


class MockVertexAI:
    """Mock Vertex AI client for ML model testing."""
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
    
    def predict(self, model_name: str, instances: List[Dict], parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """Make mock predictions."""
        if 'valuation' in model_name.lower():
            return self._predict_valuation(instances)
        elif 'risk' in model_name.lower():
            return self._predict_risk(instances)
        elif 'sentiment' in model_name.lower():
            return self._predict_sentiment(instances)
        else:
            return self._generic_prediction(instances)
    
    def _predict_valuation(self, instances: List[Dict]) -> Dict[str, Any]:
        """Mock valuation predictions."""
        predictions = []
        for instance in instances:
            predictions.append({
                'target_price': random.uniform(2.0, 8.0),
                'confidence': random.uniform(0.7, 0.95),
                'price_range': {
                    'low': random.uniform(1.5, 3.0),
                    'high': random.uniform(6.0, 10.0)
                },
                'method': random.choice(['dcf', 'multiples', 'asset_based'])
            })
        
        return {'predictions': predictions}
    
    def _predict_risk(self, instances: List[Dict]) -> Dict[str, Any]:
        """Mock risk predictions."""
        predictions = []
        for instance in instances:
            predictions.append({
                'risk_score': random.uniform(1.0, 10.0),
                'risk_factors': [
                    {
                        'category': 'market',
                        'severity': random.choice(['low', 'medium', 'high']),
                        'confidence': random.uniform(0.8, 0.99)
                    },
                    {
                        'category': 'financial',
                        'severity': random.choice(['low', 'medium', 'high']),
                        'confidence': random.uniform(0.8, 0.99)
                    }
                ]
            })
        
        return {'predictions': predictions}
    
    def _predict_sentiment(self, instances: List[Dict]) -> Dict[str, Any]:
        """Mock sentiment predictions."""
        predictions = []
        for instance in instances:
            predictions.append({
                'sentiment': random.choice(['positive', 'negative', 'neutral']),
                'confidence': random.uniform(0.75, 0.98),
                'score': random.uniform(-1.0, 1.0)
            })
        
        return {'predictions': predictions}
    
    def _generic_prediction(self, instances: List[Dict]) -> Dict[str, Any]:
        """Generic mock predictions."""
        return {
            'predictions': [
                {'result': f'mock_result_{i}', 'confidence': random.uniform(0.6, 0.9)}
                for i, _ in enumerate(instances)
            ]
        }


class MockCeleryApp:
    """Mock Celery application for task testing."""
    
    def __init__(self):
        self.tasks = {}
        self.results = {}
    
    def send_task(self, task_name: str, args: List = None, kwargs: Dict = None) -> 'MockAsyncResult':
        """Send mock task."""
        task_id = str(uuid.uuid4())
        result = MockAsyncResult(task_id, task_name, args or [], kwargs or {})
        self.results[task_id] = result
        return result
    
    def task(self, name: str):
        """Decorator for registering tasks."""
        def decorator(func):
            self.tasks[name] = func
            return func
        return decorator


class MockAsyncResult:
    """Mock Celery AsyncResult."""
    
    def __init__(self, task_id: str, task_name: str, args: List, kwargs: Dict):
        self.id = task_id
        self.task_name = task_name
        self.args = args
        self.kwargs = kwargs
        self.state = 'PENDING'
        self._result = None
    
    def get(self, timeout: Optional[float] = None):
        """Get task result."""
        # Simulate task completion
        self.state = 'SUCCESS'
        
        if 'valuation' in self.task_name.lower():
            self._result = {
                'valuation_id': self.args[0] if self.args else str(uuid.uuid4()),
                'status': 'completed',
                'target_price': random.uniform(2.0, 8.0),
                'confidence': random.uniform(0.8, 0.95)
            }
        else:
            self._result = {'status': 'completed'}
        
        return self._result
    
    @property
    def ready(self) -> bool:
        """Check if task is ready."""
        return self.state in ['SUCCESS', 'FAILURE']
    
    @property
    def successful(self) -> bool:
        """Check if task was successful."""
        return self.state == 'SUCCESS'


class MockExternalAPIClient:
    """Mock external API client for market data and other services."""
    
    def __init__(self):
        self.rate_limits = {}
        self.responses = {}
    
    async def get_asx_data(self, ticker: str) -> Dict[str, Any]:
        """Mock ASX market data."""
        await asyncio.sleep(0.01)  # Simulate network delay
        
        return {
            'ticker': ticker,
            'current_price': random.uniform(20.0, 150.0),
            'volume': random.randint(100000, 5000000),
            'market_cap': random.randint(10**8, 10**11),
            'pe_ratio': random.uniform(8.0, 25.0),
            'dividend_yield': random.uniform(0.0, 0.08),
            'last_updated': datetime.utcnow().isoformat()
        }
    
    async def get_company_fundamentals(self, identifier: str) -> Dict[str, Any]:
        """Mock company fundamental data."""
        await asyncio.sleep(0.02)  # Simulate network delay
        
        return {
            'company_id': identifier,
            'revenue': random.randint(10**6, 10**9),
            'net_income': random.randint(10**5, 10**8),
            'total_assets': random.randint(10**7, 10**10),
            'debt_to_equity': random.uniform(0.1, 2.0),
            'roe': random.uniform(0.05, 0.25),
            'revenue_growth': random.uniform(-0.1, 0.3)
        }
    
    async def search_peer_companies(self, industry: str, market_cap_range: tuple) -> List[Dict[str, Any]]:
        """Mock peer company search."""
        await asyncio.sleep(0.05)  # Simulate network delay
        
        peers = []
        for _ in range(random.randint(5, 15)):
            peers.append({
                'ticker': f"ASX:{random.choice(['ABC', 'DEF', 'XYZ', 'PQR'])}",
                'company_name': f"Mock Company {random.randint(1, 100)} Ltd",
                'industry': industry,
                'market_cap': random.randint(*market_cap_range),
                'pe_ratio': random.uniform(8.0, 30.0),
                'ev_ebitda': random.uniform(6.0, 25.0)
            })
        
        return peers


class MockGCPServices:
    """Aggregated mock GCP services."""
    
    def __init__(self):
        self.bigquery = MockBigQueryClient()
        self.storage = MockCloudStorageClient()
        self.document_ai = MockDocumentAI()
        self.vertex_ai = MockVertexAI()
    
    def setup_test_data(self):
        """Set up common test data across services."""
        # Create test dataset in BigQuery
        self.bigquery.create_dataset('test_dataset')
        
        # Create test bucket in Storage
        test_bucket = self.storage.bucket('test-bucket')
        
        # Upload sample files
        test_bucket.blob('test-prospectus.pdf').upload_from_string(
            b"Mock prospectus content", 
            content_type='application/pdf'
        )


# Utility functions for creating mock services
def create_mock_gcp_services() -> MockGCPServices:
    """Create configured mock GCP services."""
    services = MockGCPServices()
    services.setup_test_data()
    return services


def create_mock_external_apis() -> Dict[str, Any]:
    """Create mock external API clients."""
    return {
        'asx_client': MockExternalAPIClient(),
        'reuters_client': MockExternalAPIClient(),
        'bloomberg_client': MockExternalAPIClient()
    }


def create_mock_ml_services() -> Dict[str, Any]:
    """Create mock ML/AI services."""
    return {
        'vertex_ai': MockVertexAI(),
        'document_ai': MockDocumentAI(),
        'custom_models': {
            'valuation_model': Mock(),
            'risk_model': Mock(),
            'sentiment_model': Mock()
        }
    }