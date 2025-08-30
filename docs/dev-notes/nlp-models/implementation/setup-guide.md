# NLP Models Implementation Setup Guide

This guide provides step-by-step instructions for setting up the complete NLP pipeline for Australian financial document processing in the Uprez valuation system.

## Environment Setup

### System Requirements

**Minimum Requirements:**
- Python 3.9+
- 16GB RAM
- 100GB available storage
- CPU: 8 cores
- Optional: GPU with 8GB+ VRAM (recommended for training)

**Recommended Requirements:**
- Python 3.10+
- 32GB RAM
- 500GB SSD storage
- CPU: 16+ cores
- GPU: NVIDIA RTX 4090 or A100 (24GB+ VRAM)

### 1. Virtual Environment Setup

```bash
# Create project directory
mkdir uprez-nlp && cd uprez-nlp

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 2. Core Dependencies Installation

```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core NLP libraries
pip install transformers==4.35.2
pip install spacy==3.7.2
pip install nltk==3.8.1
pip install scikit-learn==1.3.2
pip install pandas==2.1.3
pip install numpy==1.24.3

# Install document processing libraries
pip install google-cloud-documentai==2.20.1
pip install PyPDF2==3.0.1
pip install pdfplumber==0.9.0
pip install opencv-python==4.8.1.78
pip install pillow==10.1.0

# Install ML/AI frameworks
pip install optuna==3.4.0
pip install wandb==0.16.0
pip install tensorboard==2.14.1

# Install web frameworks
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install pydantic==2.4.2

# Install database and caching
pip install sqlalchemy==2.0.23
pip install redis==5.0.1
pip install psycopg2-binary==2.9.9

# Install visualization and analysis
pip install matplotlib==3.8.2
pip install seaborn==0.13.0
pip install plotly==5.17.0
pip install jupyter==1.0.0

# Install additional utilities
pip install tqdm==4.66.1
pip install python-dotenv==1.0.0
pip install requests==2.31.0
pip install click==8.1.7
```

### 3. spaCy Model Installation

```bash
# Download English models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_trf

# Install spaCy transformers
pip install spacy-transformers==1.3.3
```

### 4. NLTK Data Download

```python
# Run this in Python to download NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
```

## Project Structure Setup

### Directory Structure

```bash
# Create project structure
mkdir -p uprez-nlp/{
    src/{
        models,
        preprocessing,
        training,
        evaluation,
        api,
        utils
    },
    data/{
        raw,
        processed,
        models,
        cache
    },
    config,
    tests,
    scripts,
    docs,
    logs,
    notebooks,
    docker
}
```

### Configuration Files

#### `config/config.yaml`

```yaml
# Main configuration file
project:
  name: "uprez-nlp"
  version: "1.0.0"
  description: "NLP models for Australian financial document processing"

data:
  raw_data_path: "data/raw"
  processed_data_path: "data/processed"
  cache_path: "data/cache"
  models_path: "data/models"

models:
  # Document processing models
  document_ai:
    project_id: "your-gcp-project"
    location: "us"
    processor_id: "your-processor-id"
    
  # NER model configuration
  ner:
    base_model: "en_core_web_trf"
    max_length: 512
    batch_size: 16
    learning_rate: 2e-5
    num_epochs: 10
    
  # Sentiment analysis configuration
  sentiment:
    base_model: "distilbert-base-uncased"
    num_classes: 5
    max_length: 512
    batch_size: 32
    learning_rate: 3e-5
    num_epochs: 5
    
  # Topic modeling configuration
  topic_modeling:
    num_topics: 20
    alpha: 0.02
    beta: 0.02
    iterations: 1000
    
  # BERT model configuration
  ausfinbert:
    vocab_size: 50000
    hidden_size: 768
    num_hidden_layers: 12
    num_attention_heads: 12
    intermediate_size: 3072
    max_position_embeddings: 512

training:
  device: "cuda"  # or "cpu"
  mixed_precision: true
  gradient_accumulation_steps: 4
  warmup_steps: 1000
  logging_steps: 100
  save_steps: 1000
  eval_steps: 500
  max_grad_norm: 1.0

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 300
  max_request_size: 10485760  # 10MB

database:
  url: "postgresql://user:password@localhost:5432/uprez_nlp"
  pool_size: 10
  max_overflow: 20

redis:
  host: "localhost"
  port: 6379
  db: 0
  password: null

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/uprez-nlp.log"
  max_bytes: 10485760  # 10MB
  backup_count: 5

monitoring:
  wandb:
    project: "uprez-nlp"
    entity: "your-wandb-entity"
  tensorboard:
    log_dir: "logs/tensorboard"
```

#### `requirements.txt`

```text
torch>=2.0.0
transformers>=4.35.0
spacy>=3.7.0
spacy-transformers>=1.3.0
nltk>=3.8.0
scikit-learn>=1.3.0
pandas>=2.1.0
numpy>=1.24.0
google-cloud-documentai>=2.20.0
PyPDF2>=3.0.0
pdfplumber>=0.9.0
opencv-python>=4.8.0
pillow>=10.0.0
optuna>=3.4.0
wandb>=0.16.0
tensorboard>=2.14.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.4.0
sqlalchemy>=2.0.0
redis>=5.0.0
psycopg2-binary>=2.9.0
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.17.0
jupyter>=1.0.0
tqdm>=4.66.0
python-dotenv>=1.0.0
requests>=2.31.0
click>=8.1.0
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-asyncio>=0.21.0
black>=23.9.0
flake8>=6.0.0
isort>=5.12.0
mypy>=1.6.0
```

## Google Cloud Setup

### 1. Google Cloud Project Setup

```bash
# Install Google Cloud SDK
# Follow instructions at: https://cloud.google.com/sdk/docs/install

# Initialize gcloud
gcloud init

# Create new project (optional)
gcloud projects create uprez-nlp-project --name="Uprez NLP"

# Set project
gcloud config set project uprez-nlp-project

# Enable required APIs
gcloud services enable documentai.googleapis.com
gcloud services enable language.googleapis.com
gcloud services enable translate.googleapis.com
gcloud services enable storage.googleapis.com
```

### 2. Service Account Setup

```bash
# Create service account
gcloud iam service-accounts create uprez-nlp-service \
    --description="Service account for Uprez NLP system" \
    --display-name="Uprez NLP Service Account"

# Get service account email
SERVICE_ACCOUNT_EMAIL=$(gcloud iam service-accounts list \
    --filter="displayName:Uprez NLP Service Account" \
    --format="value(email)")

# Grant necessary roles
gcloud projects add-iam-policy-binding uprez-nlp-project \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/documentai.apiUser"

gcloud projects add-iam-policy-binding uprez-nlp-project \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/language.admin"

gcloud projects add-iam-policy-binding uprez-nlp-project \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/storage.objectAdmin"

# Create and download service account key
gcloud iam service-accounts keys create credentials.json \
    --iam-account=$SERVICE_ACCOUNT_EMAIL

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
```

### 3. Document AI Processor Setup

```bash
# Create Document AI processor
gcloud documentai processors create \
    --location=us \
    --display-name="Financial Document Processor" \
    --type=FORM_PARSER_PROCESSOR

# List processors to get processor ID
gcloud documentai processors list --location=us
```

## Database Setup

### 1. PostgreSQL Installation and Setup

```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt update
sudo apt install postgresql postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql

# In PostgreSQL shell:
CREATE DATABASE uprez_nlp;
CREATE USER uprez_nlp_user WITH ENCRYPTED PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE uprez_nlp TO uprez_nlp_user;
ALTER USER uprez_nlp_user CREATEDB;
\q
```

### 2. Database Schema Creation

```sql
-- Create tables for NLP models
-- src/database/schema.sql

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(255) NOT NULL,
    content TEXT,
    document_type VARCHAR(50),
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS document_processing_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    processing_type VARCHAR(50),
    results JSONB,
    confidence_score DECIMAL(4,3),
    processing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS named_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    entity_text VARCHAR(500),
    entity_type VARCHAR(50),
    start_position INTEGER,
    end_position INTEGER,
    confidence_score DECIMAL(4,3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sentiment_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    text_segment TEXT,
    sentiment_class VARCHAR(20),
    sentiment_score DECIMAL(4,3),
    confidence_score DECIMAL(4,3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS topic_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    topic_distribution JSONB,
    dominant_topics INTEGER[],
    topic_coherence DECIMAL(4,3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_documents_type ON documents(document_type);
CREATE INDEX idx_documents_processed ON documents(processed);
CREATE INDEX idx_processing_results_document_id ON document_processing_results(document_id);
CREATE INDEX idx_processing_results_type ON document_processing_results(processing_type);
CREATE INDEX idx_entities_document_id ON named_entities(document_id);
CREATE INDEX idx_entities_type ON named_entities(entity_type);
CREATE INDEX idx_sentiment_document_id ON sentiment_analysis(document_id);
CREATE INDEX idx_topic_document_id ON topic_analysis(document_id);
```

### 3. Redis Setup

```bash
# Install Redis (Ubuntu/Debian)
sudo apt install redis-server

# Configure Redis
sudo nano /etc/redis/redis.conf

# Key settings to modify:
# maxmemory 2gb
# maxmemory-policy allkeys-lru
# save 900 1
# save 300 10
# save 60 10000

# Start Redis service
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test Redis connection
redis-cli ping
```

## Core Implementation

### 1. Base Configuration Class

```python
# src/config/config.py
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field

class Config(BaseSettings):
    """Main configuration class"""
    
    # Project settings
    project_name: str = "uprez-nlp"
    project_version: str = "1.0.0"
    debug: bool = False
    
    # Data paths
    data_dir: Path = Path("data")
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    models_dir: Path = data_dir / "models"
    cache_dir: Path = data_dir / "cache"
    
    # Database settings
    database_url: str = Field(
        default="postgresql://uprez_nlp_user:your_password@localhost:5432/uprez_nlp",
        env="DATABASE_URL"
    )
    
    # Redis settings
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Google Cloud settings
    gcp_project_id: str = Field(default="", env="GCP_PROJECT_ID")
    google_application_credentials: str = Field(default="", env="GOOGLE_APPLICATION_CREDENTIALS")
    
    # Model settings
    device: str = "cuda"
    batch_size: int = 16
    max_length: int = 512
    num_workers: int = 4
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "logs/uprez-nlp.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @classmethod
    def load_from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Flatten nested configuration
        flat_config = cls._flatten_dict(config_data)
        
        return cls(**flat_config)
    
    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(Config._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

# Create global config instance
config = Config()
```

### 2. Database Connection Manager

```python
# src/database/connection.py
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
import redis
from config.config import config

# SQLAlchemy setup
engine = create_engine(
    config.database_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=config.debug
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
metadata = MetaData()

# Redis connection
redis_client = redis.Redis(
    host=config.redis_host,
    port=config.redis_port,
    db=config.redis_db,
    password=config.redis_password,
    decode_responses=True
)

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Get database session with automatic cleanup"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def get_redis_client() -> redis.Redis:
    """Get Redis client"""
    return redis_client

# Database models
from sqlalchemy import Column, String, Text, Boolean, DateTime, JSON, Float, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    content = Column(Text)
    document_type = Column(String(50))
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    metadata = Column(JSON)

class ProcessingResult(Base):
    __tablename__ = "document_processing_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'))
    processing_type = Column(String(50))
    results = Column(JSON)
    confidence_score = Column(Float)
    processing_timestamp = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String(50))

class NamedEntity(Base):
    __tablename__ = "named_entities"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'))
    entity_text = Column(String(500))
    entity_type = Column(String(50))
    start_position = Column(Integer)
    end_position = Column(Integer)
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)
```

### 3. Logging Setup

```python
# src/utils/logging_config.py
import logging
import logging.handlers
from pathlib import Path
from config.config import config

def setup_logging():
    """Setup logging configuration"""
    
    # Create logs directory if it doesn't exist
    log_path = Path(config.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.handlers.RotatingFileHandler(
                config.log_file,
                maxBytes=10485760,  # 10MB
                backupCount=5
            ),
            logging.StreamHandler()
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('spacy').setLevel(logging.INFO)
    
    return logging.getLogger(__name__)

logger = setup_logging()
```

## Model Implementation

### 1. Base Model Interface

```python
# src/models/base_model.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import torch
import numpy as np
from pathlib import Path

class BaseNLPModel(ABC):
    """Base class for all NLP models"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device(config.get('device', 'cpu'))
        
    @abstractmethod
    def load_model(self, model_path: str):
        """Load pre-trained model"""
        pass
    
    @abstractmethod
    def train(self, train_data: List[Dict], val_data: List[Dict]) -> Dict:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, text: str) -> Dict[str, Any]:
        """Make predictions on text"""
        pass
    
    @abstractmethod
    def evaluate(self, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate model performance"""
        pass
    
    def save_model(self, save_path: str):
        """Save model to disk"""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_dir)
        else:
            torch.save(self.model.state_dict(), save_dir / 'model.pt')
        
        if hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(save_dir)
    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess text for model input"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.config.get('max_length', 512),
            return_tensors='pt'
        )
        
        return {k: v.to(self.device) for k, v in encoding.items()}
```

### 2. Model Factory

```python
# src/models/model_factory.py
from typing import Dict, Any
from .base_model import BaseNLPModel
from .sentiment_model import FinancialSentimentModel
from .ner_model import FinancialNERModel
from .document_classifier import DocumentClassifier
from .topic_model import NeuralTopicModel

class ModelFactory:
    """Factory for creating NLP models"""
    
    _models = {
        'sentiment': FinancialSentimentModel,
        'ner': FinancialNERModel,
        'document_classifier': DocumentClassifier,
        'topic_model': NeuralTopicModel
    }
    
    @classmethod
    def create_model(cls, model_type: str, config: Dict[str, Any]) -> BaseNLPModel:
        """Create model instance"""
        
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls._models[model_type]
        return model_class(config)
    
    @classmethod
    def register_model(cls, model_type: str, model_class: type):
        """Register new model type"""
        cls._models[model_type] = model_class
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List available model types"""
        return list(cls._models.keys())
```

## Testing Setup

### 1. Test Configuration

```python
# tests/conftest.py
import pytest
import tempfile
import shutil
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.connection import Base
from src.config.config import Config

@pytest.fixture(scope="session")
def test_config():
    """Create test configuration"""
    return Config(
        database_url="sqlite:///test.db",
        debug=True,
        data_dir=Path("test_data"),
        device="cpu"
    )

@pytest.fixture(scope="session")
def test_db(test_config):
    """Create test database"""
    engine = create_engine(test_config.database_url)
    Base.metadata.create_all(engine)
    
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    yield TestSessionLocal
    
    Base.metadata.drop_all(engine)

@pytest.fixture(scope="function")
def test_session(test_db):
    """Create test session"""
    session = test_db()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture
def sample_financial_documents():
    """Sample financial documents for testing"""
    return [
        {
            "text": "BHP Group Limited reported revenue of $65.1 billion for FY2023, representing a 12% increase from the previous year.",
            "document_type": "annual_report",
            "expected_entities": [
                {"text": "BHP Group Limited", "label": "COMPANY"},
                {"text": "$65.1 billion", "label": "FINANCIAL_AMOUNT"},
                {"text": "FY2023", "label": "DATE"}
            ]
        },
        {
            "text": "The company's net profit margin improved to 15.2%, driven by operational efficiency gains.",
            "document_type": "financial_statement", 
            "expected_sentiment": "positive"
        }
    ]
```

### 2. Unit Tests

```python
# tests/test_models.py
import pytest
from src.models.model_factory import ModelFactory
from src.models.sentiment_model import FinancialSentimentModel

class TestModelFactory:
    
    def test_create_sentiment_model(self, test_config):
        """Test sentiment model creation"""
        model = ModelFactory.create_model('sentiment', test_config.dict())
        assert isinstance(model, FinancialSentimentModel)
    
    def test_invalid_model_type(self, test_config):
        """Test invalid model type raises error"""
        with pytest.raises(ValueError):
            ModelFactory.create_model('invalid_model', test_config.dict())

class TestSentimentModel:
    
    @pytest.fixture
    def sentiment_model(self, test_config):
        return FinancialSentimentModel(test_config.dict())
    
    def test_preprocess_text(self, sentiment_model):
        """Test text preprocessing"""
        text = "This is a test sentence."
        processed = sentiment_model.preprocess_text(text)
        
        assert 'input_ids' in processed
        assert 'attention_mask' in processed
        assert processed['input_ids'].shape[1] <= 512  # max_length
```

## Deployment Setup

### 1. Docker Configuration

#### `Dockerfile`

```dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy models
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_trf

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/{raw,processed,models,cache} logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### `docker-compose.yml`

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://uprez_nlp_user:password@db:5432/uprez_nlp
      - REDIS_HOST=redis
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./credentials.json:/app/credentials.json
    depends_on:
      - db
      - redis
    networks:
      - uprez-nlp-network

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=uprez_nlp
      - POSTGRES_USER=uprez_nlp_user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - uprez-nlp-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - uprez-nlp-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - app
    networks:
      - uprez-nlp-network

volumes:
  postgres_data:
  redis_data:

networks:
  uprez-nlp-network:
    driver: bridge
```

### 2. API Server

```python
# src/api/main.py
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import time
from src.models.model_factory import ModelFactory
from src.database.connection import get_db_session
from src.utils.logging_config import logger
from src.config.config import config

app = FastAPI(
    title="Uprez NLP API",
    description="NLP models for Australian financial document processing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
models = {}

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    try:
        # Load pre-trained models
        models['sentiment'] = ModelFactory.create_model('sentiment', config.dict())
        models['ner'] = ModelFactory.create_model('ner', config.dict())
        models['document_classifier'] = ModelFactory.create_model('document_classifier', config.dict())
        
        # Load model weights
        for model_name, model in models.items():
            try:
                model_path = config.models_dir / model_name
                if model_path.exists():
                    model.load_model(str(model_path))
                    logger.info(f"Loaded {model_name} model")
                else:
                    logger.warning(f"No saved model found for {model_name}")
            except Exception as e:
                logger.error(f"Failed to load {model_name} model: {e}")
        
        logger.info("API startup completed")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise

class TextAnalysisRequest(BaseModel):
    text: str
    models: List[str] = ["sentiment", "ner"]
    include_confidence: bool = True

class TextAnalysisResponse(BaseModel):
    text: str
    results: Dict
    processing_time: float

@app.post("/analyze", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """Analyze text using specified models"""
    start_time = time.time()
    
    try:
        results = {}
        
        for model_name in request.models:
            if model_name not in models:
                results[model_name] = {"error": f"Model {model_name} not available"}
                continue
            
            try:
                result = models[model_name].predict(request.text)
                results[model_name] = result
            except Exception as e:
                results[model_name] = {"error": str(e)}
        
        processing_time = time.time() - start_time
        
        return TextAnalysisResponse(
            text=request.text,
            results=results,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document"""
    try:
        # Read file content
        content = await file.read()
        
        # Save to database
        with get_db_session() as session:
            # Implementation for document upload and processing
            pass
        
        return {"message": "Document uploaded successfully", "filename": file.filename}
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": list(models.keys()),
        "model_status": {name: "loaded" if model else "not_loaded" 
                        for name, model in models.items()}
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.api_host,
        port=config.api_port,
        workers=config.api_workers,
        reload=config.debug
    )
```

## Usage Examples

### 1. Basic Usage

```python
# examples/basic_usage.py
from src.models.model_factory import ModelFactory
from src.config.config import config

# Initialize sentiment model
sentiment_model = ModelFactory.create_model('sentiment', config.dict())

# Analyze text
text = "BHP reported strong quarterly results with record iron ore production."
result = sentiment_model.predict(text)

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### 2. Batch Processing

```python
# examples/batch_processing.py
import asyncio
from src.api.main import app
from httpx import AsyncClient

async def batch_analyze_documents(documents: List[str]):
    """Batch analyze multiple documents"""
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        tasks = []
        
        for doc in documents:
            task = client.post("/analyze", json={
                "text": doc,
                "models": ["sentiment", "ner", "document_classifier"]
            })
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        return [response.json() for response in responses]

# Usage
documents = [
    "Company A reported strong earnings...",
    "Regulatory changes may impact sector...",
    "Market volatility continues to affect..."
]

results = asyncio.run(batch_analyze_documents(documents))
```

## Performance Optimization

### 1. Model Optimization

```python
# src/utils/optimization.py
import torch
from torch.utils.data import DataLoader
from typing import List, Dict

class ModelOptimizer:
    """Utilities for model optimization"""
    
    @staticmethod
    def optimize_model_for_inference(model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for inference"""
        
        # Set to evaluation mode
        model.eval()
        
        # Convert to TorchScript if possible
        try:
            scripted_model = torch.jit.script(model)
            return scripted_model
        except Exception:
            # Fall back to original model
            return model
    
    @staticmethod
    def apply_quantization(model: torch.nn.Module) -> torch.nn.Module:
        """Apply dynamic quantization"""
        
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        
        return quantized_model
    
    @staticmethod
    def batch_predict(model, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """Batch prediction for efficiency"""
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Process batch
            batch_results = []
            for text in batch_texts:
                result = model.predict(text)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
```

### 2. Caching Strategy

```python
# src/utils/cache.py
import json
import hashlib
from typing import Any, Optional
from src.database.connection import get_redis_client

class ModelCache:
    """Caching for model predictions"""
    
    def __init__(self, ttl: int = 3600):  # 1 hour default TTL
        self.redis_client = get_redis_client()
        self.ttl = ttl
    
    def get_cache_key(self, model_name: str, text: str, **kwargs) -> str:
        """Generate cache key"""
        
        # Create hash of input
        input_data = {
            'model': model_name,
            'text': text,
            **kwargs
        }
        
        input_str = json.dumps(input_data, sort_keys=True)
        cache_key = hashlib.md5(input_str.encode()).hexdigest()
        
        return f"nlp_cache:{model_name}:{cache_key}"
    
    def get(self, model_name: str, text: str, **kwargs) -> Optional[Any]:
        """Get cached result"""
        
        cache_key = self.get_cache_key(model_name, text, **kwargs)
        
        try:
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
        except Exception:
            pass
        
        return None
    
    def set(self, model_name: str, text: str, result: Any, **kwargs):
        """Cache result"""
        
        cache_key = self.get_cache_key(model_name, text, **kwargs)
        
        try:
            self.redis_client.setex(
                cache_key,
                self.ttl,
                json.dumps(result, default=str)
            )
        except Exception:
            pass  # Fail silently for cache errors
```

This comprehensive setup guide provides all the necessary steps and code to implement the complete NLP pipeline for Australian financial document processing in the Uprez valuation system.