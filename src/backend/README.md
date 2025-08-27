# IPO Valuation Platform API

A comprehensive FastAPI-based API for Australian IPO valuation and analysis.

## Features

### Core API Capabilities
- **Authentication & Authorization**: JWT-based auth with OAuth2 support (Google, Microsoft)
- **User Management**: Complete user profiles and role-based access control
- **Company Data**: Comprehensive company information with ASX integration
- **Financial Data**: Multi-period financial statements with automatic ratio calculations
- **Valuation Models**: Multiple valuation methodologies (DCF, Comparables, etc.)
- **Document Processing**: Upload and extract data from financial documents
- **Real-time Collaboration**: WebSocket-based project collaboration
- **External Integrations**: ASX, ASIC, and RBA data synchronization

### Advanced Features
- **GraphQL Endpoint**: Complex queries and data relationships
- **Rate Limiting**: Redis-backed rate limiting with customizable limits
- **Security**: Comprehensive security headers and input validation
- **Monitoring**: Prometheus metrics and structured logging
- **Caching**: Multi-level caching for performance optimization
- **Background Tasks**: Celery-based async processing
- **API Versioning**: Backward compatible versioning strategy

## Quick Start

### Docker Setup (Recommended)
```bash
# Start all services
docker-compose up -d

# Initialize database
docker-compose exec api alembic upgrade head

# Access API documentation
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env

# Initialize database
alembic upgrade head

# Run development server
python -m uvicorn api.main:app --reload
```

## API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Core Endpoints

### Authentication
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/refresh` - Token refresh

### Companies
- `GET /api/v1/companies` - List companies
- `POST /api/v1/companies` - Create company
- `GET /api/v1/companies/{id}` - Get company details

### Valuations
- `POST /api/v1/valuations/calculate` - Calculate valuation
- `GET /api/v1/valuations/results` - List results
- `GET /api/v1/valuations/models` - List models

### Real-time Features
- `WebSocket /ws/collaboration/{project_id}` - Collaboration
- `WebSocket /ws/notifications/{user_id}` - Notifications

## Architecture

The API is built with:
- **FastAPI** - Modern Python web framework
- **PostgreSQL** - Primary database
- **Redis** - Caching and rate limiting
- **SQLAlchemy** - ORM with async support
- **Alembic** - Database migrations
- **Celery** - Background task processing
- **Prometheus** - Metrics and monitoring