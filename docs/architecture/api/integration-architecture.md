# API Integration Architecture & Data Flow

## Executive Summary

The IPO Valuation SaaS platform requires seamless integration with multiple external data sources and internal systems. This document outlines the API architecture, data flow patterns, and integration strategies that enable real-time valuation processing while maintaining security and reliability.

## Integration Architecture Overview

### High-Level Integration Map

```
┌─────────────────────────────────────────────────────────────────┐
│                    External Data Sources                       │
├─────────────────┬─────────────────┬─────────────────┬───────────┤
│   ASX Market    │   ASIC Company  │   Financial     │    RBA    │
│   Data Feed     │   Registry      │   Data Providers│  Economic │
│                 │                 │   (S&P, Reuters)│   Data    │
└─────────────────┴─────────────────┴─────────────────┴───────────┘
         │                   │                 │             │
         ▼                   ▼                 ▼             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     API Gateway Layer                          │
│  ┌───────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │   Rate        │  │  Security    │  │   Data             │  │
│  │   Limiting    │  │  & Auth      │  │   Transformation   │  │
│  └───────────────┘  └──────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                Core Valuation Platform                         │
│  ┌───────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │   Document    │  │  Valuation   │  │   Report            │  │
│  │   Processing  │  │  Engine      │  │   Generation        │  │
│  └───────────────┘  └──────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Client Applications                         │
│  ┌───────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │   Web App     │  │  Mobile App  │  │   Partner APIs      │  │
│  │   Frontend    │  │              │  │   (White-label)     │  │
│  └───────────────┘  └──────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## External API Integrations

### 1. ASX Market Data Integration

#### Real-time Market Data API
```python
class ASXMarketDataAPI:
    """Integration with ASX market data feeds"""
    
    def __init__(self):
        self.base_url = "https://api.asx.com.au/v1"
        self.api_key = get_secure_credential("ASX_API_KEY")
        self.rate_limiter = RateLimiter(requests_per_minute=1000)
        
    async def get_company_market_data(self, ticker: str) -> MarketData:
        """Fetch current market data for a specific ASX ticker"""
        endpoint = f"/companies/{ticker}/market-data"
        
        async with self.rate_limiter:
            response = await self.authenticated_request(
                method="GET",
                endpoint=endpoint,
                timeout=30
            )
            
        return MarketData.from_api_response(response)
    
    async def get_peer_group_data(self, tickers: List[str]) -> List[MarketData]:
        """Batch fetch market data for peer group analysis"""
        tasks = [self.get_company_market_data(ticker) for ticker in tickers]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def get_historical_data(self, ticker: str, period: str = "1Y") -> HistoricalData:
        """Fetch historical price and volume data"""
        endpoint = f"/companies/{ticker}/historical"
        params = {"period": period, "frequency": "daily"}
        
        response = await self.authenticated_request(
            method="GET",
            endpoint=endpoint,
            params=params
        )
        
        return HistoricalData.from_api_response(response)
```

#### Data Points Retrieved:
- **Current Market Data**: Market cap, share price, trading volume
- **Financial Metrics**: P/E ratio, EV/EBITDA, dividend yield
- **Trading Patterns**: 52-week high/low, average volume
- **Announcements**: Recent ASX announcements and price-sensitive information

### 2. ASIC Company Registry Integration

#### Company Registry API
```python
class ASICRegistryAPI:
    """Integration with ASIC company registry"""
    
    def __init__(self):
        self.base_url = "https://api.asic.gov.au/v2"
        self.client_credentials = get_oauth_credentials("ASIC")
        
    async def get_company_details(self, acn: str) -> CompanyDetails:
        """Fetch company registration and structural details"""
        token = await self.get_access_token()
        
        endpoint = f"/organisations/{acn}"
        headers = {"Authorization": f"Bearer {token}"}
        
        response = await self.authenticated_request(
            method="GET",
            endpoint=endpoint,
            headers=headers
        )
        
        return CompanyDetails.from_api_response(response)
    
    async def get_financial_filings(self, acn: str, years: int = 3) -> List[Filing]:
        """Retrieve recent financial filings and annual returns"""
        endpoint = f"/organisations/{acn}/filings"
        params = {"type": "financial", "years": years}
        
        response = await self.authenticated_request(
            method="GET",
            endpoint=endpoint,
            params=params
        )
        
        return [Filing.from_api_response(filing) for filing in response.get("filings", [])]
```

#### Data Points Retrieved:
- **Company Structure**: Directors, shareholders, registered office
- **Financial Filings**: Annual returns, financial statements
- **Compliance Status**: ASIC compliance history and current status
- **Corporate Actions**: Recent changes in structure or directors

### 3. Financial Data Providers Integration

#### S&P Capital IQ Integration
```python
class CapitalIQAPI:
    """Integration with S&P Capital IQ for comprehensive financial data"""
    
    def __init__(self):
        self.base_url = "https://api.capitaliq.com/v3"
        self.credentials = get_api_credentials("CAPITAL_IQ")
        
    async def get_peer_group_analysis(self, 
                                     industry_code: str, 
                                     market_cap_range: tuple) -> PeerGroupData:
        """Find and analyze peer companies based on industry and size"""
        
        search_criteria = {
            "industry_classification": industry_code,
            "market_cap_min": market_cap_range[0],
            "market_cap_max": market_cap_range[1],
            "exchange": ["ASX", "NZX"],  # Expand to regional exchanges
            "status": "active"
        }
        
        companies = await self.search_companies(search_criteria)
        
        # Fetch detailed financial data for each peer
        peer_data = []
        for company in companies:
            financial_data = await self.get_financial_metrics(company.ticker)
            peer_data.append(financial_data)
            
        return PeerGroupData(peers=peer_data)
    
    async def get_transaction_multiples(self, sector: str, timeframe: int = 24) -> List[Transaction]:
        """Retrieve M&A transaction data for precedent analysis"""
        endpoint = "/transactions/search"
        
        criteria = {
            "sector": sector,
            "timeframe_months": timeframe,
            "deal_status": "completed",
            "target_geography": ["Australia", "New Zealand"]
        }
        
        response = await self.authenticated_request(
            method="POST",
            endpoint=endpoint,
            json=criteria
        )
        
        return [Transaction.from_api_response(tx) for tx in response.get("transactions", [])]
```

### 4. Reserve Bank of Australia (RBA) Economic Data

#### RBA Statistics API
```python
class RBAStatisticsAPI:
    """Integration with RBA for macroeconomic indicators"""
    
    def __init__(self):
        self.base_url = "https://rba.gov.au/statistics/tables/json"
        # RBA API is public, no authentication required
        
    async def get_interest_rates(self) -> InterestRates:
        """Fetch current and historical interest rates"""
        endpoint = "/f1-1.json"  # Official Cash Rate
        
        response = await self.make_request(endpoint)
        
        return InterestRates.from_rba_response(response)
    
    async def get_economic_indicators(self) -> EconomicIndicators:
        """Fetch key economic indicators for valuation context"""
        indicators = {
            "gdp_growth": "/g1-1.json",
            "inflation_rate": "/g2-1.json",
            "unemployment": "/g5-1.json",
            "asx_200_index": "/f7-1.json"
        }
        
        tasks = [self.make_request(endpoint) for endpoint in indicators.values()]
        responses = await asyncio.gather(*tasks)
        
        return EconomicIndicators.from_multiple_responses(
            dict(zip(indicators.keys(), responses))
        )
```

## Internal API Architecture

### 1. Core Platform APIs

#### Valuation API
```python
from fastapi import FastAPI, BackgroundTasks, Depends
from typing import List, Optional

app = FastAPI(title="Valuation Engine API", version="1.0.0")

@app.post("/api/v1/valuations", response_model=ValuationResponse)
async def create_valuation(
    request: ValuationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Create a new valuation request"""
    
    # Validate input data
    validated_request = await validate_valuation_request(request)
    
    # Create valuation job
    valuation_job = ValuationJob(
        user_id=current_user.id,
        request_data=validated_request,
        status=JobStatus.QUEUED
    )
    
    # Save to database
    job_id = await save_valuation_job(valuation_job)
    
    # Queue for background processing
    background_tasks.add_task(process_valuation, job_id)
    
    return ValuationResponse(
        job_id=job_id,
        status="queued",
        estimated_completion_time="8-15 minutes",
        polling_url=f"/api/v1/valuations/{job_id}/status"
    )

@app.get("/api/v1/valuations/{job_id}/status", response_model=ValuationStatus)
async def get_valuation_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get status of a valuation job"""
    
    job = await get_valuation_job(job_id, current_user.id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Valuation job not found")
    
    return ValuationStatus(
        job_id=job_id,
        status=job.status,
        progress_percentage=job.progress,
        current_stage=job.current_stage,
        estimated_completion=job.estimated_completion,
        result_url=f"/api/v1/valuations/{job_id}/result" if job.status == "completed" else None
    )

@app.get("/api/v1/valuations/{job_id}/result", response_model=ValuationResult)
async def get_valuation_result(
    job_id: str,
    format: Optional[str] = "json",
    current_user: User = Depends(get_current_user)
):
    """Retrieve completed valuation results"""
    
    result = await get_valuation_result(job_id, current_user.id)
    
    if format == "pdf":
        # Generate and return PDF report
        pdf_content = await generate_pdf_report(result)
        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=valuation_{job_id}.pdf"}
        )
    
    return result
```

### 2. Document Processing API

#### Document Upload & Processing
```python
@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile,
    document_type: DocumentType,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Upload and process a document for valuation analysis"""
    
    # Validate file type and size
    await validate_uploaded_file(file, document_type)
    
    # Generate secure upload URL
    upload_url = await generate_secure_upload_url(file.filename)
    
    # Upload to cloud storage
    file_url = await upload_to_cloud_storage(file, upload_url)
    
    # Create document processing job
    doc_job = DocumentProcessingJob(
        user_id=current_user.id,
        file_url=file_url,
        document_type=document_type,
        filename=file.filename
    )
    
    job_id = await save_document_job(doc_job)
    
    # Queue for background processing
    background_tasks.add_task(process_document, job_id)
    
    return DocumentUploadResponse(
        job_id=job_id,
        status="processing",
        file_url=file_url,
        estimated_completion_time="2-5 minutes"
    )

@app.get("/api/v1/documents/{job_id}/extraction", response_model=ExtractionResult)
async def get_document_extraction(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Retrieve extracted data from processed document"""
    
    extraction = await get_document_extraction(job_id, current_user.id)
    
    return ExtractionResult(
        job_id=job_id,
        document_type=extraction.document_type,
        extracted_data=extraction.parsed_data,
        confidence_score=extraction.confidence,
        validation_errors=extraction.validation_errors
    )
```

## Data Flow Architecture

### 1. Valuation Processing Flow

```
User Request → Document Upload → OCR/NLP Processing → Data Extraction
     ↓              ↓                  ↓                   ↓
  Validation → Cloud Storage → Background Queue → Structured Data
     ↓              ↓                  ↓                   ↓
External APIs ← Market Data Fetch ← Peer Analysis ← Company Analysis
     ↓              ↓                  ↓                   ↓
 RBA/ASIC ← Economic Context ← Risk Assessment ← Growth Analysis
     ↓              ↓                  ↓                   ↓
Valuation Engine ← Synthesis ← Target Multiple ← Price Range
     ↓              ↓                  ↓                   ↓
Report Generation → PDF Creation → User Notification → Dashboard Update
```

### 2. Real-time Data Synchronization

#### Market Data Refresh Strategy
```python
class MarketDataSyncService:
    """Manages real-time synchronization of market data"""
    
    def __init__(self):
        self.refresh_intervals = {
            "market_data": 300,  # 5 minutes
            "economic_indicators": 3600,  # 1 hour
            "company_filings": 86400,  # 24 hours
            "peer_analysis": 1800  # 30 minutes
        }
    
    async def start_sync_scheduler(self):
        """Start background tasks for data synchronization"""
        
        # Schedule different refresh rates for different data types
        asyncio.create_task(self.sync_market_data())
        asyncio.create_task(self.sync_economic_indicators())
        asyncio.create_task(self.sync_company_filings())
        
    async def sync_market_data(self):
        """Continuously refresh market data for active valuations"""
        
        while True:
            try:
                # Get list of companies being actively analyzed
                active_tickers = await get_active_analysis_tickers()
                
                # Batch fetch updated market data
                updated_data = await self.asx_api.get_peer_group_data(active_tickers)
                
                # Update cache and notify active valuations
                await self.update_market_data_cache(updated_data)
                await self.notify_active_valuations(updated_data)
                
                # Wait for next refresh cycle
                await asyncio.sleep(self.refresh_intervals["market_data"])
                
            except Exception as e:
                logger.error(f"Market data sync error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute on error
```

## API Security & Authentication

### 1. Authentication Architecture

#### JWT-based Authentication
```python
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import JWTAuthentication

# JWT Authentication configuration
jwt_auth = JWTAuthentication(
    secret_key=get_secret_key(),
    lifetime_seconds=3600,  # 1 hour access token
    tokenUrl="auth/jwt/login"
)

# User manager with role-based access
class UserManager(BaseUserManager[UserCreate, UserUpdate, UserDB]):
    """Custom user manager with role-based permissions"""
    
    async def on_after_register(self, user: UserDB, request: Optional[Request] = None):
        """Post-registration actions"""
        
        # Assign default role and permissions
        await assign_default_role(user.id)
        
        # Send welcome email with onboarding information
        await send_welcome_email(user.email)
        
        # Create initial user workspace
        await create_user_workspace(user.id)

# Role-based permissions
class RolePermissions:
    """Define permissions for different user roles"""
    
    ROLES = {
        "basic_user": {
            "valuations_per_month": 2,
            "api_rate_limit": 100,
            "features": ["basic_reports", "scenario_modeling"]
        },
        "professional_user": {
            "valuations_per_month": 8,
            "api_rate_limit": 500,
            "features": ["advanced_analytics", "white_label", "api_access"]
        },
        "enterprise_user": {
            "valuations_per_month": -1,  # Unlimited
            "api_rate_limit": 2000,
            "features": ["all_features", "custom_integrations", "sla_support"]
        }
    }
```

### 2. API Rate Limiting & Throttling

#### Intelligent Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

# Create rate limiter with Redis backend
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=get_redis_url(),
    strategy="moving-window"
)

# User-specific rate limiting
class UserRateLimiter:
    """Rate limiting based on user subscription tier"""
    
    def __init__(self):
        self.tier_limits = {
            "basic": "100/hour",
            "professional": "500/hour", 
            "enterprise": "2000/hour"
        }
    
    def get_user_limit(self, user: User) -> str:
        """Get rate limit based on user's subscription tier"""
        return self.tier_limits.get(user.subscription_tier, "100/hour")
    
    async def check_rate_limit(self, user: User, request: Request) -> bool:
        """Check if user has exceeded their rate limit"""
        
        limit = self.get_user_limit(user)
        key = f"rate_limit:{user.id}"
        
        # Use Redis to track usage
        current_usage = await self.redis.get(key)
        limit_number = int(limit.split('/')[0])
        
        if current_usage and int(current_usage) >= limit_number:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Your tier allows {limit} requests."
            )
        
        # Increment usage counter
        await self.redis.incr(key)
        await self.redis.expire(key, 3600)  # 1 hour expiry
        
        return True

# Apply rate limiting to endpoints
@app.post("/api/v1/valuations")
@limiter.limit("10/minute")  # Global limit
async def create_valuation(request: Request, ...):
    # Additional user-specific rate limiting
    await user_rate_limiter.check_rate_limit(current_user, request)
    # ... rest of endpoint logic
```

## Third-party Integrations

### 1. Accounting Software Integration

#### Xero Integration
```python
class XeroIntegration:
    """Integration with Xero accounting software"""
    
    def __init__(self):
        self.oauth_config = get_xero_oauth_config()
        
    async def connect_user_account(self, user_id: str, auth_code: str) -> XeroConnection:
        """Establish connection to user's Xero account"""
        
        # Exchange auth code for access token
        token_response = await self.exchange_auth_code(auth_code)
        
        # Store encrypted credentials
        connection = XeroConnection(
            user_id=user_id,
            access_token=encrypt_token(token_response.access_token),
            refresh_token=encrypt_token(token_response.refresh_token),
            expires_at=token_response.expires_at
        )
        
        await save_xero_connection(connection)
        return connection
    
    async def import_financial_data(self, user_id: str) -> FinancialStatements:
        """Import financial statements from connected Xero account"""
        
        connection = await get_xero_connection(user_id)
        
        if not connection or connection.is_expired():
            raise HTTPException(status_code=401, detail="Xero connection expired")
        
        # Fetch financial data
        xero_client = XeroClient(connection.access_token)
        
        balance_sheet = await xero_client.get_balance_sheet()
        profit_loss = await xero_client.get_profit_loss()
        cash_flow = await xero_client.get_cash_flow()
        
        return FinancialStatements(
            balance_sheet=balance_sheet,
            profit_loss=profit_loss,
            cash_flow=cash_flow,
            source="xero",
            import_date=datetime.utcnow()
        )
```

### 2. CRM Integration

#### Salesforce Integration for Channel Partners
```python
class SalesforceIntegration:
    """Integration with Salesforce for partner channel management"""
    
    def __init__(self):
        self.sf_client = Salesforce(
            username=get_env("SALESFORCE_USERNAME"),
            password=get_env("SALESFORCE_PASSWORD"),
            security_token=get_env("SALESFORCE_TOKEN"),
            sandbox=get_env("SALESFORCE_SANDBOX", False)
        )
    
    async def sync_partner_opportunities(self) -> List[Opportunity]:
        """Sync opportunities from partner Salesforce instances"""
        
        # Query opportunities related to IPO valuations
        soql_query = """
            SELECT Id, Name, AccountId, Amount, StageName, CloseDate, 
                   Partner_Type__c, Expected_Valuation_Date__c
            FROM Opportunity 
            WHERE Product_Type__c = 'IPO Valuation' 
            AND StageName IN ('Qualified', 'Proposal', 'Negotiation')
            AND LastModifiedDate >= TODAY
        """
        
        opportunities = self.sf_client.query(soql_query)
        
        # Transform and return opportunities
        return [
            Opportunity.from_salesforce_record(record)
            for record in opportunities['records']
        ]
    
    async def create_valuation_opportunity(self, 
                                         partner_id: str, 
                                         valuation_request: ValuationRequest) -> str:
        """Create opportunity in Salesforce when valuation is requested"""
        
        opportunity_data = {
            'Name': f"IPO Valuation - {valuation_request.company_name}",
            'AccountId': partner_id,
            'Amount': self.calculate_opportunity_value(valuation_request),
            'StageName': 'Qualified',
            'CloseDate': (datetime.now() + timedelta(days=30)).date(),
            'Product_Type__c': 'IPO Valuation',
            'Expected_Valuation_Date__c': valuation_request.expected_ipo_date
        }
        
        result = self.sf_client.Opportunity.create(opportunity_data)
        return result['id']
```

## Error Handling & Resilience

### 1. Circuit Breaker Pattern

#### External API Circuit Breakers
```python
from circuitbreaker import circuit

class ExternalAPIClient:
    """Base class for external API clients with circuit breaker pattern"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.circuit_breaker = circuit(
            failure_threshold=5,    # Open after 5 failures
            recovery_timeout=60,    # Try to recover after 60 seconds
            expected_exception=requests.RequestException
        )
    
    @circuit_breaker
    async def make_api_request(self, *args, **kwargs):
        """Make API request with circuit breaker protection"""
        
        try:
            response = await self._execute_request(*args, **kwargs)
            
            # Reset circuit breaker on successful request
            self.circuit_breaker.reset()
            
            return response
            
        except requests.RequestException as e:
            logger.error(f"{self.service_name} API error: {e}")
            
            # Circuit breaker will automatically open if threshold is reached
            raise ExternalServiceError(
                service=self.service_name,
                error=str(e),
                retry_after=60
            )
    
    async def _execute_request(self, *args, **kwargs):
        """Override in subclasses to implement actual API logic"""
        raise NotImplementedError
```

### 2. Retry Logic & Exponential Backoff

#### Resilient API Calls
```python
import asyncio
from typing import Callable, Any
import random

class RetryHandler:
    """Handles retry logic with exponential backoff and jitter"""
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
    
    async def retry_async(self, 
                         func: Callable, 
                         *args, 
                         **kwargs) -> Any:
        """Execute function with retry logic"""
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    # Max retries reached, raise the last exception
                    raise e
                
                # Calculate delay with exponential backoff
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                
                # Add jitter to prevent thundering herd
                if self.jitter:
                    delay *= (0.5 + random.random() * 0.5)
                
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                
                await asyncio.sleep(delay)
        
        # This should never be reached, but just in case
        raise last_exception
```

## Performance Optimization

### 1. Caching Strategy

#### Multi-layer Caching
```python
from functools import wraps
import pickle
import hashlib

class CacheManager:
    """Multi-layer cache management for API responses"""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.memory_cache = {}
        
        # Cache TTLs for different data types
        self.cache_ttls = {
            "market_data": 300,      # 5 minutes
            "company_data": 3600,    # 1 hour
            "economic_data": 7200,   # 2 hours
            "peer_analysis": 1800,   # 30 minutes
            "transaction_data": 86400 # 24 hours
        }
    
    def cache_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate cache key from function name and parameters"""
        
        # Create deterministic hash from function parameters
        params_str = f"{args}{sorted(kwargs.items())}"
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        
        return f"{func_name}:{params_hash}"
    
    async def get_cached_result(self, key: str, data_type: str):
        """Retrieve cached result with fallback to memory cache"""
        
        # Try memory cache first (fastest)
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Try Redis cache
        cached_data = await self.redis_client.get(key)
        if cached_data:
            result = pickle.loads(cached_data)
            
            # Store in memory cache for even faster access
            self.memory_cache[key] = result
            
            return result
        
        return None
    
    async def set_cached_result(self, key: str, data: Any, data_type: str):
        """Store result in both memory and Redis cache"""
        
        # Store in memory cache
        self.memory_cache[key] = data
        
        # Store in Redis with appropriate TTL
        ttl = self.cache_ttls.get(data_type, 3600)
        serialized_data = pickle.dumps(data)
        
        await self.redis_client.setex(key, ttl, serialized_data)
    
    def cached_api_call(self, data_type: str = "default"):
        """Decorator for caching API call results"""
        
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                
                # Generate cache key
                cache_key = self.cache_key(func.__name__, *args, **kwargs)
                
                # Try to get cached result
                cached_result = await self.get_cached_result(cache_key, data_type)
                if cached_result is not None:
                    logger.info(f"Cache hit for {func.__name__}")
                    return cached_result
                
                # Execute function and cache result
                logger.info(f"Cache miss for {func.__name__}, executing API call")
                result = await func(*args, **kwargs)
                
                await self.set_cached_result(cache_key, result, data_type)
                
                return result
                
            return wrapper
        return decorator

# Usage example
cache_manager = CacheManager()

class ASXMarketDataAPI:
    @cache_manager.cached_api_call(data_type="market_data")
    async def get_company_market_data(self, ticker: str) -> MarketData:
        # API call implementation
        pass
```

### 2. Database Optimization

#### Connection Pooling and Query Optimization
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.pool import QueuePool

class DatabaseManager:
    """Optimized database connection management"""
    
    def __init__(self):
        self.engine = create_async_engine(
            get_database_url(),
            poolclass=QueuePool,
            pool_size=20,           # Number of connections to maintain
            max_overflow=30,        # Additional connections when pool is full
            pool_recycle=3600,      # Recycle connections after 1 hour
            pool_pre_ping=True,     # Validate connections before use
            echo=False              # Set to True for SQL debugging
        )
    
    async def execute_optimized_query(self, query: str, params: dict = None) -> Any:
        """Execute database query with optimization"""
        
        async with AsyncSession(self.engine) as session:
            # Use query execution plan caching
            result = await session.execute(
                text(query).execution_options(
                    compiled_cache={},  # Enable query plan caching
                    autocommit=False
                ),
                params or {}
            )
            
            await session.commit()
            return result
    
    async def batch_insert(self, table: str, records: List[dict]) -> None:
        """Optimized batch insert for large datasets"""
        
        async with AsyncSession(self.engine) as session:
            # Use bulk insert for better performance
            await session.execute(
                text(f"""
                    INSERT INTO {table} ({','.join(records[0].keys())})
                    VALUES ({','.join([f':{key}' for key in records[0].keys()])})
                """),
                records
            )
            
            await session.commit()
```

This comprehensive API integration architecture provides a robust foundation for the IPO Valuation SaaS platform, ensuring reliable data flow, security, and performance while maintaining scalability for future growth.