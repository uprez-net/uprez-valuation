"""
Test data factories for generating consistent test data across the test suite.

These factories use the Factory Boy pattern to create model instances with
realistic data for testing purposes.
"""

import random
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
import factory
from factory import Faker, SubFactory, LazyAttribute, LazyFunction
from faker.providers import BaseProvider
import uuid

# Import models (these would be actual model imports in real implementation)
# from src.backend.database.models import User, Company, Valuation, Document, MarketData


class AustralianBusinessProvider(BaseProvider):
    """Custom Faker provider for Australian business data."""
    
    def abn(self) -> str:
        """Generate a valid Australian Business Number (ABN)."""
        # Simplified ABN generation for testing
        digits = [random.randint(0, 9) for _ in range(11)]
        return ''.join(map(str, digits))
    
    def asx_ticker(self) -> str:
        """Generate a realistic ASX ticker symbol."""
        tickers = ['CBA', 'WBC', 'ANZ', 'NAB', 'BHP', 'CSL', 'WES', 'WOW', 
                  'MQG', 'TCL', 'RIO', 'QAN', 'TLS', 'STO', 'ORG']
        return random.choice(tickers)
    
    def industry_sector(self) -> str:
        """Generate a realistic industry sector."""
        sectors = [
            'Technology', 'Software & Services', 'Healthcare', 'Biotech',
            'Financial Services', 'Materials', 'Energy', 'Consumer Discretionary',
            'Consumer Staples', 'Industrials', 'Real Estate', 'Utilities',
            'Telecommunications'
        ]
        return random.choice(sectors)
    
    def company_stage(self) -> str:
        """Generate a realistic company stage."""
        stages = ['Pre-IPO', 'Early Stage', 'Growth Stage', 'Mature', 'Late Stage']
        return random.choice(stages)


# Register the custom provider
factory.Faker.add_provider(AustralianBusinessProvider)


class BaseFactory(factory.Factory):
    """Base factory with common fields."""
    
    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    created_at = factory.LazyFunction(datetime.utcnow)
    updated_at = factory.LazyFunction(datetime.utcnow)


class UserFactory(BaseFactory):
    """Factory for creating User instances."""
    
    class Meta:
        model = dict  # Replace with actual User model
    
    email = factory.Faker('email')
    first_name = factory.Faker('first_name')
    last_name = factory.Faker('last_name')
    company_name = factory.Faker('company')
    is_active = True
    is_verified = True
    subscription_tier = factory.Iterator(['basic', 'professional', 'enterprise'])
    
    # Password would be hashed in real implementation
    password_hash = factory.LazyFunction(
        lambda: '$2b$12$example.hash.for.testing.purposes'
    )
    
    preferences = factory.LazyAttribute(
        lambda obj: {
            'default_analysis_type': random.choice(['basic', 'comprehensive', 'premium']),
            'email_notifications': True,
            'webhook_notifications': False,
            'default_currency': 'AUD',
            'timezone': 'Australia/Sydney'
        }
    )


class CompanyFactory(BaseFactory):
    """Factory for creating Company instances."""
    
    class Meta:
        model = dict  # Replace with actual Company model
    
    name = factory.LazyAttribute(
        lambda obj: f"{factory.Faker('company').generate()} Pty Ltd"
    )
    abn = factory.Faker('abn')
    industry_sector = factory.Faker('industry_sector')
    business_description = factory.Faker('text', max_nb_chars=500)
    
    # Financial information
    annual_revenue = factory.LazyAttribute(
        lambda obj: Decimal(random.uniform(1000000, 100000000))
    )
    employee_count = factory.LazyAttribute(
        lambda obj: random.randint(10, 1000)
    )
    founded_year = factory.LazyAttribute(
        lambda obj: random.randint(1990, 2020)
    )
    
    # IPO information
    expected_ipo_date = factory.LazyAttribute(
        lambda obj: datetime.utcnow() + timedelta(days=random.randint(30, 365))
    )
    shares_to_be_issued = factory.LazyAttribute(
        lambda obj: random.randint(10000000, 100000000)
    )
    expected_offer_price_min = factory.LazyAttribute(
        lambda obj: Decimal(random.uniform(1.0, 5.0))
    )
    expected_offer_price_max = factory.LazyAttribute(
        lambda obj: Decimal(random.uniform(5.0, 10.0))
    )
    
    # Geographic information
    headquarters_city = factory.Faker('city')
    headquarters_state = factory.Iterator([
        'NSW', 'VIC', 'QLD', 'WA', 'SA', 'TAS', 'NT', 'ACT'
    ])
    
    # Ownership structure
    ownership_structure = factory.LazyAttribute(
        lambda obj: {
            'founders': random.randint(20, 60),
            'employees': random.randint(10, 30),
            'investors': random.randint(10, 40),
            'other': random.randint(0, 20)
        }
    )
    
    # Created by user
    created_by = factory.LazyFunction(lambda: str(uuid.uuid4()))


class ValuationFactory(BaseFactory):
    """Factory for creating Valuation instances."""
    
    class Meta:
        model = dict  # Replace with actual Valuation model
    
    company_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    created_by = factory.LazyFunction(lambda: str(uuid.uuid4()))
    
    # Job information
    job_id = factory.LazyAttribute(
        lambda obj: f"val_{uuid.uuid4().hex[:24]}"
    )
    status = factory.Iterator([
        'queued', 'processing', 'completed', 'failed', 'cancelled'
    ])
    
    # Analysis parameters
    analysis_type = factory.Iterator(['basic', 'comprehensive', 'premium'])
    include_peer_analysis = True
    risk_assessment_level = factory.Iterator(['basic', 'detailed', 'comprehensive'])
    
    # Valuation results (only if completed)
    target_price = factory.Maybe(
        'status',
        yes_declaration=factory.LazyAttribute(
            lambda obj: Decimal(random.uniform(1.0, 10.0)) if obj.status == 'completed' else None
        ),
        no_declaration=None
    )
    
    price_range_low = factory.Maybe(
        'status',
        yes_declaration=factory.LazyAttribute(
            lambda obj: Decimal(random.uniform(0.8, 2.0)) if obj.status == 'completed' else None
        ),
        no_declaration=None
    )
    
    price_range_high = factory.Maybe(
        'status',
        yes_declaration=factory.LazyAttribute(
            lambda obj: Decimal(random.uniform(2.0, 8.0)) if obj.status == 'completed' else None
        ),
        no_declaration=None
    )
    
    confidence_score = factory.Maybe(
        'status',
        yes_declaration=factory.LazyAttribute(
            lambda obj: random.uniform(0.6, 0.95) if obj.status == 'completed' else None
        ),
        no_declaration=None
    )
    
    # Processing information
    started_at = factory.Maybe(
        'status',
        yes_declaration=factory.LazyAttribute(
            lambda obj: datetime.utcnow() - timedelta(minutes=random.randint(5, 60))
            if obj.status in ['processing', 'completed'] else None
        ),
        no_declaration=None
    )
    
    completed_at = factory.Maybe(
        'status',
        yes_declaration=factory.LazyAttribute(
            lambda obj: datetime.utcnow() if obj.status == 'completed' else None
        ),
        no_declaration=None
    )
    
    # Methodology and assumptions
    methodology = factory.LazyAttribute(
        lambda obj: {
            'primary_method': random.choice(['dcf', 'multiples', 'asset_based']),
            'weighting': {
                'dcf': random.uniform(0.3, 0.7),
                'multiples': random.uniform(0.2, 0.5),
                'asset_based': random.uniform(0.1, 0.3)
            },
            'assumptions': [
                'Market growth rate: 8-12% annually',
                'Risk-free rate: 3.5%',
                'Market risk premium: 6%',
                'Terminal growth rate: 3%'
            ]
        }
    )


class DocumentFactory(BaseFactory):
    """Factory for creating Document instances."""
    
    class Meta:
        model = dict  # Replace with actual Document model
    
    job_id = factory.LazyAttribute(
        lambda obj: f"doc_{uuid.uuid4().hex[:24]}"
    )
    
    document_type = factory.Iterator([
        'prospectus', 'financial_statements', 'annual_report', 
        'offering_memorandum', 'company_profile', 'other'
    ])
    
    filename = factory.LazyAttribute(
        lambda obj: f"{obj.document_type}_{random.randint(1000, 9999)}.pdf"
    )
    
    file_size = factory.LazyAttribute(
        lambda obj: random.randint(100000, 50000000)  # 100KB to 50MB
    )
    
    mime_type = 'application/pdf'
    
    status = factory.Iterator(['processing', 'completed', 'failed'])
    
    # Processing results
    extracted_data = factory.LazyAttribute(
        lambda obj: {
            'company_info': {
                'name': factory.Faker('company').generate(),
                'industry': factory.Faker('industry_sector').generate(),
                'employees': random.randint(50, 1000)
            },
            'financial_data': {
                'revenue': random.randint(1000000, 100000000),
                'profit': random.randint(100000, 10000000),
                'assets': random.randint(5000000, 500000000)
            },
            'risk_factors': [
                'Market competition risk',
                'Regulatory compliance risk',
                'Technology disruption risk'
            ]
        } if obj.status == 'completed' else {}
    )
    
    confidence_score = factory.Maybe(
        'status',
        yes_declaration=factory.LazyAttribute(
            lambda obj: random.uniform(0.7, 0.98) if obj.status == 'completed' else None
        ),
        no_declaration=None
    )
    
    processing_time = factory.Maybe(
        'status',
        yes_declaration=factory.LazyAttribute(
            lambda obj: random.uniform(30, 180) if obj.status == 'completed' else None
        ),
        no_declaration=None
    )
    
    # File storage information
    file_url = factory.LazyAttribute(
        lambda obj: f"https://storage.googleapis.com/test-bucket/{obj.job_id}/{obj.filename}"
    )
    
    valuation_job_id = factory.LazyFunction(
        lambda: f"val_{uuid.uuid4().hex[:24]}"
    )
    
    uploaded_by = factory.LazyFunction(lambda: str(uuid.uuid4()))


class MarketDataFactory(BaseFactory):
    """Factory for creating MarketData instances."""
    
    class Meta:
        model = dict  # Replace with actual MarketData model
    
    ticker = factory.Faker('asx_ticker')
    company_name = factory.LazyAttribute(
        lambda obj: f"{factory.Faker('company').generate()} Limited"
    )
    
    # Current market data
    current_price = factory.LazyAttribute(
        lambda obj: Decimal(random.uniform(1.0, 150.0))
    )
    market_cap = factory.LazyAttribute(
        lambda obj: random.randint(100000000, 50000000000)  # $100M to $50B
    )
    shares_outstanding = factory.LazyAttribute(
        lambda obj: random.randint(50000000, 2000000000)
    )
    volume = factory.LazyAttribute(
        lambda obj: random.randint(100000, 10000000)
    )
    
    # Financial ratios
    pe_ratio = factory.LazyAttribute(
        lambda obj: round(random.uniform(5.0, 30.0), 2)
    )
    ev_ebitda = factory.LazyAttribute(
        lambda obj: round(random.uniform(8.0, 25.0), 2)
    )
    price_to_book = factory.LazyAttribute(
        lambda obj: round(random.uniform(0.8, 5.0), 2)
    )
    dividend_yield = factory.LazyAttribute(
        lambda obj: round(random.uniform(0.0, 0.08), 4)
    )
    
    # Price changes
    price_change_1d = factory.LazyAttribute(
        lambda obj: round(random.uniform(-0.1, 0.1), 4)
    )
    price_change_1m = factory.LazyAttribute(
        lambda obj: round(random.uniform(-0.2, 0.2), 4)
    )
    price_change_1y = factory.LazyAttribute(
        lambda obj: round(random.uniform(-0.5, 1.0), 4)
    )
    
    # 52-week range
    week_52_high = factory.LazyAttribute(
        lambda obj: obj.current_price * Decimal(random.uniform(1.1, 2.0))
    )
    week_52_low = factory.LazyAttribute(
        lambda obj: obj.current_price * Decimal(random.uniform(0.5, 0.9))
    )
    
    # Data source and timing
    data_source = factory.Iterator(['ASX', 'Reuters', 'Bloomberg', 'Yahoo Finance'])
    last_updated = factory.LazyFunction(
        lambda: datetime.utcnow() - timedelta(minutes=random.randint(1, 30))
    )
    
    # Industry classification
    industry_sector = factory.Faker('industry_sector')
    industry_group = factory.LazyAttribute(
        lambda obj: f"{obj.industry_sector} - {random.choice(['Large Cap', 'Mid Cap', 'Small Cap'])}"
    )


class PeerGroupFactory(BaseFactory):
    """Factory for creating peer group analysis data."""
    
    class Meta:
        model = dict
    
    industry_sector = factory.Faker('industry_sector')
    analysis_date = factory.LazyFunction(datetime.utcnow)
    
    # Peer companies (list of market data)
    peer_companies = factory.LazyAttribute(
        lambda obj: [
            MarketDataFactory.build(industry_sector=obj.industry_sector)
            for _ in range(random.randint(5, 15))
        ]
    )
    
    # Summary statistics
    summary_statistics = factory.LazyAttribute(
        lambda obj: {
            'count': len(obj.peer_companies),
            'median_market_cap': sum(p.market_cap for p in obj.peer_companies) / len(obj.peer_companies),
            'median_pe_ratio': sum(p.pe_ratio for p in obj.peer_companies) / len(obj.peer_companies),
            'median_ev_ebitda': sum(p.ev_ebitda for p in obj.peer_companies) / len(obj.peer_companies),
            'median_price_to_book': sum(p.price_to_book for p in obj.peer_companies) / len(obj.peer_companies)
        }
    )


class ProspectusFactory(BaseFactory):
    """Factory for creating prospectus document data."""
    
    class Meta:
        model = dict
    
    # Company information
    company_name = factory.LazyAttribute(
        lambda obj: f"{factory.Faker('company').generate()} Pty Ltd"
    )
    abn = factory.Faker('abn')
    incorporation_date = factory.LazyAttribute(
        lambda obj: datetime.utcnow() - timedelta(days=random.randint(365, 3650))
    )
    
    # Business overview
    business_description = factory.Faker('text', max_nb_chars=1000)
    key_products_services = factory.LazyAttribute(
        lambda obj: [factory.Faker('bs').generate() for _ in range(random.randint(2, 5))]
    )
    competitive_advantages = factory.LazyAttribute(
        lambda obj: [factory.Faker('catch_phrase').generate() for _ in range(random.randint(2, 4))]
    )
    
    # Financial information
    historical_financials = factory.LazyAttribute(
        lambda obj: {
            'revenue': [random.randint(1000000, 50000000) for _ in range(3)],
            'gross_profit': [random.randint(500000, 25000000) for _ in range(3)],
            'net_profit': [random.randint(100000, 10000000) for _ in range(3)],
            'years': [2021, 2022, 2023]
        }
    )
    
    # IPO details
    offer_price = factory.LazyAttribute(
        lambda obj: Decimal(random.uniform(1.0, 10.0))
    )
    shares_offered = factory.LazyAttribute(
        lambda obj: random.randint(10000000, 100000000)
    )
    gross_proceeds = factory.LazyAttribute(
        lambda obj: obj.offer_price * obj.shares_offered
    )
    use_of_proceeds = factory.LazyAttribute(
        lambda obj: {
            'working_capital': random.uniform(0.3, 0.5),
            'expansion': random.uniform(0.2, 0.4),
            'debt_repayment': random.uniform(0.1, 0.3),
            'other': random.uniform(0.05, 0.15)
        }
    )
    
    # Risk factors
    risk_factors = factory.LazyAttribute(
        lambda obj: [
            'Market competition may impact revenue growth',
            'Regulatory changes could affect operations',
            'Key personnel dependency risk',
            'Technology disruption risk',
            'Economic downturn impact on customer demand'
        ]
    )
    
    # Management team
    key_management = factory.LazyAttribute(
        lambda obj: [
            {
                'name': factory.Faker('name').generate(),
                'position': random.choice(['CEO', 'CFO', 'CTO', 'COO']),
                'experience': f"{random.randint(10, 30)} years in {obj.business_description[:50]}"
            }
            for _ in range(random.randint(3, 6))
        ]
    )


# Batch factories for creating multiple instances
class BatchUserFactory:
    """Factory for creating batches of users."""
    
    @staticmethod
    def create_batch(size: int = 10) -> List[Dict]:
        return [UserFactory.build() for _ in range(size)]
    
    @staticmethod
    def create_diverse_users() -> List[Dict]:
        """Create users with different subscription tiers and characteristics."""
        users = []
        
        # Basic tier users
        users.extend([
            UserFactory.build(subscription_tier='basic') 
            for _ in range(5)
        ])
        
        # Professional tier users
        users.extend([
            UserFactory.build(subscription_tier='professional') 
            for _ in range(3)
        ])
        
        # Enterprise tier users
        users.extend([
            UserFactory.build(subscription_tier='enterprise') 
            for _ in range(2)
        ])
        
        return users


class BatchCompanyFactory:
    """Factory for creating batches of companies."""
    
    @staticmethod
    def create_batch(size: int = 10) -> List[Dict]:
        return [CompanyFactory.build() for _ in range(size)]
    
    @staticmethod
    def create_diverse_companies() -> List[Dict]:
        """Create companies across different industries and stages."""
        companies = []
        
        sectors = [
            'Technology', 'Healthcare', 'Financial Services', 
            'Materials', 'Energy', 'Consumer Discretionary'
        ]
        
        for sector in sectors:
            companies.extend([
                CompanyFactory.build(industry_sector=sector)
                for _ in range(2)
            ])
        
        return companies


# Utility functions for test data
def create_complete_valuation_scenario() -> Dict[str, Any]:
    """Create a complete valuation scenario with user, company, and valuation."""
    user = UserFactory.build()
    company = CompanyFactory.build(created_by=user['id'])
    valuation = ValuationFactory.build(
        company_id=company['id'],
        created_by=user['id'],
        status='completed'
    )
    documents = [
        DocumentFactory.build(
            document_type='prospectus',
            valuation_job_id=valuation['job_id'],
            uploaded_by=user['id'],
            status='completed'
        ),
        DocumentFactory.build(
            document_type='financial_statements',
            valuation_job_id=valuation['job_id'],
            uploaded_by=user['id'],
            status='completed'
        )
    ]
    
    return {
        'user': user,
        'company': company,
        'valuation': valuation,
        'documents': documents
    }


def create_market_data_set(count: int = 20) -> List[Dict]:
    """Create a diverse set of market data for testing."""
    market_data = []
    
    # Create data for different market caps
    market_caps = ['large_cap', 'mid_cap', 'small_cap']
    
    for cap_type in market_caps:
        if cap_type == 'large_cap':
            cap_range = (10000000000, 50000000000)  # $10B - $50B
        elif cap_type == 'mid_cap':
            cap_range = (1000000000, 10000000000)   # $1B - $10B
        else:
            cap_range = (100000000, 1000000000)     # $100M - $1B
        
        for _ in range(count // len(market_caps)):
            market_data.append(
                MarketDataFactory.build(
                    market_cap=random.randint(*cap_range)
                )
            )
    
    return market_data