"""
Test data management utilities for the IPO Valuation Platform test suite.

This module provides utilities for managing test datasets, loading sample data,
and creating realistic test scenarios.
"""

import csv
import json
import os
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import tempfile
import pandas as pd
import numpy as np

from tests.utils.factories import (
    UserFactory, CompanyFactory, ValuationFactory,
    MarketDataFactory, DocumentFactory, ProspectusFactory
)


class TestDataManager:
    """Manages test data loading, creation, and cleanup."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(__file__).parent.parent / "fixtures"
        self.temp_dir = None
        self._cached_datasets = {}
    
    def setup_temp_directory(self) -> Path:
        """Create temporary directory for test files."""
        if not self.temp_dir:
            self.temp_dir = tempfile.mkdtemp(prefix="ipo_valuation_test_")
        return Path(self.temp_dir)
    
    def cleanup_temp_directory(self):
        """Clean up temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def load_sample_companies(self) -> List[Dict[str, Any]]:
        """Load sample company data for testing."""
        if 'companies' not in self._cached_datasets:
            companies = []
            
            # Create diverse company portfolio
            industries = [
                'Technology', 'Healthcare', 'Financial Services',
                'Materials', 'Energy', 'Consumer Discretionary'
            ]
            
            for industry in industries:
                # Small cap companies
                companies.extend([
                    CompanyFactory.build(
                        industry_sector=industry,
                        annual_revenue=Decimal(random.uniform(1e6, 50e6)),
                        shares_to_be_issued=random.randint(10e6, 50e6),
                        expected_offer_price_min=Decimal(random.uniform(1.0, 3.0)),
                        expected_offer_price_max=Decimal(random.uniform(3.0, 8.0))
                    )
                    for _ in range(2)
                ])
                
                # Mid cap companies
                companies.extend([
                    CompanyFactory.build(
                        industry_sector=industry,
                        annual_revenue=Decimal(random.uniform(50e6, 500e6)),
                        shares_to_be_issued=random.randint(50e6, 200e6),
                        expected_offer_price_min=Decimal(random.uniform(2.0, 5.0)),
                        expected_offer_price_max=Decimal(random.uniform(5.0, 15.0))
                    )
                    for _ in range(1)
                ])
            
            self._cached_datasets['companies'] = companies
        
        return self._cached_datasets['companies']
    
    def load_sample_market_data(self) -> List[Dict[str, Any]]:
        """Load sample market data for testing."""
        if 'market_data' not in self._cached_datasets:
            import random
            
            # Real ASX companies with realistic data ranges
            real_companies = [
                {'ticker': 'CBA', 'name': 'Commonwealth Bank', 'sector': 'Financial Services', 
                 'price_range': (80, 120), 'market_cap_range': (140e9, 180e9)},
                {'ticker': 'WBC', 'name': 'Westpac Banking Corp', 'sector': 'Financial Services',
                 'price_range': (20, 30), 'market_cap_range': (80e9, 100e9)},
                {'ticker': 'BHP', 'name': 'BHP Group', 'sector': 'Materials',
                 'price_range': (35, 55), 'market_cap_range': (180e9, 220e9)},
                {'ticker': 'CSL', 'name': 'CSL Limited', 'sector': 'Healthcare',
                 'price_range': (250, 350), 'market_cap_range': (120e9, 160e9)},
                {'ticker': 'WES', 'name': 'Wesfarmers', 'sector': 'Consumer Discretionary',
                 'price_range': (45, 65), 'market_cap_range': (50e9, 70e9)},
            ]
            
            market_data = []
            for company in real_companies:
                price = random.uniform(*company['price_range'])
                market_cap = random.uniform(*company['market_cap_range'])
                
                market_data.append({
                    'ticker': f"ASX:{company['ticker']}",
                    'company_name': company['name'],
                    'industry_sector': company['sector'],
                    'current_price': Decimal(str(round(price, 2))),
                    'market_cap': int(market_cap),
                    'pe_ratio': round(random.uniform(8.0, 25.0), 2),
                    'ev_ebitda': round(random.uniform(6.0, 20.0), 2),
                    'dividend_yield': round(random.uniform(0.02, 0.08), 4),
                    'volume': random.randint(500000, 10000000),
                    'last_updated': datetime.utcnow()
                })
            
            # Add more synthetic companies
            for _ in range(50):
                market_data.append(MarketDataFactory.build())
            
            self._cached_datasets['market_data'] = market_data
        
        return self._cached_datasets['market_data']
    
    def create_valuation_test_scenario(self, scenario_type: str = 'standard') -> Dict[str, Any]:
        """Create complete valuation test scenario."""
        import random
        
        scenarios = {
            'standard': self._create_standard_scenario,
            'complex': self._create_complex_scenario,
            'high_risk': self._create_high_risk_scenario,
            'tech_startup': self._create_tech_startup_scenario,
            'mining_company': self._create_mining_scenario,
        }
        
        if scenario_type not in scenarios:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
        
        return scenarios[scenario_type]()
    
    def _create_standard_scenario(self) -> Dict[str, Any]:
        """Create standard valuation scenario."""
        user = UserFactory.build(subscription_tier='professional')
        company = CompanyFactory.build(
            industry_sector='Software & Services',
            annual_revenue=Decimal('25000000'),
            employee_count=150,
            expected_offer_price_min=Decimal('2.50'),
            expected_offer_price_max=Decimal('4.00'),
            shares_to_be_issued=40000000,
            created_by=user['id']
        )
        
        valuation = ValuationFactory.build(
            company_id=company['id'],
            created_by=user['id'],
            analysis_type='comprehensive',
            status='completed',
            target_price=Decimal('3.25'),
            confidence_score=0.87
        )
        
        documents = [
            DocumentFactory.build(
                document_type='prospectus',
                valuation_job_id=valuation['job_id'],
                status='completed'
            ),
            DocumentFactory.build(
                document_type='financial_statements',
                valuation_job_id=valuation['job_id'],
                status='completed'
            )
        ]
        
        peer_data = [
            MarketDataFactory.build(
                industry_sector='Software & Services',
                market_cap=random.randint(100000000, 1000000000)
            )
            for _ in range(8)
        ]
        
        return {
            'user': user,
            'company': company,
            'valuation': valuation,
            'documents': documents,
            'peer_data': peer_data,
            'scenario_type': 'standard'
        }
    
    def _create_complex_scenario(self) -> Dict[str, Any]:
        """Create complex valuation scenario with multiple documents and peer groups."""
        import random
        
        user = UserFactory.build(subscription_tier='enterprise')
        company = CompanyFactory.build(
            industry_sector='Healthcare',
            annual_revenue=Decimal('75000000'),
            employee_count=450,
            expected_offer_price_min=Decimal('8.00'),
            expected_offer_price_max=Decimal('12.00'),
            shares_to_be_issued=60000000,
            created_by=user['id']
        )
        
        valuation = ValuationFactory.build(
            company_id=company['id'],
            created_by=user['id'],
            analysis_type='premium',
            status='completed',
            target_price=Decimal('10.25'),
            confidence_score=0.92
        )
        
        # Multiple document types
        documents = [
            DocumentFactory.build(
                document_type='prospectus',
                valuation_job_id=valuation['job_id'],
                status='completed'
            ),
            DocumentFactory.build(
                document_type='financial_statements',
                valuation_job_id=valuation['job_id'],
                status='completed'
            ),
            DocumentFactory.build(
                document_type='annual_report',
                valuation_job_id=valuation['job_id'],
                status='completed'
            ),
            DocumentFactory.build(
                document_type='offering_memorandum',
                valuation_job_id=valuation['job_id'],
                status='completed'
            )
        ]
        
        # Large peer group
        peer_data = [
            MarketDataFactory.build(
                industry_sector='Healthcare',
                market_cap=random.randint(500000000, 5000000000)
            )
            for _ in range(20)
        ]
        
        return {
            'user': user,
            'company': company,
            'valuation': valuation,
            'documents': documents,
            'peer_data': peer_data,
            'scenario_type': 'complex'
        }
    
    def _create_high_risk_scenario(self) -> Dict[str, Any]:
        """Create high-risk valuation scenario."""
        import random
        
        user = UserFactory.build(subscription_tier='basic')
        company = CompanyFactory.build(
            industry_sector='Biotech',
            annual_revenue=Decimal('5000000'),  # Low revenue
            employee_count=45,
            founded_year=2021,  # Very new company
            expected_offer_price_min=Decimal('0.80'),
            expected_offer_price_max=Decimal('1.50'),
            shares_to_be_issued=100000000,  # High dilution
            created_by=user['id']
        )
        
        valuation = ValuationFactory.build(
            company_id=company['id'],
            created_by=user['id'],
            analysis_type='comprehensive',
            status='completed',
            target_price=Decimal('1.15'),
            confidence_score=0.68  # Lower confidence due to risk
        )
        
        documents = [
            DocumentFactory.build(
                document_type='prospectus',
                valuation_job_id=valuation['job_id'],
                status='completed',
                confidence_score=0.75  # Lower extraction confidence
            )
        ]
        
        peer_data = [
            MarketDataFactory.build(
                industry_sector='Biotech',
                market_cap=random.randint(50000000, 500000000),
                pe_ratio=None  # Many biotech companies are pre-revenue
            )
            for _ in range(12)
        ]
        
        return {
            'user': user,
            'company': company,
            'valuation': valuation,
            'documents': documents,
            'peer_data': peer_data,
            'scenario_type': 'high_risk'
        }
    
    def _create_tech_startup_scenario(self) -> Dict[str, Any]:
        """Create tech startup valuation scenario."""
        import random
        
        user = UserFactory.build(subscription_tier='professional')
        company = CompanyFactory.build(
            industry_sector='Technology',
            annual_revenue=Decimal('12000000'),
            employee_count=85,
            founded_year=2019,
            expected_offer_price_min=Decimal('3.00'),
            expected_offer_price_max=Decimal('5.50'),
            shares_to_be_issued=50000000,
            business_description="AI-powered fintech platform for SME lending",
            created_by=user['id']
        )
        
        valuation = ValuationFactory.build(
            company_id=company['id'],
            created_by=user['id'],
            analysis_type='comprehensive',
            status='completed',
            target_price=Decimal('4.20'),
            confidence_score=0.84
        )
        
        documents = [
            DocumentFactory.build(
                document_type='prospectus',
                valuation_job_id=valuation['job_id'],
                status='completed'
            ),
            DocumentFactory.build(
                document_type='company_profile',
                valuation_job_id=valuation['job_id'],
                status='completed'
            )
        ]
        
        # Tech-focused peer group
        peer_data = [
            MarketDataFactory.build(
                industry_sector='Technology',
                market_cap=random.randint(200000000, 2000000000),
                pe_ratio=random.uniform(15.0, 35.0)  # Higher P/E for tech
            )
            for _ in range(10)
        ]
        
        return {
            'user': user,
            'company': company,
            'valuation': valuation,
            'documents': documents,
            'peer_data': peer_data,
            'scenario_type': 'tech_startup'
        }
    
    def _create_mining_scenario(self) -> Dict[str, Any]:
        """Create mining company valuation scenario."""
        import random
        
        user = UserFactory.build(subscription_tier='enterprise')
        company = CompanyFactory.build(
            industry_sector='Materials',
            annual_revenue=Decimal('180000000'),
            employee_count=850,
            expected_offer_price_min=Decimal('1.80'),
            expected_offer_price_max=Decimal('2.50'),
            shares_to_be_issued=150000000,
            business_description="Gold and copper mining operations in Western Australia",
            created_by=user['id']
        )
        
        valuation = ValuationFactory.build(
            company_id=company['id'],
            created_by=user['id'],
            analysis_type='comprehensive',
            status='completed',
            target_price=Decimal('2.15'),
            confidence_score=0.78
        )
        
        documents = [
            DocumentFactory.build(
                document_type='prospectus',
                valuation_job_id=valuation['job_id'],
                status='completed'
            ),
            DocumentFactory.build(
                document_type='annual_report',
                valuation_job_id=valuation['job_id'],
                status='completed'
            )
        ]
        
        # Mining-focused peer group
        peer_data = [
            MarketDataFactory.build(
                industry_sector='Materials',
                market_cap=random.randint(500000000, 10000000000),
                pe_ratio=random.uniform(6.0, 18.0)  # Typically lower P/E for materials
            )
            for _ in range(15)
        ]
        
        return {
            'user': user,
            'company': company,
            'valuation': valuation,
            'documents': documents,
            'peer_data': peer_data,
            'scenario_type': 'mining'
        }
    
    def create_performance_test_data(self, scale: str = 'medium') -> Dict[str, List[Dict]]:
        """Create large datasets for performance testing."""
        scales = {
            'small': {'users': 50, 'companies': 100, 'valuations': 200},
            'medium': {'users': 200, 'companies': 500, 'valuations': 1000},
            'large': {'users': 1000, 'companies': 2000, 'valuations': 5000},
            'xlarge': {'users': 5000, 'companies': 10000, 'valuations': 25000}
        }
        
        if scale not in scales:
            raise ValueError(f"Unknown scale: {scale}. Choose from {list(scales.keys())}")
        
        config = scales[scale]
        
        # Generate users
        users = [UserFactory.build() for _ in range(config['users'])]
        
        # Generate companies
        companies = []
        for i in range(config['companies']):
            user_id = users[i % len(users)]['id']
            companies.append(CompanyFactory.build(created_by=user_id))
        
        # Generate valuations
        valuations = []
        for i in range(config['valuations']):
            company = companies[i % len(companies)]
            user_id = users[i % len(users)]['id']
            valuations.append(
                ValuationFactory.build(
                    company_id=company['id'],
                    created_by=user_id
                )
            )
        
        # Generate market data
        market_data = [
            MarketDataFactory.build() 
            for _ in range(config['companies'] // 2)
        ]
        
        return {
            'users': users,
            'companies': companies,
            'valuations': valuations,
            'market_data': market_data,
            'scale': scale
        }
    
    def save_test_data_to_files(self, data: Dict[str, Any], format: str = 'json') -> Dict[str, Path]:
        """Save test data to files for external tool testing."""
        temp_dir = self.setup_temp_directory()
        file_paths = {}
        
        for data_type, records in data.items():
            if isinstance(records, list):
                if format == 'json':
                    file_path = temp_dir / f"{data_type}.json"
                    with open(file_path, 'w') as f:
                        json.dump(records, f, indent=2, default=str)
                
                elif format == 'csv':
                    file_path = temp_dir / f"{data_type}.csv"
                    if records:
                        # Convert to DataFrame and save as CSV
                        df = pd.DataFrame(records)
                        df.to_csv(file_path, index=False)
                
                file_paths[data_type] = file_path
        
        return file_paths
    
    def create_sample_documents(self) -> Dict[str, Path]:
        """Create sample document files for testing."""
        temp_dir = self.setup_temp_directory()
        document_paths = {}
        
        # Create sample prospectus
        prospectus_path = temp_dir / "sample_prospectus.pdf"
        with open(prospectus_path, 'wb') as f:
            # Mock PDF content - in reality this would be a proper PDF
            f.write(b"Mock prospectus PDF content for testing purposes")
        document_paths['prospectus'] = prospectus_path
        
        # Create sample financial statements
        financials_path = temp_dir / "financial_statements.xlsx"
        # Create sample Excel file with financial data
        financial_data = {
            'Income Statement': pd.DataFrame({
                'Year': [2021, 2022, 2023],
                'Revenue': [45000000, 52000000, 61000000],
                'Gross Profit': [28000000, 33000000, 39000000],
                'Net Income': [8500000, 11200000, 14500000]
            }),
            'Balance Sheet': pd.DataFrame({
                'Year': [2021, 2022, 2023],
                'Total Assets': [95000000, 112000000, 135000000],
                'Total Liabilities': [45000000, 52000000, 58000000],
                'Shareholders Equity': [50000000, 60000000, 77000000]
            }),
            'Cash Flow': pd.DataFrame({
                'Year': [2021, 2022, 2023],
                'Operating Cash Flow': [12000000, 15500000, 18200000],
                'Investing Cash Flow': [-8000000, -12000000, -15000000],
                'Financing Cash Flow': [-2000000, 5000000, -1000000]
            })
        }
        
        with pd.ExcelWriter(financials_path) as writer:
            for sheet_name, df in financial_data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        document_paths['financial_statements'] = financials_path
        
        # Create sample annual report
        annual_report_path = temp_dir / "annual_report.pdf"
        with open(annual_report_path, 'wb') as f:
            f.write(b"Mock annual report PDF content for testing purposes")
        document_paths['annual_report'] = annual_report_path
        
        return document_paths
    
    def load_historical_market_data(self, years: int = 5) -> pd.DataFrame:
        """Generate historical market data for backtesting."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=years * 365)
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create sample tickers
        tickers = ['CBA', 'WBC', 'ANZ', 'BHP', 'CSL', 'WES', 'WOW', 'TLS']
        
        historical_data = []
        
        for ticker in tickers:
            # Generate price series with some volatility
            np.random.seed(hash(ticker) % 2**32)  # Consistent randomness per ticker
            initial_price = np.random.uniform(20, 150)
            
            # Generate random walk with drift
            returns = np.random.normal(0.0005, 0.02, len(date_range))  # Small positive drift
            prices = [initial_price]
            
            for return_rate in returns[1:]:
                new_price = prices[-1] * (1 + return_rate)
                prices.append(max(0.01, new_price))  # Ensure positive prices
            
            for i, (date, price) in enumerate(zip(date_range, prices)):
                volume = int(np.random.lognormal(14, 0.5))  # Log-normal volume distribution
                
                historical_data.append({
                    'date': date,
                    'ticker': f'ASX:{ticker}',
                    'open': price * np.random.uniform(0.98, 1.02),
                    'high': price * np.random.uniform(1.00, 1.05),
                    'low': price * np.random.uniform(0.95, 1.00),
                    'close': price,
                    'volume': volume,
                    'adj_close': price
                })
        
        return pd.DataFrame(historical_data)
    
    def create_ml_training_data(self, samples: int = 1000) -> Dict[str, np.ndarray]:
        """Create synthetic training data for ML models."""
        np.random.seed(42)  # For reproducible results
        
        # Features for valuation model
        features = np.random.randn(samples, 15)  # 15 features
        
        # Feature names
        feature_names = [
            'revenue', 'revenue_growth', 'profit_margin', 'debt_to_equity',
            'market_share', 'employee_growth', 'r_and_d_spend', 'market_size',
            'competition_level', 'regulatory_risk', 'management_quality',
            'brand_strength', 'operational_efficiency', 'financial_stability',
            'growth_potential'
        ]
        
        # Generate target values (valuation multiples)
        # Simple linear relationship with some noise
        weights = np.random.uniform(-1, 1, 15)
        targets = np.dot(features, weights) + np.random.normal(0, 0.5, samples)
        targets = np.clip(targets, 0.5, 20.0)  # Reasonable valuation range
        
        # Classification targets for risk assessment
        risk_scores = np.random.choice([0, 1, 2], samples, p=[0.3, 0.5, 0.2])  # Low, Medium, High
        
        return {
            'features': features,
            'feature_names': feature_names,
            'valuation_targets': targets,
            'risk_targets': risk_scores,
            'sample_count': samples
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_temp_directory()


# Global test data manager instance
test_data_manager = TestDataManager()


# Utility functions for common test data scenarios
def get_standard_test_scenario() -> Dict[str, Any]:
    """Get standard test scenario."""
    return test_data_manager.create_valuation_test_scenario('standard')


def get_performance_test_data(scale: str = 'medium') -> Dict[str, List[Dict]]:
    """Get performance test data."""
    return test_data_manager.create_performance_test_data(scale)


def get_sample_market_data() -> List[Dict[str, Any]]:
    """Get sample market data."""
    return test_data_manager.load_sample_market_data()


def create_temp_files() -> Dict[str, Path]:
    """Create temporary test files."""
    return test_data_manager.create_sample_documents()