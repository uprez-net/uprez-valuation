# Integration Patterns for ASX and ASIC Data

This document provides comprehensive guidance for implementing ETL pipelines, real-time streaming architectures, and API integration patterns specifically for Australian Securities Exchange (ASX) and Australian Securities & Investments Commission (ASIC) data sources.

## 🔄 Overview

Integration patterns define how the IPO valuation platform ingests, processes, and manages data from various Australian financial data sources. This includes both real-time streaming for market data and batch processing for regulatory filings.

## 🎯 Key Components

- **ETL Pipelines**: Extract, Transform, Load processes for ASX and ASIC data
- **Real-time Streaming**: Live market data processing
- **API Integrations**: Third-party financial data providers
- **Caching Strategies**: Optimal data storage and retrieval
- **Error Handling**: Robust failure recovery mechanisms

## 1. ETL Pipelines for ASX and ASIC Data

### 1.1 ASX Market Data ETL Pipeline

```python
import pandas as pd
import requests
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import asyncio
import aiohttp
import json
from pathlib import Path
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import redis
from google.cloud import bigquery, storage, pubsub_v1
import backoff
import time

@dataclass
class DataSourceConfig:
    """Configuration for data source connections."""
    source_name: str
    base_url: str
    api_key: Optional[str] = None
    rate_limit_per_second: int = 10
    timeout_seconds: int = 30
    retry_attempts: int = 3
    cache_ttl_seconds: int = 300

class ASXDataETL:
    """
    ETL pipeline for ASX market data with real-time and batch processing capabilities.
    
    Handles stock prices, company announcements, and corporate actions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize connections
        self._setup_database_connection()
        self._setup_cache_connection()
        self._setup_cloud_connections()
        
        # Rate limiting
        self.last_request_time = 0
        self.request_interval = 1.0 / config.get('rate_limit_per_second', 10)
        
    def _setup_database_connection(self):
        """Setup database connection for storing processed data."""
        db_config = self.config['database']
        
        connection_string = (
            f"postgresql://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        self.engine = sa.create_engine(connection_string, pool_pre_ping=True)
        self.Session = sessionmaker(bind=self.engine)
        
    def _setup_cache_connection(self):
        """Setup Redis cache connection."""
        cache_config = self.config.get('cache', {})
        
        self.redis_client = redis.Redis(
            host=cache_config.get('host', 'localhost'),
            port=cache_config.get('port', 6379),
            db=cache_config.get('db', 0),
            decode_responses=True
        )
        
    def _setup_cloud_connections(self):
        """Setup Google Cloud connections."""
        gcp_config = self.config.get('gcp', {})
        
        if gcp_config.get('project_id'):
            self.bq_client = bigquery.Client(project=gcp_config['project_id'])
            self.storage_client = storage.Client(project=gcp_config['project_id'])
            self.publisher = pubsub_v1.PublisherClient()
        
    async def extract_asx_data(self, symbols: List[str], 
                              data_types: List[str] = None,
                              date_range: Tuple[str, str] = None) -> Dict[str, pd.DataFrame]:
        """
        Extract data from ASX sources.
        
        Args:
            symbols: List of ASX symbols (e.g., ['CBA', 'BHP', 'CSL'])
            data_types: Types of data to extract ['prices', 'announcements', 'dividends']
            date_range: Tuple of (start_date, end_date) in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary of DataFrames by data type
        """
        if data_types is None:
            data_types = ['prices', 'announcements']
        
        extracted_data = {}
        
        # Extract different data types
        for data_type in data_types:
            if data_type == 'prices':
                extracted_data['prices'] = await self._extract_price_data(symbols, date_range)
            elif data_type == 'announcements':
                extracted_data['announcements'] = await self._extract_announcements(symbols, date_range)
            elif data_type == 'dividends':
                extracted_data['dividends'] = await self._extract_dividend_data(symbols, date_range)
            elif data_type == 'corporate_actions':
                extracted_data['corporate_actions'] = await self._extract_corporate_actions(symbols, date_range)
        
        return extracted_data
    
    @backoff.on_exception(backoff.expo, (requests.RequestException, aiohttp.ClientError))
    async def _extract_price_data(self, symbols: List[str], 
                                 date_range: Optional[Tuple[str, str]]) -> pd.DataFrame:
        """Extract stock price data from ASX."""
        all_price_data = []
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for symbol in symbols:
                # Check cache first
                cache_key = f"asx_prices_{symbol}_{date_range[0] if date_range else 'latest'}"
                cached_data = self.redis_client.get(cache_key)
                
                if cached_data:
                    df = pd.read_json(cached_data)
                    all_price_data.append(df)
                    continue
                
                # Rate limiting
                await self._respect_rate_limit()
                
                try:
                    # ASX price data API call (example endpoint)
                    url = f"https://api.asx.com.au/v1/companies/{symbol}/prices"
                    
                    params = {}
                    if date_range:
                        params['from'] = date_range[0]
                        params['to'] = date_range[1]
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Transform to DataFrame
                            if 'data' in data:
                                price_df = pd.DataFrame(data['data'])
                                price_df['symbol'] = symbol
                                price_df['date'] = pd.to_datetime(price_df['date'])
                                
                                # Standardize column names
                                price_df = self._standardize_price_columns(price_df)
                                
                                all_price_data.append(price_df)
                                
                                # Cache the result
                                self.redis_client.setex(
                                    cache_key,
                                    self.config.get('cache_ttl_seconds', 300),
                                    price_df.to_json()
                                )
                        else:
                            self.logger.warning(f"Failed to fetch price data for {symbol}: {response.status}")
                            
                except Exception as e:
                    self.logger.error(f"Error fetching price data for {symbol}: {e}")
                    continue
        
        if all_price_data:
            combined_df = pd.concat(all_price_data, ignore_index=True)
            return combined_df.sort_values(['symbol', 'date'])
        
        return pd.DataFrame()
    
    async def _extract_announcements(self, symbols: List[str], 
                                   date_range: Optional[Tuple[str, str]]) -> pd.DataFrame:
        """Extract company announcements from ASX."""
        all_announcements = []
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for symbol in symbols:
                await self._respect_rate_limit()
                
                try:
                    # ASX announcements API (example endpoint)
                    url = f"https://api.asx.com.au/v1/companies/{symbol}/announcements"
                    
                    params = {'count': 50}  # Get recent announcements
                    if date_range:
                        params['from'] = date_range[0]
                        params['to'] = date_range[1]
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if 'data' in data:
                                announcements_df = pd.DataFrame(data['data'])
                                announcements_df['symbol'] = symbol
                                announcements_df['date'] = pd.to_datetime(announcements_df['date'])
                                
                                # Clean and categorize announcements
                                announcements_df = self._categorize_announcements(announcements_df)
                                
                                all_announcements.append(announcements_df)
                                
                except Exception as e:
                    self.logger.error(f"Error fetching announcements for {symbol}: {e}")
                    continue
        
        if all_announcements:
            return pd.concat(all_announcements, ignore_index=True)
        
        return pd.DataFrame()
    
    def _standardize_price_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize price data column names and formats."""
        column_mapping = {
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'close_price': 'close',
            'last_price': 'close',
            'volume': 'volume',
            'trading_date': 'date'
        }
        
        # Rename columns if they exist
        df = df.rename(columns={old: new for old, new in column_mapping.items() if old in df.columns})
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'date', 'symbol']
        for col in required_columns:
            if col not in df.columns:
                if col in ['open', 'high', 'low'] and 'close' in df.columns:
                    df[col] = df['close']  # Fill missing OHLC with close price
                elif col == 'volume':
                    df[col] = 0  # Default volume
        
        return df[required_columns]
    
    def _categorize_announcements(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize announcements by type and importance."""
        if 'header' not in df.columns:
            return df
        
        # Define announcement categories
        categories = {
            'financial_results': [
                'annual report', 'half year', 'quarterly', 'financial results',
                'profit', 'loss', 'earnings', 'revenue'
            ],
            'corporate_actions': [
                'dividend', 'share split', 'bonus issue', 'rights issue',
                'capital raising', 'share buyback'
            ],
            'material_changes': [
                'acquisition', 'merger', 'divestment', 'joint venture',
                'material contract', 'material agreement'
            ],
            'regulatory': [
                'substantial holder', 'director interest', 'change of director',
                'change in substantial holding'
            ],
            'operational': [
                'production update', 'operational update', 'project update',
                'exploration results'
            ]
        }
        
        # Categorize announcements
        df['category'] = 'other'  # Default category
        df['importance'] = 'low'  # Default importance
        
        for category, keywords in categories.items():
            mask = df['header'].str.lower().str.contains('|'.join(keywords), na=False)
            df.loc[mask, 'category'] = category
            
            # Set importance based on category
            if category in ['financial_results', 'material_changes']:
                df.loc[mask, 'importance'] = 'high'
            elif category in ['corporate_actions', 'regulatory']:
                df.loc[mask, 'importance'] = 'medium'
        
        return df
    
    async def _respect_rate_limit(self):
        """Implement rate limiting to avoid overwhelming APIs."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.request_interval:
            sleep_time = self.request_interval - elapsed
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def transform_asx_data(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Transform extracted ASX data for analysis.
        
        Args:
            raw_data: Dictionary of raw DataFrames
            
        Returns:
            Dictionary of transformed DataFrames
        """
        transformed_data = {}
        
        for data_type, df in raw_data.items():
            if df.empty:
                transformed_data[data_type] = df
                continue
            
            if data_type == 'prices':
                transformed_data[data_type] = self._transform_price_data(df)
            elif data_type == 'announcements':
                transformed_data[data_type] = self._transform_announcements_data(df)
            else:
                transformed_data[data_type] = df
        
        return transformed_data
    
    def _transform_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform price data with technical indicators and ratios."""
        # Sort by symbol and date
        df = df.sort_values(['symbol', 'date'])
        
        # Calculate technical indicators for each symbol
        symbols = df['symbol'].unique()
        transformed_dfs = []
        
        for symbol in symbols:
            symbol_df = df[df['symbol'] == symbol].copy()
            
            # Calculate returns
            symbol_df['daily_return'] = symbol_df['close'].pct_change()
            symbol_df['log_return'] = np.log(symbol_df['close'] / symbol_df['close'].shift(1))
            
            # Calculate moving averages
            for window in [5, 10, 20, 50]:
                symbol_df[f'sma_{window}'] = symbol_df['close'].rolling(window=window).mean()
            
            # Calculate volatility
            symbol_df['volatility_20d'] = symbol_df['daily_return'].rolling(window=20).std() * np.sqrt(252)
            
            # Calculate volume indicators
            symbol_df['volume_sma_20'] = symbol_df['volume'].rolling(window=20).mean()
            symbol_df['volume_ratio'] = symbol_df['volume'] / symbol_df['volume_sma_20']
            
            # Price position relative to range
            symbol_df['price_position'] = (symbol_df['close'] - symbol_df['low'].rolling(20).min()) / \\
                                        (symbol_df['high'].rolling(20).max() - symbol_df['low'].rolling(20).min())
            
            transformed_dfs.append(symbol_df)
        
        return pd.concat(transformed_dfs, ignore_index=True)
    
    def _transform_announcements_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform announcements data with NLP features."""
        # Basic text preprocessing and feature extraction
        df['header_length'] = df['header'].str.len()
        df['word_count'] = df['header'].str.split().str.len()
        
        # Sentiment analysis (simplified)
        positive_words = ['profit', 'growth', 'increase', 'strong', 'positive', 'improvement']
        negative_words = ['loss', 'decline', 'decrease', 'weak', 'negative', 'concern']
        
        df['positive_sentiment'] = df['header'].str.lower().str.contains('|'.join(positive_words))
        df['negative_sentiment'] = df['header'].str.lower().str.contains('|'.join(negative_words))
        
        # Calculate sentiment score
        df['sentiment_score'] = df['positive_sentiment'].astype(int) - df['negative_sentiment'].astype(int)
        
        return df
    
    def load_to_destinations(self, transformed_data: Dict[str, pd.DataFrame]):
        """
        Load transformed data to various destinations (database, BigQuery, etc.).
        
        Args:
            transformed_data: Dictionary of transformed DataFrames
        """
        for data_type, df in transformed_data.items():
            if df.empty:
                continue
            
            try:
                # Load to PostgreSQL
                self._load_to_postgresql(df, data_type)
                
                # Load to BigQuery (if configured)
                if hasattr(self, 'bq_client'):
                    self._load_to_bigquery(df, data_type)
                
                # Publish to Pub/Sub for real-time processing
                if hasattr(self, 'publisher'):
                    self._publish_to_pubsub(df, data_type)
                
                self.logger.info(f"Successfully loaded {len(df)} records of {data_type}")
                
            except Exception as e:
                self.logger.error(f"Error loading {data_type}: {e}")
    
    def _load_to_postgresql(self, df: pd.DataFrame, table_name: str):
        """Load data to PostgreSQL database."""
        try:
            # Create table name with prefix
            full_table_name = f"asx_{table_name}"
            
            # Use upsert logic for price data
            if table_name == 'prices':
                self._upsert_price_data(df, full_table_name)
            else:
                # Simple append for other data types
                df.to_sql(full_table_name, self.engine, if_exists='append', index=False)
                
        except Exception as e:
            self.logger.error(f"Error loading to PostgreSQL: {e}")
    
    def _upsert_price_data(self, df: pd.DataFrame, table_name: str):
        """Upsert price data to avoid duplicates."""
        with self.Session() as session:
            for _, row in df.iterrows():
                # Check if record exists
                existing = session.execute(
                    sa.text(f"""
                        SELECT 1 FROM {table_name} 
                        WHERE symbol = :symbol AND date = :date
                    """),
                    {"symbol": row['symbol'], "date": row['date']}
                ).fetchone()
                
                if not existing:
                    # Insert new record
                    insert_data = row.to_dict()
                    session.execute(
                        sa.text(f"""
                            INSERT INTO {table_name} ({', '.join(insert_data.keys())})
                            VALUES ({', '.join(f':{k}' for k in insert_data.keys())})
                        """),
                        insert_data
                    )
            
            session.commit()
    
    def _load_to_bigquery(self, df: pd.DataFrame, table_name: str):
        """Load data to BigQuery for analytics."""
        try:
            dataset_id = self.config['gcp']['bigquery_dataset']
            table_id = f"asx_{table_name}"
            
            table_ref = f"{self.bq_client.project}.{dataset_id}.{table_id}"
            
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                autodetect=True
            )
            
            job = self.bq_client.load_table_from_dataframe(df, table_ref, job_config=job_config)
            job.result()  # Wait for job completion
            
        except Exception as e:
            self.logger.error(f"Error loading to BigQuery: {e}")
    
    def _publish_to_pubsub(self, df: pd.DataFrame, data_type: str):
        """Publish data changes to Pub/Sub for real-time processing."""
        try:
            topic_name = f"asx-{data_type}"
            topic_path = self.publisher.topic_path(self.config['gcp']['project_id'], topic_name)
            
            # Publish each row as a separate message for real-time processing
            for _, row in df.iterrows():
                message_data = {
                    'data_type': data_type,
                    'timestamp': datetime.now().isoformat(),
                    'data': row.to_dict()
                }
                
                message_json = json.dumps(message_data, default=str)
                message_bytes = message_json.encode('utf-8')
                
                future = self.publisher.publish(topic_path, message_bytes)
                future.result()  # Wait for publish confirmation
                
        except Exception as e:
            self.logger.error(f"Error publishing to Pub/Sub: {e}")
    
    async def run_etl_pipeline(self, symbols: List[str], 
                              data_types: List[str] = None) -> Dict[str, Any]:
        """
        Run the complete ETL pipeline.
        
        Args:
            symbols: List of ASX symbols to process
            data_types: Types of data to process
            
        Returns:
            Dictionary with pipeline results and statistics
        """
        pipeline_start = datetime.now()
        
        try:
            # Extract
            self.logger.info(f"Starting ETL pipeline for {len(symbols)} symbols")
            raw_data = await self.extract_asx_data(symbols, data_types)
            
            # Transform
            self.logger.info("Transforming data")
            transformed_data = self.transform_asx_data(raw_data)
            
            # Load
            self.logger.info("Loading data to destinations")
            self.load_to_destinations(transformed_data)
            
            # Calculate statistics
            pipeline_end = datetime.now()
            duration = (pipeline_end - pipeline_start).total_seconds()
            
            total_records = sum(len(df) for df in transformed_data.values())
            
            results = {
                'status': 'success',
                'duration_seconds': duration,
                'records_processed': total_records,
                'symbols_processed': len(symbols),
                'data_types': list(transformed_data.keys()),
                'timestamp': pipeline_end
            }
            
            self.logger.info(f"ETL pipeline completed successfully in {duration:.2f} seconds")
            return results
            
        except Exception as e:
            self.logger.error(f"ETL pipeline failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'duration_seconds': (datetime.now() - pipeline_start).total_seconds(),
                'timestamp': datetime.now()
            }
```

### 1.2 ASIC Regulatory Data ETL Pipeline

```python
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import zipfile
import requests
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional, Any
import re

class ASICDataETL:
    """
    ETL pipeline for ASIC regulatory data including company extracts,
    annual returns, and regulatory filings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # ASIC API configuration
        self.asic_base_url = "https://download.asic.gov.au"
        self.asic_api_key = config.get('asic_api_key')
        
    async def extract_asic_data(self, company_numbers: List[str] = None,
                               data_types: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Extract regulatory data from ASIC sources.
        
        Args:
            company_numbers: List of ACN numbers
            data_types: Types of data ['company_extract', 'annual_returns', 'officer_details']
            
        Returns:
            Dictionary of extracted DataFrames
        """
        if data_types is None:
            data_types = ['company_extract', 'annual_returns']
        
        extracted_data = {}
        
        for data_type in data_types:
            if data_type == 'company_extract':
                extracted_data['company_extract'] = await self._extract_company_extracts(company_numbers)
            elif data_type == 'annual_returns':
                extracted_data['annual_returns'] = await self._extract_annual_returns(company_numbers)
            elif data_type == 'officer_details':
                extracted_data['officer_details'] = await self._extract_officer_details(company_numbers)
            elif data_type == 'bulk_download':
                extracted_data['bulk_download'] = await self._extract_bulk_data()
        
        return extracted_data
    
    async def _extract_company_extracts(self, company_numbers: List[str]) -> pd.DataFrame:
        """Extract individual company extract information."""
        all_extracts = []
        
        async with aiohttp.ClientSession() as session:
            for acn in company_numbers:
                try:
                    # ASIC Connect API endpoint (example)
                    url = f"{self.asic_base_url}/api/companies/{acn}/extract"
                    
                    headers = {}
                    if self.asic_api_key:
                        headers['Authorization'] = f"Bearer {self.asic_api_key}"
                    
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Parse company extract data
                            extract_data = self._parse_company_extract(data, acn)
                            all_extracts.append(extract_data)
                            
                except Exception as e:
                    self.logger.error(f"Error extracting company extract for {acn}: {e}")
                    continue
        
        if all_extracts:
            return pd.DataFrame(all_extracts)
        
        return pd.DataFrame()
    
    def _parse_company_extract(self, extract_data: Dict, acn: str) -> Dict:
        """Parse ASIC company extract data."""
        parsed_data = {
            'acn': acn,
            'company_name': extract_data.get('companyName', ''),
            'company_type': extract_data.get('companyType', ''),
            'company_status': extract_data.get('status', ''),
            'registration_date': extract_data.get('registrationDate', ''),
            'registered_office_address': self._format_address(extract_data.get('registeredOffice', {})),
            'principal_place_of_business': self._format_address(extract_data.get('principalPlace', {})),
            'share_capital': extract_data.get('shareCapital', {}),
            'annual_return_last_lodged': extract_data.get('annualReturnLastLodged', ''),
            'annual_return_due_date': extract_data.get('annualReturnDueDate', ''),
            'review_date': extract_data.get('reviewDate', ''),
            'next_statement_due': extract_data.get('nextStatementDue', '')
        }
        
        return parsed_data
    
    def _format_address(self, address_data: Dict) -> str:
        """Format address data into a single string."""
        if not address_data:
            return ''
        
        address_parts = []
        
        # Build address string
        if address_data.get('careOf'):
            address_parts.append(f"C/- {address_data['careOf']}")
        
        if address_data.get('unitNumber'):
            address_parts.append(f"Unit {address_data['unitNumber']}")
        
        if address_data.get('streetNumber'):
            address_parts.append(address_data['streetNumber'])
        
        if address_data.get('streetName'):
            address_parts.append(address_data['streetName'])
        
        if address_data.get('streetType'):
            address_parts.append(address_data['streetType'])
        
        if address_data.get('suburb'):
            address_parts.append(address_data['suburb'])
        
        if address_data.get('state'):
            address_parts.append(address_data['state'])
        
        if address_data.get('postcode'):
            address_parts.append(address_data['postcode'])
        
        return ' '.join(address_parts)
    
    async def _extract_bulk_data(self) -> pd.DataFrame:
        """Extract bulk company data from ASIC downloads."""
        try:
            # ASIC provides bulk data files (ZIP format)
            bulk_url = f"{self.asic_base_url}/bulk/companies.zip"
            
            # Download bulk file
            async with aiohttp.ClientSession() as session:
                async with session.get(bulk_url) as response:
                    if response.status == 200:
                        bulk_data = await response.read()
                        
                        # Save to temporary file
                        temp_path = Path("temp_asic_bulk.zip")
                        with open(temp_path, 'wb') as f:
                            f.write(bulk_data)
                        
                        # Extract and parse
                        return self._parse_bulk_zip_file(temp_path)
            
        except Exception as e:
            self.logger.error(f"Error extracting ASIC bulk data: {e}")
        
        return pd.DataFrame()
    
    def _parse_bulk_zip_file(self, zip_path: Path) -> pd.DataFrame:
        """Parse ASIC bulk ZIP file."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # List files in ZIP
                file_list = zip_ref.namelist()
                
                # Find company data file
                company_file = None
                for file_name in file_list:
                    if 'company' in file_name.lower() and file_name.endswith('.txt'):
                        company_file = file_name
                        break
                
                if company_file:
                    # Extract and read file
                    with zip_ref.open(company_file) as f:
                        # ASIC files are typically pipe-delimited
                        df = pd.read_csv(f, delimiter='|', dtype=str)
                        
                        # Standardize column names
                        df.columns = df.columns.str.lower().str.replace(' ', '_')
                        
                        return df
            
        except Exception as e:
            self.logger.error(f"Error parsing ASIC bulk file: {e}")
        
        finally:
            # Clean up temporary file
            if zip_path.exists():
                zip_path.unlink()
        
        return pd.DataFrame()
    
    def transform_asic_data(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Transform ASIC regulatory data for analysis."""
        transformed_data = {}
        
        for data_type, df in raw_data.items():
            if df.empty:
                transformed_data[data_type] = df
                continue
            
            if data_type == 'company_extract':
                transformed_data[data_type] = self._transform_company_extracts(df)
            else:
                transformed_data[data_type] = df
        
        return transformed_data
    
    def _transform_company_extracts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform company extract data."""
        # Convert date columns
        date_columns = ['registration_date', 'annual_return_last_lodged', 
                       'annual_return_due_date', 'review_date', 'next_statement_due']
        
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Calculate derived metrics
        if 'registration_date' in df.columns:
            df['company_age_years'] = (pd.Timestamp.now() - df['registration_date']).dt.days / 365.25
        
        # Standardize company status
        if 'company_status' in df.columns:
            df['is_active'] = df['company_status'].str.upper().eq('REGISTERED')
        
        # Extract state from addresses
        if 'registered_office_address' in df.columns:
            df['registered_state'] = df['registered_office_address'].str.extract(
                r'\\b(NSW|VIC|QLD|WA|SA|TAS|NT|ACT)\\b'
            )[0]
        
        return df
```

## 2. Real-time Streaming Architecture

### 2.1 Market Data Streaming Pipeline

```python
import asyncio
import websockets
import json
from typing import AsyncGenerator, Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from google.cloud import pubsub_v1
import pandas as pd
from kafka import KafkaProducer, KafkaConsumer
import redis
from concurrent.futures import ThreadPoolExecutor

@dataclass
class MarketDataMessage:
    """Structure for market data messages."""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_trade_time: Optional[datetime] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None

class RealTimeMarketDataStreamer:
    """
    Real-time market data streaming using WebSocket connections.
    
    Processes live ASX market data and distributes to various consumers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup streaming infrastructure
        self._setup_message_brokers()
        self._setup_cache()
        
        # Track subscriptions
        self.subscribed_symbols = set()
        self.active_connections = set()
        
        # Message processing
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    def _setup_message_brokers(self):
        """Setup message brokers for streaming data distribution."""
        # Kafka producer for high-throughput streaming
        kafka_config = self.config.get('kafka', {})
        if kafka_config:
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=kafka_config.get('bootstrap_servers', ['localhost:9092']),
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                batch_size=16384,
                linger_ms=10
            )
        
        # Google Pub/Sub for cloud-native streaming
        gcp_config = self.config.get('gcp', {})
        if gcp_config.get('project_id'):
            self.publisher = pubsub_v1.PublisherClient()
            self.topic_path = self.publisher.topic_path(
                gcp_config['project_id'], 
                'asx-market-data-stream'
            )
    
    def _setup_cache(self):
        """Setup Redis cache for latest market data."""
        cache_config = self.config.get('cache', {})
        self.redis_client = redis.Redis(
            host=cache_config.get('host', 'localhost'),
            port=cache_config.get('port', 6379),
            db=cache_config.get('db', 0),
            decode_responses=True
        )
    
    async def connect_to_market_feed(self, symbols: List[str]):
        """Connect to market data WebSocket feed."""
        websocket_config = self.config.get('websocket', {})
        ws_url = websocket_config.get('url', 'wss://api.asx.com.au/stream')
        
        try:
            async with websockets.connect(ws_url) as websocket:
                self.active_connections.add(websocket)
                
                # Subscribe to symbols
                await self._subscribe_to_symbols(websocket, symbols)
                
                # Start message processing loop
                await self._process_websocket_messages(websocket)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
        finally:
            self.active_connections.discard(websocket)
    
    async def _subscribe_to_symbols(self, websocket, symbols: List[str]):
        """Subscribe to market data for specific symbols."""
        subscription_message = {
            'action': 'subscribe',
            'symbols': symbols,
            'data_types': ['trades', 'quotes', 'orderbook']
        }
        
        await websocket.send(json.dumps(subscription_message))
        self.subscribed_symbols.update(symbols)
        
        self.logger.info(f"Subscribed to {len(symbols)} symbols")
    
    async def _process_websocket_messages(self, websocket):
        """Process incoming WebSocket messages."""
        async for message in websocket:
            try:
                data = json.loads(message)
                
                # Process message based on type
                if data.get('type') == 'trade':
                    await self._process_trade_message(data)
                elif data.get('type') == 'quote':
                    await self._process_quote_message(data)
                elif data.get('type') == 'orderbook':
                    await self._process_orderbook_message(data)
                
            except json.JSONDecodeError:
                self.logger.warning(f"Invalid JSON message: {message}")
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
    
    async def _process_trade_message(self, data: Dict):
        """Process trade data message."""
        try:
            market_data = MarketDataMessage(
                symbol=data['symbol'],
                price=float(data['price']),
                volume=int(data['volume']),
                timestamp=pd.to_datetime(data['timestamp']),
                last_trade_time=pd.to_datetime(data.get('trade_time', data['timestamp']))
            )
            
            # Distribute to various channels
            await self._distribute_market_data(market_data, 'trade')
            
        except Exception as e:
            self.logger.error(f"Error processing trade message: {e}")
    
    async def _process_quote_message(self, data: Dict):
        """Process quote (bid/ask) data message."""
        try:
            market_data = MarketDataMessage(
                symbol=data['symbol'],
                price=float(data.get('mid_price', (data.get('bid', 0) + data.get('ask', 0)) / 2)),
                volume=0,  # No volume for quotes
                timestamp=pd.to_datetime(data['timestamp']),
                bid=float(data.get('bid', 0)),
                ask=float(data.get('ask', 0))
            )
            
            await self._distribute_market_data(market_data, 'quote')
            
        except Exception as e:
            self.logger.error(f"Error processing quote message: {e}")
    
    async def _distribute_market_data(self, market_data: MarketDataMessage, data_type: str):
        """Distribute market data to various consumers."""
        # Convert to dictionary for serialization
        data_dict = asdict(market_data)
        data_dict['data_type'] = data_type
        
        # Update cache with latest data
        cache_key = f"market_data:{market_data.symbol}"
        self.redis_client.setex(
            cache_key,
            300,  # 5 minute TTL
            json.dumps(data_dict, default=str)
        )
        
        # Publish to Kafka
        if hasattr(self, 'kafka_producer'):
            self.kafka_producer.send(
                'asx-market-data',
                key=market_data.symbol.encode('utf-8'),
                value=data_dict
            )
        
        # Publish to Pub/Sub
        if hasattr(self, 'publisher'):
            message_json = json.dumps(data_dict, default=str)
            message_bytes = message_json.encode('utf-8')
            
            future = self.publisher.publish(self.topic_path, message_bytes)
            # Don't wait for result to maintain real-time performance
        
        # Trigger real-time analytics
        await self._trigger_real_time_analytics(market_data)
    
    async def _trigger_real_time_analytics(self, market_data: MarketDataMessage):
        """Trigger real-time analytics and alerts."""
        # Check for significant price movements
        await self._check_price_alerts(market_data)
        
        # Update real-time valuation models
        await self._update_valuation_models(market_data)
    
    async def _check_price_alerts(self, market_data: MarketDataMessage):
        """Check for price alerts and significant movements."""
        # Get previous price from cache
        prev_key = f"prev_price:{market_data.symbol}"
        prev_price_str = self.redis_client.get(prev_key)
        
        if prev_price_str:
            prev_price = float(prev_price_str)
            change_percent = ((market_data.price - prev_price) / prev_price) * 100
            
            # Check for significant movements (>5%)
            if abs(change_percent) > 5:
                alert = {
                    'symbol': market_data.symbol,
                    'current_price': market_data.price,
                    'previous_price': prev_price,
                    'change_percent': change_percent,
                    'timestamp': market_data.timestamp,
                    'alert_type': 'significant_movement'
                }
                
                # Send alert (could be email, webhook, etc.)
                await self._send_alert(alert)
        
        # Update previous price
        self.redis_client.setex(prev_key, 3600, str(market_data.price))  # 1 hour TTL
    
    async def _send_alert(self, alert: Dict):
        """Send alert notification."""
        self.logger.warning(f"Price alert: {alert}")
        
        # Could implement email, Slack, webhook notifications here
        # For now, just log the alert
```

This comprehensive integration patterns documentation provides the foundation for connecting with ASX and ASIC data sources. The next sections will complete the data processing documentation with configuration templates and best practices.