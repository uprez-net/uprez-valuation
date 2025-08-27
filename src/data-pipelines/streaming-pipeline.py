"""
Streaming Data Pipeline for IPO Valuation Platform
Handles real-time market data processing using Apache Beam on Google Cloud Dataflow
"""

import logging
import json
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms.window import FixedWindows, SlidingWindows
from apache_beam.transforms.trigger import AfterWatermark, AfterProcessingTime, AfterCount
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import requests

logger = logging.getLogger(__name__)

class MarketDataTransform(beam.DoFn):
    """Transform raw market data into structured format"""
    
    def process(self, element: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            # Parse incoming market data
            if isinstance(element, str):
                data = json.loads(element)
            else:
                data = element
            
            # Validate required fields
            required_fields = ['symbol', 'price', 'volume', 'timestamp']
            if not all(field in data for field in required_fields):
                logger.warning(f"Missing required fields in data: {data}")
                return []
            
            # Transform data
            transformed = {
                'symbol': data['symbol'],
                'price': float(data['price']),
                'volume': int(data['volume']),
                'timestamp': data['timestamp'],
                'market_cap': float(data.get('market_cap', 0)),
                'pe_ratio': float(data.get('pe_ratio', 0)),
                'beta': float(data.get('beta', 1.0)),
                'sector': data.get('sector', 'Unknown'),
                'processed_timestamp': datetime.utcnow().isoformat()
            }
            
            # Calculate derived metrics
            transformed['price_change_pct'] = self._calculate_price_change(data)
            transformed['volatility_indicator'] = self._calculate_volatility(data)
            transformed['trading_activity'] = self._classify_trading_activity(transformed['volume'])
            
            yield transformed
            
        except Exception as e:
            logger.error(f"Error transforming market data: {str(e)}")
            return []
    
    def _calculate_price_change(self, data: Dict[str, Any]) -> float:
        """Calculate price change percentage"""
        current_price = float(data['price'])
        previous_price = float(data.get('previous_close', current_price))
        
        if previous_price > 0:
            return ((current_price - previous_price) / previous_price) * 100
        return 0.0
    
    def _calculate_volatility(self, data: Dict[str, Any]) -> str:
        """Calculate volatility indicator"""
        price_change = abs(self._calculate_price_change(data))
        
        if price_change > 10:
            return 'HIGH'
        elif price_change > 5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _classify_trading_activity(self, volume: int) -> str:
        """Classify trading activity level"""
        if volume > 10000000:
            return 'VERY_HIGH'
        elif volume > 5000000:
            return 'HIGH'
        elif volume > 1000000:
            return 'MEDIUM'
        else:
            return 'LOW'

class IPOAnalysisTransform(beam.DoFn):
    """Analyze IPO-specific metrics and signals"""
    
    def __init__(self, ipo_companies: List[str]):
        self.ipo_companies = ipo_companies
    
    def process(self, element: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            symbol = element['symbol']
            
            # Only process IPO companies
            if symbol not in self.ipo_companies:
                return []
            
            # Calculate IPO-specific metrics
            ipo_analysis = {
                'symbol': symbol,
                'price': element['price'],
                'volume': element['volume'],
                'timestamp': element['timestamp'],
                'market_sentiment': self._analyze_market_sentiment(element),
                'valuation_signal': self._generate_valuation_signal(element),
                'risk_level': self._assess_risk_level(element),
                'institutional_interest': self._estimate_institutional_interest(element),
                'retail_interest': self._estimate_retail_interest(element)
            }
            
            yield ipo_analysis
            
        except Exception as e:
            logger.error(f"Error in IPO analysis: {str(e)}")
            return []
    
    def _analyze_market_sentiment(self, data: Dict[str, Any]) -> str:
        """Analyze market sentiment based on price and volume"""
        price_change = float(data.get('price_change_pct', 0))
        volume = int(data['volume'])
        
        # Simple sentiment analysis
        if price_change > 5 and volume > 5000000:
            return 'VERY_POSITIVE'
        elif price_change > 2:
            return 'POSITIVE'
        elif price_change < -5 and volume > 5000000:
            return 'VERY_NEGATIVE'
        elif price_change < -2:
            return 'NEGATIVE'
        else:
            return 'NEUTRAL'
    
    def _generate_valuation_signal(self, data: Dict[str, Any]) -> str:
        """Generate valuation signal"""
        pe_ratio = float(data.get('pe_ratio', 0))
        price_change = float(data.get('price_change_pct', 0))
        
        # Simple valuation logic
        if pe_ratio > 0 and pe_ratio < 15 and price_change < -2:
            return 'UNDERVALUED'
        elif pe_ratio > 50 or price_change > 10:
            return 'OVERVALUED'
        else:
            return 'FAIRLY_VALUED'
    
    def _assess_risk_level(self, data: Dict[str, Any]) -> str:
        """Assess risk level"""
        volatility = data.get('volatility_indicator', 'LOW')
        beta = float(data.get('beta', 1.0))
        
        if volatility == 'HIGH' or beta > 2.0:
            return 'HIGH'
        elif volatility == 'MEDIUM' or beta > 1.5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _estimate_institutional_interest(self, data: Dict[str, Any]) -> float:
        """Estimate institutional interest level"""
        volume = int(data['volume'])
        trading_activity = data.get('trading_activity', 'LOW')
        
        # Institutional trading typically involves large volumes
        if trading_activity in ['VERY_HIGH', 'HIGH'] and volume > 10000000:
            return 0.8
        elif trading_activity == 'MEDIUM':
            return 0.5
        else:
            return 0.2
    
    def _estimate_retail_interest(self, data: Dict[str, Any]) -> float:
        """Estimate retail interest level"""
        price_change = abs(float(data.get('price_change_pct', 0)))
        
        # Retail interest often correlates with high volatility
        if price_change > 10:
            return 0.9
        elif price_change > 5:
            return 0.7
        else:
            return 0.3

class AggregateMarketMetrics(beam.CombineFn):
    """Aggregate market metrics over time windows"""
    
    def create_accumulator(self) -> Dict[str, Any]:
        return {
            'prices': [],
            'volumes': [],
            'symbols': set(),
            'total_trades': 0,
            'sentiment_counts': {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
        }
    
    def add_input(self, accumulator: Dict[str, Any], input: Dict[str, Any]) -> Dict[str, Any]:
        accumulator['prices'].append(input['price'])
        accumulator['volumes'].append(input['volume'])
        accumulator['symbols'].add(input['symbol'])
        accumulator['total_trades'] += 1
        
        sentiment = input.get('market_sentiment', 'NEUTRAL')
        if sentiment in accumulator['sentiment_counts']:
            accumulator['sentiment_counts'][sentiment] += 1
        
        return accumulator
    
    def merge_accumulators(self, accumulators: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged = self.create_accumulator()
        
        for acc in accumulators:
            merged['prices'].extend(acc['prices'])
            merged['volumes'].extend(acc['volumes'])
            merged['symbols'].update(acc['symbols'])
            merged['total_trades'] += acc['total_trades']
            
            for sentiment, count in acc['sentiment_counts'].items():
                merged['sentiment_counts'][sentiment] += count
        
        return merged
    
    def extract_output(self, accumulator: Dict[str, Any]) -> Dict[str, Any]:
        prices = accumulator['prices']
        volumes = accumulator['volumes']
        
        if not prices:
            return {}
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'avg_price': sum(prices) / len(prices),
            'price_volatility': np.std(prices) if len(prices) > 1 else 0,
            'total_volume': sum(volumes),
            'avg_volume': sum(volumes) / len(volumes),
            'unique_symbols': len(accumulator['symbols']),
            'total_trades': accumulator['total_trades'],
            'market_sentiment_distribution': accumulator['sentiment_counts'],
            'dominant_sentiment': max(accumulator['sentiment_counts'], key=accumulator['sentiment_counts'].get)
        }

class EnrichWithExternalData(beam.DoFn):
    """Enrich data with external market indicators"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.cache = {}
        self.cache_expiry = {}
    
    def process(self, element: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            # Get market indicators
            market_data = self._get_market_indicators()
            
            # Enrich element with external data
            enriched = element.copy()
            enriched.update({
                'market_indices': market_data.get('indices', {}),
                'vix_level': market_data.get('vix', 0),
                'interest_rates': market_data.get('rates', {}),
                'economic_indicators': market_data.get('economic', {}),
                'enrichment_timestamp': datetime.utcnow().isoformat()
            })
            
            yield enriched
            
        except Exception as e:
            logger.error(f"Error enriching data: {str(e)}")
            # Return original element if enrichment fails
            yield element
    
    def _get_market_indicators(self) -> Dict[str, Any]:
        """Get market indicators from external APIs"""
        cache_key = 'market_indicators'
        now = datetime.utcnow()
        
        # Check cache
        if (cache_key in self.cache and 
            cache_key in self.cache_expiry and 
            now < self.cache_expiry[cache_key]):
            return self.cache[cache_key]
        
        try:
            # Mock external API call (replace with actual API calls)
            market_data = {
                'indices': {
                    'sp500': 4500.0,
                    'nasdaq': 15000.0,
                    'dow': 35000.0
                },
                'vix': 20.5,
                'rates': {
                    '10y_treasury': 4.2,
                    'fed_funds': 5.25
                },
                'economic': {
                    'unemployment': 3.7,
                    'inflation': 3.2
                }
            }
            
            # Cache for 5 minutes
            self.cache[cache_key] = market_data
            self.cache_expiry[cache_key] = now + timedelta(minutes=5)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market indicators: {str(e)}")
            return {}

def create_streaming_pipeline(pipeline_options: PipelineOptions) -> beam.Pipeline:
    """Create the streaming data pipeline"""
    
    # IPO companies to track (would be loaded from configuration)
    ipo_companies = ['NVDA', 'TSLA', 'AAPL', 'GOOGL', 'META']  # Example
    
    pipeline = beam.Pipeline(options=pipeline_options)
    
    # Read from Pub/Sub
    market_data = (
        pipeline
        | 'Read Market Data' >> beam.io.ReadFromPubSub(
            subscription='projects/PROJECT_ID/subscriptions/market-data-subscription'
        )
        | 'Parse JSON' >> beam.Map(json.loads)
    )
    
    # Transform and analyze data
    transformed_data = (
        market_data
        | 'Transform Market Data' >> beam.ParDo(MarketDataTransform())
        | 'Enrich with External Data' >> beam.ParDo(EnrichWithExternalData())
    )
    
    # IPO-specific analysis
    ipo_analysis = (
        transformed_data
        | 'IPO Analysis' >> beam.ParDo(IPOAnalysisTransform(ipo_companies))
    )
    
    # Real-time aggregations (5-minute windows)
    market_aggregates = (
        transformed_data
        | 'Window 5min' >> beam.WindowInto(
            FixedWindows(5 * 60),  # 5 minutes
            trigger=AfterWatermark(
                early=AfterProcessingTime(60)  # Early trigger every minute
            ),
            accumulation_mode=beam.transforms.trigger.AccumulationMode.DISCARDING
        )
        | 'Aggregate Metrics' >> beam.CombineGlobally(AggregateMarketMetrics()).without_defaults()
    )
    
    # Write to BigQuery - Raw Data
    transformed_data | 'Write Raw Data to BigQuery' >> beam.io.WriteToBigQuery(
        table='PROJECT_ID:ipo_valuation.market_data_realtime',
        schema={
            'fields': [
                {'name': 'symbol', 'type': 'STRING', 'mode': 'REQUIRED'},
                {'name': 'price', 'type': 'FLOAT', 'mode': 'REQUIRED'},
                {'name': 'volume', 'type': 'INTEGER', 'mode': 'REQUIRED'},
                {'name': 'timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'},
                {'name': 'market_cap', 'type': 'FLOAT', 'mode': 'NULLABLE'},
                {'name': 'pe_ratio', 'type': 'FLOAT', 'mode': 'NULLABLE'},
                {'name': 'beta', 'type': 'FLOAT', 'mode': 'NULLABLE'},
                {'name': 'sector', 'type': 'STRING', 'mode': 'NULLABLE'},
                {'name': 'price_change_pct', 'type': 'FLOAT', 'mode': 'NULLABLE'},
                {'name': 'volatility_indicator', 'type': 'STRING', 'mode': 'NULLABLE'},
                {'name': 'trading_activity', 'type': 'STRING', 'mode': 'NULLABLE'},
                {'name': 'market_indices', 'type': 'JSON', 'mode': 'NULLABLE'},
                {'name': 'processed_timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'}
            ]
        },
        write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
        create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
    )
    
    # Write to BigQuery - IPO Analysis
    ipo_analysis | 'Write IPO Analysis to BigQuery' >> beam.io.WriteToBigQuery(
        table='PROJECT_ID:ipo_valuation.ipo_analysis_realtime',
        schema={
            'fields': [
                {'name': 'symbol', 'type': 'STRING', 'mode': 'REQUIRED'},
                {'name': 'price', 'type': 'FLOAT', 'mode': 'REQUIRED'},
                {'name': 'volume', 'type': 'INTEGER', 'mode': 'REQUIRED'},
                {'name': 'timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'},
                {'name': 'market_sentiment', 'type': 'STRING', 'mode': 'NULLABLE'},
                {'name': 'valuation_signal', 'type': 'STRING', 'mode': 'NULLABLE'},
                {'name': 'risk_level', 'type': 'STRING', 'mode': 'NULLABLE'},
                {'name': 'institutional_interest', 'type': 'FLOAT', 'mode': 'NULLABLE'},
                {'name': 'retail_interest', 'type': 'FLOAT', 'mode': 'NULLABLE'}
            ]
        },
        write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
        create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
    )
    
    # Write to BigQuery - Aggregated Metrics
    market_aggregates | 'Write Aggregates to BigQuery' >> beam.io.WriteToBigQuery(
        table='PROJECT_ID:ipo_valuation.market_aggregates_realtime',
        schema={
            'fields': [
                {'name': 'timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'},
                {'name': 'avg_price', 'type': 'FLOAT', 'mode': 'NULLABLE'},
                {'name': 'price_volatility', 'type': 'FLOAT', 'mode': 'NULLABLE'},
                {'name': 'total_volume', 'type': 'INTEGER', 'mode': 'NULLABLE'},
                {'name': 'avg_volume', 'type': 'FLOAT', 'mode': 'NULLABLE'},
                {'name': 'unique_symbols', 'type': 'INTEGER', 'mode': 'NULLABLE'},
                {'name': 'total_trades', 'type': 'INTEGER', 'mode': 'NULLABLE'},
                {'name': 'market_sentiment_distribution', 'type': 'JSON', 'mode': 'NULLABLE'},
                {'name': 'dominant_sentiment', 'type': 'STRING', 'mode': 'NULLABLE'}
            ]
        },
        write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
        create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
    )
    
    # Write alerts to Pub/Sub
    alerts = (
        ipo_analysis
        | 'Generate Alerts' >> beam.ParDo(GenerateAlerts())
        | 'Filter Alerts' >> beam.Filter(lambda x: x is not None)
        | 'Format Alert Messages' >> beam.Map(json.dumps)
    )
    
    alerts | 'Write Alerts' >> beam.io.WriteToPubSub(
        topic='projects/PROJECT_ID/topics/valuation-alerts'
    )
    
    return pipeline

class GenerateAlerts(beam.DoFn):
    """Generate alerts for significant market events"""
    
    def process(self, element: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            alerts = []
            
            # Price movement alerts
            symbol = element['symbol']
            price = element['price']
            volume = element['volume']
            sentiment = element.get('market_sentiment', 'NEUTRAL')
            risk_level = element.get('risk_level', 'LOW')
            
            # High volatility alert
            if sentiment in ['VERY_POSITIVE', 'VERY_NEGATIVE']:
                alerts.append({
                    'alert_type': 'HIGH_VOLATILITY',
                    'symbol': symbol,
                    'price': price,
                    'sentiment': sentiment,
                    'timestamp': datetime.utcnow().isoformat(),
                    'priority': 'HIGH' if risk_level == 'HIGH' else 'MEDIUM'
                })
            
            # Volume spike alert
            if volume > 20000000:  # 20M+ volume
                alerts.append({
                    'alert_type': 'VOLUME_SPIKE',
                    'symbol': symbol,
                    'volume': volume,
                    'timestamp': datetime.utcnow().isoformat(),
                    'priority': 'MEDIUM'
                })
            
            # Valuation signal alert
            valuation_signal = element.get('valuation_signal', 'FAIRLY_VALUED')
            if valuation_signal in ['UNDERVALUED', 'OVERVALUED']:
                alerts.append({
                    'alert_type': 'VALUATION_SIGNAL',
                    'symbol': symbol,
                    'signal': valuation_signal,
                    'timestamp': datetime.utcnow().isoformat(),
                    'priority': 'LOW'
                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating alerts: {str(e)}")
            return []

def run_streaming_pipeline():
    """Run the streaming pipeline"""
    
    # Pipeline options
    options = PipelineOptions([
        '--project=PROJECT_ID',
        '--region=us-central1',
        '--runner=DataflowRunner',
        '--streaming',
        '--enable_streaming_engine',
        '--num_workers=5',
        '--max_num_workers=20',
        '--worker_machine_type=n1-standard-2',
        '--disk_size_gb=50',
        '--use_public_ips=false',
        '--network=projects/PROJECT_ID/global/networks/dataflow-network',
        '--subnetwork=projects/PROJECT_ID/regions/us-central1/subnetworks/dataflow-subnet',
        '--service_account_email=dataflow-service@PROJECT_ID.iam.gserviceaccount.com'
    ])
    
    # Create and run pipeline
    pipeline = create_streaming_pipeline(options)
    result = pipeline.run()
    
    if options.view_as(beam.options.pipeline_options.StandardOptions).runner == 'DirectRunner':
        result.wait_until_finish()
    
    return result

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_streaming_pipeline()