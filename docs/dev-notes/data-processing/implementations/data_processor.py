"""
Comprehensive Data Processing Implementation

This module provides a unified interface for all data processing operations
in the IPO valuation platform, integrating financial data preprocessing,
document processing, feature engineering, and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import asyncio
import yaml
from pathlib import Path
import warnings

# Import custom modules (these would be in separate files)
from .financial_preprocessor import FinancialDataCleaner, FinancialImputer, FinancialNormalizer
from .document_processor import PDFProcessor, OCRErrorCorrector, MultiLanguageProcessor
from .feature_engineer import FinancialRatioCalculator, TechnicalIndicatorCalculator, TimeBasedFeatureEngineer
from .text_processor import FinancialTfIdfVectorizer, FinancialPhraseAnalyzer, FinancialWordEmbeddings
from .validator import FinancialDataValidator, StatisticalIntegrityTester
from .integrations import ASXDataETL, ASICDataETL, RealTimeMarketDataStreamer

@dataclass
class ProcessingConfig:
    """Configuration for data processing pipeline."""
    
    # Financial data processing
    financial_preprocessing: Dict = field(default_factory=dict)
    feature_engineering: Dict = field(default_factory=dict)
    validation: Dict = field(default_factory=dict)
    
    # Document processing
    document_processing: Dict = field(default_factory=dict)
    text_processing: Dict = field(default_factory=dict)
    
    # Integration settings
    data_sources: Dict = field(default_factory=dict)
    
    # Output settings
    output: Dict = field(default_factory=dict)

class IPODataProcessor:
    """
    Unified data processor for IPO valuation platform.
    
    Integrates all data processing components into a single, easy-to-use interface.
    """
    
    def __init__(self, config_path: str = None, config_dict: Dict = None):
        """
        Initialize data processor with configuration.
        
        Args:
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (alternative to file)
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config_path:
            self.config = self._load_config_from_file(config_path)
        elif config_dict:
            self.config = ProcessingConfig(**config_dict)
        else:
            self.config = self._default_config()
        
        # Initialize processors
        self._initialize_processors()
        
        # Processing history
        self.processing_history = []
        
    def _load_config_from_file(self, config_path: str) -> ProcessingConfig:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            return ProcessingConfig(**config_data)
        except Exception as e:
            self.logger.warning(f"Failed to load config from {config_path}: {e}")
            return self._default_config()
    
    def _default_config(self) -> ProcessingConfig:
        """Create default configuration."""
        return ProcessingConfig(
            financial_preprocessing={
                'handle_missing': 'interpolate',
                'outlier_detection_methods': ['zscore', 'iqr'],
                'normalization_method': 'robust'
            },
            feature_engineering={
                'include_technical_indicators': True,
                'include_ratios': True,
                'include_temporal_features': True
            },
            validation={
                'run_statistical_tests': True,
                'significance_level': 0.05
            },
            document_processing={
                'ocr_correction': True,
                'multilang_support': True
            },
            text_processing={
                'tfidf_max_features': 5000,
                'create_embeddings': True
            }
        )
    
    def _initialize_processors(self):
        """Initialize all processing components."""
        # Financial data processors
        self.financial_cleaner = FinancialDataCleaner()
        self.financial_imputer = FinancialImputer()
        self.financial_normalizer = FinancialNormalizer()
        
        # Feature engineering
        self.ratio_calculator = FinancialRatioCalculator()
        self.technical_calculator = TechnicalIndicatorCalculator()
        self.temporal_engineer = TimeBasedFeatureEngineer()
        
        # Document processors
        self.pdf_processor = PDFProcessor()
        self.ocr_corrector = OCRErrorCorrector()
        self.multilang_processor = MultiLanguageProcessor()
        
        # Text processors
        self.tfidf_vectorizer = FinancialTfIdfVectorizer()
        self.phrase_analyzer = FinancialPhraseAnalyzer()
        self.embeddings_processor = FinancialWordEmbeddings()
        
        # Validators
        self.data_validator = FinancialDataValidator()
        self.statistical_tester = StatisticalIntegrityTester()
        
        # Data integrations
        if self.config.data_sources:
            self.asx_etl = ASXDataETL(self.config.data_sources)
            self.asic_etl = ASICDataETL(self.config.data_sources)
    
    def process_financial_data(self, data: pd.DataFrame, 
                             processing_steps: List[str] = None) -> Dict[str, Any]:
        """
        Process financial data through the complete pipeline.
        
        Args:
            data: Raw financial data DataFrame
            processing_steps: Specific steps to run (None = all steps)
            
        Returns:
            Dictionary with processed data and metadata
        """
        if processing_steps is None:
            processing_steps = [
                'clean_data',
                'handle_missing',
                'calculate_ratios',
                'technical_indicators',
                'temporal_features',
                'normalize',
                'validate'
            ]
        
        processing_start = datetime.now()
        results = {'original_data': data.copy()}
        
        try:
            processed_data = data.copy()
            
            # Data cleaning
            if 'clean_data' in processing_steps:
                processed_data = self.financial_cleaner.clean_price_data(processed_data)
                self.logger.info("Completed data cleaning")
            
            # Handle missing values
            if 'handle_missing' in processing_steps:
                processed_data = self.financial_imputer.impute_missing_values(
                    processed_data, 
                    method=self.config.financial_preprocessing.get('handle_missing', 'interpolate')
                )
                self.logger.info("Completed missing value imputation")
            
            # Calculate financial ratios
            if 'calculate_ratios' in processing_steps:
                processed_data = self.ratio_calculator.calculate_all_ratios(processed_data)
                self.logger.info("Completed ratio calculations")
            
            # Technical indicators
            if 'technical_indicators' in processing_steps and self.config.feature_engineering.get('include_technical_indicators'):
                if all(col in processed_data.columns for col in ['open', 'high', 'low', 'close']):
                    processed_data = self.technical_calculator.calculate_all_indicators(processed_data)
                    self.logger.info("Completed technical indicator calculations")
            
            # Temporal features
            if 'temporal_features' in processing_steps and self.config.feature_engineering.get('include_temporal_features'):
                numeric_cols = processed_data.select_dtypes(include=[np.number]).columns.tolist()[:10]  # Limit for performance
                processed_data = self.temporal_engineer.create_temporal_features(
                    processed_data, target_columns=numeric_cols
                )
                self.logger.info("Completed temporal feature engineering")
            
            # Normalization
            if 'normalize' in processing_steps:
                processed_data = self.financial_normalizer.normalize_financial_data(
                    processed_data,
                    method=self.config.financial_preprocessing.get('normalization_method', 'robust')
                )
                self.logger.info("Completed data normalization")
            
            # Validation
            if 'validate' in processing_steps:
                validation_results = self.data_validator.validate_dataframe(processed_data)
                results['validation'] = validation_results
                self.logger.info("Completed data validation")
            
            # Statistical integrity tests
            if self.config.validation.get('run_statistical_tests'):
                numeric_cols = processed_data.select_dtypes(include=[np.number]).columns.tolist()[:5]  # Limit for performance
                statistical_results = self.statistical_tester.run_comprehensive_tests(
                    processed_data, numeric_cols
                )
                results['statistical_tests'] = statistical_results
                self.logger.info("Completed statistical integrity tests")
            
            results['processed_data'] = processed_data
            results['processing_steps'] = processing_steps
            results['processing_duration'] = (datetime.now() - processing_start).total_seconds()
            results['success'] = True
            
        except Exception as e:
            self.logger.error(f"Error in financial data processing: {e}")
            results['error'] = str(e)
            results['success'] = False
        
        # Store in history
        self.processing_history.append({
            'timestamp': processing_start,
            'data_type': 'financial',
            'success': results.get('success', False),
            'duration': results.get('processing_duration', 0)
        })
        
        return results
    
    def process_documents(self, document_paths: List[str],
                         processing_options: Dict = None) -> Dict[str, Any]:
        """
        Process documents (PDFs, HTML, etc.) into structured data.
        
        Args:
            document_paths: List of document file paths
            processing_options: Additional processing options
            
        Returns:
            Dictionary with processed document data
        """
        processing_start = datetime.now()
        options = processing_options or {}
        
        results = {
            'documents_processed': 0,
            'text_content': [],
            'tables': [],
            'metadata': []
        }
        
        try:
            for doc_path in document_paths:
                doc_path = Path(doc_path)
                
                if not doc_path.exists():
                    self.logger.warning(f"Document not found: {doc_path}")
                    continue
                
                if doc_path.suffix.lower() == '.pdf':
                    # Process PDF
                    pdf_results = self.pdf_processor.extract_text_comprehensive(str(doc_path))
                    
                    # OCR correction if enabled
                    if self.config.document_processing.get('ocr_correction'):
                        corrected_results = self.ocr_corrector.correct_ocr_text(pdf_results['text'])
                        pdf_results['corrected_text'] = corrected_results['corrected_text']
                    
                    results['text_content'].append(pdf_results['text'])
                    results['tables'].extend(pdf_results['tables'])
                    results['metadata'].append(pdf_results['metadata'])
                    
                results['documents_processed'] += 1
                self.logger.info(f"Processed document: {doc_path.name}")
            
            results['processing_duration'] = (datetime.now() - processing_start).total_seconds()
            results['success'] = True
            
        except Exception as e:
            self.logger.error(f"Error in document processing: {e}")
            results['error'] = str(e)
            results['success'] = False
        
        # Store in history
        self.processing_history.append({
            'timestamp': processing_start,
            'data_type': 'documents',
            'success': results.get('success', False),
            'duration': results.get('processing_duration', 0)
        })
        
        return results
    
    def create_text_features(self, text_documents: List[str],
                           feature_types: List[str] = None) -> Dict[str, Any]:
        """
        Create text features from documents.
        
        Args:
            text_documents: List of text documents
            feature_types: Types of features to create ['tfidf', 'phrases', 'embeddings']
            
        Returns:
            Dictionary with text features
        """
        if feature_types is None:
            feature_types = ['tfidf', 'phrases']
        
        processing_start = datetime.now()
        results = {}
        
        try:
            # TF-IDF features
            if 'tfidf' in feature_types:
                tfidf_config = {
                    'max_features': self.config.text_processing.get('tfidf_max_features', 5000)
                }
                tfidf_features = self.tfidf_vectorizer.create_financial_tfidf_features(
                    text_documents, tfidf_config
                )
                results['tfidf_features'] = tfidf_features
                self.logger.info("Created TF-IDF features")
            
            # Phrase analysis
            if 'phrases' in feature_types:
                phrase_features = self.phrase_analyzer.create_phrase_features(text_documents)
                results['phrase_features'] = phrase_features
                self.logger.info("Created phrase features")
            
            # Word embeddings
            if 'embeddings' in feature_types and self.config.text_processing.get('create_embeddings'):
                embedding_results = self.embeddings_processor.train_financial_embeddings(text_documents)
                doc_embeddings = self.embeddings_processor.create_document_embeddings(text_documents)
                results['embeddings'] = {
                    'model_info': embedding_results,
                    'document_vectors': doc_embeddings
                }
                self.logger.info("Created word embeddings")
            
            results['processing_duration'] = (datetime.now() - processing_start).total_seconds()
            results['success'] = True
            
        except Exception as e:
            self.logger.error(f"Error in text feature creation: {e}")
            results['error'] = str(e)
            results['success'] = False
        
        return results
    
    def create_feature_matrix(self, financial_data: pd.DataFrame,
                            document_features: Dict = None,
                            text_features: Dict = None) -> pd.DataFrame:
        """
        Create unified feature matrix combining all feature types.
        
        Args:
            financial_data: Processed financial data
            document_features: Features from document processing
            text_features: Features from text processing
            
        Returns:
            Combined feature matrix DataFrame
        """
        try:
            feature_matrix = financial_data.copy()
            
            # Add document-based features
            if document_features:
                # Simple document count features
                feature_matrix['document_count'] = len(document_features.get('text_content', []))
                feature_matrix['table_count'] = len(document_features.get('tables', []))
            
            # Add text features
            if text_features:
                if 'tfidf_features' in text_features:
                    tfidf_df = text_features['tfidf_features']
                    if len(tfidf_df) == len(feature_matrix):
                        feature_matrix = pd.concat([feature_matrix, tfidf_df], axis=1)
                
                if 'phrase_features' in text_features:
                    phrase_df = text_features['phrase_features']
                    if len(phrase_df) == len(feature_matrix):
                        feature_matrix = pd.concat([feature_matrix, phrase_df], axis=1)
                
                if 'embeddings' in text_features:
                    embeddings = text_features['embeddings']['document_vectors']
                    if len(embeddings) == len(feature_matrix):
                        # Add embedding dimensions as features
                        embedding_df = pd.DataFrame(
                            embeddings,
                            columns=[f'embedding_{i}' for i in range(embeddings.shape[1])]
                        )
                        feature_matrix = pd.concat([feature_matrix, embedding_df], axis=1)
            
            self.logger.info(f"Created feature matrix with {len(feature_matrix.columns)} features")
            return feature_matrix
            
        except Exception as e:
            self.logger.error(f"Error creating feature matrix: {e}")
            return financial_data
    
    async def stream_market_data(self, symbols: List[str]) -> AsyncGenerator[Dict, None]:
        """
        Stream real-time market data.
        
        Args:
            symbols: List of symbols to stream
            
        Yields:
            Market data dictionaries
        """
        if not hasattr(self, 'market_streamer'):
            # Initialize streamer if not already done
            streamer_config = self.config.data_sources.get('streaming', {})
            self.market_streamer = RealTimeMarketDataStreamer(streamer_config)
        
        # This would be implemented based on the specific streaming setup
        # For now, return a placeholder
        yield {'message': 'Streaming functionality requires additional setup'}
    
    def export_processed_data(self, processed_data: Dict[str, Any],
                            output_path: str, format: str = 'parquet') -> bool:
        """
        Export processed data to various formats.
        
        Args:
            processed_data: Dictionary containing processed data
            output_path: Path for output file
            format: Output format ('parquet', 'csv', 'json', 'hdf5')
            
        Returns:
            Success status
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if 'processed_data' in processed_data:
                df = processed_data['processed_data']
                
                if format == 'parquet':
                    df.to_parquet(output_path)
                elif format == 'csv':
                    df.to_csv(output_path, index=False)
                elif format == 'json':
                    df.to_json(output_path, orient='records', date_format='iso')
                elif format == 'hdf5':
                    df.to_hdf(output_path, key='data', mode='w')
                else:
                    raise ValueError(f"Unsupported format: {format}")
                
                self.logger.info(f"Exported processed data to {output_path}")
                return True
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return False
        
        return False
    
    def generate_processing_report(self) -> Dict[str, Any]:
        """Generate comprehensive processing report."""
        return {
            'total_processing_jobs': len(self.processing_history),
            'successful_jobs': sum(1 for job in self.processing_history if job['success']),
            'failed_jobs': sum(1 for job in self.processing_history if not job['success']),
            'average_duration': np.mean([job['duration'] for job in self.processing_history]) if self.processing_history else 0,
            'data_types_processed': list(set(job['data_type'] for job in self.processing_history)),
            'last_processing': self.processing_history[-1] if self.processing_history else None,
            'config_summary': {
                'financial_preprocessing': self.config.financial_preprocessing,
                'feature_engineering': self.config.feature_engineering,
                'validation': self.config.validation
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = {
        'financial_preprocessing': {
            'handle_missing': 'interpolate',
            'outlier_detection_methods': ['zscore', 'iqr'],
            'normalization_method': 'robust'
        },
        'feature_engineering': {
            'include_technical_indicators': True,
            'include_ratios': True,
            'include_temporal_features': True
        },
        'validation': {
            'run_statistical_tests': True
        }
    }
    
    # Initialize processor
    processor = IPODataProcessor(config_dict=config)
    
    # Example usage
    # financial_data = pd.read_csv("sample_financial_data.csv")
    # results = processor.process_financial_data(financial_data)
    # print(f"Processing completed: {results['success']}")