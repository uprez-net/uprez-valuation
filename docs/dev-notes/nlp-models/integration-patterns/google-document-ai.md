# Google Document AI Integration Patterns

## Overview

This document provides comprehensive technical documentation for integrating Google Document AI with the Uprez Valuation system. Google Document AI serves as the primary OCR and document structure extraction service, providing enterprise-grade document processing capabilities for financial documents.

## Architecture Overview

### Service Integration

```python
class GoogleDocumentAIService:
    """Google Document AI integration service"""
    
    def __init__(self):
        self.project_id = settings.gcp.project_id
        self.location = settings.gcp.document_ai_location
        self.processor_id = settings.gcp.document_ai_processor_id
        
        # Initialize clients
        self.client = documentai.DocumentProcessorServiceClient()
        self.storage_client = storage.Client(project=self.project_id)
        
        # Processor configurations
        self.processors = {
            'form_parser': self._get_processor_name('FORM_PARSER_PROCESSOR'),
            'ocr_processor': self._get_processor_name('OCR_PROCESSOR'),
            'invoice_processor': self._get_processor_name('INVOICE_PROCESSOR'),
            'custom_financial': self._get_processor_name('CUSTOM_PROCESSOR')
        }
        
        # Processing configurations
        self.processing_config = {
            'enable_native_pdf_parsing': True,
            'compute_style_info': True,
            'enable_image_quality_scores': True,
            'enable_entity_extraction': True,
            'enable_layout_detection': True
        }
```

### Processor Management

```python
async def create_custom_processor(
    self,
    processor_type: str,
    processor_display_name: str,
    schema_config: Dict[str, Any] = None
) -> str:
    """Create custom Document AI processor for financial documents"""
    
    processor = documentai.Processor(
        type_=processor_type,
        display_name=processor_display_name,
        default_processor_version=documentai.ProcessorVersion(
            latest_evaluation=documentai.Evaluation(
                name="financial_documents_evaluation"
            )
        )
    )
    
    # Custom schema for financial documents
    if schema_config:
        processor.processor_version_aliases = {
            'financial-v1': self._create_financial_schema(schema_config)
        }
    
    operation = self.client.create_processor(
        parent=f"projects/{self.project_id}/locations/{self.location}",
        processor=processor
    )
    
    # Wait for operation to complete
    result = operation.result()
    processor_name = result.name
    
    logger.info(f"Created custom processor: {processor_name}")
    return processor_name

def _create_financial_schema(self, schema_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create schema for financial document processing"""
    
    financial_schema = {
        "entity_types": [
            {
                "type": "company_name",
                "description": "Company names and variations"
            },
            {
                "type": "financial_metric",
                "description": "Financial metrics and KPIs",
                "subtypes": ["revenue", "profit", "ebitda", "margin", "debt", "cash"]
            },
            {
                "type": "monetary_amount",
                "description": "Monetary values with currency",
                "pattern": r"\$[\d,]+(?:\.\d{2})?(?:[kmb])?"
            },
            {
                "type": "percentage",
                "description": "Percentage values",
                "pattern": r"\d+(?:\.\d+)?%"
            },
            {
                "type": "date_period",
                "description": "Financial periods and dates",
                "pattern": r"Q[1-4]\s+\d{4}|FY\s+\d{4}|\d{4}"
            },
            {
                "type": "stock_ticker",
                "description": "Stock ticker symbols",
                "pattern": r"[A-Z]{1,5}(?:\.[A-Z]{1,3})?"
            }
        ],
        
        "document_sections": [
            "executive_summary",
            "business_overview", 
            "risk_factors",
            "financial_statements",
            "management_discussion",
            "use_of_proceeds",
            "market_analysis"
        ]
    }
    
    return financial_schema
```

## Document Processing Workflows

### Multi-Modal Document Processing

```python
class MultiModalDocumentProcessor:
    """Process financial documents with mixed content types"""
    
    def __init__(self):
        self.document_ai_service = GoogleDocumentAIService()
        self.vision_client = vision.ImageAnnotatorClient()
        self.text_analyzer = FinancialTextAnalyzer()
        
    async def process_financial_document(
        self,
        file_path: str,
        document_type: str = 'auto_detect',
        processing_options: Dict[str, Any] = None
    ) -> FinancialDocumentResult:
        """Process financial document with comprehensive extraction"""
        
        # Auto-detect document type if needed
        if document_type == 'auto_detect':
            document_type = await self._detect_document_type(file_path)
        
        # Select appropriate processor
        processor_name = self._select_processor(document_type)
        
        # Prepare processing request
        processing_request = await self._prepare_processing_request(
            file_path, processor_name, processing_options
        )
        
        # Process document
        response = await self.document_ai_service.process_document(processing_request)
        
        # Extract structured data
        extracted_data = await self._extract_structured_data(response, document_type)
        
        # Post-process with financial domain knowledge
        enhanced_data = await self._enhance_with_financial_context(
            extracted_data, document_type
        )
        
        # Quality assessment
        quality_metrics = await self._assess_extraction_quality(enhanced_data)
        
        return FinancialDocumentResult(
            document_type=document_type,
            extracted_text=enhanced_data['text'],
            structured_data=enhanced_data['structured'],
            tables=enhanced_data['tables'],
            forms=enhanced_data['forms'],
            entities=enhanced_data['entities'],
            quality_metrics=quality_metrics,
            processing_metadata=enhanced_data['metadata']
        )
    
    async def _extract_structured_data(
        self,
        response: documentai.ProcessResponse,
        document_type: str
    ) -> Dict[str, Any]:
        """Extract structured data from Document AI response"""
        
        document = response.document
        
        structured_data = {
            'text': document.text,
            'pages': [],
            'tables': [],
            'forms': [],
            'entities': [],
            'layout_elements': [],
            'metadata': {
                'page_count': len(document.pages),
                'mime_type': response.human_review_status,
                'processing_time': response.processing_time
            }
        }
        
        # Extract pages with layout information
        for page in document.pages:
            page_data = {
                'page_number': page.page_number,
                'dimensions': {
                    'width': page.dimension.width,
                    'height': page.dimension.height,
                    'unit': page.dimension.unit
                },
                'blocks': self._extract_blocks(page, document.text),
                'paragraphs': self._extract_paragraphs(page, document.text),
                'lines': self._extract_lines(page, document.text),
                'tokens': self._extract_tokens(page, document.text),
                'visual_elements': self._extract_visual_elements(page)
            }
            structured_data['pages'].append(page_data)
        
        # Extract tables with financial context
        structured_data['tables'] = await self._extract_financial_tables(document)
        
        # Extract forms and key-value pairs
        structured_data['forms'] = self._extract_forms(document)
        
        # Extract entities with financial classification
        structured_data['entities'] = await self._extract_financial_entities(document)
        
        return structured_data
    
    async def _extract_financial_tables(self, document) -> List[FinancialTable]:
        """Extract and interpret financial tables"""
        
        financial_tables = []
        
        for page in document.pages:
            for table in page.tables:
                # Extract table structure
                table_data = {
                    'headers': [],
                    'rows': [],
                    'column_count': 0,
                    'row_count': len(table.header_rows) + len(table.body_rows)
                }
                
                # Extract header rows
                for header_row in table.header_rows:
                    header_cells = []
                    for cell in header_row.cells:
                        cell_text = self._extract_text_from_layout(
                            document.text, cell.layout.text_anchor
                        )
                        header_cells.append(cell_text.strip())
                    table_data['headers'].append(header_cells)
                    table_data['column_count'] = max(table_data['column_count'], len(header_cells))
                
                # Extract body rows
                for body_row in table.body_rows:
                    row_cells = []
                    for cell in body_row.cells:
                        cell_text = self._extract_text_from_layout(
                            document.text, cell.layout.text_anchor
                        )
                        row_cells.append(cell_text.strip())
                    table_data['rows'].append(row_cells)
                    table_data['column_count'] = max(table_data['column_count'], len(row_cells))
                
                # Classify table type
                table_type = await self._classify_financial_table(table_data)
                
                # Extract financial metrics
                financial_metrics = await self._extract_table_metrics(table_data, table_type)
                
                financial_table = FinancialTable(
                    table_type=table_type,
                    headers=table_data['headers'],
                    rows=table_data['rows'],
                    column_count=table_data['column_count'],
                    row_count=table_data['row_count'],
                    financial_metrics=financial_metrics,
                    confidence_score=self._calculate_table_confidence(table),
                    page_number=page.page_number
                )
                
                financial_tables.append(financial_table)
        
        return financial_tables
```

## Advanced Processing Features

### Layout-Aware Processing

```python
class LayoutAwareProcessor:
    """Process documents with layout understanding"""
    
    def __init__(self):
        self.layout_analyzer = DocumentLayoutAnalyzer()
        self.section_classifier = FinancialSectionClassifier()
    
    async def process_with_layout_awareness(
        self,
        document_response: documentai.ProcessResponse
    ) -> LayoutAwareResult:
        """Process document with layout structure understanding"""
        
        document = document_response.document
        
        # Analyze document layout
        layout_structure = await self.layout_analyzer.analyze_layout(document)
        
        # Classify document sections
        sections = await self._classify_document_sections(document, layout_structure)
        
        # Extract content by section
        section_content = {}
        for section in sections:
            section_text = self._extract_section_text(document, section)
            section_entities = await self._extract_section_entities(section_text)
            section_sentiment = await self._analyze_section_sentiment(section_text)
            
            section_content[section.name] = {
                'text': section_text,
                'entities': section_entities,
                'sentiment': section_sentiment,
                'layout_info': section.layout_info,
                'confidence': section.confidence
            }
        
        # Analyze cross-section relationships
        relationships = await self._analyze_section_relationships(section_content)
        
        return LayoutAwareResult(
            layout_structure=layout_structure,
            sections=section_content,
            relationships=relationships,
            document_classification=layout_structure.document_type
        )
    
    async def _classify_document_sections(
        self,
        document,
        layout_structure: DocumentLayout
    ) -> List[DocumentSection]:
        """Classify document sections based on layout and content"""
        
        sections = []
        
        for page in document.pages:
            for block in page.blocks:
                block_text = self._extract_text_from_layout(
                    document.text, block.layout.text_anchor
                )
                
                # Classify section type
                section_type = await self.section_classifier.classify(
                    block_text, layout_structure, page.page_number
                )
                
                section = DocumentSection(
                    name=section_type,
                    text=block_text,
                    page_number=page.page_number,
                    layout_info={
                        'bounding_box': block.layout.bounding_poly,
                        'orientation': block.layout.orientation,
                        'confidence': getattr(block.layout, 'confidence', 1.0)
                    },
                    confidence=section_type.confidence
                )
                
                sections.append(section)
        
        return sections
```

### Asynchronous Processing

```python
class AsyncDocumentProcessor:
    """Asynchronous document processing with Google Document AI"""
    
    def __init__(self):
        self.batch_size = 10
        self.max_concurrent = 5
        self.retry_attempts = 3
        self.retry_delay = 1.0
    
    async def process_document_batch(
        self,
        file_paths: List[str],
        processor_type: str = 'FORM_PARSER_PROCESSOR'
    ) -> List[DocumentProcessingResult]:
        """Process multiple documents concurrently"""
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_single_document(file_path: str) -> DocumentProcessingResult:
            async with semaphore:
                return await self._process_document_with_retry(file_path, processor_type)
        
        # Process all documents concurrently
        tasks = [process_single_document(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions and compile results
        successful_results = []
        failed_results = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_results.append({
                    'file_path': file_paths[i],
                    'error': str(result)
                })
            else:
                successful_results.append(result)
        
        return {
            'successful': successful_results,
            'failed': failed_results,
            'success_rate': len(successful_results) / len(file_paths)
        }
    
    async def _process_document_with_retry(
        self,
        file_path: str,
        processor_type: str
    ) -> DocumentProcessingResult:
        """Process document with retry logic"""
        
        last_exception = None
        
        for attempt in range(self.retry_attempts):
            try:
                return await self._process_single_document(file_path, processor_type)
            
            except Exception as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    logger.warning(f"Retry {attempt + 1} for {file_path}: {e}")
                else:
                    logger.error(f"Final attempt failed for {file_path}: {e}")
        
        raise last_exception
```

## Financial Document Specialization

### Custom Financial Processor

```python
class FinancialDocumentProcessor:
    """Specialized processor for financial documents"""
    
    def __init__(self):
        self.document_ai_service = GoogleDocumentAIService()
        self.financial_entity_extractor = FinancialEntityExtractor()
        self.table_interpreter = FinancialTableInterpreter()
        
        # Financial document templates
        self.document_templates = {
            'prospectus': self._load_prospectus_template(),
            '10k': self._load_10k_template(),
            '10q': self._load_10q_template(),
            'earnings_call': self._load_earnings_template()
        }
    
    async def process_financial_document(
        self,
        file_path: str,
        document_type: str,
        extraction_config: FinancialExtractionConfig
    ) -> FinancialDocumentResult:
        """Process financial document with domain-specific extraction"""
        
        # Get document template
        template = self.document_templates.get(document_type)
        if not template:
            raise ValueError(f"Unsupported document type: {document_type}")
        
        # Configure processor for document type
        processor_config = self._configure_for_document_type(document_type, template)
        
        # Process with Document AI
        base_result = await self.document_ai_service.process_document(
            file_path, processor_config
        )
        
        # Apply financial domain processing
        financial_result = await self._apply_financial_processing(
            base_result, document_type, template, extraction_config
        )
        
        return financial_result
    
    async def _apply_financial_processing(
        self,
        base_result: DocumentAIResult,
        document_type: str,
        template: DocumentTemplate,
        config: FinancialExtractionConfig
    ) -> FinancialDocumentResult:
        """Apply financial domain-specific processing"""
        
        # Extract financial entities
        entities = await self.financial_entity_extractor.extract_entities(
            base_result.text,
            entity_types=template.expected_entities
        )
        
        # Process financial tables
        financial_tables = []
        for table in base_result.tables:
            interpreted_table = await self.table_interpreter.interpret_table(
                table, document_type
            )
            financial_tables.append(interpreted_table)
        
        # Extract key financial metrics
        key_metrics = await self._extract_key_metrics(
            base_result.text, entities, financial_tables, template
        )
        
        # Section-based analysis
        sections = await self._analyze_document_sections(
            base_result, template.section_mapping
        )
        
        # Risk factor extraction
        risk_factors = await self._extract_risk_factors(
            base_result.text, sections
        )
        
        return FinancialDocumentResult(
            document_type=document_type,
            base_extraction=base_result,
            financial_entities=entities,
            financial_tables=financial_tables,
            key_metrics=key_metrics,
            sections=sections,
            risk_factors=risk_factors,
            extraction_confidence=self._calculate_extraction_confidence(
                base_result, entities, financial_tables
            )
        )
```

### Table Processing and Interpretation

```python
class FinancialTableProcessor:
    """Specialized processing for financial tables"""
    
    def __init__(self):
        self.table_classifier = FinancialTableClassifier()
        self.numeric_parser = NumericValueParser()
        self.column_mapper = FinancialColumnMapper()
    
    async def process_financial_table(
        self,
        table_data: Dict[str, Any],
        document_context: str = None
    ) -> ProcessedFinancialTable:
        """Process and interpret financial table data"""
        
        # Classify table type
        table_type = await self.table_classifier.classify_table(
            table_data, document_context
        )
        
        # Map columns to financial concepts
        column_mapping = await self.column_mapper.map_columns(
            table_data['headers'], table_type
        )
        
        # Parse numeric values
        parsed_data = []
        for row in table_data['rows']:
            parsed_row = {}
            for i, cell in enumerate(row):
                column_name = column_mapping.get(i, f'column_{i}')
                parsed_value = self.numeric_parser.parse_financial_value(cell)
                parsed_row[column_name] = parsed_value
            parsed_data.append(parsed_row)
        
        # Extract key financial metrics
        key_metrics = self._extract_table_metrics(parsed_data, table_type)
        
        # Validate data consistency
        validation_results = self._validate_table_data(parsed_data, table_type)
        
        return ProcessedFinancialTable(
            table_type=table_type,
            column_mapping=column_mapping,
            parsed_data=parsed_data,
            key_metrics=key_metrics,
            validation_results=validation_results,
            processing_confidence=self._calculate_table_confidence(
                table_data, parsed_data, validation_results
            )
        )
    
    def _extract_table_metrics(
        self,
        parsed_data: List[Dict[str, Any]],
        table_type: str
    ) -> Dict[str, Any]:
        """Extract key financial metrics from table data"""
        
        metrics = {}
        
        if table_type == 'income_statement':
            # Extract P&L metrics
            metrics.update(self._extract_income_statement_metrics(parsed_data))
        elif table_type == 'balance_sheet':
            # Extract balance sheet metrics
            metrics.update(self._extract_balance_sheet_metrics(parsed_data))
        elif table_type == 'cash_flow':
            # Extract cash flow metrics
            metrics.update(self._extract_cash_flow_metrics(parsed_data))
        elif table_type == 'ratios':
            # Extract financial ratios
            metrics.update(self._extract_ratio_metrics(parsed_data))
        
        return metrics
    
    def _extract_income_statement_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract income statement metrics"""
        
        metrics = {}
        
        # Look for key P&L items
        revenue_items = ['revenue', 'sales', 'total_revenue', 'net_sales']
        cost_items = ['cost_of_revenue', 'cost_of_sales', 'cogs']
        operating_items = ['operating_income', 'operating_profit', 'ebit']
        net_income_items = ['net_income', 'net_profit', 'net_earnings']
        
        for row in data:
            # Extract revenue
            for item in revenue_items:
                if item in row and row[item].get('numeric_value'):
                    metrics['revenue'] = row[item]['numeric_value']
                    break
            
            # Extract costs
            for item in cost_items:
                if item in row and row[item].get('numeric_value'):
                    metrics['cost_of_revenue'] = row[item]['numeric_value']
                    break
            
            # Extract operating income
            for item in operating_items:
                if item in row and row[item].get('numeric_value'):
                    metrics['operating_income'] = row[item]['numeric_value']
                    break
            
            # Extract net income
            for item in net_income_items:
                if item in row and row[item].get('numeric_value'):
                    metrics['net_income'] = row[item]['numeric_value']
                    break
        
        # Calculate derived metrics
        if 'revenue' in metrics and 'cost_of_revenue' in metrics:
            metrics['gross_profit'] = metrics['revenue'] - metrics['cost_of_revenue']
            metrics['gross_margin'] = metrics['gross_profit'] / metrics['revenue']
        
        return metrics
```

## Error Handling and Quality Assurance

### Processing Error Recovery

```python
class DocumentAIErrorHandler:
    """Handle errors and implement recovery strategies"""
    
    def __init__(self):
        self.error_strategies = {
            'QUOTA_EXCEEDED': self._handle_quota_exceeded,
            'INVALID_ARGUMENT': self._handle_invalid_argument,
            'PROCESSING_ERROR': self._handle_processing_error,
            'TIMEOUT': self._handle_timeout
        }
        
        self.fallback_processor = TesseractOCRProcessor()
    
    async def handle_processing_error(
        self,
        error: Exception,
        file_path: str,
        processor_config: Dict[str, Any]
    ) -> DocumentProcessingResult:
        """Handle Document AI processing errors"""
        
        error_type = self._classify_error(error)
        
        if error_type in self.error_strategies:
            recovery_strategy = self.error_strategies[error_type]
            return await recovery_strategy(error, file_path, processor_config)
        
        # Default fallback to local OCR
        logger.warning(f"Falling back to local OCR for {file_path}: {error}")
        return await self.fallback_processor.process_document(file_path)
    
    async def _handle_quota_exceeded(
        self,
        error: Exception,
        file_path: str,
        config: Dict[str, Any]
    ) -> DocumentProcessingResult:
        """Handle quota exceeded errors"""
        
        # Wait and retry with exponential backoff
        wait_time = self._calculate_backoff_time()
        await asyncio.sleep(wait_time)
        
        # Retry with reduced batch size
        reduced_config = config.copy()
        reduced_config['batch_size'] = max(1, config.get('batch_size', 10) // 2)
        
        return await self._retry_processing(file_path, reduced_config)
    
    async def _handle_invalid_argument(
        self,
        error: Exception,
        file_path: str,
        config: Dict[str, Any]
    ) -> DocumentProcessingResult:
        """Handle invalid argument errors"""
        
        # Validate and fix configuration
        validated_config = self._validate_and_fix_config(config, error)
        
        # Retry with corrected configuration
        return await self._retry_processing(file_path, validated_config)
```

### Quality Monitoring

```python
class ProcessingQualityMonitor:
    """Monitor and ensure processing quality"""
    
    def __init__(self):
        self.quality_thresholds = {
            'confidence': 0.8,
            'text_coverage': 0.9,
            'entity_extraction': 0.85,
            'table_accuracy': 0.9
        }
        
        self.quality_metrics = QualityMetricsCalculator()
    
    async def monitor_processing_quality(
        self,
        processing_result: DocumentProcessingResult,
        expected_content: Dict[str, Any] = None
    ) -> QualityAssessment:
        """Monitor and assess processing quality"""
        
        quality_scores = {}
        
        # Confidence score assessment
        quality_scores['confidence'] = self._assess_confidence_quality(
            processing_result.confidence_scores
        )
        
        # Text coverage assessment
        quality_scores['text_coverage'] = self._assess_text_coverage(
            processing_result.extracted_text,
            processing_result.image_quality_scores
        )
        
        # Entity extraction quality
        if processing_result.entities:
            quality_scores['entity_extraction'] = self._assess_entity_quality(
                processing_result.entities
            )
        
        # Table extraction quality
        if processing_result.tables:
            quality_scores['table_accuracy'] = self._assess_table_quality(
                processing_result.tables
            )
        
        # Overall quality assessment
        overall_quality = self._calculate_overall_quality(quality_scores)
        
        # Generate quality report
        quality_report = self._generate_quality_report(
            quality_scores, overall_quality, expected_content
        )
        
        return QualityAssessment(
            overall_quality=overall_quality,
            component_scores=quality_scores,
            quality_report=quality_report,
            meets_threshold=overall_quality >= self.quality_thresholds['confidence'],
            improvement_suggestions=self._suggest_improvements(quality_scores)
        )
```

## Integration Examples

### Basic Document Processing

```python
from src.backend.nlp_services.document_processing import GoogleDocumentAIService

# Initialize service
doc_ai_service = GoogleDocumentAIService()

# Process a financial document
file_path = "/path/to/financial_report.pdf"
result = await doc_ai_service.process_document(
    file_path=file_path,
    processor_type='FORM_PARSER_PROCESSOR',
    enable_tables=True,
    enable_entities=True
)

# Access extracted data
print(f"Extracted text length: {len(result.text)}")
print(f"Number of pages: {result.page_count}")
print(f"Number of tables: {len(result.tables)}")
print(f"Processing confidence: {result.overall_confidence:.2f}")

# Process tables
for i, table in enumerate(result.tables):
    print(f"Table {i+1}: {table.rows}x{table.columns}")
    if table.financial_metrics:
        print(f"Key metrics: {table.financial_metrics}")
```

### Batch Processing

```python
# Process multiple documents
async_processor = AsyncDocumentProcessor()

file_paths = [
    "/path/to/10k_report.pdf",
    "/path/to/quarterly_report.pdf",
    "/path/to/prospectus.pdf"
]

# Process batch
batch_results = await async_processor.process_document_batch(
    file_paths=file_paths,
    processor_type='FORM_PARSER_PROCESSOR'
)

print(f"Successfully processed: {len(batch_results['successful'])}")
print(f"Failed: {len(batch_results['failed'])}")
print(f"Success rate: {batch_results['success_rate']:.1%}")

# Process results
for result in batch_results['successful']:
    print(f"Document: {result.file_path}")
    print(f"Pages: {result.page_count}")
    print(f"Quality: {result.quality_assessment.overall_quality:.2f}")
```

### Financial Table Processing

```python
# Specialized financial table processing
table_processor = FinancialTableProcessor()

# Process extracted tables
for table in result.tables:
    processed_table = await table_processor.process_financial_table(
        table_data=table.raw_data,
        document_context=result.text[:1000]  # First 1000 chars for context
    )
    
    print(f"Table Type: {processed_table.table_type}")
    print(f"Key Metrics: {processed_table.key_metrics}")
    
    # Access specific financial data
    if processed_table.table_type == 'income_statement':
        revenue = processed_table.key_metrics.get('revenue')
        net_income = processed_table.key_metrics.get('net_income')
        
        if revenue and net_income:
            net_margin = net_income / revenue
            print(f"Net Profit Margin: {net_margin:.1%}")
```

## Configuration and Deployment

### Environment Configuration

```yaml
# config/document-ai/production.yaml
google_cloud:
  project_id: "uprez-valuation-prod"
  location: "us"
  processors:
    form_parser:
      id: "abc123def456"
      version: "stable"
    ocr_processor:
      id: "def456ghi789"
      version: "latest"
    custom_financial:
      id: "ghi789jkl012"
      version: "financial-v1"

processing:
  max_concurrent_requests: 10
  timeout_seconds: 60
  retry_attempts: 3
  enable_caching: true
  cache_ttl_hours: 24

quality:
  min_confidence_threshold: 0.75
  enable_quality_monitoring: true
  fallback_to_tesseract: true
  quality_alert_threshold: 0.6
```

### Production Deployment

```python
class ProductionDocumentAIService:
    """Production-ready Document AI service"""
    
    def __init__(self):
        self.config = load_production_config()
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.cache = RedisCache()
    
    async def process_with_monitoring(
        self,
        file_path: str,
        processor_type: str
    ) -> DocumentProcessingResult:
        """Process document with comprehensive monitoring"""
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(file_path, processor_type)
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                self.metrics_collector.increment('cache_hits')
                return cached_result
            
            # Process document
            result = await self._process_document(file_path, processor_type)
            
            # Cache result
            await self.cache.set(
                cache_key, 
                result, 
                ttl=self.config.cache_ttl_hours * 3600
            )
            
            # Collect metrics
            processing_time = time.time() - start_time
            self.metrics_collector.record_processing_time(processing_time)
            self.metrics_collector.record_confidence(result.overall_confidence)
            
            # Quality monitoring
            if result.overall_confidence < self.config.quality_alert_threshold:
                await self.alert_manager.send_quality_alert(
                    file_path, result.overall_confidence
                )
            
            return result
            
        except Exception as e:
            self.metrics_collector.increment('processing_errors')
            await self.alert_manager.send_error_alert(file_path, str(e))
            raise
```

## Best Practices

### Document AI Integration

1. **Processor Selection**: Choose appropriate processor type for document format
2. **Quality Thresholds**: Set confidence thresholds based on use case requirements
3. **Error Handling**: Implement comprehensive error handling and fallback strategies
4. **Caching**: Cache results to reduce API calls and improve performance

### Financial Domain Adaptation

1. **Custom Processors**: Train custom processors for specialized financial documents
2. **Entity Recognition**: Combine Document AI with specialized financial NER models
3. **Table Interpretation**: Implement business logic for financial table processing
4. **Quality Validation**: Validate extracted data against business rules

### Performance Optimization

1. **Batch Processing**: Process multiple documents concurrently
2. **Async Operations**: Use asynchronous processing for better throughput
3. **Resource Management**: Monitor and manage API quotas and limits
4. **Monitoring**: Implement comprehensive monitoring and alerting

---

*Last updated: 2025-08-30*
*Version: 1.0.0*