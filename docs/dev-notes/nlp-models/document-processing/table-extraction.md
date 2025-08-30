# Table Extraction and Financial Data Parsing

## Overview

This document provides comprehensive technical documentation for table extraction and financial data parsing capabilities in the Uprez Valuation system. The system combines Google Document AI's advanced table detection with specialized financial data interpretation to extract and structure financial information from complex tables in prospectuses, annual reports, and other financial documents.

## Table Processing Architecture

### Multi-Stage Table Processing Pipeline

```python
class FinancialTableProcessor:
    """Advanced financial table processing system"""
    
    def __init__(self):
        self.table_detector = DocumentAITableDetector()
        self.table_classifier = FinancialTableClassifier()
        self.column_interpreter = ColumnSemanticInterpreter()
        self.value_parser = FinancialValueParser()
        self.relationship_analyzer = TableRelationshipAnalyzer()
        self.validation_engine = FinancialTableValidator()
        
        # Financial table types and their characteristics
        self.table_types = {
            'income_statement': {
                'expected_columns': ['period', 'revenue', 'cost_of_revenue', 'gross_profit', 
                                   'operating_expenses', 'operating_income', 'net_income'],
                'required_columns': ['revenue', 'net_income'],
                'validation_rules': ['revenue_positive', 'period_consistency'],
                'metric_calculations': ['gross_margin', 'operating_margin', 'net_margin']
            },
            
            'balance_sheet': {
                'expected_columns': ['assets', 'current_assets', 'non_current_assets',
                                   'liabilities', 'current_liabilities', 'equity'],
                'required_columns': ['assets', 'liabilities', 'equity'],
                'validation_rules': ['balance_equation', 'positive_equity'],
                'metric_calculations': ['debt_to_equity', 'current_ratio', 'quick_ratio']
            },
            
            'cash_flow': {
                'expected_columns': ['operating_cash_flow', 'investing_cash_flow',
                                   'financing_cash_flow', 'net_cash_change'],
                'required_columns': ['operating_cash_flow'],
                'validation_rules': ['cash_flow_sum', 'operating_cash_reasonableness'],
                'metric_calculations': ['free_cash_flow', 'cash_conversion_ratio']
            },
            
            'segment_reporting': {
                'expected_columns': ['segment', 'revenue', 'operating_income', 'assets'],
                'required_columns': ['segment', 'revenue'],
                'validation_rules': ['segment_revenue_sum', 'segment_completeness'],
                'metric_calculations': ['segment_margins', 'segment_growth_rates']
            },
            
            'valuation_metrics': {
                'expected_columns': ['metric', 'value', 'comparable_range'],
                'required_columns': ['metric', 'value'],
                'validation_rules': ['valuation_reasonableness', 'metric_consistency'],
                'metric_calculations': ['relative_valuations', 'premium_discount']
            }
        }
    
    async def process_financial_table(
        self,
        table_data: TableData,
        document_context: str = None,
        table_index: int = None
    ) -> ProcessedFinancialTable:
        """Process a single financial table with comprehensive analysis"""
        
        # Classify table type
        table_classification = await self.table_classifier.classify_table(
            table_data, document_context, table_index
        )
        
        # Interpret column semantics
        column_interpretation = await self.column_interpreter.interpret_columns(
            table_data.headers, table_classification, document_context
        )
        
        # Parse and validate financial values
        parsed_data = await self.value_parser.parse_table_values(
            table_data, column_interpretation
        )
        
        # Extract financial metrics
        financial_metrics = await self._extract_financial_metrics(
            parsed_data, table_classification
        )
        
        # Validate table consistency
        validation_result = await self.validation_engine.validate_table(
            parsed_data, table_classification, financial_metrics
        )
        
        # Analyze relationships between metrics
        metric_relationships = await self.relationship_analyzer.analyze_relationships(
            financial_metrics, table_classification
        )
        
        # Calculate additional derived metrics
        derived_metrics = await self._calculate_derived_metrics(
            financial_metrics, table_classification
        )
        
        return ProcessedFinancialTable(
            original_table=table_data,
            table_type=table_classification.table_type,
            confidence=table_classification.confidence,
            column_interpretation=column_interpretation,
            parsed_data=parsed_data,
            financial_metrics=financial_metrics,
            derived_metrics=derived_metrics,
            validation_result=validation_result,
            metric_relationships=metric_relationships,
            processing_metadata={
                'processing_time': time.time(),
                'table_index': table_index,
                'extraction_method': 'document_ai_plus_nlp'
            }
        )
```

### Google Document AI Table Detection

```python
class DocumentAITableDetector:
    """Advanced table detection using Google Document AI"""
    
    def __init__(self):
        self.client = documentai.DocumentProcessorServiceClient()
        self.processor_name = self._get_table_processor_name()
        
        # Table detection configuration
        self.detection_config = {
            'enable_native_pdf_parsing': True,
            'compute_style_info': True,
            'enable_image_quality_scores': True,
            'table_detection_confidence': 0.8,
            'merge_adjacent_tables': False
        }
    
    async def detect_and_extract_tables(
        self,
        document_path: str,
        document_content: bytes = None
    ) -> List[DetectedTable]:
        """Detect and extract tables from financial documents"""
        
        # Prepare document for processing
        if document_content:
            raw_document = documentai.RawDocument(
                content=document_content,
                mime_type=self._get_mime_type(document_path)
            )
        else:
            with open(document_path, 'rb') as file:
                content = file.read()
                raw_document = documentai.RawDocument(
                    content=content,
                    mime_type=self._get_mime_type(document_path)
                )
        
        # Configure processing request
        request = documentai.ProcessRequest(
            name=self.processor_name,
            raw_document=raw_document,
            labels=["table_extraction", "financial_forms"]
        )
        
        # Process document
        response = self.client.process_document(request=request)
        document = response.document
        
        # Extract tables with enhanced metadata
        detected_tables = []
        
        for page_idx, page in enumerate(document.pages):
            for table_idx, table in enumerate(page.tables):
                
                # Extract table structure
                table_structure = self._extract_table_structure(table, document.text)
                
                # Calculate table quality metrics
                quality_metrics = self._calculate_table_quality(table, page)
                
                # Detect table boundaries and layout
                table_layout = self._analyze_table_layout(table, page)
                
                detected_table = DetectedTable(
                    page_number=page_idx + 1,
                    table_index=table_idx,
                    table_structure=table_structure,
                    quality_metrics=quality_metrics,
                    layout_info=table_layout,
                    raw_table_data=table,
                    extraction_confidence=quality_metrics.overall_confidence
                )
                
                detected_tables.append(detected_table)
        
        return detected_tables
    
    def _extract_table_structure(
        self, 
        table: documentai.Document.Page.Table, 
        document_text: str
    ) -> TableStructure:
        """Extract structured data from Document AI table"""
        
        # Extract headers
        headers = []
        for header_row in table.header_rows:
            header_cells = []
            for cell in header_row.cells:
                cell_text = self._extract_text_from_layout(
                    document_text, cell.layout.text_anchor
                ).strip()
                
                cell_info = TableCell(
                    text=cell_text,
                    row_span=cell.row_span,
                    col_span=cell.col_span,
                    confidence=getattr(cell.layout, 'confidence', 1.0)
                )
                header_cells.append(cell_info)
            headers.append(header_cells)
        
        # Extract body rows
        body_rows = []
        for body_row in table.body_rows:
            row_cells = []
            for cell in body_row.cells:
                cell_text = self._extract_text_from_layout(
                    document_text, cell.layout.text_anchor
                ).strip()
                
                cell_info = TableCell(
                    text=cell_text,
                    row_span=cell.row_span,
                    col_span=cell.col_span,
                    confidence=getattr(cell.layout, 'confidence', 1.0)
                )
                row_cells.append(cell_info)
            body_rows.append(row_cells)
        
        # Calculate table dimensions
        max_columns = max(
            max(len(row) for row in headers) if headers else 0,
            max(len(row) for row in body_rows) if body_rows else 0
        )
        
        return TableStructure(
            headers=headers,
            body_rows=body_rows,
            row_count=len(headers) + len(body_rows),
            column_count=max_columns,
            has_headers=len(headers) > 0,
            table_caption=self._extract_table_caption(table, document_text)
        )
```

### Financial Value Parsing

```python
class FinancialValueParser:
    """Parse and normalize financial values from table cells"""
    
    def __init__(self):
        self.currency_patterns = {
            'USD': [r'\$', r'USD', r'US\$', r'dollars?'],
            'EUR': [r'€', r'EUR', r'euros?'],
            'GBP': [r'£', r'GBP', r'pounds?'],
            'JPY': [r'¥', r'JPY', r'yen'],
            'AUD': [r'A\$', r'AUD', r'AU\$']
        }
        
        self.unit_multipliers = {
            'thousand': 1000,
            'thousands': 1000,
            'k': 1000,
            'K': 1000,
            'million': 1000000,
            'millions': 1000000,
            'm': 1000000,
            'M': 1000000,
            'billion': 1000000000,
            'billions': 1000000000,
            'b': 1000000000,
            'B': 1000000000,
            'trillion': 1000000000000,
            't': 1000000000000,
            'T': 1000000000000
        }
        
        self.percentage_patterns = [
            r'(\d+(?:\.\d+)?)%',
            r'(\d+(?:\.\d+)?)\s*percent'
        ]
        
        self.negative_indicators = ['(', ')', '-', 'loss', 'deficit', 'negative']
    
    async def parse_table_values(
        self,
        table_data: TableStructure,
        column_interpretation: ColumnInterpretation
    ) -> ParsedTableData:
        """Parse all values in a financial table"""
        
        parsed_rows = []
        
        # Parse header rows
        parsed_headers = []
        for header_row in table_data.headers:
            parsed_header_row = []
            for cell in header_row:
                parsed_cell = await self._parse_cell_value(
                    cell, 'header', column_interpretation
                )
                parsed_header_row.append(parsed_cell)
            parsed_headers.append(parsed_header_row)
        
        # Parse body rows
        for row_idx, body_row in enumerate(table_data.body_rows):
            parsed_row = []
            for col_idx, cell in enumerate(body_row):
                # Get column type from interpretation
                column_type = column_interpretation.get_column_type(col_idx)
                
                parsed_cell = await self._parse_cell_value(
                    cell, column_type, column_interpretation
                )
                parsed_row.append(parsed_cell)
            parsed_rows.append(parsed_row)
        
        return ParsedTableData(
            headers=parsed_headers,
            rows=parsed_rows,
            column_interpretation=column_interpretation,
            parsing_metadata={
                'total_cells_parsed': sum(len(row) for row in parsed_rows),
                'successful_parses': sum(1 for row in parsed_rows for cell in row if cell.parse_success),
                'parsing_confidence': self._calculate_parsing_confidence(parsed_rows)
            }
        )
    
    async def _parse_cell_value(
        self,
        cell: TableCell,
        expected_type: str,
        column_interpretation: ColumnInterpretation
    ) -> ParsedCell:
        """Parse individual cell value based on expected type"""
        
        cell_text = cell.text.strip()
        
        if not cell_text or cell_text in ['-', '—', 'N/A', 'n/a', '']:
            return ParsedCell(
                original_text=cell_text,
                parsed_value=None,
                data_type='empty',
                parse_success=True,
                confidence=1.0
            )
        
        # Parse based on expected type
        if expected_type == 'monetary':
            return await self._parse_monetary_value(cell_text)
        elif expected_type == 'percentage':
            return await self._parse_percentage_value(cell_text)
        elif expected_type == 'numeric':
            return await self._parse_numeric_value(cell_text)
        elif expected_type == 'date':
            return await self._parse_date_value(cell_text)
        elif expected_type == 'ratio':
            return await self._parse_ratio_value(cell_text)
        else:
            return await self._parse_text_value(cell_text)
    
    async def _parse_monetary_value(self, text: str) -> ParsedCell:
        """Parse monetary values with currency and units"""
        
        # Clean text
        clean_text = re.sub(r'[,\s]', '', text)
        
        # Detect currency
        detected_currency = 'USD'  # Default
        for currency, patterns in self.currency_patterns.items():
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
                detected_currency = currency
                break
        
        # Extract numeric value
        numeric_pattern = r'([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)'
        numeric_match = re.search(numeric_pattern, clean_text)
        
        if not numeric_match:
            return ParsedCell(
                original_text=text,
                parsed_value=None,
                data_type='monetary',
                parse_success=False,
                confidence=0.0,
                error='No numeric value found'
            )
        
        base_value = float(numeric_match.group(1).replace(',', ''))
        
        # Detect negative values
        is_negative = any(indicator in text for indicator in self.negative_indicators)
        if is_negative and base_value > 0:
            base_value = -base_value
        
        # Detect unit multipliers
        multiplier = 1
        for unit, mult in self.unit_multipliers.items():
            if unit in text.lower():
                multiplier = mult
                break
        
        final_value = base_value * multiplier
        
        return ParsedCell(
            original_text=text,
            parsed_value={
                'amount': final_value,
                'currency': detected_currency,
                'unit_multiplier': multiplier,
                'formatted_value': f"{detected_currency} {final_value:,.2f}"
            },
            data_type='monetary',
            parse_success=True,
            confidence=0.95 if numeric_match else 0.5
        )
    
    async def _parse_percentage_value(self, text: str) -> ParsedCell:
        """Parse percentage values"""
        
        for pattern in self.percentage_patterns:
            match = re.search(pattern, text)
            if match:
                percentage_value = float(match.group(1))
                
                # Check for negative indicators
                is_negative = any(indicator in text for indicator in self.negative_indicators)
                if is_negative and percentage_value > 0:
                    percentage_value = -percentage_value
                
                return ParsedCell(
                    original_text=text,
                    parsed_value={
                        'percentage': percentage_value,
                        'decimal': percentage_value / 100,
                        'formatted_value': f"{percentage_value}%"
                    },
                    data_type='percentage',
                    parse_success=True,
                    confidence=0.9
                )
        
        return ParsedCell(
            original_text=text,
            parsed_value=None,
            data_type='percentage',
            parse_success=False,
            confidence=0.0,
            error='No percentage pattern found'
        )
```

## Financial Table Classification

### Intelligent Table Type Detection

```python
class FinancialTableClassifier:
    """Classify financial tables based on content and structure"""
    
    def __init__(self):
        self.classification_model = FinancialBERTClassifier()
        self.pattern_matcher = TablePatternMatcher()
        
        # Table classification features
        self.classification_features = {
            'header_keywords': {
                'income_statement': [
                    'revenue', 'sales', 'cost of revenue', 'gross profit',
                    'operating income', 'net income', 'earnings per share'
                ],
                'balance_sheet': [
                    'assets', 'liabilities', 'equity', 'current assets',
                    'non-current assets', 'shareholders equity'
                ],
                'cash_flow': [
                    'operating activities', 'investing activities', 'financing activities',
                    'cash flow', 'net cash provided', 'cash and equivalents'
                ],
                'segment_reporting': [
                    'segment', 'business segment', 'geographic', 'operating segment',
                    'segment revenue', 'segment profit'
                ],
                'valuation_metrics': [
                    'multiple', 'ratio', 'valuation', 'trading multiple',
                    'price to earnings', 'enterprise value'
                ]
            },
            
            'structural_patterns': {
                'income_statement': {
                    'typical_row_count': (8, 25),
                    'typical_column_count': (2, 6),
                    'required_structure': 'hierarchical_subtotals'
                },
                'balance_sheet': {
                    'typical_row_count': (10, 40),
                    'typical_column_count': (2, 4),
                    'required_structure': 'balanced_equation'
                },
                'cash_flow': {
                    'typical_row_count': (8, 20),
                    'typical_column_count': (2, 5),
                    'required_structure': 'three_section_format'
                }
            }
        }
    
    async def classify_table(
        self,
        table_data: TableData,
        document_context: str = None,
        table_index: int = None
    ) -> TableClassification:
        """Classify financial table type with confidence scoring"""
        
        # Extract table text for analysis
        table_text = self._extract_table_text(table_data)
        
        # Keyword-based classification
        keyword_scores = self._calculate_keyword_scores(table_data.headers, table_text)
        
        # Structural analysis
        structural_scores = self._analyze_table_structure(table_data)
        
        # Context-based classification
        context_scores = {}
        if document_context:
            context_scores = await self._analyze_context_clues(
                table_text, document_context, table_index
            )
        
        # ML-based classification (if model available)
        ml_scores = {}
        if self.classification_model:
            ml_scores = await self.classification_model.classify_table(
                table_text, document_context
            )
        
        # Combine all classification signals
        combined_scores = self._combine_classification_scores(
            keyword_scores, structural_scores, context_scores, ml_scores
        )
        
        # Determine final classification
        predicted_type = max(combined_scores.items(), key=lambda x: x[1])
        
        return TableClassification(
            table_type=predicted_type[0],
            confidence=predicted_type[1],
            classification_scores=combined_scores,
            classification_features={
                'keyword_signals': keyword_scores,
                'structural_signals': structural_scores,
                'context_signals': context_scores,
                'ml_signals': ml_scores
            }
        )
    
    def _calculate_keyword_scores(
        self, 
        headers: List[List[TableCell]], 
        table_text: str
    ) -> Dict[str, float]:
        """Calculate classification scores based on keywords"""
        
        scores = {}
        
        # Combine all header text
        header_text = ' '.join([
            ' '.join([cell.text for cell in row])
            for row in headers
        ]).lower()
        
        # Combine with table text
        full_text = f"{header_text} {table_text}".lower()
        
        # Calculate scores for each table type
        for table_type, keywords in self.classification_features['header_keywords'].items():
            score = 0.0
            total_keywords = len(keywords)
            
            for keyword in keywords:
                # Exact match gets full score
                if keyword in full_text:
                    score += 1.0
                # Partial match gets partial score
                elif any(word in full_text for word in keyword.split()):
                    score += 0.5
            
            # Normalize score
            scores[table_type] = score / total_keywords if total_keywords > 0 else 0.0
        
        return scores
    
    def _analyze_table_structure(self, table_data: TableStructure) -> Dict[str, float]:
        """Analyze table structure for classification clues"""
        
        structural_scores = {}
        
        for table_type, structure_config in self.classification_features['structural_patterns'].items():
            score = 0.0
            
            # Check row count
            min_rows, max_rows = structure_config['typical_row_count']
            if min_rows <= table_data.row_count <= max_rows:
                score += 0.3
            
            # Check column count
            min_cols, max_cols = structure_config['typical_column_count']
            if min_cols <= table_data.column_count <= max_cols:
                score += 0.3
            
            # Check structural requirements
            if structure_config['required_structure'] == 'hierarchical_subtotals':
                if self._has_hierarchical_structure(table_data):
                    score += 0.4
            elif structure_config['required_structure'] == 'balanced_equation':
                if self._has_balance_structure(table_data):
                    score += 0.4
            elif structure_config['required_structure'] == 'three_section_format':
                if self._has_three_section_format(table_data):
                    score += 0.4
            
            structural_scores[table_type] = score
        
        return structural_scores
```

## Advanced Financial Calculations

### Metric Calculation Engine

```python
class FinancialMetricCalculator:
    """Calculate financial metrics and ratios from extracted table data"""
    
    def __init__(self):
        self.ratio_calculator = FinancialRatioCalculator()
        self.growth_calculator = GrowthRateCalculator()
        self.margin_calculator = MarginCalculator()
        self.valuation_calculator = ValuationMetricCalculator()
    
    async def calculate_comprehensive_metrics(
        self,
        financial_data: Dict[str, ProcessedFinancialTable],
        company_profile: CompanyProfile = None
    ) -> ComprehensiveFinancialMetrics:
        """Calculate comprehensive financial metrics from all tables"""
        
        # Extract base metrics from each table type
        base_metrics = {}
        
        if 'income_statement' in financial_data:
            base_metrics.update(
                await self._extract_income_statement_base_metrics(
                    financial_data['income_statement']
                )
            )
        
        if 'balance_sheet' in financial_data:
            base_metrics.update(
                await self._extract_balance_sheet_base_metrics(
                    financial_data['balance_sheet']
                )
            )
        
        if 'cash_flow' in financial_data:
            base_metrics.update(
                await self._extract_cash_flow_base_metrics(
                    financial_data['cash_flow']
                )
            )
        
        # Calculate derived metrics and ratios
        financial_ratios = await self.ratio_calculator.calculate_ratios(base_metrics)
        
        # Calculate growth rates
        growth_metrics = await self.growth_calculator.calculate_growth_rates(
            base_metrics, financial_data
        )
        
        # Calculate margin analysis
        margin_metrics = await self.margin_calculator.calculate_margins(base_metrics)
        
        # Calculate valuation metrics
        valuation_metrics = {}
        if company_profile and company_profile.market_data:
            valuation_metrics = await self.valuation_calculator.calculate_valuation_metrics(
                base_metrics, company_profile.market_data
            )
        
        # Industry benchmarking
        industry_comparison = await self._compare_to_industry_benchmarks(
            {**financial_ratios, **growth_metrics, **margin_metrics},
            company_profile.industry if company_profile else None
        )
        
        return ComprehensiveFinancialMetrics(
            base_metrics=base_metrics,
            financial_ratios=financial_ratios,
            growth_metrics=growth_metrics,
            margin_metrics=margin_metrics,
            valuation_metrics=valuation_metrics,
            industry_comparison=industry_comparison,
            metric_quality_score=self._assess_metric_quality(base_metrics),
            calculation_metadata={
                'calculation_timestamp': datetime.now(),
                'data_source_tables': list(financial_data.keys()),
                'metrics_calculated': len(base_metrics) + len(financial_ratios) + len(growth_metrics)
            }
        )
    
    async def _extract_income_statement_base_metrics(
        self,
        income_statement_table: ProcessedFinancialTable
    ) -> Dict[str, FinancialMetric]:
        """Extract base metrics from income statement"""
        
        metrics = {}
        parsed_data = income_statement_table.parsed_data
        
        # Revenue extraction
        revenue_row = self._find_metric_row(parsed_data, ['revenue', 'total revenue', 'net sales'])
        if revenue_row:
            metrics['revenue'] = self._create_financial_metric(
                'Revenue', revenue_row, 'income_statement'
            )
        
        # Cost of Revenue
        cost_row = self._find_metric_row(parsed_data, ['cost of revenue', 'cost of sales', 'cogs'])
        if cost_row:
            metrics['cost_of_revenue'] = self._create_financial_metric(
                'Cost of Revenue', cost_row, 'income_statement'
            )
        
        # Operating Income
        operating_row = self._find_metric_row(parsed_data, [
            'operating income', 'income from operations', 'operating profit', 'ebit'
        ])
        if operating_row:
            metrics['operating_income'] = self._create_financial_metric(
                'Operating Income', operating_row, 'income_statement'
            )
        
        # Net Income
        net_income_row = self._find_metric_row(parsed_data, [
            'net income', 'net earnings', 'profit attributable to shareholders'
        ])
        if net_income_row:
            metrics['net_income'] = self._create_financial_metric(
                'Net Income', net_income_row, 'income_statement'
            )
        
        # EBITDA (if not directly stated, calculate)
        ebitda_row = self._find_metric_row(parsed_data, ['ebitda', 'adjusted ebitda'])
        if not ebitda_row and 'operating_income' in metrics:
            # Try to calculate EBITDA
            depreciation_row = self._find_metric_row(parsed_data, [
                'depreciation', 'amortization', 'depreciation and amortization'
            ])
            if depreciation_row:
                ebitda_value = (metrics['operating_income'].value + 
                              depreciation_row.get_numeric_value())
                metrics['ebitda'] = FinancialMetric(
                    name='EBITDA',
                    value=ebitda_value,
                    currency=metrics['operating_income'].currency,
                    period=metrics['operating_income'].period,
                    unit=metrics['operating_income'].unit,
                    source='calculated',
                    calculation_method='operating_income_plus_depreciation'
                )
        
        return metrics
```

### Cross-Table Validation

```python
class CrossTableValidator:
    """Validate consistency across multiple financial tables"""
    
    def __init__(self):
        self.tolerance_threshold = 0.01  # 1% tolerance for rounding differences
        
        # Validation rules for cross-table consistency
        self.validation_rules = {
            'balance_sheet_balancing': {
                'description': 'Assets = Liabilities + Equity',
                'tables': ['balance_sheet'],
                'tolerance': 0.005  # 0.5% tolerance
            },
            
            'cash_flow_reconciliation': {
                'description': 'Net cash change matches balance sheet cash change',
                'tables': ['cash_flow', 'balance_sheet'],
                'tolerance': 0.01
            },
            
            'net_income_consistency': {
                'description': 'Net income consistent across income statement and cash flow',
                'tables': ['income_statement', 'cash_flow'],
                'tolerance': 0.005
            },
            
            'revenue_segment_reconciliation': {
                'description': 'Segment revenue sums to total revenue',
                'tables': ['income_statement', 'segment_reporting'],
                'tolerance': 0.01
            }
        }
    
    async def validate_table_consistency(
        self,
        processed_tables: Dict[str, ProcessedFinancialTable]
    ) -> CrossTableValidationResult:
        """Validate consistency across financial tables"""
        
        validation_results = {}
        overall_consistency = True
        
        for rule_name, rule_config in self.validation_rules.items():
            # Check if required tables are available
            required_tables = rule_config['tables']
            if all(table in processed_tables for table in required_tables):
                
                validation_result = await self._apply_validation_rule(
                    rule_name, rule_config, processed_tables
                )
                
                validation_results[rule_name] = validation_result
                
                if not validation_result.passed:
                    overall_consistency = False
        
        # Additional consistency checks
        additional_checks = await self._perform_additional_consistency_checks(
            processed_tables
        )
        
        validation_results.update(additional_checks)
        
        return CrossTableValidationResult(
            overall_consistency=overall_consistency,
            validation_results=validation_results,
            consistency_score=self._calculate_consistency_score(validation_results),
            data_quality_assessment=self._assess_data_quality(processed_tables),
            recommendations=self._generate_consistency_recommendations(validation_results)
        )
    
    async def _apply_validation_rule(
        self,
        rule_name: str,
        rule_config: Dict[str, Any],
        tables: Dict[str, ProcessedFinancialTable]
    ) -> ValidationResult:
        """Apply specific validation rule"""
        
        if rule_name == 'balance_sheet_balancing':
            return await self._validate_balance_sheet_equation(tables['balance_sheet'])
        
        elif rule_name == 'cash_flow_reconciliation':
            return await self._validate_cash_flow_reconciliation(
                tables['cash_flow'], tables.get('balance_sheet')
            )
        
        elif rule_name == 'net_income_consistency':
            return await self._validate_net_income_consistency(
                tables['income_statement'], tables['cash_flow']
            )
        
        elif rule_name == 'revenue_segment_reconciliation':
            return await self._validate_segment_revenue_reconciliation(
                tables['income_statement'], tables.get('segment_reporting')
            )
        
        return ValidationResult(
            rule_name=rule_name,
            passed=True,
            confidence=0.5,
            message="Rule not implemented"
        )
    
    async def _validate_balance_sheet_equation(
        self,
        balance_sheet: ProcessedFinancialTable
    ) -> ValidationResult:
        """Validate that Assets = Liabilities + Equity"""
        
        metrics = balance_sheet.financial_metrics
        
        # Extract key balance sheet components
        total_assets = metrics.get('total_assets')
        total_liabilities = metrics.get('total_liabilities')
        total_equity = metrics.get('total_equity')
        
        if not all([total_assets, total_liabilities, total_equity]):
            return ValidationResult(
                rule_name='balance_sheet_balancing',
                passed=False,
                confidence=0.0,
                message='Missing required balance sheet components',
                details={'missing_components': [
                    name for name, value in [
                        ('total_assets', total_assets),
                        ('total_liabilities', total_liabilities),
                        ('total_equity', total_equity)
                    ] if value is None
                ]}
            )
        
        # Calculate difference
        assets_value = total_assets.value
        liabilities_equity_value = total_liabilities.value + total_equity.value
        difference = abs(assets_value - liabilities_equity_value)
        relative_difference = difference / assets_value if assets_value > 0 else float('inf')
        
        # Check if within tolerance
        passed = relative_difference <= self.validation_rules['balance_sheet_balancing']['tolerance']
        
        return ValidationResult(
            rule_name='balance_sheet_balancing',
            passed=passed,
            confidence=1.0 - min(1.0, relative_difference * 10),  # Convert to confidence
            message=f"Balance sheet equation check: {relative_difference:.1%} difference",
            details={
                'assets': assets_value,
                'liabilities_plus_equity': liabilities_equity_value,
                'absolute_difference': difference,
                'relative_difference': relative_difference,
                'tolerance': self.validation_rules['balance_sheet_balancing']['tolerance']
            }
        )
```

## Integration Examples

### Basic Table Processing

```python
from src.backend.nlp_services.table_processing import FinancialTableProcessor

# Initialize table processor
table_processor = FinancialTableProcessor()

# Process tables from a financial document
tables = [
    # Tables extracted from Document AI
    table1_data,
    table2_data,
    table3_data
]

document_context = "This is a 10-K annual report for XYZ Corporation..."

# Process each table
processed_tables = {}
for i, table in enumerate(tables):
    processed_table = await table_processor.process_financial_table(
        table_data=table,
        document_context=document_context,
        table_index=i
    )
    
    if processed_table.table_type:
        processed_tables[processed_table.table_type] = processed_table
        
        print(f"Table {i+1}: {processed_table.table_type}")
        print(f"Confidence: {processed_table.confidence:.2f}")
        print(f"Key metrics: {list(processed_table.financial_metrics.keys())}")

# Calculate comprehensive metrics
metric_calculator = FinancialMetricCalculator()
comprehensive_metrics = await metric_calculator.calculate_comprehensive_metrics(
    processed_tables
)

print(f"\nCalculated {len(comprehensive_metrics.financial_ratios)} financial ratios")
print(f"Growth metrics: {list(comprehensive_metrics.growth_metrics.keys())}")
```

### Cross-Table Validation

```python
# Validate consistency across tables
validator = CrossTableValidator()

validation_result = await validator.validate_table_consistency(processed_tables)

print(f"Overall consistency: {validation_result.overall_consistency}")
print(f"Consistency score: {validation_result.consistency_score:.2f}")

# Check specific validation results
for rule_name, result in validation_result.validation_results.items():
    status = "✓" if result.passed else "✗"
    print(f"{status} {rule_name}: {result.message}")
    
# Display recommendations
if validation_result.recommendations:
    print("\nRecommendations:")
    for rec in validation_result.recommendations:
        print(f"- {rec}")
```

### Advanced Metric Calculations

```python
# Calculate advanced financial metrics
calculator = FinancialMetricCalculator()

# Extract company profile
company_profile = CompanyProfile(
    industry="Technology",
    market_cap=50000000000,  # $50B
    shares_outstanding=1000000000  # 1B shares
)

# Calculate comprehensive metrics
metrics_result = await calculator.calculate_comprehensive_metrics(
    processed_tables, company_profile
)

# Display key ratios
print("Financial Ratios:")
for ratio_name, ratio_value in metrics_result.financial_ratios.items():
    print(f"{ratio_name}: {ratio_value.value:.2f}")

# Display growth metrics
print("\nGrowth Metrics:")
for growth_name, growth_value in metrics_result.growth_metrics.items():
    print(f"{growth_name}: {growth_value.value:.1%}")

# Industry comparison
if metrics_result.industry_comparison:
    print("\nIndustry Comparison:")
    for metric, comparison in metrics_result.industry_comparison.items():
        percentile = comparison.get('percentile', 50)
        print(f"{metric}: {percentile}th percentile")
```

## Best Practices

### Table Processing Guidelines

1. **Multi-Stage Processing**: Use detection, classification, parsing, and validation stages
2. **Context Awareness**: Leverage document context for better table interpretation
3. **Error Handling**: Implement robust error handling for malformed tables
4. **Quality Assessment**: Provide confidence scores for all extracted data

### Financial Data Validation

1. **Cross-Table Consistency**: Validate data consistency across related tables
2. **Business Rule Validation**: Apply financial domain business rules
3. **Reasonableness Checks**: Flag unreasonable values for human review
4. **Completeness Assessment**: Identify missing critical financial data

### Performance Optimization

1. **Parallel Processing**: Process multiple tables concurrently
2. **Caching**: Cache parsed table results for repeated access
3. **Incremental Processing**: Only reprocess changed tables
4. **Memory Management**: Optimize memory usage for large tables

---

*Last updated: 2025-08-30*
*Version: 1.0.0*