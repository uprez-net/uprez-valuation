# Document AI Integration Guide for IPO Valuation Platform

## Overview

Google Cloud Document AI provides powerful document processing capabilities for extracting structured data from IPO prospectuses, financial statements, and regulatory filings. This guide covers the complete implementation for automated document processing in the valuation platform.

## Architecture Overview

### Core Components
- **Document AI Processors**: Specialized parsers for different document types
- **Form Parser**: Structured form data extraction
- **OCR Processing**: Text extraction from scanned documents
- **Custom Classification**: Document type identification
- **Entity Extraction**: Financial entity recognition

### Service Integration Flow

```python
# High-level document processing flow
from google.cloud import documentai
from typing import Dict, List, Any, Optional
import json

class DocumentAIProcessor:
    def __init__(self, project_id: str, location: str):
        self.project_id = project_id
        self.location = location
        self.client = documentai.DocumentProcessorServiceClient()
    
    async def process_document(
        self,
        processor_name: str,
        document_content: bytes,
        mime_type: str
    ) -> Dict[str, Any]:
        """Process document and extract structured data"""
        pass
    
    async def classify_document(self, document_content: bytes) -> str:
        """Classify document type"""
        pass
    
    async def extract_financial_entities(
        self,
        document: documentai.Document
    ) -> List[Dict[str, Any]]:
        """Extract financial entities from processed document"""
        pass
```

## Processor Configuration

### 1. Prospectus Processor Setup

```python
# processors/prospectus_processor.py
import re
from datetime import datetime
from google.cloud import documentai
from typing import Dict, List, Any, Optional

class ProspectusProcessor:
    """Specialized processor for IPO prospectuses"""
    
    def __init__(self, project_id: str, location: str, processor_id: str):
        self.project_id = project_id
        self.location = location
        self.processor_id = processor_id
        self.client = documentai.DocumentProcessorServiceClient()
        
        # Processor resource name
        self.processor_name = self.client.processor_path(
            project_id, location, processor_id
        )
    
    async def process_prospectus(
        self,
        document_content: bytes,
        mime_type: str = "application/pdf"
    ) -> Dict[str, Any]:
        """
        Process IPO prospectus and extract key information
        
        Args:
            document_content: Document bytes
            mime_type: Document MIME type
            
        Returns:
            Structured prospectus data
        """
        
        # Create request
        request = documentai.ProcessRequest(
            name=self.processor_name,
            raw_document=documentai.RawDocument(
                content=document_content,
                mime_type=mime_type
            )
        )
        
        # Process document
        result = self.client.process_document(request=request)
        document = result.document
        
        # Extract structured data
        extracted_data = {
            'company_info': self._extract_company_info(document),
            'financial_highlights': self._extract_financial_highlights(document),
            'offering_details': self._extract_offering_details(document),
            'risk_factors': self._extract_risk_factors(document),
            'use_of_proceeds': self._extract_use_of_proceeds(document),
            'competitive_position': self._extract_competitive_position(document),
            'management_team': self._extract_management_team(document),
            'financial_statements': self._extract_financial_statements(document)
        }
        
        return extracted_data
    
    def _extract_company_info(self, document: documentai.Document) -> Dict[str, Any]:
        """Extract basic company information"""
        company_info = {}
        
        # Extract company name
        company_name_patterns = [
            r'(?:Company Name|Corporation|Inc\.|LLC):?\s*([^\n]+)',
            r'^([A-Z][A-Za-z\s&,.-]+(?:Inc\.|LLC|Corporation|Corp\.|Co\.))',
        ]
        
        text = document.text
        for pattern in company_name_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                company_info['company_name'] = match.group(1).strip()
                break
        
        # Extract incorporation details
        incorporation_patterns = [
            r'incorporated.+?in\s+([A-Za-z\s]+)(?:in|on)?\s*(\d{4})',
            r'Delaware corporation.+?(\d{4})',
            r'incorporated.+?(\d{4})'
        ]
        
        for pattern in incorporation_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                company_info['incorporation_year'] = match.group(-1)
                if len(match.groups()) > 1:
                    company_info['incorporation_state'] = match.group(1).strip()
                break
        
        # Extract business description
        business_desc_patterns = [
            r'(?:Business Overview|Our Business|Company Overview):?\s*\n\s*([^.]+\.)',
            r'We (?:are|operate)\s+([^.]+\.)'
        ]
        
        for pattern in business_desc_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                company_info['business_description'] = match.group(1).strip()
                break
        
        return company_info
    
    def _extract_financial_highlights(self, document: documentai.Document) -> Dict[str, Any]:
        """Extract financial highlights and key metrics"""
        financial_data = {}
        
        text = document.text
        
        # Revenue extraction patterns
        revenue_patterns = [
            r'(?:Revenue|Net Revenue|Total Revenue):?\s*\$?([0-9,]+\.?[0-9]*)\s*(?:million|billion|M|B)?',
            r'(?:Year ended|For the year)\s+\w+\s+\d+,\s+\d+:?\s*\$?([0-9,]+\.?[0-9]*)\s*(?:million|billion)?'
        ]
        
        for pattern in revenue_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                financial_data['revenue'] = [self._parse_financial_number(match) for match in matches]
                break
        
        # Net income patterns
        income_patterns = [
            r'(?:Net Income|Net Loss):?\s*\$?\(?([0-9,]+\.?[0-9]*)\)?\s*(?:million|billion|M|B)?',
            r'(?:Income|Loss)\s+from\s+operations:?\s*\$?\(?([0-9,]+\.?[0-9]*)\)?'
        ]
        
        for pattern in income_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                financial_data['net_income'] = [self._parse_financial_number(match) for match in matches]
                break
        
        # Extract growth rates
        growth_patterns = [
            r'revenue\s+(?:grew|increased)\s+(?:by\s+)?([0-9]+)%',
            r'([0-9]+)%\s+(?:revenue\s+)?growth'
        ]
        
        for pattern in growth_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                financial_data['revenue_growth'] = float(match.group(1))
                break
        
        return financial_data
    
    def _extract_offering_details(self, document: documentai.Document) -> Dict[str, Any]:
        """Extract IPO offering details"""
        offering_details = {}
        
        text = document.text
        
        # Share price range
        price_patterns = [
            r'price\s+range:?\s*\$([0-9]+\.?[0-9]*)\s+to\s+\$([0-9]+\.?[0-9]*)',
            r'between\s+\$([0-9]+\.?[0-9]*)\s+and\s+\$([0-9]+\.?[0-9]*)\s+per\s+share'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                offering_details['price_range'] = {
                    'low': float(match.group(1)),
                    'high': float(match.group(2))
                }
                break
        
        # Number of shares
        shares_patterns = [
            r'([0-9,]+)\s+shares\s+of\s+common\s+stock',
            r'offering\s+of\s+([0-9,]+)\s+shares'
        ]
        
        for pattern in shares_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                shares_str = match.group(1).replace(',', '')
                offering_details['shares_offered'] = int(shares_str)
                break
        
        # Expected IPO date
        date_patterns = [
            r'expected\s+to\s+(?:commence\s+trading|begin\s+trading)\s+on\s+([A-Za-z]+\s+\d+,\s+\d+)',
            r'IPO\s+(?:date|expected):?\s*([A-Za-z]+\s+\d+,\s+\d+)'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    offering_details['expected_date'] = datetime.strptime(
                        match.group(1), '%B %d, %Y'
                    ).isoformat()
                except ValueError:
                    offering_details['expected_date'] = match.group(1)
                break
        
        return offering_details
    
    def _extract_risk_factors(self, document: documentai.Document) -> List[Dict[str, Any]]:
        """Extract risk factors section"""
        text = document.text
        risk_factors = []
        
        # Find risk factors section
        risk_section_pattern = r'RISK\s+FACTORS\s*\n(.*?)(?:\n[A-Z\s]{2,}\n|\n\d+\n|$)'
        risk_match = re.search(risk_section_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if risk_match:
            risk_text = risk_match.group(1)
            
            # Extract individual risk factors
            risk_patterns = [
                r'•\s*([^•]+?)(?=•|\n\n|\Z)',  # Bullet points
                r'\n([^.\n]+\.)',  # Sentences ending with periods
                r'(?:Risk|Risks?):\s*([^.\n]+\.)'  # Explicit risk statements
            ]
            
            for pattern in risk_patterns:
                matches = re.findall(pattern, risk_text, re.DOTALL)
                for match in matches:
                    risk_text_clean = match.strip()
                    if len(risk_text_clean) > 20:  # Filter out short/incomplete matches
                        risk_factors.append({
                            'risk_description': risk_text_clean,
                            'category': self._categorize_risk(risk_text_clean)
                        })
                
                if risk_factors:  # If we found risks with this pattern, stop
                    break
        
        return risk_factors[:20]  # Limit to top 20 risks
    
    def _extract_use_of_proceeds(self, document: documentai.Document) -> Dict[str, Any]:
        """Extract use of proceeds information"""
        text = document.text
        use_of_proceeds = {}
        
        # Find use of proceeds section
        proceeds_pattern = r'USE\s+OF\s+PROCEEDS\s*\n(.*?)(?:\n[A-Z\s]{2,}\n|\n\d+\n|$)'
        proceeds_match = re.search(proceeds_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if proceeds_match:
            proceeds_text = proceeds_match.group(1)
            
            # Extract specific uses with amounts
            amount_patterns = [
                r'([^.\n]+?)(?:\s+\$([0-9,]+\.?[0-9]*)\s*(?:million|billion|M|B)?)',
                r'\$([0-9,]+\.?[0-9]*)\s*(?:million|billion|M|B)?\s+(?:for|to)\s+([^.\n]+)'
            ]
            
            uses = []
            for pattern in amount_patterns:
                matches = re.findall(pattern, proceeds_text, re.IGNORECASE)
                for match in matches:
                    if len(match) == 2:
                        use_desc, amount = match
                        uses.append({
                            'description': use_desc.strip(),
                            'amount': self._parse_financial_number(amount)
                        })
            
            use_of_proceeds['specific_uses'] = uses
            use_of_proceeds['general_description'] = proceeds_text[:500].strip()
        
        return use_of_proceeds
    
    def _parse_financial_number(self, number_str: str) -> float:
        """Parse financial number string to float"""
        # Remove commas and common formatting
        cleaned = number_str.replace(',', '').replace('$', '').replace('(', '').replace(')', '')
        
        # Handle millions/billions
        multiplier = 1
        if 'billion' in number_str.lower() or 'B' in number_str:
            multiplier = 1e9
        elif 'million' in number_str.lower() or 'M' in number_str:
            multiplier = 1e6
        
        try:
            return float(cleaned) * multiplier
        except ValueError:
            return 0.0
    
    def _categorize_risk(self, risk_text: str) -> str:
        """Categorize risk factor based on content"""
        risk_categories = {
            'market': ['market', 'competition', 'demand', 'economic', 'recession'],
            'operational': ['operations', 'management', 'employees', 'supply chain'],
            'financial': ['cash', 'debt', 'financing', 'liquidity', 'revenue'],
            'regulatory': ['regulation', 'compliance', 'legal', 'lawsuit', 'government'],
            'technology': ['technology', 'cybersecurity', 'systems', 'data breach'],
            'industry': ['industry', 'sector', 'trends', 'disruption']
        }
        
        risk_lower = risk_text.lower()
        for category, keywords in risk_categories.items():
            if any(keyword in risk_lower for keyword in keywords):
                return category
        
        return 'general'
```

### 2. Financial Statement Processor

```python
# processors/financial_statement_processor.py
class FinancialStatementProcessor:
    """Processor for financial statements and regulatory filings"""
    
    def __init__(self, project_id: str, location: str, processor_id: str):
        self.project_id = project_id
        self.location = location
        self.processor_id = processor_id
        self.client = documentai.DocumentProcessorServiceClient()
        
        self.processor_name = self.client.processor_path(
            project_id, location, processor_id
        )
    
    async def process_financial_statement(
        self,
        document_content: bytes,
        statement_type: str = "unknown",
        mime_type: str = "application/pdf"
    ) -> Dict[str, Any]:
        """
        Process financial statement and extract structured data
        
        Args:
            document_content: Document bytes
            statement_type: Type of financial statement (income, balance, cash_flow)
            mime_type: Document MIME type
            
        Returns:
            Structured financial data
        """
        
        # Process document
        request = documentai.ProcessRequest(
            name=self.processor_name,
            raw_document=documentai.RawDocument(
                content=document_content,
                mime_type=mime_type
            )
        )
        
        result = self.client.process_document(request=request)
        document = result.document
        
        # Extract based on statement type
        if statement_type == "income":
            return self._extract_income_statement(document)
        elif statement_type == "balance":
            return self._extract_balance_sheet(document)
        elif statement_type == "cash_flow":
            return self._extract_cash_flow_statement(document)
        else:
            return self._extract_all_statements(document)
    
    def _extract_income_statement(self, document: documentai.Document) -> Dict[str, Any]:
        """Extract income statement data"""
        income_data = {
            'statement_type': 'income_statement',
            'periods': [],
            'line_items': {}
        }
        
        text = document.text
        
        # Extract periods (years/quarters)
        period_patterns = [
            r'(?:Year|Quarter)\s+ended\s+([A-Za-z]+\s+\d+,\s+\d+)',
            r'(\d{4})\s+(\d{4})\s+(\d{4})',  # Three year comparison
            r'(Q[1-4]\s+\d{4})\s+(Q[1-4]\s+\d{4})'  # Quarterly comparison
        ]
        
        for pattern in period_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                income_data['periods'] = [match for match in matches[0] if match]
                break
        
        # Extract key line items
        line_item_patterns = {
            'revenue': [
                r'(?:Total\s+)?(?:Revenue|Net\s+Revenue|Sales):?\s*\$?\s*([0-9,]+\.?[0-9]*)',
                r'Net\s+sales:?\s*\$?\s*([0-9,]+\.?[0-9]*)'
            ],
            'cost_of_revenue': [
                r'Cost\s+of\s+(?:revenue|sales|goods\s+sold):?\s*\$?\s*([0-9,]+\.?[0-9]*)',
                r'Cost\s+of\s+sales:?\s*\$?\s*([0-9,]+\.?[0-9]*)'
            ],
            'gross_profit': [
                r'Gross\s+profit:?\s*\$?\s*([0-9,]+\.?[0-9]*)'
            ],
            'operating_expenses': [
                r'(?:Total\s+)?Operating\s+expenses:?\s*\$?\s*([0-9,]+\.?[0-9]*)',
                r'Selling,\s+general\s+and\s+administrative:?\s*\$?\s*([0-9,]+\.?[0-9]*)'
            ],
            'operating_income': [
                r'(?:Income|Loss)\s+from\s+operations:?\s*\$?\s*\(?([0-9,]+\.?[0-9]*)\)?',
                r'Operating\s+(?:income|loss):?\s*\$?\s*\(?([0-9,]+\.?[0-9]*)\)?'
            ],
            'net_income': [
                r'Net\s+(?:income|loss):?\s*\$?\s*\(?([0-9,]+\.?[0-9]*)\)?',
                r'Net\s+(?:income|loss)\s+attributable\s+to.*:?\s*\$?\s*\(?([0-9,]+\.?[0-9]*)\)?'
            ],
            'earnings_per_share': [
                r'(?:Basic|Diluted)\s+earnings\s+per\s+share:?\s*\$?\s*([0-9]+\.?[0-9]*)',
                r'EPS:?\s*\$?\s*([0-9]+\.?[0-9]*)'
            ]
        }
        
        for item_name, patterns in line_item_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Convert to numbers and store
                    values = []
                    for match in matches:
                        try:
                            # Handle negative values in parentheses
                            if '(' in text and ')' in text:
                                value = -float(match.replace(',', ''))
                            else:
                                value = float(match.replace(',', ''))
                            values.append(value)
                        except ValueError:
                            continue
                    
                    if values:
                        income_data['line_items'][item_name] = values
                    break
        
        return income_data
    
    def _extract_balance_sheet(self, document: documentai.Document) -> Dict[str, Any]:
        """Extract balance sheet data"""
        balance_data = {
            'statement_type': 'balance_sheet',
            'periods': [],
            'assets': {},
            'liabilities': {},
            'equity': {}
        }
        
        text = document.text
        
        # Asset extraction patterns
        asset_patterns = {
            'cash_and_equivalents': [
                r'Cash\s+and\s+cash\s+equivalents:?\s*\$?\s*([0-9,]+\.?[0-9]*)',
                r'Cash\s+and\s+short-term\s+investments:?\s*\$?\s*([0-9,]+\.?[0-9]*)'
            ],
            'accounts_receivable': [
                r'(?:Trade\s+)?(?:Accounts\s+)?receivable:?\s*\$?\s*([0-9,]+\.?[0-9]*)',
                r'Receivables:?\s*\$?\s*([0-9,]+\.?[0-9]*)'
            ],
            'inventory': [
                r'Inventor(?:y|ies):?\s*\$?\s*([0-9,]+\.?[0-9]*)'
            ],
            'total_current_assets': [
                r'Total\s+current\s+assets:?\s*\$?\s*([0-9,]+\.?[0-9]*)'
            ],
            'property_plant_equipment': [
                r'Property,?\s+plant\s+and\s+equipment:?\s*\$?\s*([0-9,]+\.?[0-9]*)',
                r'PP&E:?\s*\$?\s*([0-9,]+\.?[0-9]*)'
            ],
            'total_assets': [
                r'Total\s+assets:?\s*\$?\s*([0-9,]+\.?[0-9]*)'
            ]
        }
        
        # Extract asset values
        for asset_name, patterns in asset_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    balance_data['assets'][asset_name] = [
                        float(match.replace(',', '')) for match in matches
                    ]
                    break
        
        # Liability patterns
        liability_patterns = {
            'accounts_payable': [
                r'(?:Trade\s+)?(?:Accounts\s+)?payable:?\s*\$?\s*([0-9,]+\.?[0-9]*)'
            ],
            'short_term_debt': [
                r'Short-term\s+debt:?\s*\$?\s*([0-9,]+\.?[0-9]*)',
                r'Current\s+portion\s+of\s+long-term\s+debt:?\s*\$?\s*([0-9,]+\.?[0-9]*)'
            ],
            'total_current_liabilities': [
                r'Total\s+current\s+liabilities:?\s*\$?\s*([0-9,]+\.?[0-9]*)'
            ],
            'long_term_debt': [
                r'Long-term\s+debt:?\s*\$?\s*([0-9,]+\.?[0-9]*)'
            ],
            'total_liabilities': [
                r'Total\s+liabilities:?\s*\$?\s*([0-9,]+\.?[0-9]*)'
            ]
        }
        
        # Extract liability values
        for liability_name, patterns in liability_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    balance_data['liabilities'][liability_name] = [
                        float(match.replace(',', '')) for match in matches
                    ]
                    break
        
        # Equity patterns
        equity_patterns = {
            'retained_earnings': [
                r'Retained\s+earnings:?\s*\$?\s*([0-9,]+\.?[0-9]*)'
            ],
            'stockholders_equity': [
                r'(?:Total\s+)?(?:Stockholders|Shareholders)\s+equity:?\s*\$?\s*([0-9,]+\.?[0-9]*)',
                r'Total\s+equity:?\s*\$?\s*([0-9,]+\.?[0-9]*)'
            ]
        }
        
        # Extract equity values
        for equity_name, patterns in equity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    balance_data['equity'][equity_name] = [
                        float(match.replace(',', '')) for match in matches
                    ]
                    break
        
        return balance_data
```

### 3. Document Classification

```python
# classification/document_classifier.py
import numpy as np
from typing import Dict, List, Tuple
from google.cloud import documentai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

class DocumentClassifier:
    """Classify financial documents by type"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.classifier = MultinomialNB()
        self.label_encoder = {}
        
        if model_path:
            self.load_model(model_path)
        else:
            self._initialize_default_classifier()
    
    def _initialize_default_classifier(self):
        """Initialize with predefined document patterns"""
        # Document type keywords
        document_patterns = {
            'prospectus': [
                'initial public offering', 'securities and exchange commission',
                'use of proceeds', 'risk factors', 'underwriting', 'offering price',
                'shares offered', 'lead underwriter', 'registration statement'
            ],
            'financial_statement': [
                'balance sheet', 'income statement', 'cash flow statement',
                'statement of operations', 'consolidated financial statements',
                'auditor report', 'generally accepted accounting principles'
            ],
            'annual_report': [
                'form 10-k', 'annual report', 'management discussion',
                'auditor report', 'fiscal year ended', 'business overview'
            ],
            'quarterly_report': [
                'form 10-q', 'quarterly report', 'three months ended',
                'nine months ended', 'interim financial statements'
            ],
            'proxy_statement': [
                'proxy statement', 'annual meeting', 'executive compensation',
                'board of directors', 'shareholder meeting', 'voting matters'
            ]
        }
        
        # Create training data
        texts = []
        labels = []
        
        for doc_type, keywords in document_patterns.items():
            for keyword in keywords:
                # Create synthetic training examples
                texts.append(f"This document contains {keyword} information")
                labels.append(doc_type)
        
        # Train classifier
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        
        # Create label mapping
        unique_labels = list(set(labels))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
    
    def classify_document(
        self,
        document_text: str,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Classify document type with confidence score
        
        Args:
            document_text: Text content of the document
            confidence_threshold: Minimum confidence for classification
            
        Returns:
            Classification result with confidence scores
        """
        
        # Vectorize text
        X = self.vectorizer.transform([document_text])
        
        # Get prediction probabilities
        probabilities = self.classifier.predict_proba(X)[0]
        
        # Get class labels
        classes = self.classifier.classes_
        
        # Create confidence mapping
        confidence_scores = dict(zip(classes, probabilities))
        
        # Get best prediction
        best_class = classes[np.argmax(probabilities)]
        best_confidence = max(probabilities)
        
        # Check if confidence meets threshold
        is_confident = best_confidence >= confidence_threshold
        
        result = {
            'predicted_type': best_class if is_confident else 'unknown',
            'confidence': float(best_confidence),
            'is_confident': is_confident,
            'all_scores': confidence_scores,
            'threshold': confidence_threshold
        }
        
        return result
    
    def batch_classify(
        self,
        documents: List[str],
        confidence_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Classify multiple documents"""
        return [
            self.classify_document(doc, confidence_threshold)
            for doc in documents
        ]
    
    def save_model(self, model_path: str):
        """Save trained classifier"""
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'label_encoder': self.label_encoder
        }
        joblib.dump(model_data, model_path)
    
    def load_model(self, model_path: str):
        """Load trained classifier"""
        model_data = joblib.load(model_path)
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
```

## Integration with BigQuery

### Document Storage Schema

```python
# storage/bigquery_integration.py
from google.cloud import bigquery
from typing import Dict, Any, List
import json

class DocumentDataStorage:
    """Store processed document data in BigQuery"""
    
    def __init__(self, project_id: str, dataset_id: str):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.client = bigquery.Client()
    
    def store_processed_document(
        self,
        document_id: str,
        document_type: str,
        extracted_data: Dict[str, Any],
        processing_metadata: Dict[str, Any]
    ) -> str:
        """Store processed document data"""
        
        # Prepare data for BigQuery
        row_data = {
            'document_id': document_id,
            'document_type': document_type,
            'processing_timestamp': processing_metadata.get('timestamp'),
            'processor_version': processing_metadata.get('processor_version'),
            'confidence_score': processing_metadata.get('confidence', 0.0),
            'extracted_data': json.dumps(extracted_data),
            'metadata': json.dumps(processing_metadata)
        }
        
        # Insert into processed_documents table
        table_ref = self.client.dataset(self.dataset_id).table('processed_documents')
        errors = self.client.insert_rows_json(table_ref, [row_data])
        
        if errors:
            raise Exception(f"Failed to insert document data: {errors}")
        
        return document_id
    
    def store_financial_entities(
        self,
        document_id: str,
        entities: List[Dict[str, Any]]
    ):
        """Store extracted financial entities"""
        
        rows = []
        for entity in entities:
            row = {
                'document_id': document_id,
                'entity_type': entity.get('type'),
                'entity_value': entity.get('value'),
                'confidence': entity.get('confidence', 0.0),
                'start_offset': entity.get('start_offset'),
                'end_offset': entity.get('end_offset'),
                'extraction_timestamp': entity.get('timestamp')
            }
            rows.append(row)
        
        # Insert into financial_entities table
        table_ref = self.client.dataset(self.dataset_id).table('financial_entities')
        errors = self.client.insert_rows_json(table_ref, rows)
        
        if errors:
            raise Exception(f"Failed to insert entity data: {errors}")
```

## API Integration Layer

### Document Processing API

```python
# api/document_processing_api.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Optional
import asyncio

app = FastAPI(title="Document AI Processing API")

class DocumentProcessingAPI:
    """API layer for document processing services"""
    
    def __init__(self):
        self.prospectus_processor = ProspectusProcessor(
            project_id=PROJECT_ID,
            location=LOCATION,
            processor_id=PROSPECTUS_PROCESSOR_ID
        )
        
        self.financial_processor = FinancialStatementProcessor(
            project_id=PROJECT_ID,
            location=LOCATION,
            processor_id=FINANCIAL_PROCESSOR_ID
        )
        
        self.classifier = DocumentClassifier()
        self.storage = DocumentDataStorage(PROJECT_ID, DATASET_ID)

@app.post("/process-document/")
async def process_document(
    file: UploadFile = File(...),
    document_type: Optional[str] = None
):
    """Process uploaded document"""
    
    try:
        # Read file content
        content = await file.read()
        
        # Classify document if type not provided
        if not document_type:
            # Extract text for classification (simplified)
            text = content.decode('utf-8', errors='ignore')
            classification = api.classifier.classify_document(text)
            document_type = classification['predicted_type']
        
        # Process based on document type
        if document_type == 'prospectus':
            result = await api.prospectus_processor.process_prospectus(
                content, file.content_type
            )
        elif document_type in ['financial_statement', 'annual_report', 'quarterly_report']:
            result = await api.financial_processor.process_financial_statement(
                content, document_type, file.content_type
            )
        else:
            raise HTTPException(status_code=400, f"Unsupported document type: {document_type}")
        
        # Store results
        document_id = f"{file.filename}_{datetime.utcnow().timestamp()}"
        api.storage.store_processed_document(
            document_id=document_id,
            document_type=document_type,
            extracted_data=result,
            processing_metadata={
                'filename': file.filename,
                'content_type': file.content_type,
                'timestamp': datetime.utcnow().isoformat(),
                'file_size': len(content)
            }
        )
        
        return {
            'document_id': document_id,
            'document_type': document_type,
            'status': 'processed',
            'extracted_data': result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-process/")
async def batch_process_documents(files: List[UploadFile] = File(...)):
    """Process multiple documents in batch"""
    
    results = []
    
    # Process documents concurrently
    tasks = []
    for file in files:
        task = process_document(file)
        tasks.append(task)
    
    # Wait for all processing to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return {
        'batch_id': f"batch_{datetime.utcnow().timestamp()}",
        'total_documents': len(files),
        'results': results
    }
```

## Monitoring and Error Handling

### Processing Monitoring

```python
# monitoring/document_processing_monitor.py
from google.cloud import monitoring_v3
from typing import Dict, Any
import time

class DocumentProcessingMonitor:
    """Monitor document processing performance and errors"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = monitoring_v3.MetricServiceClient()
        
    def record_processing_metrics(
        self,
        document_type: str,
        processing_time: float,
        success: bool,
        confidence_score: float
    ):
        """Record processing metrics"""
        
        # Create time series data
        now = time.time()
        seconds = int(now)
        nanos = int((now - seconds) * 10**9)
        interval = monitoring_v3.TimeInterval({
            "end_time": {"seconds": seconds, "nanos": nanos}
        })
        
        # Processing time metric
        time_series = monitoring_v3.TimeSeries()
        time_series.metric.type = "custom.googleapis.com/document_ai/processing_time"
        time_series.resource.type = "global"
        time_series.metric.labels["document_type"] = document_type
        
        point = monitoring_v3.Point()
        point.value.double_value = processing_time
        point.interval = interval
        time_series.points = [point]
        
        # Send metrics
        project_name = f"projects/{self.project_id}"
        self.client.create_time_series(
            name=project_name,
            time_series=[time_series]
        )
        
        # Success rate metric
        success_series = monitoring_v3.TimeSeries()
        success_series.metric.type = "custom.googleapis.com/document_ai/success_rate"
        success_series.resource.type = "global"
        success_series.metric.labels["document_type"] = document_type
        
        success_point = monitoring_v3.Point()
        success_point.value.double_value = 1.0 if success else 0.0
        success_point.interval = interval
        success_series.points = [success_point]
        
        # Confidence score metric
        confidence_series = monitoring_v3.TimeSeries()
        confidence_series.metric.type = "custom.googleapis.com/document_ai/confidence_score"
        confidence_series.resource.type = "global"
        confidence_series.metric.labels["document_type"] = document_type
        
        confidence_point = monitoring_v3.Point()
        confidence_point.value.double_value = confidence_score
        confidence_point.interval = interval
        confidence_series.points = [confidence_point]
        
        # Send all metrics
        self.client.create_time_series(
            name=project_name,
            time_series=[success_series, confidence_series]
        )
```

## Best Practices

### 1. Document Quality Optimization
- Implement preprocessing for scanned documents
- Use optimal image resolution (300 DPI minimum)
- Handle multi-page documents efficiently
- Implement document orientation correction

### 2. Error Handling
- Implement retry logic for API calls
- Handle rate limiting gracefully
- Provide fallback processing methods
- Log detailed error information

### 3. Performance Optimization
- Use batch processing for multiple documents
- Implement caching for repeated processing
- Optimize processor selection based on document type
- Monitor and optimize processing costs

### 4. Data Quality
- Implement validation for extracted data
- Use confidence thresholds for quality control
- Implement human-in-the-loop for low-confidence results
- Regular model retraining with new data

## Next Steps

1. **Custom Processor Training**: Train processors for specific document formats
2. **Advanced Entity Recognition**: Implement domain-specific NER models
3. **Workflow Automation**: Set up automated processing pipelines
4. **Integration Testing**: Comprehensive testing with real documents
5. **Performance Optimization**: Fine-tune for speed and accuracy

## Related Documentation

- [Vertex AI Integration](../vertex-ai/README.md)
- [Natural Language AI](../natural-language/README.md)
- [Cost Optimization](../cost-optimization/README.md)