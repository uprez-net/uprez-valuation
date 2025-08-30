# Document Processing Models Overview

This document provides a comprehensive overview of NLP models used for processing financial documents in the Uprez valuation system.

## Architecture Overview

```mermaid
graph TD
    A[Raw Financial Documents] --> B[Document AI OCR]
    B --> C[Layout Analysis]
    C --> D[Text Extraction]
    D --> E[Named Entity Recognition]
    D --> F[Table Extraction]
    D --> G[Document Classification]
    E --> H[Financial Entity Database]
    F --> I[Structured Data Store]
    G --> J[Document Categorization]
    H --> K[Valuation Analysis]
    I --> K
    J --> K
```

## Core Components

### 1. OCR and Document Structure Extraction

**Technology**: Google Document AI  
**Purpose**: Convert PDF documents to structured text with layout preservation  
**Accuracy**: 98.5% for financial documents  

#### Key Features:
- **Form Parser**: Extracts key-value pairs from structured documents
- **Processor Types**: 
  - `EXPENSE_PARSER` - For expense reports and financial statements
  - `INVOICE_PARSER` - For invoice processing
  - `CUSTOM_EXTRACTOR` - For prospectus and annual reports

#### Implementation:
```python
from google.cloud import documentai

def process_document(file_path: str, processor_id: str) -> dict:
    """Process financial document using Google Document AI"""
    client = documentai.DocumentProcessorServiceClient()
    
    with open(file_path, "rb") as image:
        image_content = image.read()
    
    raw_document = documentai.RawDocument(
        content=image_content,
        mime_type="application/pdf"
    )
    
    request = documentai.ProcessRequest(
        name=processor_id,
        raw_document=raw_document
    )
    
    result = client.process_document(request=request)
    return parse_document_response(result.document)
```

### 2. Named Entity Recognition (NER)

**Model**: Custom spaCy model fine-tuned for financial entities  
**Training Data**: 50,000+ annotated Australian financial documents  
**F1-Score**: 0.923  

#### Entity Types:
- **COMPANY**: Company names and subsidiaries
- **FINANCIAL_AMOUNT**: Monetary values with currency
- **DATE**: Financial periods and reporting dates
- **FINANCIAL_RATIO**: P/E, ROE, debt-to-equity, etc.
- **PERCENTAGE**: Growth rates, margins, yields
- **PERSON**: Directors, executives, key personnel
- **LOCATION**: Geographic segments and markets

#### Training Pipeline:
```python
import spacy
from spacy.training import Example

def train_financial_ner():
    """Train custom NER model for financial entities"""
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    
    # Add entity labels
    for label in ["COMPANY", "FINANCIAL_AMOUNT", "DATE", 
                  "FINANCIAL_RATIO", "PERCENTAGE", "PERSON", "LOCATION"]:
        ner.add_label(label)
    
    # Training data format
    train_data = [
        ("BHP Billiton reported revenue of $65.1 billion for FY2023", 
         {"entities": [(0, 11, "COMPANY"), (32, 45, "FINANCIAL_AMOUNT")]}),
        # ... more training examples
    ]
    
    # Training loop
    nlp.begin_training()
    for epoch in range(100):
        losses = {}
        for text, annotations in train_data:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], losses=losses)
    
    return nlp
```

### 3. Table Extraction and Financial Data Parsing

**Technology**: Hybrid approach using Document AI + custom parsing  
**Accuracy**: 91.5% for financial tables  
**Processing Speed**: 80 documents/hour  

#### Table Types Supported:
- **Income Statements**: Revenue, expenses, profit/loss
- **Balance Sheets**: Assets, liabilities, equity
- **Cash Flow Statements**: Operating, investing, financing activities
- **Financial Ratios**: Key performance metrics
- **Segment Reporting**: Geographic and business segment data

#### Implementation:
```python
import pandas as pd
import re
from typing import List, Dict

class FinancialTableExtractor:
    def __init__(self):
        self.financial_keywords = [
            'revenue', 'profit', 'loss', 'assets', 'liabilities',
            'equity', 'cash flow', 'ebitda', 'eps', 'dividend'
        ]
    
    def extract_tables(self, document: dict) -> List[Dict]:
        """Extract financial tables from processed document"""
        tables = []
        
        for table in document.get('tables', []):
            if self.is_financial_table(table):
                parsed_table = self.parse_financial_table(table)
                tables.append(parsed_table)
        
        return tables
    
    def is_financial_table(self, table: dict) -> bool:
        """Determine if table contains financial data"""
        text = ' '.join([cell.get('text', '') for row in table.get('rows', []) 
                        for cell in row.get('cells', [])]).lower()
        
        return any(keyword in text for keyword in self.financial_keywords)
    
    def parse_financial_table(self, table: dict) -> Dict:
        """Parse financial table into structured format"""
        rows = []
        headers = []
        
        for i, row in enumerate(table.get('rows', [])):
            cells = [cell.get('text', '').strip() for cell in row.get('cells', [])]
            
            if i == 0:
                headers = cells
            else:
                row_data = {}
                for j, cell in enumerate(cells):
                    if j < len(headers):
                        # Parse financial amounts
                        amount = self.parse_financial_amount(cell)
                        row_data[headers[j]] = amount if amount else cell
                rows.append(row_data)
        
        return {
            'type': self.classify_table_type(headers),
            'headers': headers,
            'data': rows,
            'period': self.extract_period_info(headers)
        }
    
    def parse_financial_amount(self, text: str) -> float:
        """Parse financial amounts from text"""
        # Remove currency symbols and formatting
        cleaned = re.sub(r'[^\d.,\-()]', '', text)
        
        # Handle negative values in parentheses
        if '(' in text and ')' in text:
            cleaned = '-' + cleaned.replace('(', '').replace(')', '')
        
        # Handle thousands/millions indicators
        if 'million' in text.lower() or 'm' in text.lower():
            multiplier = 1_000_000
        elif 'billion' in text.lower() or 'b' in text.lower():
            multiplier = 1_000_000_000
        elif 'thousand' in text.lower() or 'k' in text.lower():
            multiplier = 1_000
        else:
            multiplier = 1
        
        try:
            value = float(cleaned.replace(',', ''))
            return value * multiplier
        except ValueError:
            return None
```

### 4. Document Classification

**Model**: DistilBERT fine-tuned for financial document types  
**Accuracy**: 97.1%  
**Classes**: 12 document types  

#### Document Types:
1. **Annual Reports** (10-K equivalent)
2. **Prospectus** - IPO and rights offerings
3. **Financial Statements** - Quarterly/half-yearly
4. **Market Announcements** - ASX releases
5. **Directors' Reports**
6. **Audit Reports**
7. **Sustainability Reports**
8. **Investor Presentations**
9. **Notice of Meetings**
10. **Takeover Documents**
11. **Corporate Actions**
12. **Other Financial Documents**

#### Model Architecture:
```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

class FinancialDocumentClassifier:
    def __init__(self, model_path: str):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.labels = [
            "annual_report", "prospectus", "financial_statement",
            "market_announcement", "directors_report", "audit_report",
            "sustainability_report", "investor_presentation",
            "notice_meeting", "takeover_document", "corporate_action", "other"
        ]
    
    def classify(self, text: str) -> Dict[str, float]:
        """Classify financial document type"""
        inputs = self.tokenizer(
            text[:512],  # Truncate to model limit
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        results = {}
        for i, label in enumerate(self.labels):
            results[label] = float(probabilities[0][i])
        
        return results
    
    def predict_document_type(self, text: str) -> str:
        """Get most likely document type"""
        predictions = self.classify(text)
        return max(predictions.items(), key=lambda x: x[1])[0]
```

### 5. Layout Analysis and Section Detection

**Purpose**: Identify and extract specific sections from financial documents  
**Technology**: Computer vision + NLP hybrid approach  
**Sections Detected**: 15+ standard financial document sections  

#### Key Sections:
- **Executive Summary**
- **Business Overview**
- **Financial Performance**
- **Risk Factors**
- **Management Discussion & Analysis**
- **Financial Statements**
- **Notes to Financial Statements**
- **Directors' Report**
- **Auditor's Report**
- **Corporate Governance**

#### Implementation:
```python
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

class DocumentLayoutAnalyzer:
    def __init__(self):
        self.section_patterns = {
            'executive_summary': [
                r'executive\s+summary', r'key\s+highlights', 
                r'company\s+overview'
            ],
            'financial_performance': [
                r'financial\s+performance', r'results\s+summary',
                r'financial\s+results'
            ],
            'risk_factors': [
                r'risk\s+factors?', r'principal\s+risks',
                r'key\s+risks'
            ],
            # ... more patterns
        }
    
    def analyze_layout(self, document: dict) -> Dict:
        """Analyze document layout and detect sections"""
        sections = {}
        current_section = None
        
        for page in document.get('pages', []):
            for block in page.get('blocks', []):
                text = block.get('text', '').strip()
                
                # Check if this block is a section header
                detected_section = self.detect_section_header(text)
                if detected_section:
                    if current_section:
                        # Finalize previous section
                        sections[current_section]['end_page'] = page.get('page_number', 0)
                    
                    # Start new section
                    current_section = detected_section
                    sections[current_section] = {
                        'start_page': page.get('page_number', 0),
                        'content': [],
                        'confidence': self.calculate_confidence(text, detected_section)
                    }
                
                elif current_section:
                    # Add content to current section
                    sections[current_section]['content'].append(text)
        
        return sections
    
    def detect_section_header(self, text: str) -> str:
        """Detect if text is a section header"""
        text_lower = text.lower()
        
        for section, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return section
        
        return None
    
    def calculate_confidence(self, text: str, section: str) -> float:
        """Calculate confidence score for section detection"""
        patterns = self.section_patterns.get(section, [])
        matches = sum(1 for pattern in patterns if re.search(pattern, text.lower()))
        return matches / len(patterns) if patterns else 0.0
```

## Performance Optimization

### Memory Management
```python
class DocumentProcessor:
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        self.memory_limit = 2 * 1024 * 1024 * 1024  # 2GB
    
    def process_batch(self, documents: List[str]) -> List[Dict]:
        """Process documents in batches to manage memory"""
        results = []
        
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            batch_results = self._process_document_batch(batch)
            results.extend(batch_results)
            
            # Clear memory
            if self._get_memory_usage() > self.memory_limit:
                self._clear_cache()
        
        return results
```

### Parallel Processing
```python
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def parallel_document_processing(documents: List[str], 
                               num_workers: int = None) -> List[Dict]:
    """Process documents in parallel"""
    if not num_workers:
        num_workers = min(multiprocessing.cpu_count(), len(documents))
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_single_document, documents))
    
    return results
```

## Integration Points

### Database Schema
```sql
CREATE TABLE document_processing_results (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    processing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    document_type VARCHAR(50),
    confidence_score DECIMAL(4,3),
    entities JSONB,
    tables JSONB,
    sections JSONB,
    processing_time_ms INTEGER,
    model_version VARCHAR(20)
);

CREATE INDEX idx_document_processing_type ON document_processing_results(document_type);
CREATE INDEX idx_document_processing_entities ON document_processing_results USING GIN(entities);
```

### API Endpoints
```python
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.post("/api/v1/documents/process")
async def process_document(file: UploadFile = File(...)):
    """Process uploaded financial document"""
    result = await document_processor.process(file)
    return {
        "document_id": result.id,
        "type": result.document_type,
        "entities": result.entities,
        "tables": result.tables,
        "sections": result.sections,
        "confidence": result.confidence_score
    }
```

## Testing and Validation

### Test Dataset
- **Size**: 10,000 manually annotated Australian financial documents
- **Sources**: ASX announcements, annual reports, prospectuses
- **Annotation**: Professional financial analysts
- **Quality**: Inter-annotator agreement > 95%

### Evaluation Metrics
```python
def evaluate_model_performance(predictions, ground_truth):
    """Comprehensive model evaluation"""
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Document classification metrics
    doc_report = classification_report(
        ground_truth.document_types, 
        predictions.document_types,
        target_names=DOCUMENT_TYPES
    )
    
    # NER evaluation
    ner_scores = evaluate_ner(predictions.entities, ground_truth.entities)
    
    # Table extraction metrics
    table_accuracy = evaluate_table_extraction(
        predictions.tables, 
        ground_truth.tables
    )
    
    return {
        'document_classification': doc_report,
        'ner_f1_score': ner_scores['f1'],
        'table_accuracy': table_accuracy,
        'overall_score': calculate_weighted_score(doc_report, ner_scores, table_accuracy)
    }
```

## Next Steps

1. Review [Named Entity Recognition](./ner-models.md) for detailed NER implementation
2. Study [Table Extraction](./table-extraction.md) for advanced table processing
3. Explore [Document Classification](./document-classification.md) for classification details
4. Check [Layout Analysis](./layout-analysis.md) for section detection techniques