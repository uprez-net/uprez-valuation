# Named Entity Recognition (NER) for Financial Entities

## Overview

This document provides comprehensive technical documentation for Named Entity Recognition (NER) models specifically designed to extract financial entities from documents in the Uprez Valuation system. The system leverages state-of-the-art NLP models fine-tuned for financial domain applications, including specialized datasets and custom entity types relevant to financial analysis.

## Financial Entity Types

### Core Financial Entities

The system recognizes and extracts the following financial entity types:

1. **Company Information**
   - Company names and variations
   - Trading symbols and tickers
   - Stock exchanges
   - CIK (Central Index Key) numbers
   - LEI (Legal Entity Identifier) codes

2. **Financial Metrics and KPIs**
   - Revenue and sales figures
   - Profit and loss amounts
   - EBITDA and operating income
   - Debt and equity values
   - Cash flow metrics

3. **Market Data**
   - Stock prices and valuations
   - Market capitalization
   - Trading volumes
   - Price targets and ratings
   - Market indices

4. **Regulatory Information**
   - SEC filing numbers
   - CUSIP and ISIN codes
   - Regulatory dates and deadlines
   - Compliance references

5. **People and Organizations**
   - Executive names and titles
   - Board members
   - Auditors and advisors
   - Regulatory bodies

6. **Financial Instruments**
   - Bond types and ratings
   - Derivative instruments
   - Currency codes
   - Interest rates

## Model Architecture

### FiNER-139 Integration

The system integrates the FiNER-139 model, which is specifically designed for financial named entity recognition:

```python
class FinancialNER:
    def __init__(self):
        # Load FiNER-139 model for XBRL tagging
        self.finer_model = AutoModelForTokenClassification.from_pretrained(
            "nlpaueb/finer-139"
        )
        self.finer_tokenizer = AutoTokenizer.from_pretrained(
            "nlpaueb/finer-139"
        )
        
        # 139 financial entity types
        self.entity_types = self._load_entity_mapping()
        
        # Company-specific entities
        self.company_ner = spacy.load("en_core_web_sm")
        self.company_ner.add_pipe("financial_entities")
```

### Entity Label Set

The FiNER-139 model uses 139 entity types extracted from XBRL tags, including:

#### Balance Sheet Entities
- `Assets` - Total assets
- `AssetsCurrent` - Current assets
- `CashAndCashEquivalentsAtCarryingValue` - Cash and equivalents
- `AccountsReceivableNet` - Accounts receivable
- `Inventory` - Inventory values
- `PropertyPlantAndEquipmentNet` - PP&E net value
- `Goodwill` - Goodwill amounts
- `IntangibleAssetsNet` - Intangible assets

#### Income Statement Entities
- `Revenues` - Revenue figures
- `CostOfRevenue` - Cost of revenue
- `GrossProfit` - Gross profit amounts
- `OperatingIncomeLoss` - Operating income/loss
- `NetIncomeLoss` - Net income/loss
- `EarningsPerShareBasic` - Basic EPS
- `EarningsPerShareDiluted` - Diluted EPS

#### Cash Flow Entities
- `NetCashProvidedByUsedInOperatingActivities` - Operating cash flow
- `NetCashProvidedByUsedInInvestingActivities` - Investing cash flow
- `NetCashProvidedByUsedInFinancingActivities` - Financing cash flow

### Custom Financial Entity Pipeline

```python
@Language.component("financial_entities")
def financial_entities_component(doc):
    """Custom spaCy component for financial entities"""
    
    # Financial patterns
    patterns = [
        # Currency amounts
        {"label": "MONEY", "pattern": [
            {"TEXT": {"REGEX": r"^\$"}},
            {"LIKE_NUM": True},
            {"TEXT": {"IN": ["million", "billion", "thousand", "M", "B", "K"]}, "OP": "?"}
        ]},
        
        # Percentages
        {"label": "PERCENTAGE", "pattern": [
            {"LIKE_NUM": True},
            {"TEXT": {"REGEX": r"^%$|^percent$"}}
        ]},
        
        # Stock symbols
        {"label": "TICKER", "pattern": [
            {"TEXT": {"REGEX": r"^[A-Z]{1,5}$"}}
        ]},
        
        # Financial ratios
        {"label": "RATIO", "pattern": [
            {"LOWER": {"IN": ["p/e", "pe", "debt-to-equity", "roe", "roa", "current"]}},
            {"LOWER": "ratio", "OP": "?"}
        ]}
    ]
    
    return doc
```

## Implementation Architecture

### NER Processing Pipeline

```python
class FinancialNERProcessor:
    def __init__(self):
        self.models = {
            'finer': self._load_finer_model(),
            'company': self._load_company_model(),
            'general': self._load_general_ner()
        }
        
        self.entity_linker = EntityLinker()
        self.confidence_threshold = 0.8
    
    async def extract_entities(
        self, 
        text: str, 
        document_type: str = None,
        return_confidence: bool = True
    ) -> List[FinancialEntity]:
        """Extract financial entities from text"""
        
        # Preprocess text
        clean_text = self._preprocess_financial_text(text)
        
        # Multi-model entity extraction
        entities = []
        
        # FiNER-139 for comprehensive financial entities
        finer_entities = await self._extract_with_finer(clean_text)
        entities.extend(finer_entities)
        
        # Company-specific entities
        company_entities = await self._extract_company_entities(clean_text)
        entities.extend(company_entities)
        
        # General NER for people, organizations, locations
        general_entities = await self._extract_general_entities(clean_text)
        entities.extend(general_entities)
        
        # Entity linking and resolution
        linked_entities = await self._link_entities(entities, document_type)
        
        # Filter by confidence
        if return_confidence:
            linked_entities = [
                e for e in linked_entities 
                if e.confidence >= self.confidence_threshold
            ]
        
        # Deduplicate and rank
        final_entities = self._deduplicate_entities(linked_entities)
        
        return final_entities
```

### Entity Data Structure

```python
@dataclass
class FinancialEntity:
    """Financial entity with comprehensive metadata"""
    
    # Core information
    text: str                    # Original text
    label: str                   # Entity type (from 139 labels)
    start: int                   # Start position in text
    end: int                     # End position in text
    confidence: float            # Confidence score (0-1)
    
    # Financial-specific attributes
    entity_category: str         # Balance Sheet, Income Statement, etc.
    xbrl_tag: str               # XBRL tag if applicable
    numeric_value: Optional[float]  # Numeric value if extractable
    currency: Optional[str]      # Currency code
    unit: Optional[str]         # Unit (millions, thousands, etc.)
    time_period: Optional[str]   # Time period reference
    
    # Context information
    sentence_context: str        # Surrounding sentence
    document_section: str        # Document section (if known)
    related_entities: List[str]  # Related entity references
    
    # Metadata
    model_source: str           # Model that extracted the entity
    processing_timestamp: datetime
    
    # Entity linking
    linked_entity_id: Optional[str]  # Linked knowledge base ID
    canonical_name: Optional[str]    # Canonical entity name
    entity_type_hierarchy: List[str] # Hierarchical entity types
```

## Document Type Specialization

### Prospectus Documents

```python
class ProspectusNERProcessor(FinancialNERProcessor):
    """Specialized NER for IPO prospectuses"""
    
    def __init__(self):
        super().__init__()
        
        # Prospectus-specific entity types
        self.prospectus_entities = {
            'OFFERING_SIZE': r'\$[\d,.]+ (?:million|billion)',
            'SHARE_PRICE_RANGE': r'\$[\d.]+ to \$[\d.]+ per share',
            'USE_OF_PROCEEDS': r'use of proceeds|net proceeds',
            'UNDERWRITERS': r'underwriters?|lead manager',
            'LOCK_UP_PERIOD': r'lock.?up period|[\d]+ (?:days|months)',
            'TICKER_SYMBOL': r'(?:NYSE|NASDAQ):\s*[A-Z]{1,5}',
        }
    
    async def extract_ipo_entities(self, text: str) -> Dict[str, List[FinancialEntity]]:
        """Extract IPO-specific entities"""
        
        ipo_entities = {
            'company_info': [],
            'financial_metrics': [],
            'offering_details': [],
            'risk_factors': [],
            'use_of_proceeds': []
        }
        
        # Extract standard entities
        entities = await self.extract_entities(text, document_type='prospectus')
        
        # Categorize by IPO relevance
        for entity in entities:
            if entity.label in ['OFFERING_SIZE', 'SHARE_PRICE_RANGE']:
                ipo_entities['offering_details'].append(entity)
            elif entity.xbrl_tag and entity.xbrl_tag.startswith('Revenue'):
                ipo_entities['financial_metrics'].append(entity)
            elif 'risk' in entity.text.lower():
                ipo_entities['risk_factors'].append(entity)
        
        return ipo_entities
```

### Annual Reports (10-K)

```python
class TenKNERProcessor(FinancialNERProcessor):
    """Specialized NER for 10-K annual reports"""
    
    async def extract_10k_entities(self, text: str, section: str = None) -> Dict[str, Any]:
        """Extract entities from 10-K with section awareness"""
        
        section_entities = {}
        
        if section == 'business':
            # Focus on company description entities
            entities = await self._extract_business_entities(text)
        elif section == 'risk_factors':
            # Focus on risk-related entities
            entities = await self._extract_risk_entities(text)
        elif section == 'financial_statements':
            # Focus on financial metrics
            entities = await self._extract_financial_metrics(text)
        else:
            # General extraction
            entities = await self.extract_entities(text, document_type='10k')
        
        return {
            'entities': entities,
            'section': section,
            'entity_count': len(entities),
            'high_confidence_count': len([e for e in entities if e.confidence > 0.9])
        }
```

## Performance Optimization

### Batch Processing

```python
async def process_batch(
    self,
    texts: List[str],
    batch_size: int = 32,
    max_workers: int = 4
) -> List[List[FinancialEntity]]:
    """Process multiple texts in batches"""
    
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Tokenize batch
        encoded_batch = self.finer_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Model inference
        with torch.no_grad():
            outputs = self.finer_model(**encoded_batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Decode predictions
        batch_results = []
        for j, prediction in enumerate(predictions):
            entities = self._decode_predictions(
                batch[j], 
                prediction, 
                encoded_batch['input_ids'][j]
            )
            batch_results.append(entities)
        
        results.extend(batch_results)
    
    return results
```

### Caching Strategy

```python
class EntityCache:
    """Redis-based caching for extracted entities"""
    
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.cache_ttl = 3600  # 1 hour
    
    async def get_cached_entities(self, text_hash: str) -> Optional[List[FinancialEntity]]:
        """Get cached entities for text"""
        cached = await self.redis_client.get(f"ner:{text_hash}")
        if cached:
            return [FinancialEntity.from_json(e) for e in json.loads(cached)]
        return None
    
    async def cache_entities(self, text_hash: str, entities: List[FinancialEntity]):
        """Cache extracted entities"""
        entity_data = [e.to_json() for e in entities]
        await self.redis_client.setex(
            f"ner:{text_hash}", 
            self.cache_ttl, 
            json.dumps(entity_data)
        )
```

## Entity Linking and Knowledge Base Integration

### Entity Resolution

```python
class FinancialEntityLinker:
    """Link extracted entities to knowledge bases"""
    
    def __init__(self):
        self.company_db = CompanyDatabase()
        self.ticker_db = TickerDatabase()
        self.person_db = ExecutiveDatabase()
        self.similarity_threshold = 0.85
    
    async def link_entities(
        self, 
        entities: List[FinancialEntity],
        context: str = None
    ) -> List[FinancialEntity]:
        """Link entities to knowledge base entries"""
        
        linked_entities = []
        
        for entity in entities:
            if entity.label == 'ORG' and self._is_company(entity.text):
                # Link to company database
                matches = await self.company_db.search(entity.text)
                if matches and matches[0].similarity > self.similarity_threshold:
                    entity.linked_entity_id = matches[0].entity_id
                    entity.canonical_name = matches[0].canonical_name
                    
            elif entity.label == 'TICKER':
                # Link to ticker database
                ticker_info = await self.ticker_db.lookup(entity.text)
                if ticker_info:
                    entity.linked_entity_id = ticker_info.company_id
                    entity.canonical_name = ticker_info.company_name
                    
            elif entity.label == 'PERSON':
                # Link to executive database
                exec_matches = await self.person_db.search(entity.text, context)
                if exec_matches:
                    entity.linked_entity_id = exec_matches[0].person_id
                    entity.canonical_name = exec_matches[0].full_name
            
            linked_entities.append(entity)
        
        return linked_entities
```

### Knowledge Graph Integration

```python
class FinancialKnowledgeGraph:
    """Integration with financial knowledge graph"""
    
    def __init__(self):
        self.neo4j_driver = GraphDatabase.driver(
            "bolt://localhost:7687", 
            auth=("neo4j", "password")
        )
    
    async def create_entity_relationships(self, entities: List[FinancialEntity]):
        """Create relationships between entities in knowledge graph"""
        
        with self.neo4j_driver.session() as session:
            for entity in entities:
                # Create entity node
                session.run(
                    """
                    MERGE (e:FinancialEntity {id: $entity_id})
                    SET e.text = $text,
                        e.label = $label,
                        e.confidence = $confidence,
                        e.xbrl_tag = $xbrl_tag
                    """,
                    entity_id=entity.linked_entity_id or entity.text,
                    text=entity.text,
                    label=entity.label,
                    confidence=entity.confidence,
                    xbrl_tag=entity.xbrl_tag
                )
                
                # Create relationships to related entities
                for related_id in entity.related_entities:
                    session.run(
                        """
                        MATCH (e1:FinancialEntity {id: $entity1})
                        MATCH (e2:FinancialEntity {id: $entity2})
                        MERGE (e1)-[:RELATED_TO]->(e2)
                        """,
                        entity1=entity.linked_entity_id or entity.text,
                        entity2=related_id
                    )
```

## Quality Assurance and Validation

### Entity Validation Rules

```python
class EntityValidator:
    """Validate extracted financial entities"""
    
    def __init__(self):
        self.validation_rules = {
            'MONEY': self._validate_money_entity,
            'PERCENTAGE': self._validate_percentage_entity,
            'TICKER': self._validate_ticker_entity,
            'REVENUE': self._validate_revenue_entity
        }
    
    def validate_entity(self, entity: FinancialEntity) -> ValidationResult:
        """Validate a single entity"""
        
        validator = self.validation_rules.get(entity.label)
        if validator:
            return validator(entity)
        
        # Default validation
        return ValidationResult(
            is_valid=True,
            confidence_adjustment=0.0,
            validation_notes=[]
        )
    
    def _validate_money_entity(self, entity: FinancialEntity) -> ValidationResult:
        """Validate monetary entities"""
        issues = []
        confidence_adjustment = 0.0
        
        # Check for reasonable money format
        if not re.match(r'^\$?[\d,.]+ ?(million|billion|thousand|M|B|K)?', entity.text):
            issues.append("Invalid money format")
            confidence_adjustment -= 0.2
        
        # Check for context appropriateness
        if entity.numeric_value and entity.numeric_value > 1e12:  # Over $1 trillion
            issues.append("Unusually large monetary value")
            confidence_adjustment -= 0.1
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence_adjustment=confidence_adjustment,
            validation_notes=issues
        )
```

### Performance Metrics

```python
class NERPerformanceTracker:
    """Track NER performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'processing_time': 0.0,
            'entities_per_document': 0.0
        }
    
    def calculate_metrics(
        self, 
        predicted_entities: List[FinancialEntity],
        ground_truth: List[FinancialEntity]
    ) -> Dict[str, float]:
        """Calculate performance metrics"""
        
        # Convert to sets for comparison
        pred_set = {(e.text, e.label, e.start, e.end) for e in predicted_entities}
        true_set = {(e.text, e.label, e.start, e.end) for e in ground_truth}
        
        # Calculate metrics
        true_positives = len(pred_set & true_set)
        false_positives = len(pred_set - true_set)
        false_negatives = len(true_set - pred_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
```

## Integration Examples

### Basic Entity Extraction

```python
from src.backend.nlp_services.entity_extraction import FinancialNERProcessor

# Initialize NER processor
ner_processor = FinancialNERProcessor()

# Extract entities from financial text
text = """
Apple Inc. (NASDAQ: AAPL) reported revenue of $123.9 billion for Q4 2024,
representing a 8.5% increase year-over-year. Net income was $34.6 billion,
or $2.18 per diluted share. The company's gross margin was 46.2%.
"""

entities = await ner_processor.extract_entities(text)

# Process results
for entity in entities:
    print(f"Entity: {entity.text}")
    print(f"Type: {entity.label}")
    print(f"Confidence: {entity.confidence:.2f}")
    print(f"Numeric Value: {entity.numeric_value}")
    print("---")
```

### Prospectus Analysis

```python
# Specialized prospectus processing
prospectus_processor = ProspectusNERProcessor()

# Load prospectus document
with open("company_prospectus.pdf", "rb") as f:
    prospectus_text = extract_text_from_pdf(f)

# Extract IPO-specific entities
ipo_entities = await prospectus_processor.extract_ipo_entities(prospectus_text)

# Analyze offering details
for entity in ipo_entities['offering_details']:
    if entity.label == 'OFFERING_SIZE':
        print(f"Offering Size: {entity.text}")
    elif entity.label == 'SHARE_PRICE_RANGE':
        print(f"Price Range: {entity.text}")
```

### Batch Document Processing

```python
# Process multiple documents
document_texts = [
    "10-K filing text...",
    "Quarterly report text...",
    "Prospectus text..."
]

# Batch extraction
batch_results = await ner_processor.process_batch(
    document_texts,
    batch_size=16,
    max_workers=4
)

# Aggregate results
all_entities = []
for doc_entities in batch_results:
    all_entities.extend(doc_entities)

# Entity statistics
entity_counts = {}
for entity in all_entities:
    entity_counts[entity.label] = entity_counts.get(entity.label, 0) + 1

print("Entity Distribution:")
for label, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{label}: {count}")
```

## Best Practices

### Model Selection

1. **Use FiNER-139** for comprehensive financial entity extraction
2. **Combine multiple models** for better coverage and accuracy
3. **Fine-tune on domain data** for specialized use cases
4. **Validate entities** using business rules and knowledge bases

### Performance Optimization

1. **Batch processing** for large document collections
2. **Caching** for repeated document processing
3. **GPU acceleration** for transformer models
4. **Async processing** for concurrent operations

### Quality Assurance

1. **Set confidence thresholds** based on use case requirements
2. **Implement validation rules** for critical entity types
3. **Monitor performance metrics** continuously
4. **Regular model updates** with new training data

---

*Last updated: 2025-08-30*
*Version: 1.0.0*