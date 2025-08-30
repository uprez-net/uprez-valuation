# Multi-Document Summarization for Financial Reports

## Overview

This document provides comprehensive technical documentation for multi-document summarization techniques used in the Uprez Valuation system. The system leverages state-of-the-art transformer models and hierarchical approaches to synthesize information from multiple financial documents into coherent, informative summaries.

## Architecture Overview

### Multi-Document Summarization Pipeline

```python
class MultiDocumentSummarizer:
    """Advanced multi-document summarization system for financial reports"""
    
    def __init__(self):
        self.models = {
            'pegasus': self._load_pegasus_model(),
            'longformer': self._load_longformer_model(),
            't5': self._load_t5_model(),
            'led': self._load_led_model()  # Longformer Encoder-Decoder
        }
        
        self.preprocessing_pipeline = DocumentPreprocessor()
        self.hierarchical_summarizer = HierarchicalSummarizer()
        self.fusion_engine = SummaryFusionEngine()
        
        # Configuration for different summary types
        self.summary_configurations = {
            'executive_summary': {
                'max_length': 500,
                'focus_areas': ['financial_performance', 'key_metrics', 'outlook'],
                'style': 'executive'
            },
            'risk_assessment': {
                'max_length': 300,
                'focus_areas': ['risk_factors', 'uncertainties', 'challenges'],
                'style': 'analytical'
            },
            'investment_thesis': {
                'max_length': 400,
                'focus_areas': ['growth_prospects', 'competitive_position', 'valuation'],
                'style': 'investment'
            },
            'regulatory_summary': {
                'max_length': 350,
                'focus_areas': ['compliance', 'regulatory_changes', 'legal_issues'],
                'style': 'regulatory'
            }
        }
```

### Document Types and Sources

The system processes multiple types of financial documents:

1. **SEC Filings**
   - 10-K annual reports
   - 10-Q quarterly reports
   - 8-K current reports
   - DEF 14A proxy statements

2. **Analyst Reports**
   - Research reports from investment banks
   - Rating changes and price target updates
   - Sector analysis and thematic reports

3. **Company Communications**
   - Earnings transcripts
   - Investor presentations
   - Press releases

4. **Market Research**
   - Industry reports
   - Market analysis
   - Competitive intelligence

## Advanced Summarization Models

### PEGASUS-X for Long Documents

PEGASUS-X extends PEGASUS for long input summarization, handling up to 16K tokens:

```python
class PEGASUSXSummarizer:
    """PEGASUS-X implementation for long financial documents"""
    
    def __init__(self):
        self.model_name = "google/pegasus-x-large"
        self.tokenizer = PegasusXTokenizer.from_pretrained(self.model_name)
        self.model = PegasusXForConditionalGeneration.from_pretrained(self.model_name)
        
        # Extended context configuration
        self.max_input_length = 16384  # 16K tokens
        self.max_output_length = 512
        self.block_size = 512
        self.stride = 256
        
        # Financial domain adaptations
        self.financial_prompts = self._load_financial_prompts()
    
    async def summarize_long_documents(
        self,
        documents: List[str],
        summary_type: str = 'executive_summary',
        use_hierarchical: bool = True
    ) -> Dict[str, Any]:
        """Summarize multiple long financial documents"""
        
        # Combine and preprocess documents
        combined_document = self._combine_documents_strategically(documents)
        
        if use_hierarchical:
            # Hierarchical summarization for very long inputs
            summary = await self._hierarchical_summarization(
                combined_document, summary_type
            )
        else:
            # Direct summarization
            summary = await self._direct_summarization(
                combined_document, summary_type
            )
        
        return {
            'summary': summary,
            'summary_type': summary_type,
            'source_documents': len(documents),
            'processing_method': 'hierarchical' if use_hierarchical else 'direct',
            'quality_metrics': await self._calculate_quality_metrics(summary, documents)
        }
    
    async def _hierarchical_summarization(
        self, 
        document: str, 
        summary_type: str
    ) -> str:
        """Apply hierarchical summarization approach"""
        
        # Stage 1: Chunk-level summaries
        chunks = self._create_semantic_chunks(document, self.max_input_length // 4)
        chunk_summaries = []
        
        for chunk in chunks:
            chunk_summary = await self._summarize_chunk(
                chunk, 
                max_length=150,
                focus=self.summary_configurations[summary_type]['focus_areas']
            )
            chunk_summaries.append(chunk_summary)
        
        # Stage 2: Combine chunk summaries
        combined_chunk_summary = " ".join(chunk_summaries)
        
        # Stage 3: Final summarization
        final_summary = await self._summarize_chunk(
            combined_chunk_summary,
            max_length=self.summary_configurations[summary_type]['max_length'],
            style=self.summary_configurations[summary_type]['style']
        )
        
        return final_summary
    
    def _create_semantic_chunks(self, text: str, max_chunk_size: int) -> List[str]:
        """Create semantically coherent chunks"""
        
        # Use sentence boundaries and section headers
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.tokenize(sentence))
            
            if current_length + sentence_tokens > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
```

### Longformer Encoder-Decoder (LED)

LED provides efficient processing of long documents with extended attention:

```python
class LEDSummarizer:
    """Longformer Encoder-Decoder for financial document summarization"""
    
    def __init__(self):
        self.model_name = "allenai/led-large-16384"
        self.tokenizer = LEDTokenizer.from_pretrained(self.model_name)
        self.model = LEDForConditionalGeneration.from_pretrained(self.model_name)
        
        # LED-specific configuration
        self.max_input_length = 16384
        self.max_output_length = 1024
        self.global_attention_pattern = "financial_entities"
    
    async def process_financial_documents(
        self,
        documents: List[Dict[str, Any]],
        focus_query: str = None
    ) -> Dict[str, Any]:
        """Process financial documents with query-focused summarization"""
        
        # Combine documents with metadata
        structured_input = self._structure_multi_document_input(documents, focus_query)
        
        # Tokenize with global attention
        inputs = self.tokenizer(
            structured_input,
            max_length=self.max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Set global attention for important tokens
        global_attention_mask = self._create_financial_attention_mask(
            inputs, documents, focus_query
        )
        inputs['global_attention_mask'] = global_attention_mask
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                **inputs,
                max_length=self.max_output_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
        
        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return {
            'summary': summary_text,
            'attention_visualization': self._visualize_attention_patterns(inputs),
            'source_attribution': self._attribute_summary_to_sources(summary_text, documents)
        }
    
    def _create_financial_attention_mask(
        self,
        inputs: Dict[str, torch.Tensor],
        documents: List[Dict[str, Any]],
        focus_query: str = None
    ) -> torch.Tensor:
        """Create financial entity-focused global attention mask"""
        
        attention_mask = inputs['attention_mask']
        global_attention = torch.zeros_like(attention_mask)
        
        # Always attend to special tokens
        global_attention[:, 0] = 1  # [CLS]
        
        # Identify financial entities and important phrases
        input_ids = inputs['input_ids'][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        financial_patterns = [
            r'revenue', r'profit', r'earnings', r'ebitda',
            r'debt', r'cash', r'margin', r'growth',
            r'risk', r'challenge', r'opportunity',
            r'\$[\d,]+', r'\d+%', r'Q[1-4]'
        ]
        
        for i, token in enumerate(tokens):
            if any(re.search(pattern, token, re.IGNORECASE) for pattern in financial_patterns):
                global_attention[0, i] = 1
        
        # Query-specific attention
        if focus_query:
            query_tokens = self.tokenizer.tokenize(focus_query.lower())
            for i, token in enumerate(tokens):
                if token.lower() in query_tokens:
                    global_attention[0, i] = 1
        
        return global_attention
```

### Graph-Enhanced Multi-Document Summarization

```python
class GraphEnhancedSummarizer:
    """Graph-enhanced approach for multi-document summarization"""
    
    def __init__(self):
        self.entity_extractor = FinancialEntityExtractor()
        self.graph_builder = DocumentGraphBuilder()
        self.graph_nn = GraphNeuralNetwork()
        self.language_model = TransformerLanguageModel()
    
    async def summarize_with_graph_enhancement(
        self,
        documents: List[str],
        summary_type: str
    ) -> Dict[str, Any]:
        """Use graph structure to enhance summarization"""
        
        # Extract entities and build document graph
        entities = []
        for doc in documents:
            doc_entities = await self.entity_extractor.extract(doc)
            entities.extend(doc_entities)
        
        # Build heterogeneous graph
        document_graph = self.graph_builder.build_multi_document_graph(
            documents, entities
        )
        
        # Apply graph neural network to find salient information
        salient_sentences = self.graph_nn.identify_salient_sentences(
            document_graph, summary_type
        )
        
        # Generate summary from salient sentences
        summary = await self.language_model.generate_summary(
            salient_sentences,
            summary_type
        )
        
        return {
            'summary': summary,
            'graph_structure': document_graph.get_structure_info(),
            'salient_entities': document_graph.get_top_entities(10),
            'cross_document_connections': document_graph.get_cross_doc_connections()
        }
    
    class DocumentGraphBuilder:
        """Build heterogeneous graph from multiple documents"""
        
        def build_multi_document_graph(
            self,
            documents: List[str],
            entities: List[FinancialEntity]
        ) -> DocumentGraph:
            """Build graph connecting documents, sentences, and entities"""
            
            graph = DocumentGraph()
            
            # Add document nodes
            for i, doc in enumerate(documents):
                doc_node = graph.add_document_node(i, doc)
                
                # Add sentence nodes
                sentences = self._extract_sentences(doc)
                for j, sentence in enumerate(sentences):
                    sent_node = graph.add_sentence_node(i, j, sentence)
                    graph.add_edge(doc_node, sent_node, 'contains')
                    
                    # Add entity nodes and connections
                    sent_entities = self._find_sentence_entities(sentence, entities)
                    for entity in sent_entities:
                        entity_node = graph.add_entity_node(entity)
                        graph.add_edge(sent_node, entity_node, 'mentions')
            
            # Add cross-document entity connections
            self._add_cross_document_connections(graph, entities)
            
            return graph
```

## Specialized Financial Summarization

### Investment Thesis Generation

```python
class InvestmentThesisSummarizer:
    """Generate investment thesis from multiple financial documents"""
    
    def __init__(self):
        self.thesis_framework = InvestmentFramework()
        self.financial_analyzer = FinancialMetricsAnalyzer()
        self.risk_assessor = RiskAssessmentEngine()
        
        # Investment thesis structure
        self.thesis_sections = {
            'company_overview': 0.15,
            'investment_highlights': 0.25,
            'financial_performance': 0.20,
            'growth_prospects': 0.20,
            'valuation': 0.15,
            'risks': 0.15
        }
    
    async def generate_investment_thesis(
        self,
        company_documents: List[str],
        market_research: List[str],
        analyst_reports: List[str]
    ) -> Dict[str, Any]:
        """Generate comprehensive investment thesis"""
        
        # Extract key information from each document type
        company_insights = await self._extract_company_insights(company_documents)
        market_insights = await self._extract_market_insights(market_research)
        analyst_insights = await self._extract_analyst_insights(analyst_reports)
        
        # Generate thesis sections
        thesis_sections = {}
        
        for section, weight in self.thesis_sections.items():
            section_content = await self._generate_thesis_section(
                section,
                company_insights,
                market_insights,
                analyst_insights,
                weight
            )
            thesis_sections[section] = section_content
        
        # Combine into coherent thesis
        full_thesis = await self._synthesize_thesis_sections(thesis_sections)
        
        return {
            'investment_thesis': full_thesis,
            'thesis_sections': thesis_sections,
            'key_metrics': company_insights.get('key_metrics', {}),
            'risk_assessment': await self._assess_investment_risks(thesis_sections),
            'confidence_score': self._calculate_thesis_confidence(thesis_sections)
        }
```

### Risk Factor Summarization

```python
class RiskFactorSummarizer:
    """Specialized summarizer for risk factors across documents"""
    
    def __init__(self):
        self.risk_classifier = RiskClassificationModel()
        self.risk_ranker = RiskImportanceRanker()
        self.risk_categories = [
            'market_risk', 'operational_risk', 'financial_risk',
            'regulatory_risk', 'technology_risk', 'competitive_risk'
        ]
    
    async def summarize_risk_factors(
        self,
        documents: List[str],
        risk_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Extract and summarize risk factors from multiple documents"""
        
        # Extract risk-related content
        risk_segments = []
        for doc in documents:
            doc_risks = await self._extract_risk_segments(doc)
            risk_segments.extend(doc_risks)
        
        # Classify and rank risks
        classified_risks = []
        for risk_segment in risk_segments:
            risk_class = await self.risk_classifier.classify(risk_segment)
            risk_importance = await self.risk_ranker.score(risk_segment)
            
            if risk_importance > risk_threshold:
                classified_risks.append({
                    'text': risk_segment,
                    'category': risk_class,
                    'importance': risk_importance
                })
        
        # Group by category and summarize
        risk_summary = {}
        for category in self.risk_categories:
            category_risks = [r for r in classified_risks if r['category'] == category]
            if category_risks:
                category_summary = await self._summarize_risk_category(
                    category, category_risks
                )
                risk_summary[category] = category_summary
        
        # Generate overall risk assessment
        overall_assessment = await self._generate_overall_risk_assessment(risk_summary)
        
        return {
            'overall_risk_assessment': overall_assessment,
            'risk_categories': risk_summary,
            'top_risks': sorted(classified_risks, key=lambda x: x['importance'], reverse=True)[:10],
            'risk_distribution': self._calculate_risk_distribution(classified_risks)
        }
```

## Quality Assessment and Evaluation

### Summary Quality Metrics

```python
class SummaryQualityEvaluator:
    """Evaluate quality of generated financial summaries"""
    
    def __init__(self):
        self.rouge_evaluator = ROUGEEvaluator()
        self.bertscore_evaluator = BERTScoreEvaluator()
        self.financial_relevance_scorer = FinancialRelevanceScorer()
        self.factual_consistency_checker = FactualConsistencyChecker()
    
    async def evaluate_summary_quality(
        self,
        generated_summary: str,
        reference_summaries: List[str],
        source_documents: List[str]
    ) -> Dict[str, Any]:
        """Comprehensive quality evaluation of generated summary"""
        
        # ROUGE scores (lexical overlap)
        rouge_scores = self.rouge_evaluator.score(generated_summary, reference_summaries)
        
        # BERTScore (semantic similarity)
        bert_scores = self.bertscore_evaluator.score(generated_summary, reference_summaries)
        
        # Financial relevance
        relevance_score = await self.financial_relevance_scorer.score(
            generated_summary, source_documents
        )
        
        # Factual consistency
        consistency_score = await self.factual_consistency_checker.verify(
            generated_summary, source_documents
        )
        
        # Content analysis
        content_analysis = self._analyze_summary_content(generated_summary)
        
        return {
            'rouge_scores': rouge_scores,
            'bert_scores': bert_scores,
            'financial_relevance': relevance_score,
            'factual_consistency': consistency_score,
            'content_analysis': content_analysis,
            'overall_quality': self._calculate_overall_quality(
                rouge_scores, bert_scores, relevance_score, consistency_score
            )
        }
    
    def _analyze_summary_content(self, summary: str) -> Dict[str, Any]:
        """Analyze content characteristics of summary"""
        
        # Extract financial entities
        entities = self._extract_entities(summary)
        
        # Analyze sentiment
        sentiment = self._analyze_sentiment(summary)
        
        # Check completeness
        coverage = self._assess_topic_coverage(summary)
        
        # Readability metrics
        readability = self._calculate_readability(summary)
        
        return {
            'entity_count': len(entities),
            'entity_types': list(set([e.label for e in entities])),
            'sentiment': sentiment,
            'topic_coverage': coverage,
            'readability': readability,
            'length_stats': {
                'word_count': len(summary.split()),
                'sentence_count': len(summary.split('.')),
                'avg_sentence_length': len(summary.split()) / len(summary.split('.'))
            }
        }
```

### Automated Quality Feedback

```python
class QualityFeedbackSystem:
    """Automated system for providing quality feedback and improvements"""
    
    def __init__(self):
        self.quality_evaluator = SummaryQualityEvaluator()
        self.improvement_suggester = ImprovementSuggester()
        self.quality_threshold = 0.75
    
    async def assess_and_improve(
        self,
        summary: str,
        source_documents: List[str],
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """Iteratively assess and improve summary quality"""
        
        current_summary = summary
        iterations = []
        
        for i in range(max_iterations):
            # Evaluate current summary
            quality_scores = await self.quality_evaluator.evaluate_summary_quality(
                current_summary, [], source_documents
            )
            
            # Check if quality is sufficient
            overall_quality = quality_scores['overall_quality']
            
            iteration_info = {
                'iteration': i + 1,
                'summary': current_summary,
                'quality_scores': quality_scores,
                'overall_quality': overall_quality
            }
            
            if overall_quality >= self.quality_threshold:
                iteration_info['status'] = 'quality_achieved'
                iterations.append(iteration_info)
                break
            
            # Generate improvement suggestions
            improvements = await self.improvement_suggester.suggest_improvements(
                current_summary, quality_scores, source_documents
            )
            
            # Apply improvements
            improved_summary = await self._apply_improvements(
                current_summary, improvements, source_documents
            )
            
            iteration_info['improvements_applied'] = improvements
            iteration_info['status'] = 'improved'
            iterations.append(iteration_info)
            
            current_summary = improved_summary
        
        return {
            'final_summary': current_summary,
            'iterations': iterations,
            'improvement_achieved': iterations[-1]['overall_quality'] > iterations[0]['overall_quality'],
            'quality_improvement': iterations[-1]['overall_quality'] - iterations[0]['overall_quality']
        }
```

## Integration Examples

### Basic Multi-Document Summarization

```python
from src.backend.nlp_services.summarization import MultiDocumentSummarizer

# Initialize summarizer
summarizer = MultiDocumentSummarizer()

# Load financial documents
documents = [
    load_document("company_10k.pdf"),
    load_document("q4_earnings_call.txt"),
    load_document("analyst_report.pdf")
]

# Generate executive summary
result = await summarizer.summarize_documents(
    documents=documents,
    summary_type='executive_summary',
    max_length=500
)

print("Executive Summary:")
print(result['summary'])
print(f"Quality Score: {result['quality_metrics']['overall_quality']:.2f}")
```

### Investment Thesis Generation

```python
# Specialized investment thesis generation
thesis_generator = InvestmentThesisSummarizer()

# Categorized documents
company_docs = ["10-K filing", "investor presentation", "earnings transcripts"]
market_docs = ["industry report", "market analysis"]
analyst_docs = ["research reports", "rating updates"]

# Generate thesis
thesis_result = await thesis_generator.generate_investment_thesis(
    company_documents=company_docs,
    market_research=market_docs,
    analyst_reports=analyst_docs
)

print("Investment Thesis:")
for section, content in thesis_result['thesis_sections'].items():
    print(f"\n{section.upper()}:")
    print(content)
```

### Risk Assessment Summary

```python
# Risk factor analysis
risk_summarizer = RiskFactorSummarizer()

# Extract and summarize risks
risk_analysis = await risk_summarizer.summarize_risk_factors(
    documents=documents,
    risk_threshold=0.8
)

print("Risk Assessment:")
print(risk_analysis['overall_risk_assessment'])

print("\nTop Risk Factors:")
for risk in risk_analysis['top_risks'][:5]:
    print(f"- {risk['category']}: {risk['text'][:100]}...")
```

## Best Practices

### Document Preparation

1. **Document Quality**: Ensure high-quality OCR and text extraction
2. **Document Structure**: Preserve section headers and document hierarchy
3. **Metadata Preservation**: Maintain document source and date information
4. **Content Filtering**: Remove boilerplate and irrelevant content

### Model Selection

1. **Document Length**: Use LED or PEGASUS-X for long documents
2. **Multiple Documents**: Use hierarchical approaches for many documents
3. **Specific Domains**: Fine-tune models on financial corpus
4. **Quality Requirements**: Use ensemble methods for critical summaries

### Performance Optimization

1. **Chunking Strategy**: Use semantic chunking for better coherence
2. **Attention Mechanisms**: Focus attention on financial entities
3. **Caching**: Cache intermediate results for complex pipelines
4. **Parallel Processing**: Process document chunks in parallel

---

*Last updated: 2025-08-30*
*Version: 1.0.0*