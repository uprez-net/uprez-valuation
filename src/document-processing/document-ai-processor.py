"""
Document AI Processor for IPO Valuation Platform
Handles document processing using Google Cloud Document AI
"""

import logging
import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
from pathlib import Path

from google.cloud import documentai
from google.cloud import vision
from google.cloud import translate_v2 as translate
from google.cloud import language_v1 as language
from google.cloud import storage
from google.cloud import bigquery

logger = logging.getLogger(__name__)

class DocumentAIProcessor:
    """Service for processing IPO documents using Document AI"""
    
    def __init__(self, project_id: str, location: str = "us"):
        self.project_id = project_id
        self.location = location
        
        # Initialize clients
        self.doc_client = documentai.DocumentProcessorServiceClient()
        self.vision_client = vision.ImageAnnotatorClient()
        self.translate_client = translate.Client()
        self.language_client = language.LanguageServiceClient()
        self.storage_client = storage.Client(project=project_id)
        self.bq_client = bigquery.Client(project=project_id)
        
        # Processor configurations
        self.processors = {
            "general_ocr": f"projects/{project_id}/locations/{location}/processors/general-ocr-processor",
            "form_parser": f"projects/{project_id}/locations/{location}/processors/form-parser",
            "prospectus_parser": f"projects/{project_id}/locations/{location}/processors/prospectus-parser",
            "financial_statement": f"projects/{project_id}/locations/{location}/processors/financial-statement"
        }
        
        # Document type classifiers
        self.document_classifiers = {
            "prospectus": ["initial public offering", "securities and exchange commission", "use of proceeds", "risk factors"],
            "financial_statement": ["balance sheet", "income statement", "cash flow statement", "consolidated financial"],
            "annual_report": ["form 10-k", "annual report", "management discussion", "auditor's report"],
            "quarterly_report": ["form 10-q", "quarterly report", "three months ended", "nine months ended"],
            "proxy_statement": ["proxy statement", "annual meeting", "executive compensation", "board of directors"]
        }
    
    async def process_document(self, 
                             document_uri: str,
                             processor_type: str = "general_ocr",
                             enable_native_pdf_parsing: bool = True) -> Dict[str, Any]:
        """Process a single document using Document AI"""
        try:
            processor_name = self.processors.get(processor_type)
            if not processor_name:
                raise ValueError(f"Unknown processor type: {processor_type}")
            
            # Read document from GCS
            document_content = self._read_document_from_gcs(document_uri)
            
            # Prepare request
            raw_document = documentai.RawDocument(
                content=document_content,
                mime_type="application/pdf"  # Adjust based on document type
            )
            
            request = documentai.ProcessRequest(
                name=processor_name,
                raw_document=raw_document,
                skip_human_review=True
            )
            
            if enable_native_pdf_parsing:
                request.process_options = documentai.ProcessOptions(
                    ocr_config=documentai.OcrConfig(
                        enable_native_pdf_parsing=True,
                        enable_image_quality_scores=True,
                        enable_symbol=True
                    )
                )
            
            # Process document
            result = self.doc_client.process_document(request=request)
            document = result.document
            
            # Extract structured data
            extracted_data = self._extract_structured_data(document, processor_type)
            
            # Classify document type
            document_type = self._classify_document(extracted_data["text"])
            
            # Perform additional NLP analysis
            nlp_results = await self._analyze_text_with_nlp(extracted_data["text"])
            
            # Combine results
            processed_result = {
                "document_uri": document_uri,
                "document_type": document_type,
                "processor_type": processor_type,
                "processing_timestamp": datetime.utcnow().isoformat(),
                "text": extracted_data["text"],
                "entities": extracted_data["entities"],
                "tables": extracted_data["tables"],
                "form_fields": extracted_data["form_fields"],
                "nlp_analysis": nlp_results,
                "confidence_score": self._calculate_confidence_score(document)
            }
            
            # Store results in BigQuery
            await self._store_results_in_bigquery(processed_result)
            
            logger.info(f"Document processed successfully: {document_uri}")
            return processed_result
            
        except Exception as e:
            logger.error(f"Error processing document {document_uri}: {str(e)}")
            raise
    
    async def batch_process_documents(self, 
                                    document_uris: List[str],
                                    processor_type: str = "general_ocr") -> List[Dict[str, Any]]:
        """Process multiple documents in batch"""
        try:
            # Process documents concurrently
            tasks = [
                self.process_document(uri, processor_type) 
                for uri in document_uris
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing document {document_uris[i]}: {str(result)}")
                    processed_results.append({
                        "document_uri": document_uris[i],
                        "error": str(result),
                        "processing_timestamp": datetime.utcnow().isoformat()
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise
    
    def extract_financial_metrics(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract financial metrics from processed document"""
        try:
            financial_metrics = {}
            text = document_data.get("text", "")
            entities = document_data.get("entities", [])
            
            # Extract monetary amounts
            money_entities = [e for e in entities if e.get("type") == "MONEY"]
            financial_metrics["monetary_amounts"] = money_entities
            
            # Extract percentages
            percentage_entities = [e for e in entities if e.get("type") == "PERCENT"]
            financial_metrics["percentages"] = percentage_entities
            
            # Extract dates
            date_entities = [e for e in entities if e.get("type") == "DATE"]
            financial_metrics["dates"] = date_entities
            
            # Extract specific financial terms using regex
            import re
            
            # Revenue patterns
            revenue_patterns = [
                r"revenue.*?\$?([\d,]+(?:\.\d+)?)\s*(?:million|billion|thousand)?",
                r"total revenue.*?\$?([\d,]+(?:\.\d+)?)\s*(?:million|billion|thousand)?"
            ]
            
            revenues = []
            for pattern in revenue_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                revenues.extend([match.group(1) for match in matches])
            
            financial_metrics["revenue_figures"] = revenues
            
            # Net income patterns
            income_patterns = [
                r"net income.*?\$?([\d,]+(?:\.\d+)?)\s*(?:million|billion|thousand)?",
                r"net loss.*?\$?([\d,]+(?:\.\d+)?)\s*(?:million|billion|thousand)?"
            ]
            
            incomes = []
            for pattern in income_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                incomes.extend([match.group(1) for match in matches])
            
            financial_metrics["income_figures"] = incomes
            
            # Extract ratios
            ratio_patterns = [
                r"pe ratio.*?([\d.]+)",
                r"debt.to.equity.*?([\d.]+)",
                r"return on equity.*?([\d.]+)%?",
                r"profit margin.*?([\d.]+)%?"
            ]
            
            ratios = {}
            for pattern in ratio_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                ratio_name = pattern.split(".*?")[0].replace("\\", "").replace(".", "_")
                ratios[ratio_name] = [match.group(1) for match in matches]
            
            financial_metrics["financial_ratios"] = ratios
            
            return financial_metrics
            
        except Exception as e:
            logger.error(f"Error extracting financial metrics: {str(e)}")
            raise
    
    def extract_risk_factors(self, document_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract risk factors from document"""
        try:
            text = document_data.get("text", "")
            risk_factors = []
            
            # Split text into sections
            sections = text.split("\\n\\n")
            
            # Find risk factors section
            risk_section = ""
            for section in sections:
                if "risk factor" in section.lower():
                    risk_section = section
                    break
            
            if risk_section:
                # Extract individual risks
                risk_sentences = risk_section.split(". ")
                
                for sentence in risk_sentences:
                    if len(sentence.strip()) > 50:  # Filter out short fragments
                        # Analyze sentence sentiment and extract key terms
                        sentiment_result = self.language_client.analyze_sentiment(
                            document=language.Document(content=sentence, type_=language.Document.Type.PLAIN_TEXT),
                            encoding_type=language.EncodingType.UTF8
                        )
                        
                        entities_result = self.language_client.analyze_entities(
                            document=language.Document(content=sentence, type_=language.Document.Type.PLAIN_TEXT),
                            encoding_type=language.EncodingType.UTF8
                        )
                        
                        risk_factors.append({
                            "text": sentence.strip(),
                            "sentiment_score": sentiment_result.document_sentiment.score,
                            "sentiment_magnitude": sentiment_result.document_sentiment.magnitude,
                            "entities": [{"name": entity.name, "type": entity.type_.name} for entity in entities_result.entities],
                            "risk_category": self._categorize_risk(sentence)
                        })
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Error extracting risk factors: {str(e)}")
            raise
    
    def analyze_document_charts(self, document_uri: str) -> List[Dict[str, Any]]:
        """Analyze charts and graphs in documents using Vision API"""
        try:
            # Read document as images (for chart analysis)
            document_content = self._read_document_from_gcs(document_uri)
            
            # Convert PDF pages to images (would need pdf2image in production)
            # For now, assuming we have image data
            
            image = vision.Image(content=document_content)
            
            # Detect text in image
            text_response = self.vision_client.text_detection(image=image)
            texts = text_response.text_annotations
            
            # Detect objects (charts, graphs)
            objects_response = self.vision_client.object_localization(image=image)
            objects = objects_response.localized_object_annotations
            
            # Analyze detected charts
            chart_analysis = []
            for obj in objects:
                if "chart" in obj.name.lower() or "graph" in obj.name.lower():
                    # Extract text near the chart
                    chart_region = obj.bounding_poly.normalized_vertices
                    chart_text = self._extract_text_from_region(texts, chart_region)
                    
                    chart_analysis.append({
                        "type": obj.name,
                        "confidence": obj.score,
                        "location": {
                            "vertices": [(v.x, v.y) for v in chart_region]
                        },
                        "extracted_text": chart_text,
                        "analysis": self._analyze_chart_content(chart_text)
                    })
            
            return chart_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing document charts: {str(e)}")
            raise
    
    async def _analyze_text_with_nlp(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive NLP analysis on text"""
        try:
            # Language detection
            detection = self.translate_client.detect_language(text)
            detected_language = detection["language"]
            
            # Translate to English if needed
            if detected_language != "en":
                translation = self.translate_client.translate(text, target_language="en")
                analyzed_text = translation["translatedText"]
            else:
                analyzed_text = text
            
            # Sentiment analysis
            document = language.Document(content=analyzed_text, type_=language.Document.Type.PLAIN_TEXT)
            
            sentiment_response = self.language_client.analyze_sentiment(
                request={"document": document, "encoding_type": language.EncodingType.UTF8}
            )
            
            # Entity analysis
            entities_response = self.language_client.analyze_entities(
                request={"document": document, "encoding_type": language.EncodingType.UTF8}
            )
            
            # Syntax analysis
            syntax_response = self.language_client.analyze_syntax(
                request={"document": document, "encoding_type": language.EncodingType.UTF8}
            )
            
            return {
                "detected_language": detected_language,
                "sentiment": {
                    "score": sentiment_response.document_sentiment.score,
                    "magnitude": sentiment_response.document_sentiment.magnitude
                },
                "entities": [
                    {
                        "name": entity.name,
                        "type": entity.type_.name,
                        "salience": entity.salience,
                        "sentiment_score": entity.sentiment.score if entity.sentiment else None
                    }
                    for entity in entities_response.entities
                ],
                "tokens": [
                    {
                        "text": token.text.content,
                        "part_of_speech": token.part_of_speech.tag.name,
                        "lemma": token.lemma
                    }
                    for token in syntax_response.tokens[:100]  # Limit for performance
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in NLP analysis: {str(e)}")
            raise
    
    def _extract_structured_data(self, document: documentai.Document, processor_type: str) -> Dict[str, Any]:
        """Extract structured data from Document AI response"""
        try:
            # Extract text
            text = document.text
            
            # Extract entities
            entities = []
            for entity in document.entities:
                entities.append({
                    "type": entity.type_,
                    "mention_text": entity.mention_text,
                    "confidence": entity.confidence,
                    "normalized_value": getattr(entity.normalized_value, 'text', None) if hasattr(entity, 'normalized_value') else None
                })
            
            # Extract tables
            tables = []
            for table in document.pages[0].tables if document.pages else []:
                table_data = []
                for row in table.header_rows + table.body_rows:
                    row_data = []
                    for cell in row.cells:
                        cell_text = ""
                        for segment in cell.layout.text_anchor.text_segments:
                            start_index = segment.start_index or 0
                            end_index = segment.end_index
                            cell_text += text[start_index:end_index]
                        row_data.append(cell_text.strip())
                    table_data.append(row_data)
                tables.append(table_data)
            
            # Extract form fields (for form parser)
            form_fields = []
            for page in document.pages:
                for form_field in page.form_fields:
                    field_name = ""
                    field_value = ""
                    
                    if form_field.field_name:
                        for segment in form_field.field_name.text_anchor.text_segments:
                            start_index = segment.start_index or 0
                            end_index = segment.end_index
                            field_name += text[start_index:end_index]
                    
                    if form_field.field_value:
                        for segment in form_field.field_value.text_anchor.text_segments:
                            start_index = segment.start_index or 0
                            end_index = segment.end_index
                            field_value += text[start_index:end_index]
                    
                    form_fields.append({
                        "name": field_name.strip(),
                        "value": field_value.strip(),
                        "confidence": form_field.field_value.confidence if form_field.field_value else 0
                    })
            
            return {
                "text": text,
                "entities": entities,
                "tables": tables,
                "form_fields": form_fields
            }
            
        except Exception as e:
            logger.error(f"Error extracting structured data: {str(e)}")
            raise
    
    def _classify_document(self, text: str) -> str:
        """Classify document type based on content"""
        text_lower = text.lower()
        
        # Score each document type
        scores = {}
        for doc_type, keywords in self.document_classifiers.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[doc_type] = score
        
        # Return type with highest score
        if scores:
            return max(scores, key=scores.get)
        else:
            return "unknown"
    
    def _calculate_confidence_score(self, document: documentai.Document) -> float:
        """Calculate overall confidence score for document processing"""
        try:
            confidence_scores = []
            
            # Collect entity confidences
            for entity in document.entities:
                confidence_scores.append(entity.confidence)
            
            # Collect page confidence scores
            for page in document.pages:
                if hasattr(page, 'image_quality_scores'):
                    confidence_scores.append(page.image_quality_scores.quality_score)
            
            # Calculate average confidence
            if confidence_scores:
                return sum(confidence_scores) / len(confidence_scores)
            else:
                return 0.5  # Default confidence
                
        except Exception:
            return 0.5
    
    def _categorize_risk(self, risk_text: str) -> str:
        """Categorize risk based on content"""
        risk_categories = {
            "financial": ["financial", "revenue", "profit", "cash flow", "debt", "liquidity"],
            "operational": ["operational", "supply chain", "manufacturing", "production", "capacity"],
            "market": ["market", "competition", "competitive", "demand", "pricing"],
            "regulatory": ["regulatory", "compliance", "legal", "government", "regulation"],
            "technology": ["technology", "cyber", "security", "data", "system", "software"],
            "management": ["management", "key personnel", "leadership", "governance"]
        }
        
        risk_text_lower = risk_text.lower()
        
        for category, keywords in risk_categories.items():
            if any(keyword in risk_text_lower for keyword in keywords):
                return category
        
        return "general"
    
    def _read_document_from_gcs(self, document_uri: str) -> bytes:
        """Read document content from Google Cloud Storage"""
        # Parse GCS URI
        if not document_uri.startswith("gs://"):
            raise ValueError("Document URI must be a GCS path starting with gs://")
        
        uri_parts = document_uri[5:].split("/", 1)
        bucket_name = uri_parts[0]
        blob_name = uri_parts[1]
        
        # Download content
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        return blob.download_as_bytes()
    
    async def _store_results_in_bigquery(self, processed_result: Dict[str, Any]) -> None:
        """Store processing results in BigQuery"""
        try:
            table_id = f"{self.project_id}.ipo_valuation.processed_documents"
            
            # Prepare row for insertion
            row = {
                "document_uri": processed_result["document_uri"],
                "document_type": processed_result["document_type"],
                "processing_timestamp": processed_result["processing_timestamp"],
                "text": processed_result["text"][:10000],  # Truncate long text
                "entities": json.dumps(processed_result["entities"]),
                "tables": json.dumps(processed_result["tables"]),
                "form_fields": json.dumps(processed_result["form_fields"]),
                "nlp_analysis": json.dumps(processed_result["nlp_analysis"]),
                "confidence_score": processed_result["confidence_score"]
            }
            
            # Insert row
            errors = self.bq_client.insert_rows_json(table_id, [row])
            
            if errors:
                logger.error(f"Error inserting row to BigQuery: {errors}")
                
        except Exception as e:
            logger.error(f"Error storing results in BigQuery: {str(e)}")
            # Don't raise - storage failure shouldn't fail the processing