"""
OCR Document Processor
Advanced OCR processing using GCP Document AI and local OCR solutions
"""
import os
import io
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import cv2
import numpy as np
from google.cloud import documentai
from google.cloud import storage

from ...config import settings
from ...utils.metrics import track_time, track_ml_inference, ML_INFERENCE_DURATION
from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OCRResult:
    """OCR processing result"""
    text: str
    confidence: float
    language: str
    page_count: int
    processing_time: float
    method: str
    
    # Structured data
    pages: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    forms: List[Dict[str, Any]]
    
    # Quality metrics
    readability_score: float
    text_density: float
    image_quality_score: float


@dataclass
class DocumentMetadata:
    """Document metadata for processing"""
    file_path: str
    file_type: str
    file_size: int
    page_count: Optional[int] = None
    language_hint: Optional[str] = None
    processing_options: Optional[Dict[str, Any]] = None


class OCRProcessor:
    """Advanced OCR processor with multiple engines"""
    
    def __init__(self):
        self.gcp_client = None
        self.processor_name = None
        self.storage_client = None
        
        # Initialize GCP clients if configured
        if settings.gcp.project_id:
            self._initialize_gcp_clients()
    
    def _initialize_gcp_clients(self):
        """Initialize GCP Document AI and Storage clients"""
        try:
            self.gcp_client = documentai.DocumentProcessorServiceClient()
            self.storage_client = storage.Client(project=settings.gcp.project_id)
            
            # Build processor name
            if settings.gcp.document_ai_processor_id:
                self.processor_name = self.gcp_client.processor_path(
                    settings.gcp.project_id,
                    settings.gcp.document_ai_location,
                    settings.gcp.document_ai_processor_id
                )
            
            logger.info("GCP Document AI clients initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize GCP clients: {str(e)}")
    
    @track_time(ML_INFERENCE_DURATION, {"model_name": "ocr"})
    async def process_document(
        self,
        metadata: DocumentMetadata,
        use_gcp: bool = True,
        quality_enhancement: bool = True
    ) -> OCRResult:
        """
        Process document with OCR
        
        Args:
            metadata: Document metadata
            use_gcp: Whether to use GCP Document AI
            quality_enhancement: Whether to enhance image quality
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            track_ml_inference("ocr", start_time, True)
            
            # Choose processing method
            if use_gcp and self.gcp_client and self.processor_name:
                result = await self._process_with_gcp(metadata, quality_enhancement)
            else:
                result = await self._process_with_tesseract(metadata, quality_enhancement)
            
            # Post-process results
            result = await self._post_process_result(result, metadata)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            result.processing_time = processing_time
            
            logger.info(
                f"OCR processing completed",
                file_path=metadata.file_path,
                method=result.method,
                confidence=result.confidence,
                processing_time=processing_time
            )
            
            return result
            
        except Exception as e:
            track_ml_inference("ocr", start_time, False)
            logger.error(f"OCR processing failed: {str(e)}", file_path=metadata.file_path)
            raise
    
    async def _process_with_gcp(self, metadata: DocumentMetadata, enhance_quality: bool) -> OCRResult:
        """Process document using GCP Document AI"""
        
        # Read document
        with open(metadata.file_path, 'rb') as file:
            document_content = file.read()
        
        # Prepare request
        raw_document = documentai.RawDocument(
            content=document_content,
            mime_type=self._get_mime_type(metadata.file_type)
        )
        
        request = documentai.ProcessRequest(
            name=self.processor_name,
            raw_document=raw_document
        )
        
        # Process document
        response = self.gcp_client.process_document(request=request)
        document = response.document
        
        # Extract text and structure
        full_text = document.text
        pages = self._extract_pages_gcp(document)
        tables = self._extract_tables_gcp(document)
        forms = self._extract_forms_gcp(document)
        
        # Calculate confidence
        confidence = self._calculate_confidence_gcp(document)
        
        # Detect language
        language = self._detect_language(full_text)
        
        return OCRResult(
            text=full_text,
            confidence=confidence,
            language=language,
            page_count=len(pages),
            processing_time=0,  # Will be set by caller
            method="gcp_document_ai",
            pages=pages,
            tables=tables,
            forms=forms,
            readability_score=0,  # Will be calculated in post-processing
            text_density=0,      # Will be calculated in post-processing
            image_quality_score=0 # Will be calculated in post-processing
        )
    
    async def _process_with_tesseract(self, metadata: DocumentMetadata, enhance_quality: bool) -> OCRResult:
        """Process document using Tesseract OCR"""
        
        # Convert document to images if needed
        if metadata.file_type.lower() == 'pdf':
            images = self._pdf_to_images(metadata.file_path)
        else:
            # Direct image processing
            images = [Image.open(metadata.file_path)]
        
        pages = []
        all_text = []
        total_confidence = 0
        
        for i, image in enumerate(images):
            # Enhance image quality if requested
            if enhance_quality:
                image = self._enhance_image_quality(image)
            
            # Convert PIL Image to numpy array for OpenCV operations
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Extract text with detailed data
            ocr_data = pytesseract.image_to_data(
                cv_image,
                output_type=pytesseract.Output.DICT,
                config='--oem 3 --psm 6'
            )
            
            # Extract page text
            page_text = pytesseract.image_to_string(cv_image, config='--oem 3 --psm 6')
            all_text.append(page_text)
            
            # Calculate page confidence
            confidences = [conf for conf in ocr_data['conf'] if conf > 0]
            page_confidence = np.mean(confidences) if confidences else 0
            total_confidence += page_confidence
            
            # Build page structure
            page_data = {
                'page_number': i + 1,
                'text': page_text,
                'confidence': page_confidence,
                'words': self._extract_words_tesseract(ocr_data),
                'lines': self._extract_lines_tesseract(ocr_data)
            }
            pages.append(page_data)
        
        # Combine all text
        full_text = '\n\n'.join(all_text)
        
        # Calculate overall confidence
        avg_confidence = total_confidence / len(pages) if pages else 0
        
        # Detect language
        language = self._detect_language(full_text)
        
        return OCRResult(
            text=full_text,
            confidence=avg_confidence / 100.0,  # Convert to 0-1 scale
            language=language,
            page_count=len(pages),
            processing_time=0,
            method="tesseract",
            pages=pages,
            tables=[],  # Tesseract doesn't extract tables directly
            forms=[],   # Tesseract doesn't extract forms directly
            readability_score=0,
            text_density=0,
            image_quality_score=0
        )
    
    def _pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[Image.Image]:
        """Convert PDF pages to images"""
        pdf_document = fitz.open(pdf_path)
        images = []
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            
            # Convert to image
            mat = fitz.Matrix(dpi/72, dpi/72)  # Scale factor
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("ppm")
            image = Image.open(io.BytesIO(img_data))
            images.append(image)
        
        pdf_document.close()
        return images
    
    def _enhance_image_quality(self, image: Image.Image) -> Image.Image:
        """Enhance image quality for better OCR"""
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to PIL Image
        return Image.fromarray(cleaned)
    
    def _extract_pages_gcp(self, document) -> List[Dict[str, Any]]:
        """Extract page information from GCP Document AI response"""
        pages = []
        
        for page in document.pages:
            page_data = {
                'page_number': page.page_number,
                'dimensions': {
                    'width': page.dimension.width,
                    'height': page.dimension.height
                },
                'blocks': [],
                'paragraphs': [],
                'lines': [],
                'tokens': []
            }
            
            # Extract blocks
            for block in page.blocks:
                block_text = self._extract_text_from_layout(document.text, block.layout.text_anchor)
                page_data['blocks'].append({
                    'text': block_text,
                    'confidence': block.layout.confidence if hasattr(block.layout, 'confidence') else 1.0
                })
            
            # Extract paragraphs
            for paragraph in page.paragraphs:
                para_text = self._extract_text_from_layout(document.text, paragraph.layout.text_anchor)
                page_data['paragraphs'].append({
                    'text': para_text,
                    'confidence': paragraph.layout.confidence if hasattr(paragraph.layout, 'confidence') else 1.0
                })
            
            pages.append(page_data)
        
        return pages
    
    def _extract_tables_gcp(self, document) -> List[Dict[str, Any]]:
        """Extract table information from GCP Document AI response"""
        tables = []
        
        for table in document.pages[0].tables if document.pages else []:
            table_data = {
                'rows': len(table.header_rows) + len(table.body_rows),
                'columns': 0,
                'header_rows': [],
                'body_rows': []
            }
            
            # Extract header rows
            for row in table.header_rows:
                row_data = []
                for cell in row.cells:
                    cell_text = self._extract_text_from_layout(document.text, cell.layout.text_anchor)
                    row_data.append(cell_text.strip())
                table_data['header_rows'].append(row_data)
                table_data['columns'] = max(table_data['columns'], len(row_data))
            
            # Extract body rows
            for row in table.body_rows:
                row_data = []
                for cell in row.cells:
                    cell_text = self._extract_text_from_layout(document.text, cell.layout.text_anchor)
                    row_data.append(cell_text.strip())
                table_data['body_rows'].append(row_data)
                table_data['columns'] = max(table_data['columns'], len(row_data))
            
            tables.append(table_data)
        
        return tables
    
    def _extract_forms_gcp(self, document) -> List[Dict[str, Any]]:
        """Extract form information from GCP Document AI response"""
        forms = []
        
        for page in document.pages:
            for form_field in page.form_fields:
                field_name = self._extract_text_from_layout(
                    document.text, form_field.field_name.text_anchor
                ).strip()
                field_value = self._extract_text_from_layout(
                    document.text, form_field.field_value.text_anchor
                ).strip()
                
                forms.append({
                    'field_name': field_name,
                    'field_value': field_value,
                    'confidence': form_field.field_name.confidence if hasattr(form_field.field_name, 'confidence') else 1.0
                })
        
        return forms
    
    def _extract_text_from_layout(self, document_text: str, text_anchor) -> str:
        """Extract text from GCP Document AI text anchor"""
        if not text_anchor or not text_anchor.text_segments:
            return ""
        
        text_segments = []
        for segment in text_anchor.text_segments:
            start_index = segment.start_index
            end_index = segment.end_index
            text_segments.append(document_text[start_index:end_index])
        
        return ''.join(text_segments)
    
    def _calculate_confidence_gcp(self, document) -> float:
        """Calculate overall confidence from GCP Document AI response"""
        confidences = []
        
        for page in document.pages:
            for paragraph in page.paragraphs:
                if hasattr(paragraph.layout, 'confidence'):
                    confidences.append(paragraph.layout.confidence)
        
        return np.mean(confidences) if confidences else 1.0
    
    def _extract_words_tesseract(self, ocr_data: Dict) -> List[Dict[str, Any]]:
        """Extract word-level information from Tesseract output"""
        words = []
        
        for i in range(len(ocr_data['text'])):
            if ocr_data['conf'][i] > 0:  # Only include confident detections
                words.append({
                    'text': ocr_data['text'][i],
                    'confidence': ocr_data['conf'][i] / 100.0,
                    'bbox': {
                        'left': ocr_data['left'][i],
                        'top': ocr_data['top'][i],
                        'width': ocr_data['width'][i],
                        'height': ocr_data['height'][i]
                    }
                })
        
        return words
    
    def _extract_lines_tesseract(self, ocr_data: Dict) -> List[Dict[str, Any]]:
        """Extract line-level information from Tesseract output"""
        lines = []
        current_line = []
        current_line_num = -1
        
        for i in range(len(ocr_data['text'])):
            if ocr_data['level'][i] == 4:  # Line level
                if ocr_data['line_num'][i] != current_line_num:
                    if current_line:
                        line_text = ' '.join([word['text'] for word in current_line])
                        line_confidence = np.mean([word['confidence'] for word in current_line])
                        lines.append({
                            'text': line_text,
                            'confidence': line_confidence,
                            'words': current_line
                        })
                    current_line = []
                    current_line_num = ocr_data['line_num'][i]
                
                if ocr_data['conf'][i] > 0:
                    current_line.append({
                        'text': ocr_data['text'][i],
                        'confidence': ocr_data['conf'][i] / 100.0
                    })
        
        # Add final line
        if current_line:
            line_text = ' '.join([word['text'] for word in current_line])
            line_confidence = np.mean([word['confidence'] for word in current_line])
            lines.append({
                'text': line_text,
                'confidence': line_confidence,
                'words': current_line
            })
        
        return lines
    
    def _detect_language(self, text: str) -> str:
        """Detect language of the text"""
        try:
            from langdetect import detect
            return detect(text)
        except:
            return 'en'  # Default to English
    
    def _get_mime_type(self, file_type: str) -> str:
        """Get MIME type for file type"""
        mime_types = {
            'pdf': 'application/pdf',
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'tiff': 'image/tiff',
            'gif': 'image/gif'
        }
        return mime_types.get(file_type.lower(), 'application/octet-stream')
    
    async def _post_process_result(self, result: OCRResult, metadata: DocumentMetadata) -> OCRResult:
        """Post-process OCR result with additional metrics"""
        
        # Calculate readability score
        result.readability_score = self._calculate_readability_score(result.text)
        
        # Calculate text density
        result.text_density = self._calculate_text_density(result.text, metadata)
        
        # Calculate image quality score (placeholder)
        result.image_quality_score = 0.85  # This would be calculated from actual image analysis
        
        return result
    
    def _calculate_readability_score(self, text: str) -> float:
        """Calculate readability score (Flesch Reading Ease)"""
        try:
            import textstat
            return textstat.flesch_reading_ease(text) / 100.0
        except:
            # Fallback calculation
            words = len(text.split())
            sentences = text.count('.') + text.count('!') + text.count('?')
            if words == 0 or sentences == 0:
                return 0.5
            
            avg_sentence_length = words / sentences
            # Simple readability approximation
            return max(0, min(1, (20 - avg_sentence_length) / 20))
    
    def _calculate_text_density(self, text: str, metadata: DocumentMetadata) -> float:
        """Calculate text density (characters per page)"""
        if metadata.page_count and metadata.page_count > 0:
            return len(text) / metadata.page_count
        return len(text)


def create_ocr_processor() -> OCRProcessor:
    """Create configured OCR processor"""
    return OCRProcessor()