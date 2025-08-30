# Document Preprocessing Pipeline for IPO Valuation

This document provides comprehensive guidance for preprocessing various document types in the IPO valuation platform, including PDF prospectuses, HTML filings, and multi-language regulatory documents.

## 📋 Overview

Document preprocessing is crucial for extracting valuable insights from IPO prospectuses, ASIC filings, and other regulatory documents. This pipeline handles PDF text extraction, HTML parsing, OCR error correction, and multi-language support for international filings.

## 🎯 Key Components

- **PDF Processing**: Extract text from complex prospectuses with tables and charts
- **HTML/XML Parsing**: Clean web-scraped financial data and regulatory filings
- **OCR Enhancement**: Correct scanning artifacts and improve text quality
- **Multi-language Support**: Handle international IPO filings
- **Structure Preservation**: Maintain document hierarchy and metadata

## 1. PDF Text Extraction and Cleaning

### 1.1 Advanced PDF Processing

```python
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import tabula
import camelot
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from pathlib import Path

@dataclass
class DocumentMetadata:
    """Metadata for processed documents."""
    filename: str
    document_type: str  # 'prospectus', 'annual_report', 'filing'
    company_name: str
    filing_date: str
    total_pages: int
    language: str = 'en'
    currency: str = 'AUD'
    
class PDFProcessor:
    """
    Advanced PDF processing for IPO prospectuses and regulatory documents.
    
    Handles complex layouts, tables, and financial data extraction.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
    def _default_config(self) -> Dict:
        """Default configuration for PDF processing."""
        return {
            'extraction_methods': ['pdfplumber', 'pymupdf', 'pypdf2'],
            'table_extraction': 'camelot',  # or 'tabula'
            'ocr_fallback': True,
            'language_detection': True,
            'preserve_layout': True,
            'min_confidence': 0.8
        }
    
    def extract_text_comprehensive(self, pdf_path: str) -> Dict:
        """
        Extract text using multiple methods and combine results.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted content and metadata
        """
        results = {
            'text': '',
            'tables': [],
            'metadata': {},
            'sections': {},
            'extraction_quality': 0.0
        }
        
        # Try multiple extraction methods
        extraction_results = {}
        
        for method in self.config['extraction_methods']:
            try:
                if method == 'pdfplumber':
                    extraction_results[method] = self._extract_with_pdfplumber(pdf_path)
                elif method == 'pymupdf':
                    extraction_results[method] = self._extract_with_pymupdf(pdf_path)
                elif method == 'pypdf2':
                    extraction_results[method] = self._extract_with_pypdf2(pdf_path)
                    
            except Exception as e:
                self.logger.warning(f"Method {method} failed: {e}")
                continue
        
        # Combine results using best quality text
        best_result = self._select_best_extraction(extraction_results)
        results.update(best_result)
        
        # Extract tables separately
        results['tables'] = self._extract_tables(pdf_path)
        
        # Extract metadata
        results['metadata'] = self._extract_metadata(pdf_path)
        
        # Identify document sections
        results['sections'] = self._identify_sections(results['text'])
        
        return results
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> Dict:
        """Extract text using pdfplumber (best for complex layouts)."""
        text_content = []
        page_data = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text with layout preservation
                text = page.extract_text(layout=self.config['preserve_layout'])
                
                if text:
                    text_content.append(text)
                    
                    # Extract additional page information
                    page_info = {
                        'page_number': page_num + 1,
                        'text': text,
                        'width': page.width,
                        'height': page.height,
                        'rotation': page.rotation if hasattr(page, 'rotation') else 0
                    }
                    page_data.append(page_info)
        
        return {
            'text': '\\n'.join(text_content),
            'pages': page_data,
            'method': 'pdfplumber',
            'quality_score': self._calculate_quality_score('\\n'.join(text_content))
        }
    
    def _extract_with_pymupdf(self, pdf_path: str) -> Dict:
        """Extract text using PyMuPDF (good for scanned documents)."""
        doc = fitz.open(pdf_path)
        text_content = []
        page_data = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # Extract text
            text = page.get_text()
            
            if text.strip():
                text_content.append(text)
                
                # Get page information
                page_info = {
                    'page_number': page_num + 1,
                    'text': text,
                    'rect': page.rect,
                    'rotation': page.rotation
                }
                page_data.append(page_info)
        
        doc.close()
        
        return {
            'text': '\\n'.join(text_content),
            'pages': page_data,
            'method': 'pymupdf',
            'quality_score': self._calculate_quality_score('\\n'.join(text_content))
        }
    
    def _extract_with_pypdf2(self, pdf_path: str) -> Dict:
        """Extract text using PyPDF2 (fallback method)."""
        text_content = []
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
                except Exception as e:
                    self.logger.warning(f"Failed to extract page {page_num}: {e}")
                    continue
        
        return {
            'text': '\\n'.join(text_content),
            'method': 'pypdf2',
            'quality_score': self._calculate_quality_score('\\n'.join(text_content))
        }
    
    def _extract_tables(self, pdf_path: str) -> List[pd.DataFrame]:
        """Extract tables from PDF using specialized tools."""
        tables = []
        
        try:
            if self.config['table_extraction'] == 'camelot':
                # Camelot is better for complex table structures
                table_list = camelot.read_pdf(pdf_path, pages='all')
                
                for table in table_list:
                    if table.parsing_report['accuracy'] > self.config['min_confidence']:
                        df = table.df
                        # Clean table data
                        df = self._clean_table_data(df)
                        tables.append(df)
                        
            elif self.config['table_extraction'] == 'tabula':
                # Tabula is faster for simple tables
                table_list = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
                
                for df in table_list:
                    if not df.empty:
                        df = self._clean_table_data(df)
                        tables.append(df)
                        
        except Exception as e:
            self.logger.warning(f"Table extraction failed: {e}")
        
        return tables
    
    def _clean_table_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean extracted table data."""
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Handle merged cells and formatting issues
        df = df.replace(r'^\\s*$', np.nan, regex=True)  # Replace whitespace with NaN
        
        # Try to identify numeric columns and clean them
        for col in df.columns:
            if df[col].dtype == 'object':
                # Clean currency symbols and thousands separators
                cleaned_col = df[col].astype(str).str.replace('[,$AUD\\s]', '', regex=True)
                
                # Try to convert to numeric
                try:
                    numeric_col = pd.to_numeric(cleaned_col, errors='coerce')
                    if numeric_col.notna().sum() > len(df) * 0.5:  # If >50% are numeric
                        df[col] = numeric_col
                except:
                    continue
        
        return df
    
    def _calculate_quality_score(self, text: str) -> float:
        """Calculate quality score for extracted text."""
        if not text:
            return 0.0
        
        # Factors that indicate good quality text
        total_chars = len(text)
        alphabetic_chars = sum(c.isalpha() for c in text)
        numeric_chars = sum(c.isdigit() for c in text)
        whitespace_chars = sum(c.isspace() for c in text)
        
        # Calculate ratios
        alpha_ratio = alphabetic_chars / total_chars if total_chars > 0 else 0
        numeric_ratio = numeric_chars / total_chars if total_chars > 0 else 0
        whitespace_ratio = whitespace_chars / total_chars if total_chars > 0 else 0
        
        # Good text should have reasonable ratios
        quality_score = 0.0
        
        # Alphabetic content should be substantial
        if 0.3 <= alpha_ratio <= 0.8:
            quality_score += 0.4
        
        # Some numeric content is expected in financial documents
        if 0.05 <= numeric_ratio <= 0.3:
            quality_score += 0.3
        
        # Reasonable whitespace
        if 0.1 <= whitespace_ratio <= 0.3:
            quality_score += 0.3
        
        return min(quality_score, 1.0)
```

### 1.2 OCR Error Correction

```python
import pytesseract
from PIL import Image
import cv2
import numpy as np
from textdistance import levenshtein
from autocorrect import Speller
import language_tool_python

class OCRErrorCorrector:
    """
    Advanced OCR error correction for scanned financial documents.
    
    Handles common OCR errors in financial text and numbers.
    """
    
    def __init__(self, language: str = 'en'):
        self.language = language
        self.spell_checker = Speller(lang=language)
        self.grammar_checker = language_tool_python.LanguageTool(language)
        self.financial_vocab = self._load_financial_vocabulary()
        
    def _load_financial_vocabulary(self) -> set:
        """Load financial terms and company names for correction."""
        financial_terms = {
            # Financial terms
            'revenue', 'profit', 'loss', 'assets', 'liabilities', 'equity',
            'dividend', 'earnings', 'ebitda', 'cash', 'debt', 'valuation',
            'ipo', 'prospectus', 'offering', 'shares', 'securities',
            
            # Australian specific
            'asic', 'asx', 'aasb', 'apra', 'aud', 'australian',
            
            # Common OCR confusions
            'financial', 'statement', 'balance', 'income', 'directors'
        }
        
        # Add currency codes
        currencies = {'AUD', 'USD', 'EUR', 'GBP', 'JPY', 'NZD'}
        financial_terms.update(currencies)
        
        return financial_terms
    
    def correct_ocr_text(self, text: str, confidence_threshold: float = 0.8) -> Dict:
        """
        Correct OCR errors in extracted text.
        
        Args:
            text: Raw OCR text
            confidence_threshold: Minimum confidence for corrections
            
        Returns:
            Dictionary with corrected text and correction statistics
        """
        original_text = text
        corrected_text = text
        corrections_made = []
        
        # 1. Fix common OCR character substitutions
        corrected_text = self._fix_character_substitutions(corrected_text)
        
        # 2. Correct numerical errors
        corrected_text = self._correct_numerical_errors(corrected_text)
        
        # 3. Fix word-level errors using dictionary
        words = corrected_text.split()
        corrected_words = []
        
        for word in words:
            corrected_word, correction_info = self._correct_word(word, confidence_threshold)
            corrected_words.append(corrected_word)
            
            if correction_info:
                corrections_made.append(correction_info)
        
        corrected_text = ' '.join(corrected_words)
        
        # 4. Grammar correction for sentences
        corrected_text = self._correct_grammar(corrected_text)
        
        return {
            'original_text': original_text,
            'corrected_text': corrected_text,
            'corrections_made': corrections_made,
            'correction_count': len(corrections_made),
            'confidence_score': self._calculate_correction_confidence(
                original_text, corrected_text
            )
        }
    
    def _fix_character_substitutions(self, text: str) -> str:
        """Fix common OCR character recognition errors."""
        substitutions = {
            # Numbers often confused with letters
            'O': '0',  # Letter O -> Zero (in numerical contexts)
            'l': '1',  # Lowercase l -> One (in numerical contexts)
            'I': '1',  # Capital I -> One (in numerical contexts)
            'S': '5',  # S -> 5 (in numerical contexts)
            'B': '8',  # B -> 8 (in numerical contexts)
            
            # Letters often confused with numbers
            '0': 'O',  # Zero -> Letter O (in text contexts)
            '1': 'I',  # One -> Capital I (in text contexts)
            '5': 'S',  # 5 -> S (in text contexts)
            
            # Common character confusions
            'rn': 'm',  # rn -> m
            'cl': 'd',  # cl -> d
            'vv': 'w',  # vv -> w
            'ii': 'ii', # Keep as is
            
            # Punctuation errors
            ',': '.',  # In decimal numbers
            ';': ':',  # Colons
        }
        
        corrected = text
        
        # Apply contextual substitutions
        # For numbers (look for digit contexts)
        number_pattern = r'\\b\\d*[OlISB]\\d*\\b'
        for match in re.finditer(number_pattern, text):
            number_text = match.group()
            corrected_number = number_text
            for char, replacement in substitutions.items():
                if char in 'OlISB':  # Only apply to letters in number context
                    corrected_number = corrected_number.replace(char, replacement)
            corrected = corrected.replace(number_text, corrected_number)
        
        return corrected
    
    def _correct_numerical_errors(self, text: str) -> str:
        """Correct errors in numerical values and currency amounts."""
        # Pattern for currency amounts (with possible OCR errors)
        currency_pattern = r'\\$?\\s*([0-9,OlISB]+\\.?[0-9OlISB]*)'
        
        def correct_number(match):
            number_str = match.group(1)
            
            # Fix character substitutions in numbers
            corrections = {'O': '0', 'l': '1', 'I': '1', 'S': '5', 'B': '8'}
            for char, replacement in corrections.items():
                number_str = number_str.replace(char, replacement)
            
            return match.group(0).replace(match.group(1), number_str)
        
        corrected = re.sub(currency_pattern, correct_number, text)
        
        # Fix percentage values
        percentage_pattern = r'([0-9OlISB]+\\.?[0-9OlISB]*)\\s*%'
        corrected = re.sub(percentage_pattern, correct_number, corrected)
        
        return corrected
    
    def _correct_word(self, word: str, confidence_threshold: float) -> Tuple[str, Dict]:
        """Correct individual word using multiple strategies."""
        if not word or word.isdigit() or len(word) < 3:
            return word, None
        
        original_word = word
        
        # Check if word is in financial vocabulary (case-insensitive)
        if word.lower() in self.financial_vocab:
            return word, None
        
        # Try spell correction
        suggested = self.spell_checker(word)
        
        if suggested != word:
            # Calculate confidence based on edit distance
            distance = levenshtein.distance(word, suggested)
            confidence = max(0, 1 - (distance / len(word)))
            
            if confidence >= confidence_threshold:
                correction_info = {
                    'original': original_word,
                    'corrected': suggested,
                    'confidence': confidence,
                    'method': 'spell_check'
                }
                return suggested, correction_info
        
        # Check against financial terms with fuzzy matching
        best_match = None
        best_score = 0
        
        for term in self.financial_vocab:
            if abs(len(word) - len(term)) > 2:  # Skip very different lengths
                continue
                
            similarity = 1 - (levenshtein.distance(word.lower(), term.lower()) / 
                             max(len(word), len(term)))
            
            if similarity > best_score and similarity >= confidence_threshold:
                best_score = similarity
                best_match = term
        
        if best_match:
            correction_info = {
                'original': original_word,
                'corrected': best_match,
                'confidence': best_score,
                'method': 'financial_vocab'
            }
            return best_match, correction_info
        
        return word, None
    
    def _correct_grammar(self, text: str) -> str:
        """Apply grammar corrections to improve sentence structure."""
        try:
            matches = self.grammar_checker.check(text)
            
            # Apply corrections in reverse order to maintain positions
            for match in reversed(matches):
                if match.replacements and len(match.replacements) > 0:
                    # Only apply high-confidence corrections
                    if len(match.replacements[0]) > 0:
                        text = (text[:match.offset] + 
                               match.replacements[0] + 
                               text[match.offset + match.errorLength:])
            
            return text
            
        except Exception as e:
            self.logger.warning(f"Grammar correction failed: {e}")
            return text
    
    def _calculate_correction_confidence(self, original: str, corrected: str) -> float:
        """Calculate overall confidence in the correction process."""
        if original == corrected:
            return 1.0  # No changes needed
        
        # Calculate based on the ratio of changes to total text
        distance = levenshtein.distance(original, corrected)
        max_length = max(len(original), len(corrected))
        
        if max_length == 0:
            return 1.0
        
        similarity = 1 - (distance / max_length)
        
        # Confidence should be high if changes are reasonable
        change_ratio = distance / max_length
        
        if change_ratio < 0.1:  # Less than 10% changed
            return min(similarity + 0.2, 1.0)
        elif change_ratio < 0.3:  # Less than 30% changed
            return similarity
        else:  # Major changes - lower confidence
            return max(similarity - 0.2, 0.0)
```

## 2. HTML/XML Parsing for Web-scraped Data

### 2.1 Financial Data Web Scraping

```python
import requests
from bs4 import BeautifulSoup
import lxml
import html5lib
from urllib.parse import urljoin, urlparse
import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class WebDataParser:
    """
    Parse financial data from web sources including ASX, ASIC, and company websites.
    
    Handles dynamic content, JavaScript-rendered pages, and complex HTML structures.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def _default_config(self) -> Dict:
        """Default configuration for web parsing."""
        return {
            'timeout': 30,
            'retry_attempts': 3,
            'delay_between_requests': 1,
            'use_selenium_for_js': True,
            'parsers': ['lxml', 'html5lib', 'html.parser'],
            'extract_tables': True,
            'extract_links': True,
            'clean_text': True
        }
    
    def parse_asx_announcements(self, company_code: str, 
                              start_date: str = None) -> List[Dict]:
        """
        Parse ASX announcements for a specific company.
        
        Args:
            company_code: ASX company code (e.g., 'CBA')
            start_date: Start date for announcements (YYYY-MM-DD)
            
        Returns:
            List of announcement dictionaries
        """
        base_url = f"https://www.asx.com.au/asxpdf/{company_code}"
        announcements = []
        
        try:
            # This would integrate with ASX API or scrape announcement pages
            response = self.session.get(base_url, timeout=self.config['timeout'])
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Extract announcement data (structure varies by ASX page layout)
            announcement_rows = soup.find_all('tr', class_='announcement-row')
            
            for row in announcement_rows:
                try:
                    announcement = self._parse_announcement_row(row, company_code)
                    if announcement:
                        announcements.append(announcement)
                except Exception as e:
                    self.logger.warning(f"Failed to parse announcement row: {e}")
                    continue
            
        except Exception as e:
            self.logger.error(f"Failed to fetch ASX announcements: {e}")
        
        return announcements
    
    def parse_asic_company_extract(self, company_number: str) -> Dict:
        """
        Parse ASIC company extract information.
        
        Args:
            company_number: ASIC company number (ACN)
            
        Returns:
            Dictionary with company information
        """
        # Note: This would require proper ASIC API integration
        # This is a simplified example structure
        
        company_info = {
            'company_number': company_number,
            'company_name': '',
            'registration_date': '',
            'company_type': '',
            'status': '',
            'registered_office': {},
            'directors': [],
            'share_structure': {},
            'annual_return_date': ''
        }
        
        # Implementation would integrate with ASIC Connect API
        # or parse company extract documents
        
        return company_info
    
    def parse_financial_tables(self, url: str, table_selectors: List[str] = None) -> List[pd.DataFrame]:
        """
        Extract financial tables from web pages.
        
        Args:
            url: URL containing financial tables
            table_selectors: CSS selectors for specific tables
            
        Returns:
            List of DataFrames containing financial data
        """
        tables = []
        
        try:
            response = self.session.get(url, timeout=self.config['timeout'])
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            if table_selectors:
                # Use specific selectors
                for selector in table_selectors:
                    table_elements = soup.select(selector)
                    for table_elem in table_elements:
                        df = self._parse_html_table(table_elem)
                        if not df.empty:
                            tables.append(df)
            else:
                # Find all tables
                table_elements = soup.find_all('table')
                for table_elem in table_elements:
                    df = self._parse_html_table(table_elem)
                    if not df.empty:
                        tables.append(df)
        
        except Exception as e:
            self.logger.error(f"Failed to parse financial tables: {e}")
        
        return tables
    
    def _parse_html_table(self, table_element) -> pd.DataFrame:
        """Convert HTML table element to DataFrame."""
        try:
            # Extract headers
            headers = []
            header_row = table_element.find('tr')
            
            if header_row:
                for th in header_row.find_all(['th', 'td']):
                    headers.append(th.get_text(strip=True))
            
            # Extract data rows
            rows_data = []
            for row in table_element.find_all('tr')[1:]:  # Skip header row
                row_data = []
                for cell in row.find_all(['td', 'th']):
                    cell_text = cell.get_text(strip=True)
                    # Clean financial data
                    cell_text = self._clean_financial_cell(cell_text)
                    row_data.append(cell_text)
                
                if row_data:  # Only add non-empty rows
                    rows_data.append(row_data)
            
            # Create DataFrame
            if headers and rows_data:
                # Ensure all rows have the same number of columns
                max_cols = len(headers)
                for i, row in enumerate(rows_data):
                    while len(row) < max_cols:
                        row.append('')
                    if len(row) > max_cols:
                        row = row[:max_cols]
                    rows_data[i] = row
                
                df = pd.DataFrame(rows_data, columns=headers)
                return self._clean_financial_dataframe(df)
            
        except Exception as e:
            self.logger.warning(f"Failed to parse HTML table: {e}")
        
        return pd.DataFrame()
    
    def _clean_financial_cell(self, cell_text: str) -> str:
        """Clean individual table cell text."""
        if not cell_text:
            return ''
        
        # Remove extra whitespace
        cell_text = ' '.join(cell_text.split())
        
        # Clean currency symbols and formatting
        cell_text = cell_text.replace('$', '').replace('AUD', '').replace(',', '')
        
        # Handle negative values in parentheses
        if cell_text.startswith('(') and cell_text.endswith(')'):
            cell_text = '-' + cell_text[1:-1]
        
        return cell_text.strip()
    
    def _clean_financial_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean financial DataFrame."""
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Try to convert numeric columns
        for col in df.columns:
            if col and df[col].dtype == 'object':
                # Check if column contains mostly numeric data
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.notna().sum() > len(df) * 0.7:  # If >70% are numeric
                    df[col] = numeric_series
        
        return df
```

## 3. Multi-language Support

### 3.1 Language Detection and Translation

```python
import langdetect
from googletrans import Translator
from textblob import TextBlob
import polyglot
from polyglot.detect import Detector
from polyglot.text import Text

class MultiLanguageProcessor:
    """
    Handle multi-language document processing for international IPOs.
    
    Supports language detection, translation, and cross-language financial term extraction.
    """
    
    def __init__(self):
        self.translator = Translator()
        self.supported_languages = {
            'en': 'English',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'de': 'German',
            'fr': 'French',
            'es': 'Spanish',
            'pt': 'Portuguese',
            'it': 'Italian',
            'ru': 'Russian'
        }
        
        # Financial terms in different languages
        self.financial_terms_multilang = {
            'en': ['revenue', 'profit', 'assets', 'liabilities', 'equity'],
            'zh': ['收入', '利润', '资产', '负债', '股权'],
            'ja': ['収益', '利益', '資産', '負債', '株式'],
            'de': ['Umsatz', 'Gewinn', 'Vermögen', 'Verbindlichkeiten', 'Eigenkapital'],
            'fr': ['chiffre d\\'affaires', 'profit', 'actifs', 'passifs', 'capitaux propres']
        }
    
    def detect_language(self, text: str, confidence_threshold: float = 0.9) -> Dict:
        """
        Detect the language of the input text.
        
        Args:
            text: Input text
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Dictionary with language code and confidence
        """
        try:
            # Use multiple detection methods for better accuracy
            results = {}
            
            # Method 1: langdetect
            try:
                lang_detect_result = langdetect.detect_langs(text)
                if lang_detect_result:
                    results['langdetect'] = {
                        'language': lang_detect_result[0].lang,
                        'confidence': lang_detect_result[0].prob
                    }
            except:
                pass
            
            # Method 2: TextBlob
            try:
                blob = TextBlob(text)
                textblob_lang = blob.detect_language()
                results['textblob'] = {
                    'language': textblob_lang,
                    'confidence': 0.8  # TextBlob doesn't provide confidence
                }
            except:
                pass
            
            # Method 3: Polyglot
            try:
                detector = Detector(text)
                results['polyglot'] = {
                    'language': detector.language.code,
                    'confidence': detector.language.confidence
                }
            except:
                pass
            
            # Combine results using weighted voting
            language_votes = {}
            total_weight = 0
            
            for method, result in results.items():
                lang = result['language']
                confidence = result['confidence']
                
                if lang not in language_votes:
                    language_votes[lang] = 0
                language_votes[lang] += confidence
                total_weight += confidence
            
            if language_votes:
                # Get the language with highest weighted score
                best_language = max(language_votes, key=language_votes.get)
                best_confidence = language_votes[best_language] / total_weight
                
                return {
                    'language': best_language,
                    'confidence': best_confidence,
                    'supported': best_language in self.supported_languages,
                    'language_name': self.supported_languages.get(best_language, 'Unknown'),
                    'detection_methods': results
                }
            
        except Exception as e:
            self.logger.warning(f"Language detection failed: {e}")
        
        # Default to English if detection fails
        return {
            'language': 'en',
            'confidence': 0.5,
            'supported': True,
            'language_name': 'English',
            'detection_methods': {}
        }
    
    def translate_financial_document(self, text: str, 
                                   target_language: str = 'en',
                                   preserve_financial_terms: bool = True) -> Dict:
        """
        Translate financial document while preserving key financial terms.
        
        Args:
            text: Input text to translate
            target_language: Target language code
            preserve_financial_terms: Whether to preserve financial terms
            
        Returns:
            Dictionary with translated text and metadata
        """
        # Detect source language
        source_lang_info = self.detect_language(text)
        source_language = source_lang_info['language']
        
        if source_language == target_language:
            return {
                'original_text': text,
                'translated_text': text,
                'source_language': source_language,
                'target_language': target_language,
                'translation_needed': False
            }
        
        try:
            if preserve_financial_terms:
                # Extract and preserve financial terms
                financial_terms = self._extract_financial_terms(text, source_language)
                protected_text = self._protect_financial_terms(text, financial_terms)
                
                # Translate protected text
                translated = self.translator.translate(
                    protected_text, 
                    src=source_language, 
                    dest=target_language
                ).text
                
                # Restore financial terms
                final_text = self._restore_financial_terms(translated, financial_terms, target_language)
            else:
                # Direct translation
                translated = self.translator.translate(
                    text, 
                    src=source_language, 
                    dest=target_language
                ).text
                final_text = translated
            
            return {
                'original_text': text,
                'translated_text': final_text,
                'source_language': source_language,
                'target_language': target_language,
                'translation_needed': True,
                'confidence': source_lang_info['confidence']
            }
            
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            return {
                'original_text': text,
                'translated_text': text,
                'source_language': source_language,
                'target_language': target_language,
                'translation_needed': True,
                'error': str(e)
            }
    
    def _extract_financial_terms(self, text: str, language: str) -> List[str]:
        """Extract financial terms from text in specific language."""
        terms = []
        
        if language in self.financial_terms_multilang:
            lang_terms = self.financial_terms_multilang[language]
            
            for term in lang_terms:
                if term in text:
                    terms.append(term)
        
        # Also extract numbers, currency symbols, and percentages
        import re
        
        # Currency amounts
        currency_pattern = r'[\\$€£¥₹]\\s*[0-9,]+\\.?[0-9]*'
        currency_matches = re.findall(currency_pattern, text)
        terms.extend(currency_matches)
        
        # Percentages
        percentage_pattern = r'[0-9]+\\.?[0-9]*\\s*%'
        percentage_matches = re.findall(percentage_pattern, text)
        terms.extend(percentage_matches)
        
        return list(set(terms))  # Remove duplicates
    
    def _protect_financial_terms(self, text: str, terms: List[str]) -> str:
        """Replace financial terms with placeholders before translation."""
        protected_text = text
        term_map = {}
        
        for i, term in enumerate(terms):
            placeholder = f"__FINANCIAL_TERM_{i}__"
            term_map[placeholder] = term
            protected_text = protected_text.replace(term, placeholder)
        
        self._term_map = term_map  # Store for restoration
        return protected_text
    
    def _restore_financial_terms(self, text: str, original_terms: List[str], 
                                target_language: str) -> str:
        """Restore financial terms after translation."""
        restored_text = text
        
        if hasattr(self, '_term_map'):
            for placeholder, original_term in self._term_map.items():
                # Try to translate the financial term appropriately
                if target_language in self.financial_terms_multilang:
                    # Use predefined translations if available
                    target_terms = self.financial_terms_multilang[target_language]
                    # This would need a more sophisticated mapping
                    translated_term = original_term  # Fallback to original
                else:
                    translated_term = original_term
                
                restored_text = restored_text.replace(placeholder, translated_term)
        
        return restored_text
```

## 4. Document Structure Preservation

### 4.1 Hierarchical Document Analysis

```python
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class SectionType(Enum):
    """Types of document sections in financial documents."""
    TITLE = "title"
    EXECUTIVE_SUMMARY = "executive_summary"
    COMPANY_OVERVIEW = "company_overview"
    FINANCIAL_INFORMATION = "financial_information"
    RISK_FACTORS = "risk_factors"
    MANAGEMENT = "management"
    SHARE_OFFER = "share_offer"
    USE_OF_FUNDS = "use_of_funds"
    FINANCIAL_STATEMENTS = "financial_statements"
    NOTES = "notes"
    APPENDIX = "appendix"
    TABLE_OF_CONTENTS = "table_of_contents"
    FOOTER = "footer"
    OTHER = "other"

@dataclass
class DocumentSection:
    """Represents a section of a financial document."""
    section_type: SectionType
    title: str
    content: str
    page_number: int
    level: int  # Hierarchy level (1 = main section, 2 = subsection, etc.)
    start_position: int
    end_position: int
    subsections: List['DocumentSection'] = None
    
    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []

class DocumentStructureAnalyzer:
    """
    Analyze and preserve document structure in financial documents.
    
    Identifies sections, maintains hierarchy, and extracts metadata.
    """
    
    def __init__(self):
        self.section_patterns = self._compile_section_patterns()
        self.table_patterns = self._compile_table_patterns()
        
    def _compile_section_patterns(self) -> Dict[SectionType, List[re.Pattern]]:
        """Compile regex patterns for identifying document sections."""
        patterns = {
            SectionType.EXECUTIVE_SUMMARY: [
                re.compile(r'executive\\s+summary', re.IGNORECASE),
                re.compile(r'investment\\s+highlights?', re.IGNORECASE),
                re.compile(r'key\\s+information', re.IGNORECASE)
            ],
            SectionType.COMPANY_OVERVIEW: [
                re.compile(r'company\\s+overview', re.IGNORECASE),
                re.compile(r'about\\s+the\\s+company', re.IGNORECASE),
                re.compile(r'business\\s+description', re.IGNORECASE),
                re.compile(r'our\\s+business', re.IGNORECASE)
            ],
            SectionType.FINANCIAL_INFORMATION: [
                re.compile(r'financial\\s+information', re.IGNORECASE),
                re.compile(r'financial\\s+summary', re.IGNORECASE),
                re.compile(r'selected\\s+financial\\s+data', re.IGNORECASE)
            ],
            SectionType.RISK_FACTORS: [
                re.compile(r'risk\\s+factors?', re.IGNORECASE),
                re.compile(r'investment\\s+risks?', re.IGNORECASE),
                re.compile(r'principal\\s+risks?', re.IGNORECASE)
            ],
            SectionType.MANAGEMENT: [
                re.compile(r'management', re.IGNORECASE),
                re.compile(r'directors?\\s+and\\s+officers?', re.IGNORECASE),
                re.compile(r'board\\s+of\\s+directors?', re.IGNORECASE),
                re.compile(r'key\\s+personnel', re.IGNORECASE)
            ],
            SectionType.SHARE_OFFER: [
                re.compile(r'share\\s+offer', re.IGNORECASE),
                re.compile(r'public\\s+offer', re.IGNORECASE),
                re.compile(r'offer\\s+details', re.IGNORECASE),
                re.compile(r'securities\\s+offered?', re.IGNORECASE)
            ],
            SectionType.USE_OF_FUNDS: [
                re.compile(r'use\\s+of\\s+funds', re.IGNORECASE),
                re.compile(r'use\\s+of\\s+proceeds', re.IGNORECASE),
                re.compile(r'application\\s+of\\s+funds', re.IGNORECASE)
            ],
            SectionType.FINANCIAL_STATEMENTS: [
                re.compile(r'financial\\s+statements?', re.IGNORECASE),
                re.compile(r'audited\\s+financial', re.IGNORECASE),
                re.compile(r'balance\\s+sheet', re.IGNORECASE),
                re.compile(r'income\\s+statement', re.IGNORECASE),
                re.compile(r'cash\\s+flow', re.IGNORECASE)
            ]
        }
        
        return patterns
    
    def analyze_document_structure(self, text: str, 
                                 metadata: Dict = None) -> Dict:
        """
        Analyze document structure and extract sections.
        
        Args:
            text: Full document text
            metadata: Additional document metadata
            
        Returns:
            Dictionary with structured document analysis
        """
        # Split text into paragraphs/lines for analysis
        lines = text.split('\\n')
        
        # Identify section breaks and headers
        sections = self._identify_sections(text, lines)
        
        # Build hierarchical structure
        structured_sections = self._build_hierarchy(sections)
        
        # Extract tables and figures
        tables = self._extract_table_references(text)
        figures = self._extract_figure_references(text)
        
        # Analyze document metadata
        doc_metadata = self._extract_document_metadata(text, metadata)
        
        return {
            'sections': structured_sections,
            'tables': tables,
            'figures': figures,
            'metadata': doc_metadata,
            'structure_confidence': self._calculate_structure_confidence(structured_sections),
            'total_sections': len(sections),
            'hierarchy_levels': self._count_hierarchy_levels(structured_sections)
        }
    
    def _identify_sections(self, full_text: str, lines: List[str]) -> List[DocumentSection]:
        """Identify document sections based on patterns and formatting."""
        sections = []
        current_position = 0
        
        for line_num, line in enumerate(lines):
            line_clean = line.strip()
            
            if not line_clean:
                current_position += len(line) + 1
                continue
            
            # Check if line is a section header
            section_type, confidence = self._classify_section_header(line_clean)
            
            if confidence > 0.7:  # High confidence threshold
                # Determine section level based on formatting
                level = self._determine_section_level(line, line_num, lines)
                
                # Find section content
                content_start = current_position
                content_end = self._find_section_end(line_num, lines, full_text, current_position)
                
                section = DocumentSection(
                    section_type=section_type,
                    title=line_clean,
                    content=full_text[content_start:content_end],
                    page_number=self._estimate_page_number(line_num, lines),
                    level=level,
                    start_position=content_start,
                    end_position=content_end
                )
                
                sections.append(section)
            
            current_position += len(line) + 1
        
        return sections
    
    def _classify_section_header(self, line: str) -> Tuple[SectionType, float]:
        """Classify a line as a specific section type."""
        best_match = SectionType.OTHER
        best_score = 0.0
        
        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                if pattern.search(line):
                    # Calculate match quality
                    score = len(pattern.pattern) / len(line) if len(line) > 0 else 0
                    score = min(score, 1.0)  # Cap at 1.0
                    
                    if score > best_score:
                        best_score = score
                        best_match = section_type
        
        # Additional heuristics for section identification
        if best_score < 0.5:
            # Check for numbered sections (e.g., "1. Executive Summary")
            if re.match(r'^\\d+\\.?\\s+', line):
                best_score += 0.3
                if best_match == SectionType.OTHER:
                    best_match = self._infer_section_from_context(line)
            
            # Check for all-caps headers
            if line.isupper() and len(line) > 5:
                best_score += 0.2
        
        return best_match, best_score
    
    def _determine_section_level(self, line: str, line_num: int, lines: List[str]) -> int:
        """Determine the hierarchical level of a section."""
        # Check for numbering patterns
        if re.match(r'^\\d+\\.\\s+', line):  # 1. Main section
            return 1
        elif re.match(r'^\\d+\\.\\d+\\.?\\s+', line):  # 1.1. Subsection
            return 2
        elif re.match(r'^\\d+\\.\\d+\\.\\d+\\.?\\s+', line):  # 1.1.1. Sub-subsection
            return 3
        
        # Check formatting cues
        if line.isupper():
            return 1  # All caps usually indicates main section
        
        # Check indentation (approximate)
        leading_spaces = len(line) - len(line.lstrip())
        if leading_spaces == 0:
            return 1
        elif leading_spaces <= 4:
            return 2
        else:
            return 3
    
    def _build_hierarchy(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        """Build hierarchical structure from flat section list."""
        if not sections:
            return []
        
        # Sort sections by position
        sections.sort(key=lambda s: s.start_position)
        
        # Build hierarchy
        root_sections = []
        section_stack = []
        
        for section in sections:
            # Pop sections from stack that are at same or higher level
            while section_stack and section_stack[-1].level >= section.level:
                section_stack.pop()
            
            # Add as subsection to parent if exists
            if section_stack:
                parent = section_stack[-1]
                parent.subsections.append(section)
            else:
                root_sections.append(section)
            
            section_stack.append(section)
        
        return root_sections
    
    def _extract_table_references(self, text: str) -> List[Dict]:
        """Extract references to tables in the document."""
        table_pattern = r'table\\s+(\\d+(?:\\.\\d+)?):?\\s*([^\\n]+)'
        tables = []
        
        for match in re.finditer(table_pattern, text, re.IGNORECASE):
            table_number = match.group(1)
            table_title = match.group(2).strip()
            
            tables.append({
                'number': table_number,
                'title': table_title,
                'position': match.start(),
                'reference_text': match.group(0)
            })
        
        return tables
    
    def _extract_figure_references(self, text: str) -> List[Dict]:
        """Extract references to figures in the document."""
        figure_pattern = r'figure\\s+(\\d+(?:\\.\\d+)?):?\\s*([^\\n]+)'
        figures = []
        
        for match in re.finditer(figure_pattern, text, re.IGNORECASE):
            figure_number = match.group(1)
            figure_title = match.group(2).strip()
            
            figures.append({
                'number': figure_number,
                'title': figure_title,
                'position': match.start(),
                'reference_text': match.group(0)
            })
        
        return figures
```

This comprehensive document preprocessing pipeline provides the foundation for extracting and cleaning data from various document types in the IPO valuation platform. The next sections will cover feature engineering and integration patterns.