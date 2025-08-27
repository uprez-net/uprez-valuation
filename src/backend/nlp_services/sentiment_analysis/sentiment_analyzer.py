"""
Sentiment Analysis Service
Advanced sentiment analysis for financial documents with aspect-based analysis
"""
import re
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import numpy as np

import spacy
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertForSequenceClassification
)
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ...config import settings
from ...utils.metrics import track_time, track_ml_inference, ML_INFERENCE_DURATION
from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    overall_sentiment: str  # positive, negative, neutral
    overall_score: float    # -1 to 1
    confidence: float       # 0 to 1
    
    # Detailed scores
    positive_score: float
    negative_score: float
    neutral_score: float
    
    # Aspect-based sentiment
    aspect_sentiments: Dict[str, Dict[str, float]]
    
    # Risk-specific sentiment
    risk_sentiment_score: float
    opportunity_sentiment_score: float
    
    # Model information
    model_name: str
    processing_time: float
    
    # Detailed analysis
    sentence_sentiments: List[Dict[str, Any]]
    key_phrases: List[Dict[str, Any]]


@dataclass
class AspectDefinition:
    """Definition of aspects for aspect-based sentiment analysis"""
    name: str
    keywords: List[str]
    patterns: List[str]
    weight: float = 1.0


class FinancialSentimentAnalyzer:
    """Advanced sentiment analyzer for financial documents"""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.nlp = None
        self.finbert_tokenizer = None
        self.finbert_model = None
        self.sentiment_pipeline = None
        
        # Financial aspects for analysis
        self.financial_aspects = self._define_financial_aspects()
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load NLP models"""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded")
            
            # Load FinBERT for financial sentiment
            try:
                self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                logger.info("FinBERT model loaded")
            except Exception as e:
                logger.warning(f"Failed to load FinBERT: {str(e)}")
            
            # Load general sentiment pipeline as fallback
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                logger.info("RoBERTa sentiment pipeline loaded")
            except Exception as e:
                logger.warning(f"Failed to load RoBERTa pipeline: {str(e)}")
                
        except Exception as e:
            logger.error(f"Failed to load NLP models: {str(e)}")
    
    def _define_financial_aspects(self) -> List[AspectDefinition]:
        """Define financial aspects for analysis"""
        return [
            AspectDefinition(
                name="revenue_growth",
                keywords=["revenue", "sales", "income", "earnings", "growth"],
                patterns=[r"revenue.*(?:growth|increase|decrease|decline)", r"sales.*(?:up|down|growth)"],
                weight=1.2
            ),
            AspectDefinition(
                name="profitability",
                keywords=["profit", "margin", "ebitda", "operating income", "net income"],
                patterns=[r"profit.*(?:margin|increase|decrease)", r"ebitda.*(?:growth|decline)"],
                weight=1.3
            ),
            AspectDefinition(
                name="market_position",
                keywords=["market share", "competition", "competitive", "leadership", "market position"],
                patterns=[r"market.*(?:share|position|leader)", r"competitive.*(?:advantage|position)"],
                weight=1.1
            ),
            AspectDefinition(
                name="financial_health",
                keywords=["debt", "cash", "liquidity", "solvency", "financial position"],
                patterns=[r"debt.*(?:ratio|level|burden)", r"cash.*(?:flow|position)"],
                weight=1.4
            ),
            AspectDefinition(
                name="management_quality",
                keywords=["management", "leadership", "strategy", "execution", "governance"],
                patterns=[r"management.*(?:team|quality|strategy)", r"leadership.*(?:strong|weak)"],
                weight=1.0
            ),
            AspectDefinition(
                name="risk_factors",
                keywords=["risk", "uncertainty", "volatility", "threat", "challenge"],
                patterns=[r"risk.*(?:factor|assessment|management)", r"uncertainty.*(?:market|economic)"],
                weight=1.5
            ),
            AspectDefinition(
                name="growth_prospects",
                keywords=["growth", "expansion", "opportunity", "potential", "outlook"],
                patterns=[r"growth.*(?:prospects|potential|outlook)", r"expansion.*(?:plans|opportunity)"],
                weight=1.3
            ),
            AspectDefinition(
                name="valuation",
                keywords=["valuation", "price", "value", "expensive", "cheap", "fair value"],
                patterns=[r"(?:over|under)valued", r"fair.*value", r"price.*(?:target|objective)"],
                weight=1.4
            )
        ]
    
    @track_time(ML_INFERENCE_DURATION, {"model_name": "sentiment"})
    async def analyze_sentiment(
        self,
        text: str,
        use_finbert: bool = True,
        include_aspects: bool = True,
        include_sentences: bool = True
    ) -> SentimentResult:
        """
        Comprehensive sentiment analysis
        
        Args:
            text: Text to analyze
            use_finbert: Whether to use FinBERT model
            include_aspects: Whether to perform aspect-based analysis
            include_sentences: Whether to analyze sentence-level sentiment
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            track_ml_inference("sentiment", start_time, True)
            
            # Clean and preprocess text
            clean_text = self._preprocess_text(text)
            
            # Get overall sentiment using best available model
            if use_finbert and self.finbert_model:
                overall_sentiment = await self._analyze_with_finbert(clean_text)
                model_name = "finbert"
            elif self.sentiment_pipeline:
                overall_sentiment = await self._analyze_with_roberta(clean_text)
                model_name = "roberta"
            else:
                overall_sentiment = self._analyze_with_vader(clean_text)
                model_name = "vader"
            
            # Aspect-based sentiment analysis
            aspect_sentiments = {}
            if include_aspects:
                aspect_sentiments = await self._analyze_aspects(clean_text)
            
            # Sentence-level analysis
            sentence_sentiments = []
            if include_sentences:
                sentence_sentiments = await self._analyze_sentences(clean_text)
            
            # Risk and opportunity analysis
            risk_score = self._analyze_risk_sentiment(clean_text)
            opportunity_score = self._analyze_opportunity_sentiment(clean_text)
            
            # Extract key phrases
            key_phrases = self._extract_key_phrases(clean_text)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            result = SentimentResult(
                overall_sentiment=overall_sentiment["label"],
                overall_score=overall_sentiment["score"],
                confidence=overall_sentiment["confidence"],
                positive_score=overall_sentiment.get("positive", 0),
                negative_score=overall_sentiment.get("negative", 0),
                neutral_score=overall_sentiment.get("neutral", 0),
                aspect_sentiments=aspect_sentiments,
                risk_sentiment_score=risk_score,
                opportunity_sentiment_score=opportunity_score,
                model_name=model_name,
                processing_time=processing_time,
                sentence_sentiments=sentence_sentiments,
                key_phrases=key_phrases
            )
            
            logger.info(
                f"Sentiment analysis completed",
                model=model_name,
                overall_sentiment=result.overall_sentiment,
                score=result.overall_score,
                confidence=result.confidence
            )
            
            return result
            
        except Exception as e:
            track_ml_inference("sentiment", start_time, False)
            logger.error(f"Sentiment analysis failed: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,!?;:\-()$%]', '', text)
        
        # Normalize financial terms
        text = self._normalize_financial_terms(text)
        
        return text.strip()
    
    def _normalize_financial_terms(self, text: str) -> str:
        """Normalize financial terms for better analysis"""
        # Common financial abbreviations
        replacements = {
            r'\$(\d+)([kmb])\b': lambda m: f"${m.group(1)} {'thousand' if m.group(2).lower() == 'k' else 'million' if m.group(2).lower() == 'm' else 'billion'}",
            r'\b(\d+)%': r'\1 percent',
            r'\bQ(\d)\b': r'quarter \1',
            r'\bYoY\b': 'year over year',
            r'\bQoQ\b': 'quarter over quarter',
        }
        
        for pattern, replacement in replacements.items():
            if callable(replacement):
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            else:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    async def _analyze_with_finbert(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using FinBERT"""
        try:
            # Tokenize
            inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT labels: negative, neutral, positive
            labels = ["negative", "neutral", "positive"]
            scores = predictions[0].tolist()
            
            # Find dominant sentiment
            max_idx = np.argmax(scores)
            dominant_sentiment = labels[max_idx]
            confidence = scores[max_idx]
            
            # Convert to standardized format
            sentiment_score = scores[2] - scores[0]  # positive - negative
            
            return {
                "label": dominant_sentiment,
                "score": sentiment_score,
                "confidence": confidence,
                "positive": scores[2],
                "neutral": scores[1],
                "negative": scores[0]
            }
            
        except Exception as e:
            logger.warning(f"FinBERT analysis failed: {str(e)}")
            return self._analyze_with_vader(text)
    
    async def _analyze_with_roberta(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using RoBERTa"""
        try:
            results = self.sentiment_pipeline(text)[0]
            
            # Convert results to standardized format
            sentiment_map = {}
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                if 'positive' in label:
                    sentiment_map['positive'] = score
                elif 'negative' in label:
                    sentiment_map['negative'] = score
                else:
                    sentiment_map['neutral'] = score
            
            # Determine dominant sentiment
            dominant = max(sentiment_map.items(), key=lambda x: x[1])
            
            # Calculate sentiment score (-1 to 1)
            pos_score = sentiment_map.get('positive', 0)
            neg_score = sentiment_map.get('negative', 0)
            sentiment_score = pos_score - neg_score
            
            return {
                "label": dominant[0],
                "score": sentiment_score,
                "confidence": dominant[1],
                "positive": sentiment_map.get('positive', 0),
                "neutral": sentiment_map.get('neutral', 0),
                "negative": sentiment_map.get('negative', 0)
            }
            
        except Exception as e:
            logger.warning(f"RoBERTa analysis failed: {str(e)}")
            return self._analyze_with_vader(text)
    
    def _analyze_with_vader(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using VADER (fallback)"""
        scores = self.vader_analyzer.polarity_scores(text)
        
        # Determine dominant sentiment
        if scores['compound'] >= 0.05:
            dominant = 'positive'
            confidence = scores['pos']
        elif scores['compound'] <= -0.05:
            dominant = 'negative'
            confidence = scores['neg']
        else:
            dominant = 'neutral'
            confidence = scores['neu']
        
        return {
            "label": dominant,
            "score": scores['compound'],
            "confidence": confidence,
            "positive": scores['pos'],
            "neutral": scores['neu'],
            "negative": scores['neg']
        }
    
    async def _analyze_aspects(self, text: str) -> Dict[str, Dict[str, float]]:
        """Perform aspect-based sentiment analysis"""
        aspect_sentiments = {}
        
        # Split text into sentences for better aspect identification
        if self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]
        else:
            sentences = text.split('.')
        
        for aspect in self.financial_aspects:
            aspect_sentences = []
            
            # Find sentences related to this aspect
            for sentence in sentences:
                sentence_lower = sentence.lower()
                
                # Check keywords
                keyword_match = any(keyword in sentence_lower for keyword in aspect.keywords)
                
                # Check patterns
                pattern_match = any(re.search(pattern, sentence_lower) for pattern in aspect.patterns)
                
                if keyword_match or pattern_match:
                    aspect_sentences.append(sentence)
            
            if aspect_sentences:
                # Analyze sentiment for aspect-related sentences
                aspect_text = ' '.join(aspect_sentences)
                
                if self.finbert_model:
                    aspect_sentiment = await self._analyze_with_finbert(aspect_text)
                else:
                    aspect_sentiment = self._analyze_with_vader(aspect_text)
                
                aspect_sentiments[aspect.name] = {
                    "sentiment": aspect_sentiment["label"],
                    "score": aspect_sentiment["score"] * aspect.weight,
                    "confidence": aspect_sentiment["confidence"],
                    "sentence_count": len(aspect_sentences)
                }
        
        return aspect_sentiments
    
    async def _analyze_sentences(self, text: str) -> List[Dict[str, Any]]:
        """Analyze sentiment for individual sentences"""
        sentence_sentiments = []
        
        if self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        else:
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
        
        for i, sentence in enumerate(sentences):
            if len(sentence) > 5:  # Skip very short sentences
                sentiment = self._analyze_with_vader(sentence)
                
                sentence_sentiments.append({
                    "sentence_index": i,
                    "text": sentence,
                    "sentiment": sentiment["label"],
                    "score": sentiment["score"],
                    "confidence": sentiment["confidence"]
                })
        
        return sentence_sentiments
    
    def _analyze_risk_sentiment(self, text: str) -> float:
        """Analyze risk-related sentiment"""
        risk_keywords = [
            "risk", "risks", "risky", "dangerous", "threat", "threats", "vulnerable",
            "uncertainty", "uncertain", "volatile", "volatility", "unpredictable",
            "challenge", "challenges", "difficult", "difficulties", "problem", "problems",
            "concern", "concerns", "worry", "worries", "caution", "cautious"
        ]
        
        risk_score = 0.0
        word_count = 0
        
        words = text.lower().split()
        for word in words:
            if word in risk_keywords:
                # Use VADER to get sentiment of surrounding context
                word_idx = words.index(word)
                start_idx = max(0, word_idx - 5)
                end_idx = min(len(words), word_idx + 6)
                context = ' '.join(words[start_idx:end_idx])
                
                context_sentiment = self.vader_analyzer.polarity_scores(context)
                # Higher negative sentiment around risk words = higher risk score
                risk_score += max(0, -context_sentiment['compound']) * 0.5
                word_count += 1
        
        # Normalize risk score
        if word_count > 0:
            risk_score = min(1.0, risk_score / word_count + (word_count / len(words)) * 0.5)
        
        return risk_score
    
    def _analyze_opportunity_sentiment(self, text: str) -> float:
        """Analyze opportunity-related sentiment"""
        opportunity_keywords = [
            "opportunity", "opportunities", "growth", "expansion", "potential",
            "promising", "positive", "strong", "robust", "healthy", "improving",
            "optimistic", "confident", "bullish", "upside", "benefit", "benefits",
            "advantage", "advantages", "profit", "profits", "success", "successful"
        ]
        
        opportunity_score = 0.0
        word_count = 0
        
        words = text.lower().split()
        for word in words:
            if word in opportunity_keywords:
                # Use VADER to get sentiment of surrounding context
                word_idx = words.index(word)
                start_idx = max(0, word_idx - 5)
                end_idx = min(len(words), word_idx + 6)
                context = ' '.join(words[start_idx:end_idx])
                
                context_sentiment = self.vader_analyzer.polarity_scores(context)
                # Higher positive sentiment around opportunity words = higher opportunity score
                opportunity_score += max(0, context_sentiment['compound']) * 0.5
                word_count += 1
        
        # Normalize opportunity score
        if word_count > 0:
            opportunity_score = min(1.0, opportunity_score / word_count + (word_count / len(words)) * 0.5)
        
        return opportunity_score
    
    def _extract_key_phrases(self, text: str) -> List[Dict[str, Any]]:
        """Extract key phrases with sentiment"""
        key_phrases = []
        
        if not self.nlp:
            return key_phrases
        
        doc = self.nlp(text)
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 5 and len(chunk.text.split()) >= 2:
                phrase_sentiment = self._analyze_with_vader(chunk.text)
                
                key_phrases.append({
                    "phrase": chunk.text,
                    "sentiment": phrase_sentiment["label"],
                    "score": phrase_sentiment["score"],
                    "start": chunk.start_char,
                    "end": chunk.end_char
                })
        
        # Sort by absolute sentiment score
        key_phrases.sort(key=lambda x: abs(x["score"]), reverse=True)
        
        return key_phrases[:20]  # Return top 20 key phrases


def create_sentiment_analyzer() -> FinancialSentimentAnalyzer:
    """Create configured sentiment analyzer"""
    return FinancialSentimentAnalyzer()