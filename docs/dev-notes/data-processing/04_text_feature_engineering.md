# Text Feature Engineering for Financial NLP

This document provides comprehensive guidance for extracting and engineering text features from financial documents in the IPO valuation platform, including TF-IDF vectorization, word embeddings, sentiment analysis, and named entity recognition.

## 📝 Overview

Text feature engineering transforms unstructured text from prospectuses, annual reports, and regulatory filings into numerical features that machine learning models can process. This is critical for extracting insights from qualitative information that traditional financial metrics cannot capture.

## 🎯 Key Objectives

- **Information Extraction**: Extract meaningful signals from unstructured text
- **Sentiment Analysis**: Quantify sentiment and tone in financial communications
- **Entity Recognition**: Identify key financial entities and relationships
- **Topic Modeling**: Discover latent themes in financial documents
- **Risk Signal Detection**: Identify textual indicators of financial risk

## 1. TF-IDF Vectorization for Financial Documents

### 1.1 Financial-Specific TF-IDF Implementation

```python
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import spacy
from collections import Counter
import logging

class FinancialTfIdfVectorizer:
    """
    Specialized TF-IDF vectorization for financial documents.
    
    Includes financial term preservation, domain-specific preprocessing,
    and feature selection optimized for valuation models.
    """
    
    def __init__(self, financial_lexicon_path: str = None):
        self.financial_terms = self._load_financial_lexicon(financial_lexicon_path)
        self.vectorizers = {}
        self.feature_names = {}
        self.logger = logging.getLogger(__name__)
        
        # Load NLP models
        self.nlp = spacy.load('en_core_web_sm')
        self.lemmatizer = WordNetLemmatizer()
        
        # Financial stopwords (in addition to standard ones)
        self.financial_stopwords = set([
            'company', 'business', 'financial', 'year', 'period', 'management',
            'board', 'director', 'pursuant', 'accordance', 'respect', 'regard',
            'particular', 'certain', 'various', 'including', 'related', 'based'
        ])
        
    def _load_financial_lexicon(self, lexicon_path: str) -> Dict[str, List[str]]:
        """Load financial domain lexicon."""
        # Default financial terms by category
        default_lexicon = {
            'positive_financial': [
                'profit', 'revenue', 'growth', 'increase', 'strong', 'robust',
                'improvement', 'expansion', 'success', 'opportunity', 'advantage',
                'efficient', 'optimize', 'enhance', 'exceed', 'outperform'
            ],
            'negative_financial': [
                'loss', 'decline', 'decrease', 'weak', 'risk', 'challenge',
                'uncertainty', 'volatility', 'default', 'bankruptcy', 'impairment',
                'restructuring', 'downturn', 'adverse', 'deterioration'
            ],
            'risk_indicators': [
                'contingent', 'litigation', 'regulatory', 'compliance', 'penalty',
                'investigation', 'violation', 'breach', 'default', 'covenant',
                'material', 'adverse', 'significant', 'substantial'
            ],
            'financial_metrics': [
                'ebitda', 'revenue', 'earnings', 'margin', 'ratio', 'return',
                'yield', 'dividend', 'cash', 'debt', 'equity', 'asset',
                'liability', 'valuation', 'multiple'
            ],
            'forward_looking': [
                'expect', 'anticipate', 'forecast', 'project', 'estimate',
                'guidance', 'outlook', 'target', 'goal', 'plan', 'strategy',
                'future', 'forward', 'prospective'
            ]
        }
        
        if lexicon_path:
            try:
                # Load custom lexicon from file
                import json
                with open(lexicon_path, 'r') as f:
                    custom_lexicon = json.load(f)
                    default_lexicon.update(custom_lexicon)
            except Exception as e:
                self.logger.warning(f"Could not load custom lexicon: {e}")
        
        return default_lexicon
    
    def preprocess_financial_text(self, text: str) -> str:
        """
        Preprocess financial text with domain-specific rules.
        
        Args:
            text: Raw financial text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags and special characters
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\\s\\-\\.,%$]', ' ', text)
        
        # Normalize financial amounts
        # Convert currency amounts to normalized form
        text = re.sub(r'\\$([0-9,]+(?:\\.[0-9]{2})?)', r'currency_amount_\\1', text)
        text = re.sub(r'([0-9,]+(?:\\.[0-9]{2})?)\\s*million', r'\\1_million', text)
        text = re.sub(r'([0-9,]+(?:\\.[0-9]{2})?)\\s*billion', r'\\1_billion', text)
        
        # Normalize percentages
        text = re.sub(r'([0-9\\.]+)\\s*%', r'\\1_percent', text)
        
        # Preserve important financial terms (prevent splitting)
        for category, terms in self.financial_terms.items():
            for term in terms:
                if ' ' in term:
                    # Replace multi-word terms with underscore versions
                    pattern = term.replace(' ', '\\\\s+')
                    replacement = term.replace(' ', '_')
                    text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def create_financial_tfidf_features(self, documents: List[str],
                                      feature_config: Dict = None) -> pd.DataFrame:
        """
        Create TF-IDF features optimized for financial analysis.
        
        Args:
            documents: List of financial document texts
            feature_config: Configuration for feature extraction
            
        Returns:
            DataFrame with TF-IDF features
        """
        config = feature_config or self._default_feature_config()
        
        # Preprocess all documents
        preprocessed_docs = [self.preprocess_financial_text(doc) for doc in documents]
        
        # Create different types of TF-IDF features
        features_dict = {}
        
        # 1. Standard TF-IDF with financial preprocessing
        standard_features = self._create_standard_tfidf(preprocessed_docs, config)
        features_dict.update(standard_features)
        
        # 2. N-gram features for financial phrases
        ngram_features = self._create_ngram_tfidf(preprocessed_docs, config)
        features_dict.update(ngram_features)
        
        # 3. Financial domain-specific features
        domain_features = self._create_domain_specific_features(documents, config)
        features_dict.update(domain_features)
        
        # 4. Sentiment-based TF-IDF features
        sentiment_features = self._create_sentiment_tfidf(documents, config)
        features_dict.update(sentiment_features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_dict)
        
        # Feature selection and dimensionality reduction
        if config.get('apply_feature_selection', True):
            features_df = self._apply_feature_selection(features_df, config)
        
        return features_df
    
    def _default_feature_config(self) -> Dict:
        """Default configuration for TF-IDF feature extraction."""
        return {
            'max_features': 5000,
            'min_df': 2,
            'max_df': 0.95,
            'ngram_range': (1, 3),
            'use_idf': True,
            'smooth_idf': True,
            'sublinear_tf': True,
            'apply_feature_selection': True,
            'feature_selection_k': 1000,
            'svd_components': 100
        }
    
    def _create_standard_tfidf(self, documents: List[str], config: Dict) -> Dict:
        """Create standard TF-IDF features."""
        # Combine standard and financial stopwords
        all_stopwords = set(stopwords.words('english')) | self.financial_stopwords
        
        vectorizer = TfidfVectorizer(
            max_features=config['max_features'],
            min_df=config['min_df'],
            max_df=config['max_df'],
            ngram_range=(1, 2),  # Unigrams and bigrams
            stop_words=list(all_stopwords),
            use_idf=config['use_idf'],
            smooth_idf=config['smooth_idf'],
            sublinear_tf=config['sublinear_tf']
        )
        
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        
        # Store vectorizer for later use
        self.vectorizers['standard'] = vectorizer
        self.feature_names['standard'] = feature_names
        
        # Convert to dictionary
        features = {}
        for i, feature_name in enumerate(feature_names):
            features[f'tfidf_{feature_name}'] = tfidf_matrix[:, i].toarray().flatten()
        
        return features
    
    def _create_ngram_tfidf(self, documents: List[str], config: Dict) -> Dict:
        """Create N-gram TF-IDF features for financial phrases."""
        # Focus on longer phrases that capture financial concepts
        vectorizer = TfidfVectorizer(
            max_features=config['max_features'] // 2,
            min_df=config['min_df'],
            max_df=config['max_df'],
            ngram_range=(2, 4),  # Bigrams to 4-grams
            stop_words=None,  # Don't remove stopwords for phrases
            use_idf=config['use_idf'],
            smooth_idf=config['smooth_idf'],
            sublinear_tf=config['sublinear_tf']
        )
        
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        
        self.vectorizers['ngram'] = vectorizer
        self.feature_names['ngram'] = feature_names
        
        features = {}
        for i, feature_name in enumerate(feature_names):
            features[f'ngram_{feature_name}'] = tfidf_matrix[:, i].toarray().flatten()
        
        return features
    
    def _create_domain_specific_features(self, documents: List[str], config: Dict) -> Dict:
        """Create features based on financial domain knowledge."""
        features = {}
        
        for doc_idx, doc in enumerate(documents):
            doc_lower = doc.lower()
            
            # Count features for each financial category
            for category, terms in self.financial_terms.items():
                count = 0
                for term in terms:
                    count += len(re.findall(rf'\\b{re.escape(term)}\\b', doc_lower))
                
                if f'{category}_count' not in features:
                    features[f'{category}_count'] = [0] * len(documents)
                features[f'{category}_count'][doc_idx] = count
        
        # Financial metrics density
        financial_metrics_density = []
        for doc in documents:
            doc_lower = doc.lower()
            total_words = len(doc_lower.split())
            financial_words = 0
            
            for terms in self.financial_terms.values():
                for term in terms:
                    financial_words += len(re.findall(rf'\\b{re.escape(term)}\\b', doc_lower))
            
            density = financial_words / total_words if total_words > 0 else 0
            financial_metrics_density.append(density)
        
        features['financial_density'] = financial_metrics_density
        
        return features
    
    def _create_sentiment_tfidf(self, documents: List[str], config: Dict) -> Dict:
        """Create TF-IDF features weighted by sentiment."""
        from textblob import TextBlob
        
        # Calculate sentiment for each document
        sentiments = []
        for doc in documents:
            blob = TextBlob(doc)
            sentiment = blob.sentiment.polarity  # -1 to 1
            sentiments.append(sentiment)
        
        # Create sentiment-weighted documents
        sentiment_weighted_docs = []
        for doc, sentiment in zip(documents, sentiments):
            # Weight positive/negative words based on overall sentiment
            words = doc.split()
            weighted_words = []
            
            for word in words:
                word_lower = word.lower()
                weight = 1.0
                
                # Boost positive financial terms in positive documents
                if sentiment > 0 and word_lower in self.financial_terms['positive_financial']:
                    weight = 1.5
                # Boost negative financial terms in negative documents
                elif sentiment < 0 and word_lower in self.financial_terms['negative_financial']:
                    weight = 1.5
                # Boost risk terms based on magnitude
                elif word_lower in self.financial_terms['risk_indicators']:
                    weight = 1.2
                
                # Repeat word based on weight
                weighted_words.extend([word] * int(weight))
            
            sentiment_weighted_docs.append(' '.join(weighted_words))
        
        # Apply TF-IDF to sentiment-weighted documents
        vectorizer = TfidfVectorizer(
            max_features=config['max_features'] // 3,
            min_df=config['min_df'],
            max_df=config['max_df'],
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        tfidf_matrix = vectorizer.fit_transform(sentiment_weighted_docs)
        feature_names = vectorizer.get_feature_names_out()
        
        self.vectorizers['sentiment'] = vectorizer
        self.feature_names['sentiment'] = feature_names
        
        features = {}
        for i, feature_name in enumerate(feature_names):
            features[f'sentiment_tfidf_{feature_name}'] = tfidf_matrix[:, i].toarray().flatten()
        
        # Add sentiment scores as features
        features['document_sentiment'] = sentiments
        
        return features
    
    def _apply_feature_selection(self, features_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Apply feature selection to reduce dimensionality."""
        from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
        from sklearn.decomposition import TruncatedSVD
        
        # Separate TF-IDF features from other features
        tfidf_cols = [col for col in features_df.columns if 'tfidf' in col.lower()]
        other_cols = [col for col in features_df.columns if col not in tfidf_cols]
        
        if not tfidf_cols:
            return features_df
        
        # Apply SVD to TF-IDF features for dimensionality reduction
        svd = TruncatedSVD(n_components=min(config['svd_components'], len(tfidf_cols)))
        tfidf_reduced = svd.fit_transform(features_df[tfidf_cols])
        
        # Create DataFrame with reduced TF-IDF features
        svd_feature_names = [f'tfidf_svd_{i}' for i in range(tfidf_reduced.shape[1])]
        svd_df = pd.DataFrame(tfidf_reduced, columns=svd_feature_names, index=features_df.index)
        
        # Combine with other features
        if other_cols:
            final_df = pd.concat([svd_df, features_df[other_cols]], axis=1)
        else:
            final_df = svd_df
        
        return final_df
```

### 1.2 Financial Phrase Detection and N-gram Analysis

```python
from collections import Counter, defaultdict
import re
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

class FinancialPhraseAnalyzer:
    """
    Analyze and extract meaningful financial phrases and n-grams.
    
    Identifies important financial phrases, risk indicators, and forward-looking statements.
    """
    
    def __init__(self):
        # Financial phrase patterns
        self.phrase_patterns = {
            'risk_phrases': [
                r'material\\s+adverse\\s+effect',
                r'significant\\s+risk',
                r'may\\s+adversely\\s+affect',
                r'subject\\s+to\\s+risks?',
                r'uncertainty\\s+regarding',
                r'potential\\s+for\\s+loss',
                r'regulatory\\s+action',
                r'litigation\\s+risk',
                r'market\\s+volatility',
                r'economic\\s+downturn'
            ],
            'growth_phrases': [
                r'growth\\s+strategy',
                r'expansion\\s+plans?',
                r'market\\s+opportunity',
                r'revenue\\s+growth',
                r'business\\s+growth',
                r'increase\\s+market\\s+share',
                r'strategic\\s+initiative',
                r'competitive\\s+advantage',
                r'strong\\s+performance',
                r'positive\\s+outlook'
            ],
            'financial_performance': [
                r'strong\\s+financial\\s+position',
                r'improved\\s+profitability',
                r'increased\\s+revenue',
                r'positive\\s+cash\\s+flow',
                r'return\\s+on\\s+investment',
                r'earnings\\s+per\\s+share',
                r'dividend\\s+yield',
                r'debt\\s+to\\s+equity',
                r'working\\s+capital',
                r'operating\\s+margin'
            ],
            'forward_looking': [
                r'expect\\s+to',
                r'anticipate\\s+that',
                r'believe\\s+that',
                r'estimate\\s+that',
                r'forecast\\s+for',
                r'guidance\\s+for',
                r'outlook\\s+for',
                r'target\\s+of',
                r'plan\\s+to',
                r'intend\\s+to'
            ]
        }
        
        self.phrase_counts = defaultdict(Counter)
        
    def extract_financial_phrases(self, documents: List[str], 
                                 min_frequency: int = 2) -> Dict[str, Dict[str, int]]:
        """
        Extract financial phrases from documents.
        
        Args:
            documents: List of financial document texts
            min_frequency: Minimum frequency for phrase inclusion
            
        Returns:
            Dictionary of phrase categories and their frequencies
        """
        phrase_results = {}
        
        for category, patterns in self.phrase_patterns.items():
            category_matches = Counter()
            
            for doc in documents:
                doc_lower = doc.lower()
                
                for pattern in patterns:
                    matches = re.findall(pattern, doc_lower)
                    for match in matches:
                        category_matches[match] += 1
            
            # Filter by minimum frequency
            filtered_matches = {
                phrase: count for phrase, count in category_matches.items()
                if count >= min_frequency
            }
            
            phrase_results[category] = filtered_matches
            
        return phrase_results
    
    def analyze_ngram_patterns(self, documents: List[str],
                             ngram_range: Tuple[int, int] = (2, 4),
                             min_df: int = 2,
                             max_features: int = 1000) -> pd.DataFrame:
        """
        Analyze n-gram patterns in financial documents.
        
        Args:
            documents: List of financial document texts
            ngram_range: Range of n-gram sizes
            min_df: Minimum document frequency
            max_features: Maximum number of features
            
        Returns:
            DataFrame with n-gram analysis results
        """
        # Preprocess documents
        processed_docs = []
        for doc in documents:
            # Simple preprocessing
            doc_clean = re.sub(r'[^a-zA-Z\\s]', ' ', doc.lower())
            doc_clean = ' '.join(doc_clean.split())
            processed_docs.append(doc_clean)
        
        # Extract n-grams
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            stop_words='english'
        )
        
        ngram_matrix = vectorizer.fit_transform(processed_docs)
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate statistics
        ngram_counts = np.array(ngram_matrix.sum(axis=0)).flatten()
        doc_frequencies = np.array((ngram_matrix > 0).sum(axis=0)).flatten()
        
        # Create results DataFrame
        results = pd.DataFrame({
            'ngram': feature_names,
            'total_count': ngram_counts,
            'document_frequency': doc_frequencies,
            'avg_per_document': ngram_counts / len(documents)
        })
        
        # Add n-gram length
        results['ngram_length'] = results['ngram'].apply(lambda x: len(x.split()))
        
        # Sort by frequency
        results = results.sort_values('total_count', ascending=False)
        
        return results
    
    def identify_key_financial_concepts(self, documents: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Identify key financial concepts using co-occurrence analysis.
        
        Args:
            documents: List of financial document texts
            
        Returns:
            Dictionary of concept categories with weighted terms
        """
        # Define seed terms for each concept
        concept_seeds = {
            'profitability': ['profit', 'earnings', 'margin', 'return'],
            'growth': ['growth', 'expansion', 'increase', 'development'],
            'risk': ['risk', 'uncertainty', 'volatility', 'adverse'],
            'liquidity': ['cash', 'liquid', 'working capital', 'current ratio'],
            'leverage': ['debt', 'leverage', 'gearing', 'borrowing'],
            'market': ['market', 'competition', 'industry', 'sector']
        }
        
        concept_results = {}
        
        for concept, seeds in concept_seeds.items():
            concept_terms = Counter()
            
            for doc in documents:
                doc_lower = doc.lower()
                words = doc_lower.split()
                
                # Find sentences containing seed terms
                sentences = re.split(r'[.!?]+', doc_lower)
                
                for sentence in sentences:
                    if any(seed in sentence for seed in seeds):
                        # Extract all words from sentences containing seed terms
                        sentence_words = sentence.split()
                        for word in sentence_words:
                            if (len(word) > 3 and 
                                word.isalpha() and 
                                word not in stopwords.words('english')):
                                concept_terms[word] += 1
            
            # Calculate TF-IDF-like scores
            total_docs = len(documents)
            concept_scores = []
            
            for term, freq in concept_terms.most_common(50):  # Top 50 terms
                # Simple scoring based on frequency and relevance
                score = freq / total_docs
                concept_scores.append((term, score))
            
            concept_results[concept] = concept_scores[:20]  # Top 20 per concept
        
        return concept_results
    
    def create_phrase_features(self, documents: List[str]) -> pd.DataFrame:
        """
        Create features based on financial phrase analysis.
        
        Args:
            documents: List of financial document texts
            
        Returns:
            DataFrame with phrase-based features
        """
        # Extract phrases
        phrase_data = self.extract_financial_phrases(documents)
        
        # Create feature matrix
        features = {}
        
        for doc_idx, doc in enumerate(documents):
            doc_lower = doc.lower()
            
            # Count phrases by category
            for category, phrases in phrase_data.items():
                category_count = 0
                for phrase in phrases.keys():
                    category_count += len(re.findall(rf'\\b{re.escape(phrase)}\\b', doc_lower))
                
                if f'{category}_phrase_count' not in features:
                    features[f'{category}_phrase_count'] = [0] * len(documents)
                features[f'{category}_phrase_count'][doc_idx] = category_count
        
        # Additional phrase-based features
        for doc_idx, doc in enumerate(documents):
            doc_words = doc.lower().split()
            total_words = len(doc_words)
            
            # Phrase density features
            for category in phrase_data.keys():
                category_phrases = sum(features[f'{category}_phrase_count'])
                density = category_phrases / total_words if total_words > 0 else 0
                
                if f'{category}_phrase_density' not in features:
                    features[f'{category}_phrase_density'] = [0] * len(documents)
                features[f'{category}_phrase_density'][doc_idx] = density
        
        return pd.DataFrame(features)
```

## 2. Word Embeddings for Financial Terms

### 2.1 Financial Word Embeddings Implementation

```python
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import pickle

class FinancialWordEmbeddings:
    """
    Create and manage word embeddings specialized for financial text.
    
    Includes pre-trained financial embeddings, custom training, and
    domain-specific similarity computations.
    """
    
    def __init__(self, model_type: str = 'word2vec'):
        self.model_type = model_type
        self.models = {}
        self.vocabularies = {}
        
        # Financial domain terms for similarity validation
        self.financial_analogies = [
            ('revenue', 'sales', 'profit', 'earnings'),
            ('debt', 'liability', 'equity', 'asset'),
            ('risk', 'uncertainty', 'opportunity', 'potential'),
            ('growth', 'expansion', 'decline', 'contraction')
        ]
        
    def train_financial_embeddings(self, documents: List[str],
                                 model_config: Dict = None) -> Dict:
        """
        Train word embeddings on financial corpus.
        
        Args:
            documents: List of financial documents
            model_config: Configuration for embedding training
            
        Returns:
            Dictionary with training results and model info
        """
        config = model_config or self._default_embedding_config()
        
        # Preprocess documents for training
        processed_docs = self._preprocess_for_embeddings(documents)
        
        if self.model_type == 'word2vec':
            model = self._train_word2vec(processed_docs, config)
        elif self.model_type == 'fasttext':
            model = self._train_fasttext(processed_docs, config)
        elif self.model_type == 'doc2vec':
            model = self._train_doc2vec(processed_docs, config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.models[self.model_type] = model
        
        # Evaluate embeddings
        evaluation_results = self._evaluate_embeddings(model)
        
        return {
            'model': model,
            'vocabulary_size': len(model.wv.key_to_index) if hasattr(model, 'wv') else len(model.dv),
            'evaluation': evaluation_results
        }
    
    def _default_embedding_config(self) -> Dict:
        """Default configuration for embedding training."""
        return {
            'vector_size': 300,
            'window': 10,
            'min_count': 5,
            'workers': 4,
            'sg': 1,  # Skip-gram
            'epochs': 20,
            'alpha': 0.025,
            'min_alpha': 0.00025
        }
    
    def _preprocess_for_embeddings(self, documents: List[str]) -> List[List[str]]:
        """Preprocess documents for embedding training."""
        processed_docs = []
        
        for doc in documents:
            # Clean and tokenize
            doc_clean = re.sub(r'[^a-zA-Z\\s]', ' ', doc.lower())
            words = doc_clean.split()
            
            # Filter words
            filtered_words = [
                word for word in words 
                if len(word) > 2 and word not in stopwords.words('english')
            ]
            
            processed_docs.append(filtered_words)
        
        return processed_docs
    
    def _train_word2vec(self, processed_docs: List[List[str]], config: Dict) -> Word2Vec:
        """Train Word2Vec model."""
        model = Word2Vec(
            sentences=processed_docs,
            vector_size=config['vector_size'],
            window=config['window'],
            min_count=config['min_count'],
            workers=config['workers'],
            sg=config['sg'],
            epochs=config['epochs'],
            alpha=config['alpha'],
            min_alpha=config['min_alpha']
        )
        
        return model
    
    def _train_fasttext(self, processed_docs: List[List[str]], config: Dict) -> FastText:
        """Train FastText model."""
        model = FastText(
            sentences=processed_docs,
            vector_size=config['vector_size'],
            window=config['window'],
            min_count=config['min_count'],
            workers=config['workers'],
            sg=config['sg'],
            epochs=config['epochs'],
            alpha=config['alpha'],
            min_alpha=config['min_alpha']
        )
        
        return model
    
    def _train_doc2vec(self, processed_docs: List[List[str]], config: Dict) -> Doc2Vec:
        """Train Doc2Vec model."""
        # Create tagged documents
        tagged_docs = [
            TaggedDocument(words=doc, tags=[str(i)]) 
            for i, doc in enumerate(processed_docs)
        ]
        
        model = Doc2Vec(
            documents=tagged_docs,
            vector_size=config['vector_size'],
            window=config['window'],
            min_count=config['min_count'],
            workers=config['workers'],
            epochs=config['epochs'],
            alpha=config['alpha'],
            min_alpha=config['min_alpha']
        )
        
        return model
    
    def create_document_embeddings(self, documents: List[str],
                                 aggregation_method: str = 'mean') -> np.ndarray:
        """
        Create document-level embeddings from word embeddings.
        
        Args:
            documents: List of documents
            aggregation_method: Method to aggregate word vectors ('mean', 'sum', 'max')
            
        Returns:
            Array of document embeddings
        """
        if self.model_type not in self.models:
            raise ValueError("No trained model available. Train embeddings first.")
        
        model = self.models[self.model_type]
        doc_embeddings = []
        
        for doc in documents:
            doc_clean = re.sub(r'[^a-zA-Z\\s]', ' ', doc.lower())
            words = [word for word in doc_clean.split() if len(word) > 2]
            
            # Get word vectors
            word_vectors = []
            for word in words:
                try:
                    if hasattr(model, 'wv'):  # Word2Vec, FastText
                        vector = model.wv[word]
                        word_vectors.append(vector)
                except KeyError:
                    continue  # Word not in vocabulary
            
            if word_vectors:
                word_vectors = np.array(word_vectors)
                
                if aggregation_method == 'mean':
                    doc_vector = np.mean(word_vectors, axis=0)
                elif aggregation_method == 'sum':
                    doc_vector = np.sum(word_vectors, axis=0)
                elif aggregation_method == 'max':
                    doc_vector = np.max(word_vectors, axis=0)
                else:
                    doc_vector = np.mean(word_vectors, axis=0)  # Default to mean
                
                doc_embeddings.append(doc_vector)
            else:
                # Create zero vector if no words found
                vector_size = model.wv.vector_size if hasattr(model, 'wv') else model.vector_size
                doc_embeddings.append(np.zeros(vector_size))
        
        return np.array(doc_embeddings)
    
    def get_financial_word_similarities(self, target_word: str, 
                                      n_similar: int = 10) -> List[Tuple[str, float]]:
        """
        Get most similar words to a target financial term.
        
        Args:
            target_word: Target financial term
            n_similar: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        if self.model_type not in self.models:
            raise ValueError("No trained model available.")
        
        model = self.models[self.model_type]
        
        try:
            if hasattr(model, 'wv'):
                similar_words = model.wv.most_similar(target_word, topn=n_similar)
            else:
                similar_words = model.most_similar(target_word, topn=n_similar)
            
            return similar_words
            
        except KeyError:
            return []  # Word not in vocabulary
    
    def _evaluate_embeddings(self, model) -> Dict:
        """Evaluate quality of trained embeddings."""
        evaluation_results = {}
        
        # Test financial analogies
        analogy_scores = []
        for analogy in self.financial_analogies:
            try:
                word1, word2, word3, expected = analogy
                if hasattr(model, 'wv'):
                    predicted = model.wv.most_similar(
                        positive=[word2, word3],
                        negative=[word1],
                        topn=1
                    )[0][0]
                    
                    # Simple scoring: 1 if exact match, 0.5 if similar, 0 otherwise
                    if predicted == expected:
                        score = 1.0
                    elif predicted in model.wv.most_similar(expected, topn=5):
                        score = 0.5
                    else:
                        score = 0.0
                    
                    analogy_scores.append(score)
                    
            except (KeyError, IndexError):
                analogy_scores.append(0.0)  # Words not in vocabulary
        
        evaluation_results['analogy_accuracy'] = np.mean(analogy_scores)
        
        # Vocabulary coverage
        total_vocab = len(model.wv.key_to_index) if hasattr(model, 'wv') else len(model.dv)
        evaluation_results['vocabulary_size'] = total_vocab
        
        return evaluation_results
```

This comprehensive text feature engineering documentation provides the foundation for extracting meaningful features from financial documents. The next sections will cover data quality validation and integration patterns to complete the data processing pipeline.