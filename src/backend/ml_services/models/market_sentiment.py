"""
Market Sentiment Analysis for IPO Valuation
Advanced sentiment analysis using FinBERT, social media monitoring, and market timing
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
import json

# ML and NLP libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Time series and forecasting
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats as stats

# Data sources (in practice would use actual APIs)
# import yfinance as yf
# import tweepy
# import praw  # Reddit API
# import feedparser  # RSS feeds

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Individual news article data"""
    title: str
    content: str
    source: str
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    url: Optional[str] = None
    sentiment_score: float = 0.0
    relevance_score: float = 0.0
    credibility_score: float = 1.0
    impact_score: float = 0.0
    
@dataclass
class SocialMediaPost:
    """Social media post data"""
    platform: str  # twitter, reddit, linkedin, etc.
    content: str
    author: str
    posted_date: datetime
    engagement_metrics: Dict[str, int] = field(default_factory=dict)  # likes, shares, comments
    sentiment_score: float = 0.0
    influence_score: float = 0.0  # Based on author's follower count, etc.
    
@dataclass
class AnalystReport:
    """Financial analyst report/recommendation"""
    analyst_firm: str
    analyst_name: str
    report_date: datetime
    recommendation: str  # Buy, Hold, Sell, etc.
    price_target: Optional[float] = None
    previous_recommendation: Optional[str] = None
    report_summary: str = ""
    sentiment_score: float = 0.0
    accuracy_history: float = 0.5  # Historical accuracy of this analyst

@dataclass
class MarketIndicators:
    """Market timing and sentiment indicators"""
    # Market sentiment indices
    vix_level: float = 20.0  # Volatility index
    put_call_ratio: float = 1.0
    insider_trading_ratio: float = 0.5
    
    # Sector-specific indicators  
    sector_momentum: float = 0.0
    sector_relative_strength: float = 1.0
    sector_rotation_signal: str = "neutral"
    
    # IPO market indicators
    ipo_volume: int = 0  # Number of IPOs in recent period
    ipo_performance: float = 0.0  # Average IPO performance
    ipo_withdrawal_rate: float = 0.0
    
    # Economic indicators
    consumer_confidence: float = 100.0
    business_confidence: float = 100.0
    gdp_growth: float = 2.0
    unemployment_rate: float = 4.0
    inflation_rate: float = 2.0

@dataclass
class SentimentAnalysisInputs:
    """Inputs for comprehensive sentiment analysis"""
    company_name: str
    ticker_symbol: Optional[str] = None
    sector: str = ""
    industry: str = ""
    
    # Data collection parameters
    analysis_period_days: int = 90
    news_sources: List[str] = field(default_factory=lambda: ['reuters', 'bloomberg', 'wsj', 'ft'])
    social_platforms: List[str] = field(default_factory=lambda: ['twitter', 'reddit', 'linkedin'])
    
    # Analysis configuration
    include_finbert: bool = True
    include_social_sentiment: bool = True
    include_analyst_sentiment: bool = True
    include_market_timing: bool = True
    
    # Keywords and search terms
    primary_keywords: List[str] = field(default_factory=list)
    competitor_keywords: List[str] = field(default_factory=list)
    industry_keywords: List[str] = field(default_factory=list)

@dataclass
class SentimentAnalysisResults:
    """Comprehensive sentiment analysis results"""
    # Overall sentiment metrics
    composite_sentiment_score: float  # -1 to 1 scale
    sentiment_confidence: float  # 0 to 1
    sentiment_trend: str  # improving, stable, deteriorating
    sentiment_volatility: float
    
    # Component scores
    news_sentiment: float
    social_sentiment: float
    analyst_sentiment: float
    market_timing_score: float
    
    # Detailed analysis
    news_analysis: Dict[str, Any]
    social_analysis: Dict[str, Any]
    analyst_analysis: Dict[str, Any]
    market_analysis: Dict[str, Any]
    
    # Predictive metrics
    sentiment_momentum: float  # Rate of change
    predicted_sentiment_direction: str
    sentiment_support_resistance: Dict[str, float]
    
    # Risk metrics
    sentiment_risk_score: float
    narrative_consistency: float
    information_quality: float
    
    # Time series data
    sentiment_time_series: pd.DataFrame
    sentiment_decomposition: Dict[str, Any]
    
    # Actionable insights
    key_sentiment_drivers: List[str]
    sentiment_catalysts: List[str]
    risk_factors: List[str]
    opportunities: List[str]

class MarketSentimentAnalyzer:
    """
    Comprehensive Market Sentiment Analyzer for IPO Valuation
    
    Features:
    - FinBERT-based financial news sentiment analysis
    - Social media sentiment monitoring (Twitter, Reddit, LinkedIn)
    - Analyst recommendation analysis
    - Market timing optimization
    - Sector rotation detection
    - Volatility forecasting
    - Sentiment-driven valuation adjustments
    """
    
    def __init__(self, use_finbert: bool = True, cache_models: bool = True):
        self.use_finbert = use_finbert
        self.cache_models = cache_models
        
        # Initialize models
        self.finbert_model = None
        self.finbert_tokenizer = None
        self.sentiment_pipeline = None
        
        # Traditional sentiment analyzers as backup
        self.sia = SentimentIntensityAnalyzer()
        
        # Load models if specified
        if self.use_finbert:
            asyncio.create_task(self._initialize_finbert())
        
        # Sentiment weights by source
        self.source_weights = {
            'finbert_news': 0.35,
            'analyst_reports': 0.30,
            'social_media': 0.20,
            'market_indicators': 0.15
        }
        
        # News source credibility scores
        self.source_credibility = {
            'reuters': 0.95,
            'bloomberg': 0.95,
            'wsj': 0.90,
            'ft': 0.90,
            'cnbc': 0.85,
            'yahoo_finance': 0.75,
            'seeking_alpha': 0.70,
            'marketwatch': 0.75,
            'twitter': 0.60,
            'reddit': 0.55,
            'linkedin': 0.65
        }
    
    async def analyze_market_sentiment(
        self,
        inputs: SentimentAnalysisInputs,
        return_detailed_breakdown: bool = True,
        include_forecasting: bool = True
    ) -> SentimentAnalysisResults:
        """
        Comprehensive market sentiment analysis
        """
        try:
            logger.info(f"Starting sentiment analysis for {inputs.company_name}")
            
            # Collect data from various sources
            news_data = await self._collect_news_data(inputs)
            social_data = await self._collect_social_data(inputs)
            analyst_data = await self._collect_analyst_data(inputs)
            market_data = await self._collect_market_indicators(inputs)
            
            # Analyze sentiment from each source
            news_analysis = await self._analyze_news_sentiment(news_data, inputs)
            social_analysis = await self._analyze_social_sentiment(social_data, inputs)
            analyst_analysis = await self._analyze_analyst_sentiment(analyst_data, inputs)
            market_analysis = await self._analyze_market_timing(market_data, inputs)
            
            # Calculate composite sentiment
            composite_sentiment = await self._calculate_composite_sentiment(
                news_analysis, social_analysis, analyst_analysis, market_analysis
            )
            
            # Time series analysis
            sentiment_timeseries = await self._create_sentiment_timeseries(
                news_data, social_data, analyst_data, inputs.analysis_period_days
            )
            
            # Forecasting
            forecasting_results = {}
            if include_forecasting:
                forecasting_results = await self._forecast_sentiment_trends(sentiment_timeseries)
            
            # Risk assessment
            risk_assessment = await self._assess_sentiment_risks(
                composite_sentiment, sentiment_timeseries, news_analysis, social_analysis
            )
            
            # Generate insights and recommendations
            insights = await self._generate_sentiment_insights(
                composite_sentiment, news_analysis, social_analysis, 
                analyst_analysis, market_analysis, forecasting_results
            )
            
            # Compile comprehensive results
            return await self._compile_sentiment_results(
                composite_sentiment, news_analysis, social_analysis,
                analyst_analysis, market_analysis, sentiment_timeseries,
                forecasting_results, risk_assessment, insights
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            raise
    
    async def _initialize_finbert(self):
        """Initialize FinBERT model for financial sentiment analysis"""
        try:
            model_name = "ProsusAI/finbert"  # Pre-trained FinBERT model
            
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Create pipeline for easier usage
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.finbert_model,
                tokenizer=self.finbert_tokenizer,
                return_all_scores=True
            )
            
            logger.info("FinBERT model initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize FinBERT: {str(e)}. Using fallback sentiment analysis.")
            self.use_finbert = False
    
    async def _collect_news_data(self, inputs: SentimentAnalysisInputs) -> List[NewsArticle]:
        """Collect news articles from various sources"""
        news_articles = []
        
        # In a real implementation, this would use actual news APIs
        # For demonstration, we'll create synthetic news data
        
        synthetic_articles = [
            {
                'title': f"{inputs.company_name} announces strong Q4 earnings",
                'content': f"Company {inputs.company_name} reported better than expected earnings with strong revenue growth in the {inputs.sector} sector.",
                'source': 'reuters',
                'published_date': datetime.now() - timedelta(days=5)
            },
            {
                'title': f"Analysts bullish on {inputs.company_name} IPO prospects",
                'content': f"Multiple analysts have raised price targets for {inputs.company_name} citing strong market position and growth potential.",
                'source': 'bloomberg',
                'published_date': datetime.now() - timedelta(days=3)
            },
            {
                'title': f"Market volatility affects {inputs.sector} sector outlook",
                'content': f"Recent market turbulence has created uncertainty in the {inputs.sector} sector, affecting investor sentiment.",
                'source': 'wsj',
                'published_date': datetime.now() - timedelta(days=7)
            }
        ]
        
        # Convert to NewsArticle objects
        for article_data in synthetic_articles:
            article = NewsArticle(
                title=article_data['title'],
                content=article_data['content'],
                source=article_data['source'],
                published_date=article_data['published_date'],
                credibility_score=self.source_credibility.get(article_data['source'], 0.75)
            )
            news_articles.append(article)
        
        # In practice, would implement real news collection:
        # - RSS feeds
        # - News APIs (NewsAPI, Alpha Vantage, etc.)
        # - Web scraping (where legally permitted)
        
        return news_articles
    
    async def _collect_social_data(self, inputs: SentimentAnalysisInputs) -> List[SocialMediaPost]:
        """Collect social media data"""
        social_posts = []
        
        # Synthetic social media data for demonstration
        synthetic_posts = [
            {
                'platform': 'twitter',
                'content': f"Excited about {inputs.company_name} IPO! Strong fundamentals and great market opportunity #IPO #investing",
                'author': '@investor123',
                'posted_date': datetime.now() - timedelta(hours=12),
                'engagement_metrics': {'likes': 45, 'retweets': 12, 'replies': 8}
            },
            {
                'platform': 'reddit',
                'content': f"Analysis of {inputs.company_name}: Revenue growth looks promising but valuation seems stretched",
                'author': 'financial_analyst_pro',
                'posted_date': datetime.now() - timedelta(days=1),
                'engagement_metrics': {'upvotes': 127, 'comments': 34}
            },
            {
                'platform': 'linkedin',
                'content': f"Professional perspective on {inputs.company_name}: Industry leadership position makes it an interesting investment opportunity",
                'author': 'Industry Expert',
                'posted_date': datetime.now() - timedelta(days=2),
                'engagement_metrics': {'likes': 89, 'comments': 15, 'shares': 23}
            }
        ]
        
        for post_data in synthetic_posts:
            post = SocialMediaPost(
                platform=post_data['platform'],
                content=post_data['content'],
                author=post_data['author'],
                posted_date=post_data['posted_date'],
                engagement_metrics=post_data['engagement_metrics'],
                influence_score=0.7  # Would calculate based on author metrics
            )
            social_posts.append(post)
        
        return social_posts
    
    async def _collect_analyst_data(self, inputs: SentimentAnalysisInputs) -> List[AnalystReport]:
        """Collect analyst reports and recommendations"""
        analyst_reports = []
        
        # Synthetic analyst data
        synthetic_reports = [
            {
                'analyst_firm': 'Goldman Sachs',
                'analyst_name': 'John Smith',
                'report_date': datetime.now() - timedelta(days=10),
                'recommendation': 'Buy',
                'price_target': 45.0,
                'report_summary': f'{inputs.company_name} shows strong growth potential with expanding market share'
            },
            {
                'analyst_firm': 'Morgan Stanley',
                'analyst_name': 'Sarah Johnson',
                'report_date': datetime.now() - timedelta(days=15),
                'recommendation': 'Overweight',
                'price_target': 42.0,
                'report_summary': f'Positive outlook for {inputs.company_name} driven by operational efficiency'
            },
            {
                'analyst_firm': 'JP Morgan',
                'analyst_name': 'Michael Chen',
                'report_date': datetime.now() - timedelta(days=20),
                'recommendation': 'Neutral',
                'price_target': 38.0,
                'report_summary': f'Mixed signals for {inputs.company_name} with both opportunities and challenges'
            }
        ]
        
        for report_data in synthetic_reports:
            report = AnalystReport(
                analyst_firm=report_data['analyst_firm'],
                analyst_name=report_data['analyst_name'],
                report_date=report_data['report_date'],
                recommendation=report_data['recommendation'],
                price_target=report_data['price_target'],
                report_summary=report_data['report_summary'],
                accuracy_history=0.65  # Would track historical accuracy
            )
            analyst_reports.append(report)
        
        return analyst_reports
    
    async def _collect_market_indicators(self, inputs: SentimentAnalysisInputs) -> MarketIndicators:
        """Collect market timing and sentiment indicators"""
        
        # In practice, would collect from financial data APIs
        # Synthetic data for demonstration
        indicators = MarketIndicators(
            vix_level=22.5,  # Moderate volatility
            put_call_ratio=0.95,  # Slightly optimistic
            insider_trading_ratio=0.4,  # More buying than selling
            sector_momentum=0.15,  # Positive sector momentum
            sector_relative_strength=1.08,  # Outperforming market
            sector_rotation_signal="positive",
            ipo_volume=25,  # Recent IPO activity
            ipo_performance=0.08,  # 8% average first-day pop
            ipo_withdrawal_rate=0.15,  # 15% withdrawal rate
            consumer_confidence=108.0,
            business_confidence=105.0
        )
        
        return indicators
    
    async def _analyze_news_sentiment(
        self, 
        news_articles: List[NewsArticle],
        inputs: SentimentAnalysisInputs
    ) -> Dict[str, Any]:
        """Analyze sentiment from news articles using FinBERT and traditional methods"""
        
        if not news_articles:
            return {'overall_sentiment': 0.0, 'confidence': 0.0, 'article_count': 0}
        
        sentiment_scores = []
        confidence_scores = []
        
        for article in news_articles:
            # Combine title and content for analysis
            text = f"{article.title}. {article.content}"
            
            # Use FinBERT if available
            if self.use_finbert and self.sentiment_pipeline:
                try:
                    results = self.sentiment_pipeline(text[:512])  # Limit text length
                    
                    # FinBERT returns: positive, negative, neutral
                    sentiment_map = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}
                    
                    # Get the highest confidence prediction
                    best_result = max(results, key=lambda x: x['score'])
                    finbert_sentiment = sentiment_map.get(best_result['label'].lower(), 0.0)
                    finbert_confidence = best_result['score']
                    
                    # Weight by source credibility
                    weighted_sentiment = finbert_sentiment * article.credibility_score
                    sentiment_scores.append(weighted_sentiment)
                    confidence_scores.append(finbert_confidence * article.credibility_score)
                    
                    # Store individual scores
                    article.sentiment_score = weighted_sentiment
                    
                except Exception as e:
                    logger.warning(f"FinBERT analysis failed for article, using fallback: {str(e)}")
                    # Fall back to traditional sentiment analysis
                    fallback_sentiment = self._traditional_sentiment_analysis(text)
                    article.sentiment_score = fallback_sentiment * article.credibility_score
                    sentiment_scores.append(article.sentiment_score)
                    confidence_scores.append(0.6)  # Lower confidence for fallback
            else:
                # Traditional sentiment analysis
                traditional_sentiment = self._traditional_sentiment_analysis(text)
                article.sentiment_score = traditional_sentiment * article.credibility_score
                sentiment_scores.append(article.sentiment_score)
                confidence_scores.append(0.6)
        
        # Calculate aggregate metrics
        overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        sentiment_std = np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0.0
        average_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Sentiment distribution
        positive_count = sum(1 for score in sentiment_scores if score > 0.1)
        negative_count = sum(1 for score in sentiment_scores if score < -0.1)
        neutral_count = len(sentiment_scores) - positive_count - negative_count
        
        # Time-based trend analysis
        recent_articles = [a for a in news_articles if a.published_date and 
                          a.published_date > datetime.now() - timedelta(days=7)]
        recent_sentiment = np.mean([a.sentiment_score for a in recent_articles]) if recent_articles else 0.0
        
        # Momentum calculation
        older_articles = [a for a in news_articles if a.published_date and 
                         a.published_date <= datetime.now() - timedelta(days=7)]
        older_sentiment = np.mean([a.sentiment_score for a in older_articles]) if older_articles else recent_sentiment
        
        sentiment_momentum = recent_sentiment - older_sentiment
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_std': sentiment_std,
            'confidence': average_confidence,
            'article_count': len(news_articles),
            'sentiment_distribution': {
                'positive': positive_count,
                'neutral': neutral_count,
                'negative': negative_count
            },
            'recent_sentiment': recent_sentiment,
            'sentiment_momentum': sentiment_momentum,
            'top_sources': self._get_top_sources(news_articles),
            'key_topics': self._extract_key_topics(news_articles)
        }
    
    def _traditional_sentiment_analysis(self, text: str) -> float:
        """Fallback traditional sentiment analysis"""
        try:
            # Use NLTK VADER sentiment analyzer
            scores = self.sia.polarity_scores(text)
            compound_score = scores['compound']
            
            # Also use TextBlob as secondary measure
            blob = TextBlob(text)
            textblob_sentiment = blob.sentiment.polarity
            
            # Average the two approaches
            combined_sentiment = (compound_score + textblob_sentiment) / 2.0
            
            return combined_sentiment
            
        except Exception as e:
            logger.warning(f"Traditional sentiment analysis failed: {str(e)}")
            return 0.0
    
    async def _analyze_social_sentiment(
        self,
        social_posts: List[SocialMediaPost],
        inputs: SentimentAnalysisInputs
    ) -> Dict[str, Any]:
        """Analyze sentiment from social media posts"""
        
        if not social_posts:
            return {'overall_sentiment': 0.0, 'confidence': 0.0, 'post_count': 0}
        
        sentiment_scores = []
        platform_breakdown = {}
        
        for post in social_posts:
            # Analyze sentiment
            sentiment = self._traditional_sentiment_analysis(post.content)
            
            # Weight by influence score and platform credibility
            platform_credibility = self.source_credibility.get(post.platform, 0.5)
            weighted_sentiment = sentiment * post.influence_score * platform_credibility
            
            sentiment_scores.append(weighted_sentiment)
            post.sentiment_score = weighted_sentiment
            
            # Platform breakdown
            if post.platform not in platform_breakdown:
                platform_breakdown[post.platform] = []
            platform_breakdown[post.platform].append(weighted_sentiment)
        
        # Calculate aggregate metrics
        overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        
        # Platform-specific analysis
        platform_sentiments = {}
        for platform, scores in platform_breakdown.items():
            platform_sentiments[platform] = {
                'sentiment': np.mean(scores),
                'count': len(scores),
                'std': np.std(scores)
            }
        
        # Engagement-weighted sentiment
        engagement_weights = []
        for post in social_posts:
            total_engagement = sum(post.engagement_metrics.values())
            engagement_weights.append(total_engagement)
        
        if engagement_weights:
            total_engagement = sum(engagement_weights)
            engagement_weighted_sentiment = sum(
                post.sentiment_score * (sum(post.engagement_metrics.values()) / total_engagement)
                for post in social_posts
            )
        else:
            engagement_weighted_sentiment = overall_sentiment
        
        return {
            'overall_sentiment': overall_sentiment,
            'engagement_weighted_sentiment': engagement_weighted_sentiment,
            'confidence': 0.7,  # Lower confidence than news
            'post_count': len(social_posts),
            'platform_breakdown': platform_sentiments,
            'viral_content': self._identify_viral_content(social_posts),
            'sentiment_drivers': self._extract_social_sentiment_drivers(social_posts)
        }
    
    async def _analyze_analyst_sentiment(
        self,
        analyst_reports: List[AnalystReport],
        inputs: SentimentAnalysisInputs
    ) -> Dict[str, Any]:
        """Analyze sentiment from analyst reports and recommendations"""
        
        if not analyst_reports:
            return {'overall_sentiment': 0.0, 'confidence': 0.0, 'report_count': 0}
        
        # Recommendation mapping
        recommendation_scores = {
            'strong buy': 1.0,
            'buy': 0.7,
            'overweight': 0.5,
            'outperform': 0.5,
            'neutral': 0.0,
            'hold': 0.0,
            'underweight': -0.5,
            'underperform': -0.5,
            'sell': -0.7,
            'strong sell': -1.0
        }
        
        sentiment_scores = []
        price_targets = []
        firm_breakdown = {}
        
        for report in analyst_reports:
            # Recommendation sentiment
            rec_sentiment = recommendation_scores.get(report.recommendation.lower(), 0.0)
            
            # Text sentiment from summary
            text_sentiment = self._traditional_sentiment_analysis(report.report_summary)
            
            # Combined sentiment weighted by analyst accuracy
            combined_sentiment = (rec_sentiment * 0.7 + text_sentiment * 0.3) * report.accuracy_history
            
            sentiment_scores.append(combined_sentiment)
            report.sentiment_score = combined_sentiment
            
            if report.price_target:
                price_targets.append(report.price_target)
            
            # Firm breakdown
            if report.analyst_firm not in firm_breakdown:
                firm_breakdown[report.analyst_firm] = []
            firm_breakdown[report.analyst_firm].append(combined_sentiment)
        
        # Calculate metrics
        overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        
        # Recommendation distribution
        recommendations = [report.recommendation.lower() for report in analyst_reports]
        rec_distribution = {}
        for rec in recommendations:
            rec_distribution[rec] = rec_distribution.get(rec, 0) + 1
        
        # Price target analysis
        price_target_stats = {}
        if price_targets:
            price_target_stats = {
                'mean': np.mean(price_targets),
                'median': np.median(price_targets),
                'std': np.std(price_targets),
                'min': np.min(price_targets),
                'max': np.max(price_targets),
                'count': len(price_targets)
            }
        
        # Recent vs older sentiment trend
        recent_reports = [r for r in analyst_reports if 
                         r.report_date > datetime.now() - timedelta(days=30)]
        recent_sentiment = np.mean([r.sentiment_score for r in recent_reports]) if recent_reports else overall_sentiment
        
        return {
            'overall_sentiment': overall_sentiment,
            'recent_sentiment': recent_sentiment,
            'confidence': 0.85,  # High confidence in analyst sentiment
            'report_count': len(analyst_reports),
            'recommendation_distribution': rec_distribution,
            'price_target_stats': price_target_stats,
            'firm_breakdown': {firm: np.mean(scores) for firm, scores in firm_breakdown.items()},
            'consensus_recommendation': self._calculate_consensus_recommendation(analyst_reports),
            'sentiment_trend': 'improving' if recent_sentiment > overall_sentiment else 'stable'
        }
    
    async def _analyze_market_timing(
        self,
        market_indicators: MarketIndicators,
        inputs: SentimentAnalysisInputs
    ) -> Dict[str, Any]:
        """Analyze market timing and conditions for IPO"""
        
        # Market sentiment indicators (normalize to -1 to 1 scale)
        vix_sentiment = max(-1, min(1, (25 - market_indicators.vix_level) / 15))  # Lower VIX = better sentiment
        put_call_sentiment = max(-1, min(1, (1 - market_indicators.put_call_ratio) * 2))  # Lower P/C = bullish
        insider_sentiment = max(-1, min(1, (market_indicators.insider_trading_ratio - 0.5) * 4))  # More buying = bullish
        
        # Sector indicators
        sector_sentiment = max(-1, min(1, market_indicators.sector_momentum * 2))
        sector_strength_sentiment = max(-1, min(1, (market_indicators.sector_relative_strength - 1) * 2))
        
        # IPO market indicators
        ipo_volume_sentiment = max(-1, min(1, (market_indicators.ipo_volume - 20) / 30))  # Optimal IPO volume ~20-50
        ipo_performance_sentiment = max(-1, min(1, market_indicators.ipo_performance * 5))  # Performance as sentiment
        
        # Economic indicators
        confidence_sentiment = max(-1, min(1, (market_indicators.consumer_confidence - 100) / 50))
        
        # Component weights
        component_weights = {
            'market_volatility': 0.25,
            'sector_conditions': 0.25,
            'ipo_market': 0.25,
            'economic_backdrop': 0.25
        }
        
        # Calculate component scores
        market_volatility_score = (vix_sentiment + put_call_sentiment) / 2
        sector_conditions_score = (sector_sentiment + sector_strength_sentiment) / 2
        ipo_market_score = (ipo_volume_sentiment + ipo_performance_sentiment) / 2
        economic_backdrop_score = confidence_sentiment
        
        # Overall market timing score
        overall_timing_score = (
            market_volatility_score * component_weights['market_volatility'] +
            sector_conditions_score * component_weights['sector_conditions'] +
            ipo_market_score * component_weights['ipo_market'] +
            economic_backdrop_score * component_weights['economic_backdrop']
        )
        
        # Market timing recommendation
        if overall_timing_score > 0.3:
            timing_recommendation = "Favorable"
        elif overall_timing_score > -0.3:
            timing_recommendation = "Neutral"
        else:
            timing_recommendation = "Unfavorable"
        
        return {
            'overall_timing_score': overall_timing_score,
            'timing_recommendation': timing_recommendation,
            'component_scores': {
                'market_volatility': market_volatility_score,
                'sector_conditions': sector_conditions_score,
                'ipo_market': ipo_market_score,
                'economic_backdrop': economic_backdrop_score
            },
            'key_indicators': {
                'vix_level': market_indicators.vix_level,
                'sector_momentum': market_indicators.sector_momentum,
                'ipo_performance': market_indicators.ipo_performance,
                'consumer_confidence': market_indicators.consumer_confidence
            },
            'timing_risks': self._identify_timing_risks(market_indicators),
            'optimal_timing_window': self._estimate_optimal_timing(market_indicators)
        }
    
    async def _calculate_composite_sentiment(
        self,
        news_analysis: Dict[str, Any],
        social_analysis: Dict[str, Any],
        analyst_analysis: Dict[str, Any],
        market_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate composite sentiment score from all sources"""
        
        # Extract sentiment scores
        news_sentiment = news_analysis.get('overall_sentiment', 0.0)
        social_sentiment = social_analysis.get('overall_sentiment', 0.0)
        analyst_sentiment = analyst_analysis.get('overall_sentiment', 0.0)
        market_sentiment = market_analysis.get('overall_timing_score', 0.0)
        
        # Apply source weights
        composite_sentiment = (
            news_sentiment * self.source_weights['finbert_news'] +
            analyst_sentiment * self.source_weights['analyst_reports'] +
            social_sentiment * self.source_weights['social_media'] +
            market_sentiment * self.source_weights['market_indicators']
        )
        
        # Calculate confidence based on data availability and quality
        data_completeness = sum([
            1 if news_analysis.get('article_count', 0) > 0 else 0,
            1 if social_analysis.get('post_count', 0) > 0 else 0,
            1 if analyst_analysis.get('report_count', 0) > 0 else 0,
            1  # Market data always available
        ]) / 4.0
        
        # Weighted confidence
        confidence = (
            news_analysis.get('confidence', 0) * self.source_weights['finbert_news'] +
            analyst_analysis.get('confidence', 0) * self.source_weights['analyst_reports'] +
            social_analysis.get('confidence', 0) * self.source_weights['social_media'] +
            0.9 * self.source_weights['market_indicators']  # Market data has high confidence
        ) * data_completeness
        
        # Sentiment consistency check
        sentiments = [news_sentiment, social_sentiment, analyst_sentiment, market_sentiment]
        sentiment_std = np.std([s for s in sentiments if s != 0])
        consistency = max(0, 1 - sentiment_std / 2)  # Higher std = lower consistency
        
        return {
            'composite_sentiment': composite_sentiment,
            'confidence': confidence,
            'consistency': consistency,
            'component_sentiments': {
                'news': news_sentiment,
                'social': social_sentiment,
                'analyst': analyst_sentiment,
                'market': market_sentiment
            },
            'sentiment_range': (min(sentiments), max(sentiments)),
            'dominant_source': max(
                [('news', news_sentiment), ('social', social_sentiment), 
                 ('analyst', analyst_sentiment), ('market', market_sentiment)],
                key=lambda x: abs(x[1])
            )[0]
        }
    
    async def _create_sentiment_timeseries(
        self,
        news_articles: List[NewsArticle],
        social_posts: List[SocialMediaPost],
        analyst_reports: List[AnalystReport],
        period_days: int
    ) -> pd.DataFrame:
        """Create time series of sentiment data"""
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Initialize time series dataframe
        ts_data = []
        
        for date in date_range:
            date_data = {'date': date, 'news_sentiment': 0, 'social_sentiment': 0, 
                        'analyst_sentiment': 0, 'composite_sentiment': 0}
            
            # News sentiment for this date
            daily_news = [a for a in news_articles if a.published_date and 
                         a.published_date.date() == date.date()]
            if daily_news:
                date_data['news_sentiment'] = np.mean([a.sentiment_score for a in daily_news])
            
            # Social sentiment for this date
            daily_social = [p for p in social_posts if p.posted_date.date() == date.date()]
            if daily_social:
                date_data['social_sentiment'] = np.mean([p.sentiment_score for p in daily_social])
            
            # Analyst sentiment (use most recent report)
            recent_reports = [r for r in analyst_reports if r.report_date.date() <= date.date()]
            if recent_reports:
                latest_report = max(recent_reports, key=lambda x: x.report_date)
                date_data['analyst_sentiment'] = latest_report.sentiment_score
            
            # Composite sentiment
            sentiments = [date_data['news_sentiment'], date_data['social_sentiment'], 
                         date_data['analyst_sentiment']]
            non_zero_sentiments = [s for s in sentiments if s != 0]
            if non_zero_sentiments:
                date_data['composite_sentiment'] = np.mean(non_zero_sentiments)
            
            ts_data.append(date_data)
        
        return pd.DataFrame(ts_data)
    
    async def _forecast_sentiment_trends(self, sentiment_timeseries: pd.DataFrame) -> Dict[str, Any]:
        """Forecast sentiment trends using time series analysis"""
        
        if len(sentiment_timeseries) < 10:  # Need minimum data
            return {'forecast_available': False, 'reason': 'Insufficient data'}
        
        try:
            # Use composite sentiment for forecasting
            sentiment_values = sentiment_timeseries['composite_sentiment'].fillna(method='ffill')
            
            if sentiment_values.isna().all():
                return {'forecast_available': False, 'reason': 'No valid sentiment data'}
            
            # Simple trend analysis
            # Calculate moving averages
            sentiment_values = sentiment_values.dropna()
            if len(sentiment_values) < 5:
                return {'forecast_available': False, 'reason': 'Insufficient valid data'}
            
            ma_5 = sentiment_values.rolling(window=5).mean()
            ma_10 = sentiment_values.rolling(window=min(10, len(sentiment_values))).mean()
            
            # Trend direction
            recent_trend = sentiment_values.iloc[-5:].mean() - sentiment_values.iloc[-10:-5].mean()
            
            # Momentum
            momentum = sentiment_values.iloc[-3:].mean() - sentiment_values.iloc[-6:-3].mean()
            
            # Volatility
            volatility = sentiment_values.rolling(window=7).std().iloc[-1]
            
            # Simple linear trend forecast (7 days ahead)
            if len(sentiment_values) >= 7:
                x = np.arange(len(sentiment_values))
                y = sentiment_values.values
                
                # Linear regression for trend
                coeffs = np.polyfit(x, y, 1)
                trend_slope = coeffs[0]
                
                # Forecast next 7 days
                forecast_x = np.arange(len(sentiment_values), len(sentiment_values) + 7)
                forecast_values = np.polyval(coeffs, forecast_x)
                
                return {
                    'forecast_available': True,
                    'trend_direction': 'positive' if recent_trend > 0.05 else 'negative' if recent_trend < -0.05 else 'neutral',
                    'momentum': momentum,
                    'volatility': volatility,
                    'trend_slope': trend_slope,
                    '7_day_forecast': forecast_values.tolist(),
                    'forecast_confidence': max(0, 1 - volatility * 2),  # Lower volatility = higher confidence
                    'support_resistance': {
                        'support': sentiment_values.quantile(0.25),
                        'resistance': sentiment_values.quantile(0.75)
                    }
                }
            
        except Exception as e:
            logger.warning(f"Sentiment forecasting failed: {str(e)}")
            
        return {'forecast_available': False, 'reason': 'Forecasting error'}
    
    async def _assess_sentiment_risks(
        self,
        composite_sentiment: Dict[str, Any],
        sentiment_timeseries: pd.DataFrame,
        news_analysis: Dict[str, Any],
        social_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess risks related to sentiment"""
        
        risks = []
        risk_score = 0.0
        
        # Low sentiment consistency
        if composite_sentiment.get('consistency', 1.0) < 0.6:
            risks.append("Inconsistent sentiment across sources")
            risk_score += 0.2
        
        # High sentiment volatility
        if len(sentiment_timeseries) > 7:
            sentiment_vol = sentiment_timeseries['composite_sentiment'].std()
            if sentiment_vol > 0.5:
                risks.append("High sentiment volatility")
                risk_score += 0.15
        
        # Negative momentum
        news_momentum = news_analysis.get('sentiment_momentum', 0)
        if news_momentum < -0.2:
            risks.append("Declining news sentiment trend")
            risk_score += 0.1
        
        # Low confidence
        if composite_sentiment.get('confidence', 1.0) < 0.5:
            risks.append("Low confidence in sentiment data quality")
            risk_score += 0.1
        
        # Social media risks
        if social_analysis.get('post_count', 0) > 0:
            social_sentiment = social_analysis.get('overall_sentiment', 0)
            if social_sentiment < -0.3:
                risks.append("Negative social media sentiment")
                risk_score += 0.15
        
        # Information quality risks
        info_quality = min(1.0, (news_analysis.get('article_count', 0) / 10) * 
                          composite_sentiment.get('confidence', 0.5))
        
        if info_quality < 0.4:
            risks.append("Limited information availability")
            risk_score += 0.1
        
        return {
            'sentiment_risk_score': min(1.0, risk_score),
            'risk_factors': risks,
            'information_quality': info_quality,
            'narrative_consistency': composite_sentiment.get('consistency', 0.5),
            'monitoring_required': len(risks) > 2,
            'risk_mitigation_suggestions': self._suggest_sentiment_risk_mitigations(risks)
        }
    
    async def _generate_sentiment_insights(
        self,
        composite_sentiment: Dict[str, Any],
        news_analysis: Dict[str, Any],
        social_analysis: Dict[str, Any],
        analyst_analysis: Dict[str, Any],
        market_analysis: Dict[str, Any],
        forecasting_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate actionable sentiment insights"""
        
        insights = {
            'key_drivers': [],
            'catalysts': [],
            'risks': [],
            'opportunities': [],
            'recommendations': []
        }
        
        # Key sentiment drivers
        component_sentiments = composite_sentiment.get('component_sentiments', {})
        dominant_source = composite_sentiment.get('dominant_source', 'unknown')
        
        if dominant_source == 'news':
            insights['key_drivers'].append("Media coverage is the primary sentiment driver")
        elif dominant_source == 'analyst':
            insights['key_drivers'].append("Analyst recommendations are driving sentiment")
        elif dominant_source == 'social':
            insights['key_drivers'].append("Social media discussion is influencing sentiment")
        elif dominant_source == 'market':
            insights['key_drivers'].append("Market conditions are the key sentiment factor")
        
        # Sentiment catalysts
        if news_analysis.get('sentiment_momentum', 0) > 0.1:
            insights['catalysts'].append("Positive news momentum building")
        
        if analyst_analysis.get('sentiment_trend', '') == 'improving':
            insights['catalysts'].append("Improving analyst sentiment")
        
        if market_analysis.get('timing_recommendation', '') == 'Favorable':
            insights['catalysts'].append("Favorable market timing conditions")
        
        # Opportunities
        overall_sentiment = composite_sentiment.get('composite_sentiment', 0)
        
        if overall_sentiment > 0.3:
            insights['opportunities'].append("Strong positive sentiment supports premium valuation")
        elif overall_sentiment < -0.2:
            insights['opportunities'].append("Negative sentiment may create buying opportunity")
        
        # Recommendations
        timing_rec = market_analysis.get('timing_recommendation', 'Neutral')
        if timing_rec == 'Favorable':
            insights['recommendations'].append("Market timing is favorable for IPO launch")
        elif timing_rec == 'Unfavorable':
            insights['recommendations'].append("Consider delaying IPO until market conditions improve")
        
        if forecasting_results.get('trend_direction', '') == 'positive':
            insights['recommendations'].append("Sentiment trends are favorable for near-term launch")
        
        return insights
    
    async def _compile_sentiment_results(
        self,
        composite_sentiment: Dict[str, Any],
        news_analysis: Dict[str, Any],
        social_analysis: Dict[str, Any],
        analyst_analysis: Dict[str, Any],
        market_analysis: Dict[str, Any],
        sentiment_timeseries: pd.DataFrame,
        forecasting_results: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        insights: Dict[str, Any]
    ) -> SentimentAnalysisResults:
        """Compile comprehensive sentiment analysis results"""
        
        # Determine overall sentiment trend
        if forecasting_results.get('trend_direction') == 'positive':
            sentiment_trend = 'improving'
        elif forecasting_results.get('trend_direction') == 'negative':
            sentiment_trend = 'deteriorating'
        else:
            sentiment_trend = 'stable'
        
        # Calculate sentiment volatility
        if len(sentiment_timeseries) > 1:
            sentiment_volatility = sentiment_timeseries['composite_sentiment'].std()
        else:
            sentiment_volatility = 0.0
        
        # Sentiment momentum
        sentiment_momentum = forecasting_results.get('momentum', 0.0)
        
        # Predicted direction
        if sentiment_momentum > 0.1:
            predicted_direction = 'positive'
        elif sentiment_momentum < -0.1:
            predicted_direction = 'negative'
        else:
            predicted_direction = 'neutral'
        
        # Support and resistance levels
        support_resistance = forecasting_results.get('support_resistance', {
            'support': composite_sentiment.get('composite_sentiment', 0) - 0.2,
            'resistance': composite_sentiment.get('composite_sentiment', 0) + 0.2
        })
        
        # Decompose sentiment time series (simplified)
        sentiment_decomposition = {
            'trend': forecasting_results.get('trend_slope', 0.0),
            'volatility': sentiment_volatility,
            'momentum': sentiment_momentum
        }
        
        return SentimentAnalysisResults(
            composite_sentiment_score=composite_sentiment.get('composite_sentiment', 0.0),
            sentiment_confidence=composite_sentiment.get('confidence', 0.0),
            sentiment_trend=sentiment_trend,
            sentiment_volatility=sentiment_volatility,
            news_sentiment=news_analysis.get('overall_sentiment', 0.0),
            social_sentiment=social_analysis.get('overall_sentiment', 0.0),
            analyst_sentiment=analyst_analysis.get('overall_sentiment', 0.0),
            market_timing_score=market_analysis.get('overall_timing_score', 0.0),
            news_analysis=news_analysis,
            social_analysis=social_analysis,
            analyst_analysis=analyst_analysis,
            market_analysis=market_analysis,
            sentiment_momentum=sentiment_momentum,
            predicted_sentiment_direction=predicted_direction,
            sentiment_support_resistance=support_resistance,
            sentiment_risk_score=risk_assessment.get('sentiment_risk_score', 0.0),
            narrative_consistency=risk_assessment.get('narrative_consistency', 1.0),
            information_quality=risk_assessment.get('information_quality', 1.0),
            sentiment_time_series=sentiment_timeseries,
            sentiment_decomposition=sentiment_decomposition,
            key_sentiment_drivers=insights.get('key_drivers', []),
            sentiment_catalysts=insights.get('catalysts', []),
            risk_factors=insights.get('risks', []),
            opportunities=insights.get('opportunities', [])
        )
    
    # Helper methods
    def _get_top_sources(self, news_articles: List[NewsArticle]) -> List[Dict[str, Any]]:
        """Get top news sources by article count and sentiment"""
        source_stats = {}
        for article in news_articles:
            if article.source not in source_stats:
                source_stats[article.source] = {'count': 0, 'sentiment_sum': 0.0}
            source_stats[article.source]['count'] += 1
            source_stats[article.source]['sentiment_sum'] += article.sentiment_score
        
        top_sources = []
        for source, stats in source_stats.items():
            avg_sentiment = stats['sentiment_sum'] / stats['count'] if stats['count'] > 0 else 0
            top_sources.append({
                'source': source,
                'article_count': stats['count'],
                'average_sentiment': avg_sentiment
            })
        
        return sorted(top_sources, key=lambda x: x['article_count'], reverse=True)[:5]
    
    def _extract_key_topics(self, news_articles: List[NewsArticle]) -> List[str]:
        """Extract key topics from news articles (simplified)"""
        # In practice, would use topic modeling (LDA, BERT-based, etc.)
        common_terms = ['earnings', 'growth', 'market', 'revenue', 'profit', 'expansion', 'investment']
        return common_terms[:3]  # Placeholder
    
    def _identify_viral_content(self, social_posts: List[SocialMediaPost]) -> List[Dict[str, Any]]:
        """Identify viral social media content"""
        viral_threshold = 100  # Total engagement threshold
        
        viral_posts = []
        for post in social_posts:
            total_engagement = sum(post.engagement_metrics.values())
            if total_engagement > viral_threshold:
                viral_posts.append({
                    'platform': post.platform,
                    'content_preview': post.content[:100],
                    'engagement': total_engagement,
                    'sentiment': post.sentiment_score
                })
        
        return sorted(viral_posts, key=lambda x: x['engagement'], reverse=True)[:5]
    
    def _extract_social_sentiment_drivers(self, social_posts: List[SocialMediaPost]) -> List[str]:
        """Extract key sentiment drivers from social media"""
        # Simplified implementation - would use more sophisticated NLP
        positive_keywords = ['bullish', 'buy', 'strong', 'growth', 'opportunity']
        negative_keywords = ['bearish', 'sell', 'weak', 'decline', 'risk']
        
        drivers = []
        for post in social_posts:
            content_lower = post.content.lower()
            if any(keyword in content_lower for keyword in positive_keywords):
                drivers.append("Positive investor sentiment")
            elif any(keyword in content_lower for keyword in negative_keywords):
                drivers.append("Negative investor concerns")
        
        return list(set(drivers))[:3]
    
    def _calculate_consensus_recommendation(self, analyst_reports: List[AnalystReport]) -> str:
        """Calculate consensus analyst recommendation"""
        if not analyst_reports:
            return "No consensus"
        
        rec_scores = {
            'strong buy': 5, 'buy': 4, 'overweight': 4, 'outperform': 4,
            'hold': 3, 'neutral': 3, 'underweight': 2, 'underperform': 2,
            'sell': 1, 'strong sell': 0
        }
        
        scores = [rec_scores.get(report.recommendation.lower(), 3) for report in analyst_reports]
        avg_score = np.mean(scores)
        
        if avg_score >= 4.5:
            return "Strong Buy"
        elif avg_score >= 3.5:
            return "Buy"
        elif avg_score >= 2.5:
            return "Hold"
        elif avg_score >= 1.5:
            return "Sell"
        else:
            return "Strong Sell"
    
    def _identify_timing_risks(self, market_indicators: MarketIndicators) -> List[str]:
        """Identify market timing risks"""
        risks = []
        
        if market_indicators.vix_level > 30:
            risks.append("Elevated market volatility")
        if market_indicators.ipo_withdrawal_rate > 0.25:
            risks.append("High IPO withdrawal rate")
        if market_indicators.consumer_confidence < 90:
            risks.append("Weak consumer confidence")
        
        return risks
    
    def _estimate_optimal_timing(self, market_indicators: MarketIndicators) -> str:
        """Estimate optimal timing window"""
        # Simplified timing estimation
        if (market_indicators.vix_level < 25 and 
            market_indicators.ipo_performance > 0.05 and
            market_indicators.consumer_confidence > 100):
            return "Next 3-6 months"
        elif market_indicators.vix_level > 35:
            return "Wait 6-12 months"
        else:
            return "Next 6-9 months"
    
    def _suggest_sentiment_risk_mitigations(self, risks: List[str]) -> List[str]:
        """Suggest mitigations for sentiment risks"""
        suggestions = []
        
        if "Inconsistent sentiment across sources" in risks:
            suggestions.append("Coordinate messaging across all communication channels")
        if "High sentiment volatility" in risks:
            suggestions.append("Implement proactive investor relations strategy")
        if "Declining news sentiment trend" in risks:
            suggestions.append("Enhance media outreach and thought leadership")
        if "Negative social media sentiment" in risks:
            suggestions.append("Monitor and respond to social media discussions")
        
        return suggestions

# Factory function
def create_market_sentiment_analyzer(**kwargs) -> MarketSentimentAnalyzer:
    """Factory function for creating market sentiment analyzer"""
    return MarketSentimentAnalyzer(**kwargs)

# Utility functions
def sentiment_to_valuation_adjustment(sentiment_score: float) -> float:
    """Convert sentiment score to valuation adjustment factor"""
    # Sentiment score is -1 to 1, convert to adjustment factor
    # Neutral sentiment (0) = no adjustment (1.0)
    # Strong positive sentiment (1) = +20% adjustment (1.2)
    # Strong negative sentiment (-1) = -20% adjustment (0.8)
    
    adjustment = 1.0 + (sentiment_score * 0.2)
    return max(0.5, min(1.5, adjustment))  # Cap between -50% and +50%

def calculate_sentiment_risk_premium(sentiment_volatility: float, consistency: float) -> float:
    """Calculate risk premium based on sentiment characteristics"""
    base_premium = 0.02  # 2% base premium
    
    # Higher volatility increases premium
    volatility_premium = sentiment_volatility * 0.03
    
    # Lower consistency increases premium
    consistency_penalty = (1 - consistency) * 0.02
    
    total_premium = base_premium + volatility_premium + consistency_penalty
    return min(0.10, total_premium)  # Cap at 10%