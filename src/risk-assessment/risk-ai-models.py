"""
Risk Assessment AI Models for IPO Valuation Platform
Comprehensive risk analysis using machine learning and regulatory compliance
"""

import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from google.cloud import aiplatform
from google.cloud import bigquery
from google.cloud import monitoring_v3
from google.cloud import language_v1 as language

logger = logging.getLogger(__name__)

class RiskCategory(Enum):
    """Risk categories for IPO valuation"""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    MARKET = "market"
    REGULATORY = "regulatory"
    GOVERNANCE = "governance"
    TECHNOLOGY = "technology"
    ESG = "esg"
    MACROECONOMIC = "macroeconomic"

class RiskLevel(Enum):
    """Risk severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskFactor:
    """Individual risk factor data structure"""
    category: RiskCategory
    level: RiskLevel
    description: str
    probability: float
    impact: float
    mitigation_strategies: List[str]
    regulatory_implications: List[str]
    confidence_score: float
    data_sources: List[str]

class FinancialRiskAnalyzer:
    """Analyze financial risks using ML models"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.bq_client = bigquery.Client(project=project_id)
        self.scaler = StandardScaler()
        self.model = None
        
    def train_financial_risk_model(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Train financial risk assessment model"""
        try:
            # Prepare features
            financial_features = [
                'debt_to_equity', 'current_ratio', 'quick_ratio', 'interest_coverage',
                'asset_turnover', 'inventory_turnover', 'receivables_turnover',
                'return_on_assets', 'return_on_equity', 'gross_margin',
                'operating_margin', 'net_margin', 'revenue_growth',
                'earnings_growth', 'cash_flow_growth', 'working_capital_ratio'
            ]
            
            X = training_data[financial_features].fillna(0)
            y = training_data['financial_risk_level']  # LOW, MEDIUM, HIGH, CRITICAL
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
            
            self.model.fit(X_scaled, y)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': financial_features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Model performance metrics
            y_pred = self.model.predict(X_scaled)
            performance = classification_report(y, y_pred, output_dict=True)
            
            logger.info("Financial risk model trained successfully")
            
            return {
                'model_type': 'financial_risk',
                'performance_metrics': performance,
                'feature_importance': feature_importance.to_dict('records'),
                'training_samples': len(training_data),
                'model_version': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error training financial risk model: {str(e)}")
            raise
    
    def assess_financial_risk(self, financial_data: Dict[str, Any]) -> RiskFactor:
        """Assess financial risk for a company"""
        try:
            if self.model is None:
                raise ValueError("Model not trained. Call train_financial_risk_model first.")
            
            # Prepare features
            features = np.array([[
                financial_data.get('debt_to_equity', 0),
                financial_data.get('current_ratio', 0),
                financial_data.get('quick_ratio', 0),
                financial_data.get('interest_coverage', 0),
                financial_data.get('asset_turnover', 0),
                financial_data.get('inventory_turnover', 0),
                financial_data.get('receivables_turnover', 0),
                financial_data.get('return_on_assets', 0),
                financial_data.get('return_on_equity', 0),
                financial_data.get('gross_margin', 0),
                financial_data.get('operating_margin', 0),
                financial_data.get('net_margin', 0),
                financial_data.get('revenue_growth', 0),
                financial_data.get('earnings_growth', 0),
                financial_data.get('cash_flow_growth', 0),
                financial_data.get('working_capital_ratio', 0)
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict risk level
            risk_prediction = self.model.predict(features_scaled)[0]
            risk_probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Calculate impact and probability scores
            impact_score = self._calculate_financial_impact(financial_data)
            probability_score = max(risk_probabilities)
            
            # Generate mitigation strategies
            mitigation_strategies = self._generate_financial_mitigations(
                financial_data, risk_prediction
            )
            
            return RiskFactor(
                category=RiskCategory.FINANCIAL,
                level=RiskLevel(risk_prediction.lower()),
                description=f"Financial risk assessment based on key ratios and metrics",
                probability=probability_score,
                impact=impact_score,
                mitigation_strategies=mitigation_strategies,
                regulatory_implications=self._get_regulatory_implications('financial'),
                confidence_score=probability_score,
                data_sources=['financial_statements', 'market_data']
            )
            
        except Exception as e:
            logger.error(f"Error assessing financial risk: {str(e)}")
            raise
    
    def _calculate_financial_impact(self, financial_data: Dict[str, Any]) -> float:
        """Calculate potential financial impact score"""
        # Weighted impact based on various financial metrics
        weights = {
            'debt_to_equity': 0.25,
            'current_ratio': 0.20,
            'interest_coverage': 0.15,
            'return_on_equity': 0.15,
            'operating_margin': 0.15,
            'cash_flow_growth': 0.10
        }
        
        impact_score = 0.0
        for metric, weight in weights.items():
            value = financial_data.get(metric, 0)
            # Normalize and calculate impact (implementation specific to each metric)
            normalized_impact = min(abs(value) / 100, 1.0)  # Simplified normalization
            impact_score += normalized_impact * weight
        
        return min(impact_score, 1.0)
    
    def _generate_financial_mitigations(self, financial_data: Dict[str, Any], risk_level: str) -> List[str]:
        """Generate financial risk mitigation strategies"""
        mitigations = []
        
        if financial_data.get('debt_to_equity', 0) > 2.0:
            mitigations.append("Reduce debt levels through equity financing or debt paydown")
            mitigations.append("Renegotiate debt terms to improve cash flow")
        
        if financial_data.get('current_ratio', 0) < 1.0:
            mitigations.append("Improve liquidity through better working capital management")
            mitigations.append("Establish credit facilities for short-term funding needs")
        
        if financial_data.get('interest_coverage', 0) < 2.0:
            mitigations.append("Focus on improving EBITDA through operational efficiency")
            mitigations.append("Consider refinancing high-cost debt")
        
        if risk_level in ['HIGH', 'CRITICAL']:
            mitigations.extend([
                "Implement comprehensive financial monitoring and reporting",
                "Engage financial advisory services for restructuring options",
                "Consider strategic partnerships or asset sales to improve financial position"
            ])
        
        return mitigations
    
    def _get_regulatory_implications(self, risk_type: str) -> List[str]:
        """Get regulatory implications for risk type"""
        financial_regulations = [
            "Enhanced financial disclosure requirements under SEC regulations",
            "Potential covenant compliance issues with existing debt agreements",
            "Increased scrutiny from credit rating agencies",
            "SOX compliance requirements for internal controls over financial reporting"
        ]
        
        return financial_regulations

class MarketRiskAnalyzer:
    """Analyze market and systematic risks"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.bq_client = bigquery.Client(project=project_id)
        
    def assess_market_risk(self, market_data: Dict[str, Any]) -> RiskFactor:
        """Assess market risk factors"""
        try:
            # Calculate market risk metrics
            beta = market_data.get('beta', 1.0)
            correlation = market_data.get('market_correlation', 0.5)
            sector_volatility = market_data.get('sector_volatility', 0.2)
            market_cap = market_data.get('market_cap', 0)
            
            # Market risk scoring
            risk_score = 0.0
            risk_score += min(abs(beta - 1.0), 2.0) * 0.3  # Beta deviation
            risk_score += abs(correlation) * 0.2  # Market correlation
            risk_score += sector_volatility * 0.3  # Sector volatility
            risk_score += (1.0 - min(market_cap / 10_000_000_000, 1.0)) * 0.2  # Size risk
            
            # Determine risk level
            if risk_score > 0.8:
                risk_level = RiskLevel.CRITICAL
            elif risk_score > 0.6:
                risk_level = RiskLevel.HIGH
            elif risk_score > 0.4:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            # Generate market-specific mitigations
            mitigation_strategies = self._generate_market_mitigations(market_data, risk_score)
            
            return RiskFactor(
                category=RiskCategory.MARKET,
                level=risk_level,
                description=f"Market risk assessment including beta, correlation, and sector risks",
                probability=min(risk_score, 1.0),
                impact=self._calculate_market_impact(market_data),
                mitigation_strategies=mitigation_strategies,
                regulatory_implications=self._get_regulatory_implications('market'),
                confidence_score=0.85,  # High confidence in market data
                data_sources=['market_data', 'sector_analysis', 'economic_indicators']
            )
            
        except Exception as e:
            logger.error(f"Error assessing market risk: {str(e)}")
            raise
    
    def _calculate_market_impact(self, market_data: Dict[str, Any]) -> float:
        """Calculate market risk impact"""
        beta = market_data.get('beta', 1.0)
        market_cap = market_data.get('market_cap', 0)
        
        # Higher beta and smaller market cap = higher impact
        impact = min(abs(beta - 1.0) * 0.5, 1.0)
        impact += (1.0 - min(market_cap / 50_000_000_000, 1.0)) * 0.3
        
        return min(impact, 1.0)
    
    def _generate_market_mitigations(self, market_data: Dict[str, Any], risk_score: float) -> List[str]:
        """Generate market risk mitigation strategies"""
        mitigations = []
        
        beta = market_data.get('beta', 1.0)
        if beta > 1.5:
            mitigations.append("Diversify business lines to reduce market sensitivity")
            mitigations.append("Implement hedging strategies for systematic risk")
        
        if market_data.get('sector_volatility', 0) > 0.3:
            mitigations.append("Develop counter-cyclical revenue streams")
            mitigations.append("Build strategic cash reserves for market downturns")
        
        if risk_score > 0.6:
            mitigations.extend([
                "Consider timing of market entry based on market conditions",
                "Implement dynamic pricing strategies to adapt to market changes",
                "Establish partnerships in less volatile markets"
            ])
        
        return mitigations

class RegulatoryRiskAnalyzer:
    """Analyze regulatory and compliance risks"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.language_client = language.LanguageServiceClient()
        
    def assess_regulatory_risk(self, company_data: Dict[str, Any], document_texts: List[str]) -> RiskFactor:
        """Assess regulatory compliance risks"""
        try:
            # Analyze regulatory mentions in documents
            regulatory_analysis = self._analyze_regulatory_content(document_texts)
            
            # Industry-specific regulatory risks
            industry_risks = self._assess_industry_regulatory_risks(
                company_data.get('industry', ''),
                company_data.get('geographic_presence', [])
            )
            
            # Compliance history analysis
            compliance_score = self._assess_compliance_history(company_data)
            
            # Calculate overall regulatory risk
            risk_score = (
                regulatory_analysis['risk_score'] * 0.4 +
                industry_risks['risk_score'] * 0.4 +
                (1.0 - compliance_score) * 0.2
            )
            
            # Determine risk level
            if risk_score > 0.8:
                risk_level = RiskLevel.CRITICAL
            elif risk_score > 0.6:
                risk_level = RiskLevel.HIGH
            elif risk_score > 0.4:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            # Combine mitigation strategies
            mitigation_strategies = (
                regulatory_analysis['mitigations'] +
                industry_risks['mitigations'] +
                self._generate_compliance_mitigations(compliance_score)
            )
            
            return RiskFactor(
                category=RiskCategory.REGULATORY,
                level=risk_level,
                description="Regulatory and compliance risk assessment",
                probability=risk_score,
                impact=self._calculate_regulatory_impact(company_data, risk_score),
                mitigation_strategies=mitigation_strategies,
                regulatory_implications=regulatory_analysis['implications'],
                confidence_score=regulatory_analysis['confidence'],
                data_sources=['regulatory_filings', 'industry_analysis', 'compliance_records']
            )
            
        except Exception as e:
            logger.error(f"Error assessing regulatory risk: {str(e)}")
            raise
    
    def _analyze_regulatory_content(self, document_texts: List[str]) -> Dict[str, Any]:
        """Analyze regulatory content in documents using NLP"""
        try:
            regulatory_keywords = [
                'regulation', 'compliance', 'sec', 'fda', 'epa', 'osha',
                'gdpr', 'ccpa', 'sox', 'dodd-frank', 'mifid', 'basel',
                'investigation', 'enforcement', 'penalty', 'fine',
                'lawsuit', 'litigation', 'settlement', 'consent decree'
            ]
            
            risk_indicators = []
            total_mentions = 0
            sentiment_scores = []
            
            for text in document_texts:
                # Count regulatory mentions
                text_lower = text.lower()
                mentions = sum(1 for keyword in regulatory_keywords if keyword in text_lower)
                total_mentions += mentions
                
                # Analyze sentiment of regulatory mentions
                if mentions > 0:
                    document = language.Document(content=text, type_=language.Document.Type.PLAIN_TEXT)
                    sentiment = self.language_client.analyze_sentiment(
                        request={"document": document, "encoding_type": language.EncodingType.UTF8}
                    ).document_sentiment
                    sentiment_scores.append(sentiment.score)
                    
                    # Extract specific risk indicators
                    for keyword in regulatory_keywords:
                        if keyword in text_lower:
                            # Extract sentence containing keyword
                            sentences = text.split('.')
                            for sentence in sentences:
                                if keyword in sentence.lower():
                                    risk_indicators.append({
                                        'keyword': keyword,
                                        'context': sentence.strip(),
                                        'severity': self._assess_keyword_severity(keyword)
                                    })
            
            # Calculate risk score
            mention_density = min(total_mentions / max(len(document_texts), 1), 10) / 10
            avg_sentiment = sum(sentiment_scores) / max(len(sentiment_scores), 1) if sentiment_scores else 0
            sentiment_risk = max(0, -avg_sentiment)  # Negative sentiment = higher risk
            
            risk_score = min(mention_density * 0.6 + sentiment_risk * 0.4, 1.0)
            
            # Generate mitigations based on findings
            mitigations = self._generate_regulatory_mitigations(risk_indicators, risk_score)
            
            return {
                'risk_score': risk_score,
                'total_mentions': total_mentions,
                'risk_indicators': risk_indicators,
                'sentiment_score': avg_sentiment,
                'mitigations': mitigations,
                'implications': self._extract_regulatory_implications(risk_indicators),
                'confidence': min(0.8, 0.3 + len(document_texts) * 0.1)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing regulatory content: {str(e)}")
            return {'risk_score': 0.5, 'mitigations': [], 'implications': [], 'confidence': 0.5}
    
    def _assess_industry_regulatory_risks(self, industry: str, geographic_presence: List[str]) -> Dict[str, Any]:
        """Assess industry-specific regulatory risks"""
        # Industry risk mappings
        industry_risks = {
            'financial_services': 0.8,
            'healthcare': 0.9,
            'pharmaceuticals': 0.9,
            'energy': 0.7,
            'telecommunications': 0.6,
            'technology': 0.5,
            'manufacturing': 0.6,
            'retail': 0.4,
            'real_estate': 0.5
        }
        
        # Geographic regulatory complexity
        geographic_risk = {
            'us': 0.6,
            'eu': 0.7,
            'china': 0.8,
            'india': 0.6,
            'brazil': 0.7,
            'global': 0.8
        }
        
        industry_lower = industry.lower()
        base_risk = 0.5  # Default risk
        
        for ind, risk in industry_risks.items():
            if ind in industry_lower:
                base_risk = risk
                break
        
        # Adjust for geographic complexity
        geo_adjustment = 0.0
        for geo in geographic_presence:
            geo_lower = geo.lower()
            for region, risk in geographic_risk.items():
                if region in geo_lower:
                    geo_adjustment = max(geo_adjustment, risk * 0.3)
                    break
        
        final_risk_score = min(base_risk + geo_adjustment, 1.0)
        
        # Industry-specific mitigations
        mitigations = self._get_industry_mitigations(industry_lower)
        
        return {
            'risk_score': final_risk_score,
            'mitigations': mitigations
        }
    
    def _assess_compliance_history(self, company_data: Dict[str, Any]) -> float:
        """Assess historical compliance performance"""
        # This would typically query compliance databases
        # For now, using available data points
        
        compliance_indicators = {
            'sec_violations': company_data.get('sec_violations', 0),
            'regulatory_fines': company_data.get('regulatory_fines', 0),
            'compliance_certifications': company_data.get('compliance_certifications', []),
            'audit_findings': company_data.get('audit_findings', 0)
        }
        
        # Calculate compliance score (higher = better compliance)
        score = 1.0
        score -= min(compliance_indicators['sec_violations'] * 0.2, 0.5)
        score -= min(compliance_indicators['regulatory_fines'] / 10_000_000 * 0.3, 0.4)
        score += min(len(compliance_indicators['compliance_certifications']) * 0.1, 0.2)
        score -= min(compliance_indicators['audit_findings'] * 0.05, 0.3)
        
        return max(score, 0.0)
    
    def _assess_keyword_severity(self, keyword: str) -> str:
        """Assess severity of regulatory keyword"""
        high_severity = ['investigation', 'enforcement', 'penalty', 'fine', 'lawsuit', 'litigation']
        medium_severity = ['violation', 'non-compliance', 'warning', 'notice']
        
        if keyword in high_severity:
            return 'HIGH'
        elif keyword in medium_severity:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_regulatory_mitigations(self, risk_indicators: List[Dict], risk_score: float) -> List[str]:
        """Generate regulatory risk mitigations"""
        mitigations = []
        
        # General compliance mitigations
        if risk_score > 0.6:
            mitigations.extend([
                "Establish dedicated compliance function with appropriate resources",
                "Implement comprehensive compliance monitoring and reporting systems",
                "Engage external regulatory counsel for specialized expertise"
            ])
        
        # Specific mitigations based on risk indicators
        high_risk_keywords = [indicator['keyword'] for indicator in risk_indicators 
                             if indicator['severity'] == 'HIGH']
        
        if 'investigation' in high_risk_keywords or 'enforcement' in high_risk_keywords:
            mitigations.append("Develop crisis management and regulatory response protocols")
            
        if 'litigation' in high_risk_keywords or 'lawsuit' in high_risk_keywords:
            mitigations.append("Maintain adequate legal reserves and insurance coverage")
        
        return mitigations
    
    def _extract_regulatory_implications(self, risk_indicators: List[Dict]) -> List[str]:
        """Extract regulatory implications"""
        implications = []
        
        for indicator in risk_indicators:
            if indicator['severity'] == 'HIGH':
                implications.append(f"High-severity regulatory concern: {indicator['keyword']}")
            
        if not implications:
            implications.append("Standard regulatory compliance requirements for public companies")
            
        return implications
    
    def _get_industry_mitigations(self, industry: str) -> List[str]:
        """Get industry-specific mitigation strategies"""
        industry_mitigations = {
            'financial_services': [
                "Implement robust AML and KYC procedures",
                "Maintain compliance with capital adequacy requirements",
                "Establish comprehensive risk management framework"
            ],
            'healthcare': [
                "Ensure HIPAA compliance for patient data protection",
                "Maintain FDA regulatory compliance for products",
                "Implement clinical trial oversight and safety monitoring"
            ],
            'technology': [
                "Implement data privacy compliance (GDPR, CCPA)",
                "Establish cybersecurity and data protection measures",
                "Comply with platform and content moderation regulations"
            ]
        }
        
        return industry_mitigations.get(industry, ["Implement industry best practices for compliance"])
    
    def _calculate_regulatory_impact(self, company_data: Dict[str, Any], risk_score: float) -> float:
        """Calculate regulatory risk impact"""
        # Base impact from risk score
        base_impact = risk_score
        
        # Adjust based on company size (larger companies have higher impact)
        revenue = company_data.get('revenue', 0)
        size_multiplier = min(revenue / 1_000_000_000, 2.0)  # Up to 2x for $1B+ revenue
        
        return min(base_impact * (1 + size_multiplier * 0.3), 1.0)
    
    def _generate_compliance_mitigations(self, compliance_score: float) -> List[str]:
        """Generate compliance-specific mitigations"""
        mitigations = []
        
        if compliance_score < 0.7:
            mitigations.extend([
                "Conduct comprehensive compliance audit and remediation",
                "Implement enhanced internal controls and monitoring",
                "Provide regular compliance training for all employees"
            ])
        
        if compliance_score < 0.5:
            mitigations.extend([
                "Consider hiring Chief Compliance Officer",
                "Engage third-party compliance consulting services",
                "Implement whistleblower and incident reporting systems"
            ])
        
        return mitigations

class AnomalyDetectionService:
    """Detect anomalies in financial and operational data"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        
    def detect_financial_anomalies(self, financial_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in financial data"""
        try:
            # Prepare features for anomaly detection
            anomaly_features = [
                'revenue', 'net_income', 'total_assets', 'total_liabilities',
                'cash_and_equivalents', 'operating_cash_flow', 'free_cash_flow',
                'debt_to_equity', 'current_ratio', 'roe', 'roa'
            ]
            
            # Handle missing values
            X = financial_data[anomaly_features].fillna(financial_data[anomaly_features].median())
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Detect anomalies
            anomaly_labels = self.isolation_forest.fit_predict(X_scaled)
            anomaly_scores = self.isolation_forest.score_samples(X_scaled)
            
            # Identify anomalous records
            anomalies = financial_data[anomaly_labels == -1].copy()
            anomalies['anomaly_score'] = anomaly_scores[anomaly_labels == -1]
            
            # Analyze anomaly patterns
            anomaly_analysis = self._analyze_anomaly_patterns(anomalies, anomaly_features)
            
            return {
                'total_records': len(financial_data),
                'anomaly_count': len(anomalies),
                'anomaly_rate': len(anomalies) / len(financial_data),
                'anomalous_records': anomalies.to_dict('records'),
                'anomaly_patterns': anomaly_analysis,
                'risk_implications': self._assess_anomaly_risks(anomaly_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error detecting financial anomalies: {str(e)}")
            raise
    
    def _analyze_anomaly_patterns(self, anomalies: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """Analyze patterns in detected anomalies"""
        if anomalies.empty:
            return {}
        
        patterns = {}
        
        # Feature-wise analysis
        for feature in features:
            if feature in anomalies.columns:
                feature_stats = {
                    'mean': anomalies[feature].mean(),
                    'median': anomalies[feature].median(),
                    'std': anomalies[feature].std(),
                    'outlier_threshold': anomalies[feature].quantile(0.95)
                }
                patterns[feature] = feature_stats
        
        # Temporal patterns (if date column exists)
        if 'date' in anomalies.columns:
            anomalies['date'] = pd.to_datetime(anomalies['date'])
            temporal_patterns = anomalies.groupby(anomalies['date'].dt.quarter).size()
            patterns['temporal_distribution'] = temporal_patterns.to_dict()
        
        return patterns
    
    def _assess_anomaly_risks(self, anomaly_patterns: Dict[str, Any]) -> List[str]:
        """Assess risks implied by anomaly patterns"""
        risks = []
        
        # Check for concerning patterns
        for feature, stats in anomaly_patterns.items():
            if isinstance(stats, dict) and 'std' in stats:
                if stats['std'] > stats.get('mean', 0) * 2:  # High variability
                    risks.append(f"High variability detected in {feature}")
                    
                if feature in ['revenue', 'net_income'] and stats.get('mean', 0) < 0:
                    risks.append(f"Negative {feature} anomalies detected")
        
        if not risks:
            risks.append("No significant risk patterns identified in anomalies")
        
        return risks

class ComprehensiveRiskAssessment:
    """Main risk assessment orchestrator"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.financial_analyzer = FinancialRiskAnalyzer(project_id)
        self.market_analyzer = MarketRiskAnalyzer(project_id)
        self.regulatory_analyzer = RegulatoryRiskAnalyzer(project_id)
        self.anomaly_detector = AnomalyDetectionService(project_id)
        
    def conduct_comprehensive_assessment(self, 
                                       company_data: Dict[str, Any],
                                       financial_data: pd.DataFrame,
                                       market_data: Dict[str, Any],
                                       document_texts: List[str]) -> Dict[str, Any]:
        """Conduct comprehensive risk assessment"""
        try:
            logger.info("Starting comprehensive risk assessment")
            
            risk_factors = []
            
            # Financial risk assessment
            try:
                financial_risk = self.financial_analyzer.assess_financial_risk(
                    financial_data.iloc[-1].to_dict() if not financial_data.empty else {}
                )
                risk_factors.append(financial_risk)
            except Exception as e:
                logger.error(f"Financial risk assessment failed: {str(e)}")
            
            # Market risk assessment
            try:
                market_risk = self.market_analyzer.assess_market_risk(market_data)
                risk_factors.append(market_risk)
            except Exception as e:
                logger.error(f"Market risk assessment failed: {str(e)}")
            
            # Regulatory risk assessment
            try:
                regulatory_risk = self.regulatory_analyzer.assess_regulatory_risk(
                    company_data, document_texts
                )
                risk_factors.append(regulatory_risk)
            except Exception as e:
                logger.error(f"Regulatory risk assessment failed: {str(e)}")
            
            # Anomaly detection
            try:
                anomaly_results = self.anomaly_detector.detect_financial_anomalies(financial_data)
            except Exception as e:
                logger.error(f"Anomaly detection failed: {str(e)}")
                anomaly_results = {}
            
            # Calculate overall risk score
            overall_risk_score = self._calculate_overall_risk_score(risk_factors)
            overall_risk_level = self._determine_overall_risk_level(overall_risk_score)
            
            # Generate comprehensive recommendations
            recommendations = self._generate_comprehensive_recommendations(risk_factors, anomaly_results)
            
            assessment_result = {
                'assessment_timestamp': datetime.utcnow().isoformat(),
                'company_id': company_data.get('company_id', 'unknown'),
                'overall_risk_score': overall_risk_score,
                'overall_risk_level': overall_risk_level.value,
                'risk_factors': [self._serialize_risk_factor(rf) for rf in risk_factors],
                'anomaly_detection': anomaly_results,
                'recommendations': recommendations,
                'assessment_metadata': {
                    'financial_data_points': len(financial_data),
                    'document_count': len(document_texts),
                    'assessment_version': '1.0'
                }
            }
            
            logger.info("Comprehensive risk assessment completed")
            return assessment_result
            
        except Exception as e:
            logger.error(f"Error in comprehensive risk assessment: {str(e)}")
            raise
    
    def _calculate_overall_risk_score(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate weighted overall risk score"""
        if not risk_factors:
            return 0.5
        
        # Weight factors by category importance
        category_weights = {
            RiskCategory.FINANCIAL: 0.30,
            RiskCategory.MARKET: 0.25,
            RiskCategory.REGULATORY: 0.20,
            RiskCategory.OPERATIONAL: 0.15,
            RiskCategory.GOVERNANCE: 0.10
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for risk_factor in risk_factors:
            weight = category_weights.get(risk_factor.category, 0.1)
            # Convert risk level to numeric score
            level_score = {
                RiskLevel.LOW: 0.25,
                RiskLevel.MEDIUM: 0.50,
                RiskLevel.HIGH: 0.75,
                RiskLevel.CRITICAL: 1.0
            }.get(risk_factor.level, 0.5)
            
            weighted_score += level_score * weight * risk_factor.confidence_score
            total_weight += weight
        
        return min(weighted_score / max(total_weight, 0.1), 1.0)
    
    def _determine_overall_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine overall risk level from score"""
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_comprehensive_recommendations(self, 
                                              risk_factors: List[RiskFactor],
                                              anomaly_results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations"""
        recommendations = []
        
        # Collect all mitigation strategies
        all_mitigations = []
        for rf in risk_factors:
            all_mitigations.extend(rf.mitigation_strategies)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_mitigations = []
        for mitigation in all_mitigations:
            if mitigation not in seen:
                unique_mitigations.append(mitigation)
                seen.add(mitigation)
        
        recommendations.extend(unique_mitigations)
        
        # Add anomaly-specific recommendations
        if anomaly_results.get('anomaly_rate', 0) > 0.15:
            recommendations.append("Investigate and address financial data anomalies identified")
        
        # Add overall recommendations
        high_risk_categories = [rf.category.value for rf in risk_factors if rf.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        if len(high_risk_categories) > 2:
            recommendations.append("Consider delaying IPO until high-risk issues are addressed")
            recommendations.append("Engage specialized risk management consultants")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _serialize_risk_factor(self, risk_factor: RiskFactor) -> Dict[str, Any]:
        """Serialize risk factor for JSON output"""
        return {
            'category': risk_factor.category.value,
            'level': risk_factor.level.value,
            'description': risk_factor.description,
            'probability': risk_factor.probability,
            'impact': risk_factor.impact,
            'mitigation_strategies': risk_factor.mitigation_strategies,
            'regulatory_implications': risk_factor.regulatory_implications,
            'confidence_score': risk_factor.confidence_score,
            'data_sources': risk_factor.data_sources
        }