"""
Multi-Factor Risk Assessment Model for IPO Valuation
Comprehensive risk scoring with industry-specific models and ESG integration
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import scipy.stats as stats
from scipy.optimize import minimize
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class RiskCategory(Enum):
    MARKET = "market"
    OPERATIONAL = "operational" 
    FINANCIAL = "financial"
    ESG = "esg"
    REGULATORY = "regulatory"
    INDUSTRY_SPECIFIC = "industry_specific"

@dataclass
class RiskFactor:
    """Individual risk factor definition"""
    name: str
    category: RiskCategory
    weight: float
    current_score: float = 0.0
    historical_scores: List[float] = field(default_factory=list)
    benchmark_score: float = 50.0  # Industry benchmark (0-100 scale)
    volatility: float = 0.0
    trend: float = 0.0  # Positive = improving, negative = deteriorating
    confidence: float = 1.0  # Confidence in the score (0-1)

@dataclass
class ESGMetrics:
    """Environmental, Social, and Governance metrics"""
    # Environmental
    carbon_intensity: float = 0.0
    energy_efficiency: float = 0.0
    waste_management: float = 0.0
    water_usage: float = 0.0
    environmental_score: float = 50.0
    
    # Social
    employee_satisfaction: float = 0.0
    diversity_metrics: Dict[str, float] = field(default_factory=dict)
    community_impact: float = 0.0
    product_safety: float = 0.0
    social_score: float = 50.0
    
    # Governance
    board_independence: float = 0.0
    executive_compensation: float = 0.0
    audit_quality: float = 0.0
    shareholder_rights: float = 0.0
    governance_score: float = 50.0
    
    # Overall ESG
    esg_score: float = 50.0
    esg_rating: str = "BB"
    esg_trend: float = 0.0

@dataclass
class MarketRiskFactors:
    """Market-related risk factors"""
    beta: float = 1.0
    volatility: float = 0.0
    correlation_with_market: float = 0.0
    sector_beta: float = 1.0
    liquidity_risk: float = 0.0
    market_cap_risk: float = 0.0
    
    # Market sentiment indicators
    analyst_coverage: int = 0
    institutional_ownership: float = 0.0
    short_interest: float = 0.0
    
    # Macroeconomic sensitivity
    interest_rate_sensitivity: float = 0.0
    inflation_sensitivity: float = 0.0
    currency_risk: float = 0.0

@dataclass
class OperationalRiskFactors:
    """Operational risk factors"""
    # Business model risks
    revenue_concentration: float = 0.0  # Customer concentration
    geographic_concentration: float = 0.0
    product_concentration: float = 0.0
    supplier_dependence: float = 0.0
    
    # Operational metrics
    operating_leverage: float = 0.0
    scalability_score: float = 50.0
    technology_risk: float = 0.0
    cybersecurity_score: float = 50.0
    
    # Management and organizational
    management_experience: float = 0.0
    employee_turnover: float = 0.0
    succession_planning: float = 50.0
    organizational_depth: float = 50.0

@dataclass
class FinancialRiskFactors:
    """Financial risk factors"""
    # Liquidity risks
    current_ratio: float = 1.0
    quick_ratio: float = 1.0
    cash_conversion_cycle: float = 0.0
    working_capital_ratio: float = 0.0
    
    # Leverage and solvency
    debt_to_equity: float = 0.0
    interest_coverage: float = 0.0
    debt_service_coverage: float = 0.0
    
    # Profitability stability
    earnings_volatility: float = 0.0
    margin_stability: float = 0.0
    cash_flow_predictability: float = 0.0
    
    # Capital allocation
    capex_intensity: float = 0.0
    dividend_sustainability: float = 0.0
    acquisition_risk: float = 0.0

@dataclass
class RegulatoryRiskFactors:
    """Regulatory and compliance risk factors"""
    regulatory_environment_score: float = 50.0
    compliance_history: float = 50.0
    regulatory_change_risk: float = 0.0
    
    # Industry-specific regulatory risks
    licensing_risk: float = 0.0
    environmental_compliance: float = 50.0
    data_privacy_compliance: float = 50.0
    antitrust_risk: float = 0.0
    
    # Geographic regulatory risks
    political_risk: Dict[str, float] = field(default_factory=dict)
    tax_policy_risk: float = 0.0

@dataclass
class RiskAssessmentInputs:
    """Comprehensive risk assessment inputs"""
    company_name: str
    ticker: Optional[str] = None
    sector: str = ""
    industry: str = ""
    
    # Risk factor categories
    market_risks: MarketRiskFactors = field(default_factory=MarketRiskFactors)
    operational_risks: OperationalRiskFactors = field(default_factory=OperationalRiskFactors)
    financial_risks: FinancialRiskFactors = field(default_factory=FinancialRiskFactors)
    regulatory_risks: RegulatoryRiskFactors = field(default_factory=RegulatoryRiskFactors)
    esg_metrics: ESGMetrics = field(default_factory=ESGMetrics)
    
    # Industry benchmarks
    industry_risk_benchmarks: Dict[str, float] = field(default_factory=dict)
    peer_risk_comparison: Dict[str, float] = field(default_factory=dict)
    
    # Contextual factors
    company_stage: str = "mature"  # startup, growth, mature
    market_conditions: str = "normal"  # bull, normal, bear
    economic_cycle: str = "expansion"  # recession, recovery, expansion, peak

@dataclass
class RiskAssessmentResults:
    """Comprehensive risk assessment results"""
    # Overall risk metrics
    composite_risk_score: float  # 0-100 scale (lower is better)
    risk_grade: str  # AAA, AA, A, BBB, BB, B, CCC, CC, C, D
    risk_category: str  # Low, Medium-Low, Medium, Medium-High, High, Very High
    
    # Category scores
    category_scores: Dict[RiskCategory, float]
    category_percentiles: Dict[RiskCategory, float]  # Relative to industry peers
    
    # Factor analysis
    risk_factor_scores: Dict[str, RiskFactor]
    top_risk_factors: List[Tuple[str, float]]  # Top 10 risk factors
    risk_drivers: List[str]  # Key risk drivers
    
    # Comparative analysis
    industry_comparison: Dict[str, float]
    peer_comparison: Dict[str, float]
    historical_trend: Dict[str, List[float]]
    
    # Predictive metrics
    risk_trajectory: str  # improving, stable, deteriorating
    probability_of_distress: float  # 0-1 probability
    expected_risk_adjusted_return: float
    risk_premium: float
    
    # ESG integration
    esg_risk_contribution: float
    esg_risk_mitigation: float
    
    # Scenario analysis
    stress_test_results: Dict[str, Dict[str, float]]
    risk_sensitivity: Dict[str, float]
    
    # Recommendations
    risk_mitigation_priorities: List[str]
    monitoring_indicators: List[str]

class MultiFactorRiskModel:
    """
    Multi-Factor Risk Assessment Model for IPO Valuation
    
    Features:
    - Comprehensive risk scoring across 6 categories
    - Industry-specific risk models
    - ESG integration
    - Machine learning risk prediction
    - Stress testing and scenario analysis
    - Risk-adjusted return calculations
    """
    
    def __init__(self, model_type: str = 'ensemble'):
        self.model_type = model_type
        self.risk_models = {}
        self.scaler = StandardScaler()
        self.industry_models = {}
        
        # Risk weights by category (can be industry-adjusted)
        self.default_category_weights = {
            RiskCategory.MARKET: 0.25,
            RiskCategory.FINANCIAL: 0.25,
            RiskCategory.OPERATIONAL: 0.20,
            RiskCategory.REGULATORY: 0.15,
            RiskCategory.ESG: 0.10,
            RiskCategory.INDUSTRY_SPECIFIC: 0.05
        }
        
        # Load industry risk models
        self._initialize_industry_models()
    
    async def assess_comprehensive_risk(
        self,
        inputs: RiskAssessmentInputs,
        include_stress_testing: bool = True,
        include_scenario_analysis: bool = True,
        peer_companies: Optional[List[Dict]] = None
    ) -> RiskAssessmentResults:
        """
        Perform comprehensive risk assessment
        """
        try:
            # Calculate individual risk factor scores
            risk_factors = await self._calculate_risk_factors(inputs)
            
            # Calculate category scores
            category_scores = await self._calculate_category_scores(risk_factors, inputs)
            
            # Calculate composite risk score
            composite_score = await self._calculate_composite_risk_score(
                category_scores, inputs.industry
            )
            
            # Perform peer comparison
            peer_comparison = await self._perform_peer_comparison(
                inputs, category_scores, peer_companies
            )
            
            # ESG risk integration
            esg_analysis = await self._analyze_esg_risks(inputs.esg_metrics)
            
            # Stress testing
            stress_results = {}
            if include_stress_testing:
                stress_results = await self._perform_stress_testing(inputs, risk_factors)
            
            # Scenario analysis
            scenario_results = {}
            if include_scenario_analysis:
                scenario_results = await self._perform_scenario_analysis(inputs)
            
            # Risk prediction and trajectory
            risk_prediction = await self._predict_risk_trajectory(inputs, risk_factors)
            
            # Generate recommendations
            recommendations = await self._generate_risk_recommendations(
                risk_factors, category_scores
            )
            
            # Compile comprehensive results
            return await self._compile_risk_results(
                composite_score, category_scores, risk_factors,
                peer_comparison, esg_analysis, stress_results,
                risk_prediction, recommendations
            )
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {str(e)}")
            raise
    
    async def _calculate_risk_factors(
        self, 
        inputs: RiskAssessmentInputs
    ) -> Dict[str, RiskFactor]:
        """Calculate individual risk factor scores"""
        
        risk_factors = {}
        
        # Market risk factors
        market_factors = await self._assess_market_risks(inputs.market_risks)
        risk_factors.update(market_factors)
        
        # Financial risk factors
        financial_factors = await self._assess_financial_risks(inputs.financial_risks)
        risk_factors.update(financial_factors)
        
        # Operational risk factors
        operational_factors = await self._assess_operational_risks(inputs.operational_risks)
        risk_factors.update(operational_factors)
        
        # Regulatory risk factors
        regulatory_factors = await self._assess_regulatory_risks(inputs.regulatory_risks)
        risk_factors.update(regulatory_factors)
        
        # ESG risk factors
        esg_factors = await self._assess_esg_risks(inputs.esg_metrics)
        risk_factors.update(esg_factors)
        
        # Industry-specific risk factors
        industry_factors = await self._assess_industry_risks(inputs)
        risk_factors.update(industry_factors)
        
        return risk_factors
    
    async def _assess_market_risks(self, market_risks: MarketRiskFactors) -> Dict[str, RiskFactor]:
        """Assess market-related risks"""
        factors = {}
        
        # Beta risk - systematic risk
        beta_score = min(100, max(0, (market_risks.beta - 0.5) / 1.5 * 100))
        factors['systematic_risk'] = RiskFactor(
            name="Systematic Risk (Beta)",
            category=RiskCategory.MARKET,
            weight=0.3,
            current_score=beta_score,
            benchmark_score=60.0
        )
        
        # Volatility risk
        vol_score = min(100, market_risks.volatility * 1000)  # Convert to 0-100 scale
        factors['volatility_risk'] = RiskFactor(
            name="Price Volatility",
            category=RiskCategory.MARKET,
            weight=0.25,
            current_score=vol_score,
            benchmark_score=40.0
        )
        
        # Liquidity risk
        liquidity_score = market_risks.liquidity_risk * 100
        factors['liquidity_risk'] = RiskFactor(
            name="Liquidity Risk",
            category=RiskCategory.MARKET,
            weight=0.2,
            current_score=liquidity_score,
            benchmark_score=30.0
        )
        
        # Size risk (market cap)
        size_score = market_risks.market_cap_risk * 100
        factors['size_risk'] = RiskFactor(
            name="Size Risk",
            category=RiskCategory.MARKET,
            weight=0.15,
            current_score=size_score,
            benchmark_score=35.0
        )
        
        # Interest rate sensitivity
        interest_score = abs(market_risks.interest_rate_sensitivity) * 50
        factors['interest_rate_risk'] = RiskFactor(
            name="Interest Rate Sensitivity",
            category=RiskCategory.MARKET,
            weight=0.1,
            current_score=interest_score,
            benchmark_score=25.0
        )
        
        return factors
    
    async def _assess_financial_risks(self, financial_risks: FinancialRiskFactors) -> Dict[str, RiskFactor]:
        """Assess financial risks"""
        factors = {}
        
        # Leverage risk
        leverage_score = min(100, financial_risks.debt_to_equity * 20)  # Scale to 0-100
        factors['leverage_risk'] = RiskFactor(
            name="Leverage Risk",
            category=RiskCategory.FINANCIAL,
            weight=0.25,
            current_score=leverage_score,
            benchmark_score=40.0
        )
        
        # Liquidity risk
        if financial_risks.current_ratio > 0:
            liquidity_score = max(0, min(100, (2.0 - financial_risks.current_ratio) / 1.5 * 100))
        else:
            liquidity_score = 100
            
        factors['financial_liquidity'] = RiskFactor(
            name="Financial Liquidity",
            category=RiskCategory.FINANCIAL,
            weight=0.2,
            current_score=liquidity_score,
            benchmark_score=30.0
        )
        
        # Interest coverage risk
        if financial_risks.interest_coverage > 0:
            coverage_score = max(0, min(100, (5.0 - financial_risks.interest_coverage) / 4.0 * 100))
        else:
            coverage_score = 100
            
        factors['interest_coverage_risk'] = RiskFactor(
            name="Interest Coverage Risk",
            category=RiskCategory.FINANCIAL,
            weight=0.2,
            current_score=coverage_score,
            benchmark_score=35.0
        )
        
        # Earnings volatility
        earnings_vol_score = financial_risks.earnings_volatility * 100
        factors['earnings_volatility'] = RiskFactor(
            name="Earnings Volatility",
            category=RiskCategory.FINANCIAL,
            weight=0.15,
            current_score=earnings_vol_score,
            benchmark_score=45.0
        )
        
        # Cash flow predictability (inverse score - lower is better)
        cf_predictability_score = (1.0 - financial_risks.cash_flow_predictability) * 100
        factors['cash_flow_unpredictability'] = RiskFactor(
            name="Cash Flow Unpredictability",
            category=RiskCategory.FINANCIAL,
            weight=0.2,
            current_score=cf_predictability_score,
            benchmark_score=40.0
        )
        
        return factors
    
    async def _assess_operational_risks(self, operational_risks: OperationalRiskFactors) -> Dict[str, RiskFactor]:
        """Assess operational risks"""
        factors = {}
        
        # Revenue concentration risk
        revenue_conc_score = operational_risks.revenue_concentration * 100
        factors['revenue_concentration'] = RiskFactor(
            name="Revenue Concentration",
            category=RiskCategory.OPERATIONAL,
            weight=0.25,
            current_score=revenue_conc_score,
            benchmark_score=30.0
        )
        
        # Operating leverage risk
        op_leverage_score = min(100, operational_risks.operating_leverage * 25)
        factors['operating_leverage'] = RiskFactor(
            name="Operating Leverage",
            category=RiskCategory.OPERATIONAL,
            weight=0.2,
            current_score=op_leverage_score,
            benchmark_score=40.0
        )
        
        # Technology risk
        tech_risk_score = operational_risks.technology_risk * 100
        factors['technology_risk'] = RiskFactor(
            name="Technology Risk",
            category=RiskCategory.OPERATIONAL,
            weight=0.15,
            current_score=tech_risk_score,
            benchmark_score=35.0
        )
        
        # Cybersecurity risk (inverse of score)
        cyber_risk_score = (100 - operational_risks.cybersecurity_score)
        factors['cybersecurity_risk'] = RiskFactor(
            name="Cybersecurity Risk",
            category=RiskCategory.OPERATIONAL,
            weight=0.15,
            current_score=cyber_risk_score,
            benchmark_score=30.0
        )
        
        # Management risk
        mgmt_risk_score = max(0, 100 - operational_risks.management_experience * 100)
        factors['management_risk'] = RiskFactor(
            name="Management Risk",
            category=RiskCategory.OPERATIONAL,
            weight=0.15,
            current_score=mgmt_risk_score,
            benchmark_score=25.0
        )
        
        # Employee turnover risk
        turnover_score = min(100, operational_risks.employee_turnover * 200)  # 50% turnover = max score
        factors['employee_turnover_risk'] = RiskFactor(
            name="Employee Turnover Risk",
            category=RiskCategory.OPERATIONAL,
            weight=0.1,
            current_score=turnover_score,
            benchmark_score=20.0
        )
        
        return factors
    
    async def _assess_regulatory_risks(self, regulatory_risks: RegulatoryRiskFactors) -> Dict[str, RiskFactor]:
        """Assess regulatory risks"""
        factors = {}
        
        # Regulatory environment risk (inverse of score)
        reg_env_score = 100 - regulatory_risks.regulatory_environment_score
        factors['regulatory_environment'] = RiskFactor(
            name="Regulatory Environment Risk",
            category=RiskCategory.REGULATORY,
            weight=0.3,
            current_score=reg_env_score,
            benchmark_score=40.0
        )
        
        # Compliance risk (inverse of score)
        compliance_score = 100 - regulatory_risks.compliance_history
        factors['compliance_risk'] = RiskFactor(
            name="Compliance Risk",
            category=RiskCategory.REGULATORY,
            weight=0.25,
            current_score=compliance_score,
            benchmark_score=30.0
        )
        
        # Regulatory change risk
        reg_change_score = regulatory_risks.regulatory_change_risk * 100
        factors['regulatory_change_risk'] = RiskFactor(
            name="Regulatory Change Risk",
            category=RiskCategory.REGULATORY,
            weight=0.2,
            current_score=reg_change_score,
            benchmark_score=35.0
        )
        
        # Environmental compliance risk (inverse of score)
        env_compliance_score = 100 - regulatory_risks.environmental_compliance
        factors['environmental_compliance_risk'] = RiskFactor(
            name="Environmental Compliance Risk",
            category=RiskCategory.REGULATORY,
            weight=0.15,
            current_score=env_compliance_score,
            benchmark_score=25.0
        )
        
        # Tax policy risk
        tax_risk_score = regulatory_risks.tax_policy_risk * 100
        factors['tax_policy_risk'] = RiskFactor(
            name="Tax Policy Risk",
            category=RiskCategory.REGULATORY,
            weight=0.1,
            current_score=tax_risk_score,
            benchmark_score=20.0
        )
        
        return factors
    
    async def _assess_esg_risks(self, esg_metrics: ESGMetrics) -> Dict[str, RiskFactor]:
        """Assess ESG-related risks"""
        factors = {}
        
        # Environmental risk (inverse of score)
        env_risk_score = 100 - esg_metrics.environmental_score
        factors['environmental_risk'] = RiskFactor(
            name="Environmental Risk",
            category=RiskCategory.ESG,
            weight=0.4,
            current_score=env_risk_score,
            benchmark_score=40.0
        )
        
        # Social risk (inverse of score)
        social_risk_score = 100 - esg_metrics.social_score
        factors['social_risk'] = RiskFactor(
            name="Social Risk",
            category=RiskCategory.ESG,
            weight=0.3,
            current_score=social_risk_score,
            benchmark_score=35.0
        )
        
        # Governance risk (inverse of score)
        gov_risk_score = 100 - esg_metrics.governance_score
        factors['governance_risk'] = RiskFactor(
            name="Governance Risk",
            category=RiskCategory.ESG,
            weight=0.3,
            current_score=gov_risk_score,
            benchmark_score=30.0
        )
        
        return factors
    
    async def _assess_industry_risks(self, inputs: RiskAssessmentInputs) -> Dict[str, RiskFactor]:
        """Assess industry-specific risks"""
        factors = {}
        
        # Get industry-specific risk model
        if inputs.industry in self.industry_models:
            industry_model = self.industry_models[inputs.industry]
            
            # Apply industry-specific risk calculations
            for risk_name, risk_config in industry_model.items():
                # This would be implemented based on specific industry characteristics
                risk_score = self._calculate_industry_specific_risk(
                    inputs, risk_name, risk_config
                )
                
                factors[f'industry_{risk_name}'] = RiskFactor(
                    name=f"Industry {risk_name.title()} Risk",
                    category=RiskCategory.INDUSTRY_SPECIFIC,
                    weight=risk_config.get('weight', 0.2),
                    current_score=risk_score,
                    benchmark_score=risk_config.get('benchmark', 40.0)
                )
        
        return factors
    
    def _calculate_industry_specific_risk(
        self, 
        inputs: RiskAssessmentInputs, 
        risk_name: str, 
        risk_config: Dict
    ) -> float:
        """Calculate industry-specific risk score"""
        
        # Placeholder implementation - would be customized per industry
        if inputs.industry == 'Technology':
            if risk_name == 'disruption':
                # Technology disruption risk
                return 60.0  # Higher risk in tech
            elif risk_name == 'talent':
                # Talent retention risk
                return 45.0
        elif inputs.industry == 'Healthcare':
            if risk_name == 'regulatory':
                # FDA approval risks
                return 70.0
            elif risk_name == 'reimbursement':
                # Insurance reimbursement risk
                return 55.0
        
        return 40.0  # Default risk score
    
    async def _calculate_category_scores(
        self, 
        risk_factors: Dict[str, RiskFactor],
        inputs: RiskAssessmentInputs
    ) -> Dict[RiskCategory, float]:
        """Calculate risk scores by category"""
        
        category_scores = {}
        
        for category in RiskCategory:
            # Get factors for this category
            category_factors = [
                factor for factor in risk_factors.values() 
                if factor.category == category
            ]
            
            if not category_factors:
                category_scores[category] = 50.0  # Default neutral score
                continue
            
            # Calculate weighted average
            total_weight = sum(factor.weight for factor in category_factors)
            if total_weight == 0:
                category_scores[category] = 50.0
                continue
            
            weighted_score = sum(
                factor.current_score * factor.weight for factor in category_factors
            ) / total_weight
            
            category_scores[category] = weighted_score
        
        return category_scores
    
    async def _calculate_composite_risk_score(
        self, 
        category_scores: Dict[RiskCategory, float],
        industry: str
    ) -> float:
        """Calculate composite risk score"""
        
        # Get industry-adjusted weights
        weights = self._get_industry_adjusted_weights(industry)
        
        # Calculate weighted composite score
        total_weight = sum(weights.values())
        composite_score = sum(
            score * weights.get(category, 0) 
            for category, score in category_scores.items()
        ) / total_weight
        
        return composite_score
    
    def _get_industry_adjusted_weights(self, industry: str) -> Dict[RiskCategory, float]:
        """Get industry-adjusted category weights"""
        
        # Industry-specific weight adjustments
        industry_adjustments = {
            'Technology': {
                RiskCategory.OPERATIONAL: 1.2,  # Higher operational risk weight
                RiskCategory.REGULATORY: 0.8,   # Lower regulatory risk weight
                RiskCategory.ESG: 1.1          # Slightly higher ESG weight
            },
            'Healthcare': {
                RiskCategory.REGULATORY: 1.4,   # Much higher regulatory risk
                RiskCategory.OPERATIONAL: 0.9,  # Slightly lower operational risk
                RiskCategory.ESG: 1.2          # Higher ESG weight
            },
            'Financial Services': {
                RiskCategory.REGULATORY: 1.3,   # Higher regulatory risk
                RiskCategory.FINANCIAL: 1.2,    # Higher financial risk
                RiskCategory.MARKET: 1.1       # Higher market risk
            }
        }
        
        # Start with default weights
        adjusted_weights = self.default_category_weights.copy()
        
        # Apply industry adjustments
        if industry in industry_adjustments:
            adjustments = industry_adjustments[industry]
            for category, adjustment in adjustments.items():
                if category in adjusted_weights:
                    adjusted_weights[category] *= adjustment
        
        # Normalize weights to sum to 1.0
        total_weight = sum(adjusted_weights.values())
        adjusted_weights = {
            category: weight / total_weight 
            for category, weight in adjusted_weights.items()
        }
        
        return adjusted_weights
    
    async def _perform_peer_comparison(
        self,
        inputs: RiskAssessmentInputs,
        category_scores: Dict[RiskCategory, float],
        peer_companies: Optional[List[Dict]]
    ) -> Dict[str, Any]:
        """Perform peer risk comparison"""
        
        if not peer_companies:
            return {'percentiles': {}, 'relative_scores': {}}
        
        peer_comparison = {
            'percentiles': {},
            'relative_scores': {},
            'peer_stats': {}
        }
        
        # Calculate percentiles for each category
        for category, score in category_scores.items():
            # Simulate peer scores (in practice, would use actual peer data)
            peer_scores = [
                np.random.normal(50, 15) for _ in peer_companies
            ]  # Placeholder - would use actual peer risk scores
            
            # Calculate percentile
            percentile = stats.percentileofscore(peer_scores, score)
            peer_comparison['percentiles'][category.value] = percentile
            
            # Calculate relative score
            peer_mean = np.mean(peer_scores)
            relative_score = (score - peer_mean) / peer_mean * 100 if peer_mean != 0 else 0
            peer_comparison['relative_scores'][category.value] = relative_score
            
            # Peer statistics
            peer_comparison['peer_stats'][category.value] = {
                'mean': peer_mean,
                'median': np.median(peer_scores),
                'std': np.std(peer_scores),
                'min': np.min(peer_scores),
                'max': np.max(peer_scores)
            }
        
        return peer_comparison
    
    async def _analyze_esg_risks(self, esg_metrics: ESGMetrics) -> Dict[str, Any]:
        """Analyze ESG risk contributions and mitigations"""
        
        esg_analysis = {
            'risk_contribution': 0.0,
            'risk_mitigation': 0.0,
            'esg_risk_factors': {},
            'improvement_opportunities': []
        }
        
        # Calculate ESG risk contribution (higher ESG scores reduce risk)
        esg_risk_score = 100 - esg_metrics.esg_score
        esg_analysis['risk_contribution'] = esg_risk_score
        
        # Risk mitigation from strong ESG
        if esg_metrics.esg_score > 70:
            esg_analysis['risk_mitigation'] = (esg_metrics.esg_score - 50) * 0.3
        
        # Identify specific ESG risk factors
        if esg_metrics.environmental_score < 40:
            esg_analysis['esg_risk_factors']['environmental'] = 'High environmental risk'
        if esg_metrics.social_score < 40:
            esg_analysis['esg_risk_factors']['social'] = 'High social risk'
        if esg_metrics.governance_score < 40:
            esg_analysis['esg_risk_factors']['governance'] = 'High governance risk'
        
        # Improvement opportunities
        if esg_metrics.environmental_score < 60:
            esg_analysis['improvement_opportunities'].append('Improve environmental practices')
        if esg_metrics.social_score < 60:
            esg_analysis['improvement_opportunities'].append('Enhance social responsibility')
        if esg_metrics.governance_score < 60:
            esg_analysis['improvement_opportunities'].append('Strengthen governance structures')
        
        return esg_analysis
    
    async def _perform_stress_testing(
        self,
        inputs: RiskAssessmentInputs,
        risk_factors: Dict[str, RiskFactor]
    ) -> Dict[str, Dict[str, float]]:
        """Perform comprehensive stress testing"""
        
        stress_scenarios = {
            'market_crash': {
                'market_beta_shock': 1.5,
                'volatility_shock': 2.0,
                'liquidity_shock': 1.8
            },
            'recession': {
                'revenue_shock': 0.7,
                'margin_shock': 0.8,
                'credit_shock': 1.5
            },
            'sector_crisis': {
                'industry_shock': 2.0,
                'regulatory_shock': 1.6,
                'reputation_shock': 1.4
            },
            'company_specific': {
                'operational_shock': 1.8,
                'management_shock': 1.5,
                'technology_shock': 1.3
            }
        }
        
        stress_results = {}
        
        for scenario_name, shocks in stress_scenarios.items():
            scenario_results = {}
            
            # Apply stress multipliers to relevant risk factors
            for risk_name, risk_factor in risk_factors.items():
                stressed_score = risk_factor.current_score
                
                # Apply relevant shocks
                for shock_type, multiplier in shocks.items():
                    if self._is_shock_relevant(risk_name, shock_type):
                        stressed_score *= multiplier
                
                # Cap at 100
                stressed_score = min(100, stressed_score)
                scenario_results[risk_name] = stressed_score
            
            # Calculate composite stress score
            stressed_categories = {}
            for category in RiskCategory:
                category_factors = [
                    (risk_name, score) for risk_name, score in scenario_results.items()
                    if risk_factors[risk_name].category == category
                ]
                
                if category_factors:
                    # Weighted average
                    total_weight = sum(risk_factors[risk_name].weight for risk_name, _ in category_factors)
                    weighted_score = sum(
                        score * risk_factors[risk_name].weight 
                        for risk_name, score in category_factors
                    ) / total_weight if total_weight > 0 else 50.0
                    
                    stressed_categories[category.value] = weighted_score
            
            # Calculate stressed composite score
            weights = self._get_industry_adjusted_weights(inputs.industry)
            stressed_composite = sum(
                score * weights.get(RiskCategory(category), 0)
                for category, score in stressed_categories.items()
            ) / sum(weights.values())
            
            stressed_categories['composite'] = stressed_composite
            stress_results[scenario_name] = stressed_categories
        
        return stress_results
    
    def _is_shock_relevant(self, risk_name: str, shock_type: str) -> bool:
        """Determine if a shock is relevant to a risk factor"""
        
        relevance_map = {
            'market_beta_shock': ['systematic_risk', 'volatility_risk'],
            'volatility_shock': ['volatility_risk', 'market_risk'],
            'liquidity_shock': ['liquidity_risk', 'financial_liquidity'],
            'revenue_shock': ['revenue_concentration', 'earnings_volatility'],
            'margin_shock': ['operating_leverage', 'earnings_volatility'],
            'credit_shock': ['leverage_risk', 'interest_coverage_risk'],
            'industry_shock': ['industry_' in risk_name],
            'regulatory_shock': ['regulatory' in risk_name, 'compliance_risk'],
            'reputation_shock': ['social_risk', 'governance_risk'],
            'operational_shock': ['operational' in risk_name],
            'management_shock': ['management_risk', 'governance_risk'],
            'technology_shock': ['technology_risk', 'cybersecurity_risk']
        }
        
        if shock_type in relevance_map:
            relevant_patterns = relevance_map[shock_type]
            return any(
                pattern in risk_name if isinstance(pattern, str) else pattern
                for pattern in relevant_patterns
            )
        
        return False
    
    async def _perform_scenario_analysis(self, inputs: RiskAssessmentInputs) -> Dict[str, Any]:
        """Perform scenario analysis"""
        
        # Economic scenarios
        scenarios = {
            'bull_market': {
                'market_conditions': 'bull',
                'risk_adjustment': 0.85
            },
            'bear_market': {
                'market_conditions': 'bear',
                'risk_adjustment': 1.25
            },
            'recession': {
                'economic_cycle': 'recession',
                'risk_adjustment': 1.4
            },
            'expansion': {
                'economic_cycle': 'expansion',
                'risk_adjustment': 0.9
            }
        }
        
        scenario_results = {}
        
        for scenario_name, scenario_params in scenarios.items():
            # Apply scenario adjustments
            adjusted_inputs = self._create_scenario_inputs(inputs, scenario_params)
            
            # Recalculate risk (simplified for this example)
            risk_adjustment = scenario_params['risk_adjustment']
            
            scenario_results[scenario_name] = {
                'risk_multiplier': risk_adjustment,
                'expected_risk_score': inputs.market_risks.beta * 50 * risk_adjustment  # Simplified
            }
        
        return scenario_results
    
    def _create_scenario_inputs(
        self, 
        base_inputs: RiskAssessmentInputs, 
        scenario_params: Dict[str, Any]
    ) -> RiskAssessmentInputs:
        """Create scenario-adjusted inputs"""
        
        # In practice, this would create comprehensive scenario adjustments
        # For now, return base inputs with market conditions updated
        import copy
        scenario_inputs = copy.deepcopy(base_inputs)
        scenario_inputs.market_conditions = scenario_params.get('market_conditions', base_inputs.market_conditions)
        scenario_inputs.economic_cycle = scenario_params.get('economic_cycle', base_inputs.economic_cycle)
        
        return scenario_inputs
    
    async def _predict_risk_trajectory(
        self,
        inputs: RiskAssessmentInputs,
        risk_factors: Dict[str, RiskFactor]
    ) -> Dict[str, Any]:
        """Predict risk trajectory and probability of distress"""
        
        # Calculate risk trend
        trend_indicators = []
        for factor in risk_factors.values():
            if factor.trend != 0:
                trend_indicators.append(factor.trend)
        
        if trend_indicators:
            overall_trend = np.mean(trend_indicators)
            if overall_trend > 0.1:
                trajectory = "deteriorating"
            elif overall_trend < -0.1:
                trajectory = "improving"
            else:
                trajectory = "stable"
        else:
            trajectory = "stable"
        
        # Calculate probability of distress (simplified model)
        composite_risk = np.mean([factor.current_score for factor in risk_factors.values()])
        
        # Logistic function for probability of distress
        distress_probability = 1 / (1 + np.exp(-(composite_risk - 60) / 10))
        
        # Expected risk-adjusted return
        base_return = 0.10  # 10% base return
        risk_adjustment = (composite_risk - 50) / 50 * 0.05  # Adjust by +/- 5%
        expected_return = base_return - risk_adjustment
        
        # Risk premium
        risk_premium = max(0, composite_risk / 100 * 0.08)  # Up to 8% risk premium
        
        return {
            'trajectory': trajectory,
            'probability_of_distress': distress_probability,
            'expected_risk_adjusted_return': expected_return,
            'risk_premium': risk_premium,
            'trend_score': overall_trend if trend_indicators else 0.0
        }
    
    async def _generate_risk_recommendations(
        self,
        risk_factors: Dict[str, RiskFactor],
        category_scores: Dict[RiskCategory, float]
    ) -> Dict[str, List[str]]:
        """Generate risk mitigation recommendations"""
        
        recommendations = {
            'risk_mitigation_priorities': [],
            'monitoring_indicators': []
        }
        
        # Identify top risk factors
        sorted_risks = sorted(
            risk_factors.items(),
            key=lambda x: x[1].current_score,
            reverse=True
        )
        
        top_risks = sorted_risks[:5]  # Top 5 risks
        
        for risk_name, risk_factor in top_risks:
            if risk_factor.current_score > 70:  # High risk
                if 'leverage' in risk_name:
                    recommendations['risk_mitigation_priorities'].append(
                        'Reduce financial leverage through debt reduction'
                    )
                elif 'liquidity' in risk_name:
                    recommendations['risk_mitigation_priorities'].append(
                        'Improve liquidity through cash management'
                    )
                elif 'concentration' in risk_name:
                    recommendations['risk_mitigation_priorities'].append(
                        'Diversify revenue sources and customer base'
                    )
                elif 'regulatory' in risk_name:
                    recommendations['risk_mitigation_priorities'].append(
                        'Strengthen regulatory compliance and monitoring'
                    )
                elif 'esg' in risk_name or 'environmental' in risk_name:
                    recommendations['risk_mitigation_priorities'].append(
                        'Improve ESG practices and sustainability'
                    )
        
        # Monitoring indicators
        recommendations['monitoring_indicators'] = [
            'Debt-to-equity ratio trends',
            'Cash flow volatility',
            'Customer concentration metrics',
            'Regulatory environment changes',
            'ESG score improvements',
            'Market volatility measures'
        ]
        
        return recommendations
    
    async def _compile_risk_results(
        self,
        composite_score: float,
        category_scores: Dict[RiskCategory, float],
        risk_factors: Dict[str, RiskFactor],
        peer_comparison: Dict[str, Any],
        esg_analysis: Dict[str, Any],
        stress_results: Dict[str, Dict[str, float]],
        risk_prediction: Dict[str, Any],
        recommendations: Dict[str, List[str]]
    ) -> RiskAssessmentResults:
        """Compile comprehensive risk assessment results"""
        
        # Risk grade mapping
        if composite_score <= 20:
            risk_grade = "AAA"
            risk_category = "Very Low"
        elif composite_score <= 30:
            risk_grade = "AA"
            risk_category = "Low"
        elif composite_score <= 40:
            risk_grade = "A"
            risk_category = "Medium-Low"
        elif composite_score <= 55:
            risk_grade = "BBB"
            risk_category = "Medium"
        elif composite_score <= 70:
            risk_grade = "BB"
            risk_category = "Medium-High"
        elif composite_score <= 85:
            risk_grade = "B"
            risk_category = "High"
        else:
            risk_grade = "CCC"
            risk_category = "Very High"
        
        # Top risk factors
        top_risks = sorted(
            risk_factors.items(),
            key=lambda x: x[1].current_score,
            reverse=True
        )[:10]
        
        # Risk drivers (categories with scores > 60)
        risk_drivers = [
            category.value for category, score in category_scores.items()
            if score > 60
        ]
        
        return RiskAssessmentResults(
            composite_risk_score=composite_score,
            risk_grade=risk_grade,
            risk_category=risk_category,
            category_scores=category_scores,
            category_percentiles=peer_comparison.get('percentiles', {}),
            risk_factor_scores=risk_factors,
            top_risk_factors=[(name, factor.current_score) for name, factor in top_risks],
            risk_drivers=risk_drivers,
            industry_comparison={},  # Placeholder
            peer_comparison=peer_comparison.get('relative_scores', {}),
            historical_trend={},  # Placeholder
            risk_trajectory=risk_prediction.get('trajectory', 'stable'),
            probability_of_distress=risk_prediction.get('probability_of_distress', 0.0),
            expected_risk_adjusted_return=risk_prediction.get('expected_risk_adjusted_return', 0.0),
            risk_premium=risk_prediction.get('risk_premium', 0.0),
            esg_risk_contribution=esg_analysis.get('risk_contribution', 0.0),
            esg_risk_mitigation=esg_analysis.get('risk_mitigation', 0.0),
            stress_test_results=stress_results,
            risk_sensitivity={},  # Placeholder
            risk_mitigation_priorities=recommendations.get('risk_mitigation_priorities', []),
            monitoring_indicators=recommendations.get('monitoring_indicators', [])
        )
    
    def _initialize_industry_models(self):
        """Initialize industry-specific risk models"""
        
        self.industry_models = {
            'Technology': {
                'disruption': {'weight': 0.3, 'benchmark': 60.0},
                'talent': {'weight': 0.25, 'benchmark': 45.0},
                'ip_protection': {'weight': 0.2, 'benchmark': 40.0},
                'platform': {'weight': 0.15, 'benchmark': 35.0},
                'scalability': {'weight': 0.1, 'benchmark': 30.0}
            },
            'Healthcare': {
                'regulatory': {'weight': 0.35, 'benchmark': 70.0},
                'reimbursement': {'weight': 0.25, 'benchmark': 55.0},
                'clinical_trial': {'weight': 0.2, 'benchmark': 65.0},
                'ip_patent': {'weight': 0.15, 'benchmark': 50.0},
                'manufacturing': {'weight': 0.05, 'benchmark': 40.0}
            },
            'Financial Services': {
                'credit': {'weight': 0.3, 'benchmark': 50.0},
                'interest_rate': {'weight': 0.25, 'benchmark': 45.0},
                'regulatory': {'weight': 0.2, 'benchmark': 55.0},
                'operational': {'weight': 0.15, 'benchmark': 40.0},
                'reputation': {'weight': 0.1, 'benchmark': 35.0}
            }
        }

# Factory function
def create_multi_factor_risk_model(**kwargs) -> MultiFactorRiskModel:
    """Factory function for creating multi-factor risk models"""
    return MultiFactorRiskModel(**kwargs)

# Utility functions for risk scoring
def calculate_z_score_risk(value: float, benchmark: float, std: float) -> float:
    """Calculate risk score using Z-score normalization"""
    if std == 0:
        return 50.0
    
    z_score = (value - benchmark) / std
    # Convert to 0-100 scale (50 = benchmark)
    risk_score = 50 + z_score * 15  # 1 std dev = 15 points
    return np.clip(risk_score, 0, 100)

def risk_grade_to_numeric(grade: str) -> float:
    """Convert risk grade to numeric score"""
    grade_mapping = {
        'AAA': 15, 'AA': 25, 'A': 35, 'BBB': 50,
        'BB': 65, 'B': 80, 'CCC': 90, 'CC': 95, 'C': 98, 'D': 100
    }
    return grade_mapping.get(grade, 50)

def numeric_to_risk_grade(score: float) -> str:
    """Convert numeric score to risk grade"""
    if score <= 20: return 'AAA'
    elif score <= 30: return 'AA'
    elif score <= 40: return 'A'
    elif score <= 55: return 'BBB'
    elif score <= 70: return 'BB'
    elif score <= 85: return 'B'
    else: return 'CCC'