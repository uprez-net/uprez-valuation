"""
Enhanced Comparable Company Analysis (CCA) with ML-powered peer selection
Advanced statistical modeling and regression-based multiple prediction
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor
import joblib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class CompanyFinancialData:
    """Financial data for a single company"""
    company_name: str
    ticker: Optional[str] = None
    sector: str = ""
    industry: str = ""
    
    # Market data
    market_cap: float = 0
    enterprise_value: float = 0
    share_price: float = 0
    shares_outstanding: float = 0
    
    # Financial metrics (TTM or most recent)
    revenue: float = 0
    ebitda: float = 0
    ebit: float = 0
    net_income: float = 0
    free_cash_flow: float = 0
    total_debt: float = 0
    cash: float = 0
    
    # Growth metrics (historical)
    revenue_growth_1y: float = 0
    revenue_growth_3y: float = 0
    revenue_growth_5y: float = 0
    ebitda_growth_1y: float = 0
    ebitda_growth_3y: float = 0
    
    # Profitability ratios
    ebitda_margin: float = 0
    net_margin: float = 0
    roe: float = 0
    roic: float = 0
    
    # Valuation multiples
    ev_revenue: float = 0
    ev_ebitda: float = 0
    pe_ratio: float = 0
    peg_ratio: float = 0
    
    # Risk metrics
    beta: float = 1.0
    debt_to_equity: float = 0
    interest_coverage: float = 0
    
    # Size and scale metrics
    employee_count: Optional[int] = None
    geographic_presence: List[str] = field(default_factory=list)
    business_model: str = ""
    
    # Data quality indicators
    data_quality_score: float = 1.0
    last_updated: Optional[datetime] = None

@dataclass
class PeerSelectionCriteria:
    """Criteria for peer company selection"""
    target_company: str
    
    # Industry filters
    same_sector_required: bool = True
    same_industry_preferred: bool = True
    
    # Size filters
    size_multiple_range: Tuple[float, float] = (0.5, 2.0)  # 0.5x to 2x revenue
    market_cap_range: Optional[Tuple[float, float]] = None
    
    # Geographic filters
    same_geography_preferred: bool = False
    exclude_regions: List[str] = field(default_factory=list)
    
    # Business model filters
    same_business_model_preferred: bool = True
    exclude_business_models: List[str] = field(default_factory=list)
    
    # Quality filters
    min_data_quality: float = 0.7
    min_trading_history: int = 252  # Days
    exclude_distressed: bool = True
    
    # Statistical filters
    max_statistical_distance: float = 2.0
    min_correlation: float = 0.3
    
    # Output controls
    max_peers: int = 15
    min_peers: int = 5

@dataclass
class CCAModelResults:
    """Results from CCA model analysis"""
    # Target company info
    target_company: str
    
    # Selected peers
    peer_companies: List[CompanyFinancialData]
    peer_selection_score: float
    
    # Valuation estimates
    estimated_multiples: Dict[str, float]
    implied_valuations: Dict[str, float]
    
    # Statistical analysis
    multiple_statistics: Dict[str, Dict[str, float]]
    regression_results: Dict[str, Any]
    
    # Confidence metrics
    confidence_intervals: Dict[str, Tuple[float, float]]
    prediction_intervals: Dict[str, Tuple[float, float]]
    model_r_squared: Dict[str, float]
    
    # Risk assessment
    valuation_range: Tuple[float, float]
    risk_adjusted_value: float
    outlier_adjusted_value: float
    
    # Peer analysis
    peer_cluster_analysis: Dict[str, Any]
    similarity_scores: Dict[str, float]
    
    # Time series analysis
    multiple_trends: Dict[str, List[float]]
    seasonality_analysis: Dict[str, Any]

class EnhancedCCAModel:
    """
    Enhanced Comparable Company Analysis with ML-powered features
    
    Features:
    - ML-powered peer selection using clustering and similarity metrics
    - Statistical outlier detection and removal
    - Regression-based multiple prediction with confidence intervals
    - Time-series analysis of multiples
    - Cross-validation and model validation
    - Risk-adjusted valuations
    """
    
    def __init__(self, use_ml_clustering: bool = True, validation_method: str = 'cross_validation'):
        self.use_ml_clustering = use_ml_clustering
        self.validation_method = validation_method
        self.scaler = StandardScaler()
        self.outlier_detector = IsolationForest(contamination=0.1, random_state=42)
        self.peer_selector_model = None
        self.multiple_predictors = {}
        
        # Standard multiples to analyze
        self.key_multiples = [
            'ev_revenue', 'ev_ebitda', 'pe_ratio', 'peg_ratio'
        ]
        
        # Features for peer selection
        self.peer_selection_features = [
            'revenue', 'ebitda_margin', 'revenue_growth_3y', 
            'roic', 'debt_to_equity', 'beta'
        ]
    
    async def analyze_comparable_companies(
        self,
        target_company: CompanyFinancialData,
        universe_companies: List[CompanyFinancialData],
        selection_criteria: PeerSelectionCriteria,
        include_time_series: bool = True,
        include_regression_analysis: bool = True
    ) -> CCAModelResults:
        """
        Comprehensive comparable company analysis
        """
        try:
            # Data preprocessing and quality checks
            cleaned_universe = await self._preprocess_universe_data(universe_companies)
            
            # Select peer companies using ML
            selected_peers = await self._ml_peer_selection(
                target_company, cleaned_universe, selection_criteria
            )
            
            # Statistical analysis of multiples
            multiple_stats = await self._analyze_multiples_statistics(selected_peers)
            
            # Regression-based predictions
            regression_results = {}
            if include_regression_analysis:
                regression_results = await self._regression_based_multiple_prediction(
                    target_company, selected_peers
                )
            
            # Time series analysis
            time_series_results = {}
            if include_time_series:
                time_series_results = await self._time_series_multiple_analysis(selected_peers)
            
            # Calculate implied valuations
            implied_valuations = await self._calculate_implied_valuations(
                target_company, multiple_stats, regression_results
            )
            
            # Risk assessment and adjustments
            risk_assessment = await self._risk_adjusted_valuation(
                target_company, selected_peers, implied_valuations
            )
            
            # Compile comprehensive results
            return await self._compile_cca_results(
                target_company, selected_peers, multiple_stats, 
                regression_results, implied_valuations, risk_assessment,
                time_series_results
            )
            
        except Exception as e:
            logger.error(f"CCA analysis failed: {str(e)}")
            raise
    
    async def _preprocess_universe_data(
        self, 
        universe_companies: List[CompanyFinancialData]
    ) -> List[CompanyFinancialData]:
        """Preprocess and clean universe data"""
        cleaned_companies = []
        
        for company in universe_companies:
            # Data quality checks
            if company.data_quality_score < 0.5:
                continue
                
            # Remove companies with missing critical data
            if (company.revenue <= 0 or company.market_cap <= 0 or 
                company.enterprise_value <= 0):
                continue
                
            # Calculate derived metrics if missing
            if company.ev_revenue == 0 and company.enterprise_value > 0:
                company.ev_revenue = company.enterprise_value / company.revenue
                
            if company.ev_ebitda == 0 and company.enterprise_value > 0 and company.ebitda > 0:
                company.ev_ebitda = company.enterprise_value / company.ebitda
                
            if company.ebitda_margin == 0 and company.revenue > 0:
                company.ebitda_margin = company.ebitda / company.revenue
                
            # Add to cleaned list
            cleaned_companies.append(company)
        
        logger.info(f"Cleaned universe: {len(cleaned_companies)} companies from {len(universe_companies)}")
        return cleaned_companies
    
    async def _ml_peer_selection(
        self,
        target_company: CompanyFinancialData,
        universe_companies: List[CompanyFinancialData],
        criteria: PeerSelectionCriteria
    ) -> List[CompanyFinancialData]:
        """ML-powered peer company selection"""
        
        # Step 1: Apply basic filters
        filtered_companies = await self._apply_basic_filters(
            target_company, universe_companies, criteria
        )
        
        if len(filtered_companies) <= criteria.max_peers:
            return filtered_companies
        
        # Step 2: Feature-based similarity analysis
        if self.use_ml_clustering:
            similar_companies = await self._cluster_based_selection(
                target_company, filtered_companies, criteria
            )
        else:
            similar_companies = await self._distance_based_selection(
                target_company, filtered_companies, criteria
            )
        
        # Step 3: Final ranking and selection
        final_peers = await self._rank_and_select_peers(
            target_company, similar_companies, criteria
        )
        
        return final_peers
    
    async def _apply_basic_filters(
        self,
        target_company: CompanyFinancialData,
        universe_companies: List[CompanyFinancialData],
        criteria: PeerSelectionCriteria
    ) -> List[CompanyFinancialData]:
        """Apply basic filtering criteria"""
        filtered = []
        
        for company in universe_companies:
            # Skip self
            if company.company_name == target_company.company_name:
                continue
                
            # Sector filter
            if criteria.same_sector_required and company.sector != target_company.sector:
                continue
                
            # Size filters
            revenue_ratio = company.revenue / target_company.revenue if target_company.revenue > 0 else 0
            if not (criteria.size_multiple_range[0] <= revenue_ratio <= criteria.size_multiple_range[1]):
                continue
                
            # Market cap filter
            if criteria.market_cap_range:
                if not (criteria.market_cap_range[0] <= company.market_cap <= criteria.market_cap_range[1]):
                    continue
                    
            # Data quality filter
            if company.data_quality_score < criteria.min_data_quality:
                continue
                
            # Exclude distressed companies
            if criteria.exclude_distressed:
                if (company.debt_to_equity > 5 or company.interest_coverage < 1.5 or
                    company.ebitda_margin < -0.1):
                    continue
            
            filtered.append(company)
        
        return filtered
    
    async def _cluster_based_selection(
        self,
        target_company: CompanyFinancialData,
        companies: List[CompanyFinancialData],
        criteria: PeerSelectionCriteria
    ) -> List[CompanyFinancialData]:
        """Use clustering to identify similar companies"""
        
        # Prepare feature matrix
        features_data = []
        company_list = [target_company] + companies
        
        for company in company_list:
            features = []
            for feature in self.peer_selection_features:
                value = getattr(company, feature, 0)
                features.append(value)
            features_data.append(features)
        
        # Scale features
        features_array = np.array(features_data)
        scaled_features = self.scaler.fit_transform(features_array)
        
        # Perform clustering
        n_clusters = min(5, len(company_list) // 3)  # Adaptive cluster count
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Find target company's cluster
        target_cluster = cluster_labels[0]
        
        # Select companies from same cluster
        similar_companies = []
        for i, (company, label) in enumerate(zip(company_list[1:], cluster_labels[1:]), 1):
            if label == target_cluster:
                similar_companies.append(company)
        
        # If too few in same cluster, add nearest companies from other clusters
        if len(similar_companies) < criteria.min_peers:
            # Calculate distances to all companies
            target_features = scaled_features[0].reshape(1, -1)
            distances = np.linalg.norm(scaled_features[1:] - target_features, axis=1)
            
            # Sort by distance and add closest companies
            sorted_indices = np.argsort(distances)
            for idx in sorted_indices:
                company = companies[idx]
                if company not in similar_companies:
                    similar_companies.append(company)
                    if len(similar_companies) >= criteria.max_peers:
                        break
        
        return similar_companies[:criteria.max_peers]
    
    async def _distance_based_selection(
        self,
        target_company: CompanyFinancialData,
        companies: List[CompanyFinancialData],
        criteria: PeerSelectionCriteria
    ) -> List[CompanyFinancialData]:
        """Distance-based peer selection"""
        
        # Calculate similarity scores
        similarity_scores = []
        
        for company in companies:
            score = self._calculate_similarity_score(target_company, company)
            similarity_scores.append((company, score))
        
        # Sort by similarity (higher is better)
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top companies
        selected_companies = [company for company, score in similarity_scores[:criteria.max_peers]]
        
        return selected_companies
    
    def _calculate_similarity_score(
        self, 
        target: CompanyFinancialData, 
        candidate: CompanyFinancialData
    ) -> float:
        """Calculate similarity score between two companies"""
        
        weights = {
            'industry_match': 0.25,
            'size_similarity': 0.20,
            'profitability_similarity': 0.20,
            'growth_similarity': 0.15,
            'risk_similarity': 0.10,
            'business_model_match': 0.10
        }
        
        score = 0.0
        
        # Industry match
        if candidate.industry == target.industry:
            score += weights['industry_match']
        elif candidate.sector == target.sector:
            score += weights['industry_match'] * 0.5
        
        # Size similarity (log scale)
        if target.revenue > 0 and candidate.revenue > 0:
            log_ratio = abs(np.log(candidate.revenue) - np.log(target.revenue))
            size_sim = max(0, 1 - log_ratio / 2)  # Normalized to 0-1
            score += weights['size_similarity'] * size_sim
        
        # Profitability similarity
        margin_diff = abs(candidate.ebitda_margin - target.ebitda_margin)
        prof_sim = max(0, 1 - margin_diff / 0.5)  # Normalized assuming 50% max difference
        score += weights['profitability_similarity'] * prof_sim
        
        # Growth similarity
        growth_diff = abs(candidate.revenue_growth_3y - target.revenue_growth_3y)
        growth_sim = max(0, 1 - growth_diff / 1.0)  # Normalized assuming 100% max difference
        score += weights['growth_similarity'] * growth_sim
        
        # Risk similarity (beta and leverage)
        beta_diff = abs(candidate.beta - target.beta)
        leverage_diff = abs(candidate.debt_to_equity - target.debt_to_equity)
        risk_sim = max(0, 1 - (beta_diff / 2 + leverage_diff / 5) / 2)
        score += weights['risk_similarity'] * risk_sim
        
        # Business model match
        if candidate.business_model == target.business_model:
            score += weights['business_model_match']
        
        return score
    
    async def _rank_and_select_peers(
        self,
        target_company: CompanyFinancialData,
        candidates: List[CompanyFinancialData],
        criteria: PeerSelectionCriteria
    ) -> List[CompanyFinancialData]:
        """Final ranking and selection of peer companies"""
        
        # Calculate comprehensive scores
        scored_candidates = []
        
        for candidate in candidates:
            # Base similarity score
            similarity_score = self._calculate_similarity_score(target_company, candidate)
            
            # Adjust for data quality
            quality_adjustment = candidate.data_quality_score
            
            # Adjust for liquidity/trading activity (if available)
            liquidity_adjustment = 1.0  # Placeholder
            
            # Combined score
            final_score = similarity_score * quality_adjustment * liquidity_adjustment
            scored_candidates.append((candidate, final_score))
        
        # Sort by final score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select top candidates within range
        min_peers = max(criteria.min_peers, 3)
        max_peers = min(criteria.max_peers, len(scored_candidates))
        
        selected_count = max(min_peers, min(max_peers, len(scored_candidates)))
        selected_peers = [candidate for candidate, score in scored_candidates[:selected_count]]
        
        return selected_peers
    
    async def _analyze_multiples_statistics(
        self, 
        peer_companies: List[CompanyFinancialData]
    ) -> Dict[str, Dict[str, float]]:
        """Comprehensive statistical analysis of valuation multiples"""
        
        multiple_stats = {}
        
        for multiple in self.key_multiples:
            values = []
            
            # Collect multiple values
            for company in peer_companies:
                value = getattr(company, multiple, 0)
                if value > 0:  # Exclude negative/zero multiples
                    values.append(value)
            
            if not values:
                continue
                
            values = np.array(values)
            
            # Remove outliers using IQR method
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            clean_values = values[(values >= lower_bound) & (values <= upper_bound)]
            
            # Calculate comprehensive statistics
            stats_dict = {
                'count': len(clean_values),
                'mean': np.mean(clean_values),
                'median': np.median(clean_values),
                'std': np.std(clean_values),
                'min': np.min(clean_values),
                'max': np.max(clean_values),
                'q25': np.percentile(clean_values, 25),
                'q75': np.percentile(clean_values, 75),
                'skewness': stats.skew(clean_values),
                'kurtosis': stats.kurtosis(clean_values),
                'cv': np.std(clean_values) / np.mean(clean_values) if np.mean(clean_values) != 0 else 0
            }
            
            # Confidence intervals
            confidence_interval = stats.t.interval(
                0.95, len(clean_values) - 1, 
                loc=np.mean(clean_values), 
                scale=stats.sem(clean_values)
            )
            
            stats_dict['ci_lower'] = confidence_interval[0]
            stats_dict['ci_upper'] = confidence_interval[1]
            
            multiple_stats[multiple] = stats_dict
        
        return multiple_stats
    
    async def _regression_based_multiple_prediction(
        self,
        target_company: CompanyFinancialData,
        peer_companies: List[CompanyFinancialData]
    ) -> Dict[str, Any]:
        """Regression-based multiple prediction with confidence intervals"""
        
        regression_results = {}
        
        # Prepare feature matrix
        feature_names = [
            'revenue_growth_3y', 'ebitda_margin', 'roic', 
            'debt_to_equity', 'beta', 'revenue'
        ]
        
        # Build dataset
        X_data = []
        y_data = {}
        
        # Initialize y_data dictionaries
        for multiple in self.key_multiples:
            y_data[multiple] = []
        
        for company in peer_companies:
            features = []
            for feature in feature_names:
                value = getattr(company, feature, 0)
                features.append(value)
            
            # Only add if we have valid features
            if any(f != 0 for f in features):
                X_data.append(features)
                
                # Add multiple values
                for multiple in self.key_multiples:
                    multiple_value = getattr(company, multiple, 0)
                    y_data[multiple].append(multiple_value)
        
        if len(X_data) < 3:  # Need minimum observations
            return {}
        
        X = np.array(X_data)
        
        # Standardize features
        X_scaled = StandardScaler().fit_transform(X)
        
        # Train models for each multiple
        for multiple in self.key_multiples:
            y = np.array(y_data[multiple])
            
            # Remove zero/negative values
            valid_mask = y > 0
            if np.sum(valid_mask) < 3:
                continue
                
            X_valid = X_scaled[valid_mask]
            y_valid = y[valid_mask]
            
            try:
                # Use Ridge regression with cross-validation
                ridge = Ridge(alpha=1.0)
                
                # Cross-validation scores
                cv_scores = cross_val_score(ridge, X_valid, y_valid, cv=min(5, len(y_valid)), 
                                          scoring='r2')
                
                # Fit model
                ridge.fit(X_valid, y_valid)
                
                # Predict for target company
                target_features = []
                for feature in feature_names:
                    value = getattr(target_company, feature, 0)
                    target_features.append(value)
                
                target_scaled = StandardScaler().fit(X).transform([target_features])
                predicted_multiple = ridge.predict(target_scaled)[0]
                
                # Calculate prediction intervals (simplified)
                y_pred = ridge.predict(X_valid)
                residuals = y_valid - y_pred
                mse = np.mean(residuals ** 2)
                
                # Simplified prediction interval
                prediction_std = np.sqrt(mse * (1 + 1/len(y_valid)))
                t_stat = stats.t.ppf(0.975, len(y_valid) - 2)
                
                regression_results[multiple] = {
                    'predicted_value': predicted_multiple,
                    'r_squared': ridge.score(X_valid, y_valid),
                    'cv_score_mean': np.mean(cv_scores),
                    'cv_score_std': np.std(cv_scores),
                    'feature_importance': ridge.coef_.tolist(),
                    'prediction_interval': (
                        predicted_multiple - t_stat * prediction_std,
                        predicted_multiple + t_stat * prediction_std
                    ),
                    'model_mse': mse,
                    'n_observations': len(y_valid)
                }
                
            except Exception as e:
                logger.warning(f"Regression failed for {multiple}: {str(e)}")
                continue
        
        return regression_results
    
    async def _time_series_multiple_analysis(
        self, 
        peer_companies: List[CompanyFinancialData]
    ) -> Dict[str, Any]:
        """Time series analysis of multiples (placeholder for future enhancement)"""
        
        # This would analyze historical multiple trends
        # For now, return placeholder structure
        time_series_results = {
            'trend_analysis': {},
            'seasonality': {},
            'volatility_metrics': {},
            'mean_reversion_indicators': {}
        }
        
        return time_series_results
    
    async def _calculate_implied_valuations(
        self,
        target_company: CompanyFinancialData,
        multiple_stats: Dict[str, Dict[str, float]],
        regression_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate implied valuations using different methods"""
        
        implied_valuations = {}
        
        # Method 1: Statistical multiples (mean, median)
        for multiple, stats_dict in multiple_stats.items():
            # Get the relevant fundamental metric
            if multiple == 'ev_revenue':
                fundamental_value = target_company.revenue
            elif multiple == 'ev_ebitda':
                fundamental_value = target_company.ebitda
            elif multiple == 'pe_ratio':
                fundamental_value = target_company.net_income
            else:
                continue
                
            if fundamental_value <= 0:
                continue
            
            # Calculate valuations using different statistics
            mean_multiple = stats_dict.get('mean', 0)
            median_multiple = stats_dict.get('median', 0)
            
            if multiple in ['ev_revenue', 'ev_ebitda']:
                # Enterprise value multiples
                if mean_multiple > 0:
                    ev = fundamental_value * mean_multiple
                    equity_value = ev - target_company.total_debt + target_company.cash
                    implied_valuations[f'{multiple}_mean'] = equity_value / target_company.shares_outstanding
                
                if median_multiple > 0:
                    ev = fundamental_value * median_multiple
                    equity_value = ev - target_company.total_debt + target_company.cash
                    implied_valuations[f'{multiple}_median'] = equity_value / target_company.shares_outstanding
                    
            elif multiple == 'pe_ratio':
                # P/E ratio
                if mean_multiple > 0:
                    market_value = fundamental_value * mean_multiple
                    implied_valuations[f'{multiple}_mean'] = market_value / target_company.shares_outstanding
                
                if median_multiple > 0:
                    market_value = fundamental_value * median_multiple
                    implied_valuations[f'{multiple}_median'] = market_value / target_company.shares_outstanding
        
        # Method 2: Regression-based predictions
        for multiple, reg_result in regression_results.items():
            predicted_multiple = reg_result['predicted_value']
            
            if multiple == 'ev_revenue' and target_company.revenue > 0:
                ev = target_company.revenue * predicted_multiple
                equity_value = ev - target_company.total_debt + target_company.cash
                implied_valuations[f'{multiple}_regression'] = equity_value / target_company.shares_outstanding
                
            elif multiple == 'ev_ebitda' and target_company.ebitda > 0:
                ev = target_company.ebitda * predicted_multiple
                equity_value = ev - target_company.total_debt + target_company.cash
                implied_valuations[f'{multiple}_regression'] = equity_value / target_company.shares_outstanding
                
            elif multiple == 'pe_ratio' and target_company.net_income > 0:
                market_value = target_company.net_income * predicted_multiple
                implied_valuations[f'{multiple}_regression'] = market_value / target_company.shares_outstanding
        
        return implied_valuations
    
    async def _risk_adjusted_valuation(
        self,
        target_company: CompanyFinancialData,
        peer_companies: List[CompanyFinancialData],
        implied_valuations: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate risk-adjusted valuations"""
        
        risk_assessment = {}
        
        if not implied_valuations:
            return risk_assessment
        
        # Calculate basic statistics
        valuations = list(implied_valuations.values())
        mean_valuation = np.mean(valuations)
        std_valuation = np.std(valuations)
        
        # Risk adjustments based on target vs peer characteristics
        risk_multipliers = {}
        
        # Size risk
        peer_revenues = [p.revenue for p in peer_companies if p.revenue > 0]
        if peer_revenues and target_company.revenue > 0:
            median_peer_revenue = np.median(peer_revenues)
            size_ratio = target_company.revenue / median_peer_revenue
            
            if size_ratio < 0.5:  # Smaller company
                risk_multipliers['size_discount'] = 0.9
            elif size_ratio > 2.0:  # Larger company
                risk_multipliers['size_premium'] = 1.05
            else:
                risk_multipliers['size_neutral'] = 1.0
        
        # Profitability risk
        peer_margins = [p.ebitda_margin for p in peer_companies if p.ebitda_margin > 0]
        if peer_margins:
            median_peer_margin = np.median(peer_margins)
            if target_company.ebitda_margin < median_peer_margin * 0.8:
                risk_multipliers['profitability_discount'] = 0.95
            elif target_company.ebitda_margin > median_peer_margin * 1.2:
                risk_multipliers['profitability_premium'] = 1.05
        
        # Apply risk adjustments
        risk_adjusted_value = mean_valuation
        for risk_type, multiplier in risk_multipliers.items():
            risk_adjusted_value *= multiplier
        
        # Calculate valuation ranges
        valuation_range = (
            mean_valuation - 1.96 * std_valuation,
            mean_valuation + 1.96 * std_valuation
        )
        
        # Outlier-adjusted value (remove extreme outliers)
        Q1 = np.percentile(valuations, 25)
        Q3 = np.percentile(valuations, 75)
        IQR = Q3 - Q1
        
        filtered_valuations = [
            v for v in valuations 
            if Q1 - 1.5 * IQR <= v <= Q3 + 1.5 * IQR
        ]
        
        outlier_adjusted_value = np.mean(filtered_valuations) if filtered_valuations else mean_valuation
        
        risk_assessment = {
            'mean_valuation': mean_valuation,
            'valuation_std': std_valuation,
            'risk_adjusted_value': risk_adjusted_value,
            'outlier_adjusted_value': outlier_adjusted_value,
            'valuation_range': valuation_range,
            'risk_multipliers': risk_multipliers,
            'confidence_score': min(len(peer_companies) / 10, 1.0)  # Max confidence with 10+ peers
        }
        
        return risk_assessment
    
    async def _compile_cca_results(
        self,
        target_company: CompanyFinancialData,
        peer_companies: List[CompanyFinancialData],
        multiple_stats: Dict[str, Dict[str, float]],
        regression_results: Dict[str, Any],
        implied_valuations: Dict[str, float],
        risk_assessment: Dict[str, Any],
        time_series_results: Dict[str, Any]
    ) -> CCAModelResults:
        """Compile comprehensive CCA results"""
        
        # Calculate peer selection score
        peer_selection_score = min(len(peer_companies) / 15, 1.0)  # Normalize to 0-1
        
        # Extract estimated multiples
        estimated_multiples = {}
        for multiple, stats_dict in multiple_stats.items():
            estimated_multiples[f'{multiple}_mean'] = stats_dict.get('mean', 0)
            estimated_multiples[f'{multiple}_median'] = stats_dict.get('median', 0)
        
        for multiple, reg_result in regression_results.items():
            estimated_multiples[f'{multiple}_predicted'] = reg_result.get('predicted_value', 0)
        
        # Confidence intervals
        confidence_intervals = {}
        prediction_intervals = {}
        model_r_squared = {}
        
        for multiple, stats_dict in multiple_stats.items():
            confidence_intervals[multiple] = (
                stats_dict.get('ci_lower', 0),
                stats_dict.get('ci_upper', 0)
            )
        
        for multiple, reg_result in regression_results.items():
            prediction_intervals[multiple] = reg_result.get('prediction_interval', (0, 0))
            model_r_squared[multiple] = reg_result.get('r_squared', 0)
        
        # Similarity scores
        similarity_scores = {}
        for peer in peer_companies:
            similarity_scores[peer.company_name] = self._calculate_similarity_score(
                target_company, peer
            )
        
        return CCAModelResults(
            target_company=target_company.company_name,
            peer_companies=peer_companies,
            peer_selection_score=peer_selection_score,
            estimated_multiples=estimated_multiples,
            implied_valuations=implied_valuations,
            multiple_statistics=multiple_stats,
            regression_results=regression_results,
            confidence_intervals=confidence_intervals,
            prediction_intervals=prediction_intervals,
            model_r_squared=model_r_squared,
            valuation_range=risk_assessment.get('valuation_range', (0, 0)),
            risk_adjusted_value=risk_assessment.get('risk_adjusted_value', 0),
            outlier_adjusted_value=risk_assessment.get('outlier_adjusted_value', 0),
            peer_cluster_analysis={},  # Placeholder
            similarity_scores=similarity_scores,
            multiple_trends={},  # From time series analysis
            seasonality_analysis=time_series_results
        )

# Factory function
def create_enhanced_cca_model(**kwargs) -> EnhancedCCAModel:
    """Factory function for creating enhanced CCA models"""
    return EnhancedCCAModel(**kwargs)

# Utility functions for data loading and preprocessing
async def load_company_data_from_api(ticker: str) -> Optional[CompanyFinancialData]:
    """Load company data from external API (placeholder)"""
    # This would integrate with data providers like Bloomberg, Refinitiv, etc.
    return None

async def batch_load_universe_data(tickers: List[str]) -> List[CompanyFinancialData]:
    """Batch load universe data (placeholder)"""
    # This would efficiently load data for many companies
    return []