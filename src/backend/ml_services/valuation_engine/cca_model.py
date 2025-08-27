"""
Comparable Company Analysis (CCA) Model
Advanced peer comparison valuation with ML-enhanced peer selection
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from decimal import Decimal
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

from ...config import settings
from ...utils.metrics import track_ml_inference, track_time, ML_INFERENCE_DURATION

logger = logging.getLogger(__name__)


@dataclass
class ComparableCompany:
    """Comparable company data structure"""
    ticker: str
    name: str
    market_cap: float
    enterprise_value: float
    revenue: float
    ebitda: float
    net_income: float
    
    # Financial ratios
    ev_revenue: float
    ev_ebitda: float
    pe_ratio: float
    pb_ratio: float
    peg_ratio: float
    
    # Business characteristics
    sector: str
    industry: str
    geography: str
    business_model: str
    size_category: str
    
    # Quality scores
    similarity_score: float = 0.0
    data_quality_score: float = 1.0
    liquidity_score: float = 1.0


@dataclass
class CCAInputs:
    """CCA model input parameters"""
    target_company: Dict[str, Any]
    comparable_companies: List[ComparableCompany]
    
    # Selection criteria
    min_similarity_score: float = 0.6
    max_peers: int = 15
    min_peers: int = 5
    
    # Weighting preferences
    weight_by_similarity: bool = True
    weight_by_size: bool = True
    weight_by_liquidity: bool = True
    
    # Multiple preferences
    primary_multiples: List[str] = None
    secondary_multiples: List[str] = None


@dataclass
class CCAOutputs:
    """CCA model outputs"""
    # Selected peers
    selected_peers: List[ComparableCompany]
    peer_statistics: Dict[str, Dict[str, float]]
    
    # Valuation results
    implied_valuations: Dict[str, float]
    weighted_valuation: float
    valuation_range: Tuple[float, float]
    
    # Multiple analysis
    multiple_statistics: Dict[str, Dict[str, float]]
    peer_premiums_discounts: Dict[str, float]
    
    # Quality metrics
    peer_correlation_matrix: Dict[str, Dict[str, float]]
    selection_confidence: float
    data_completeness: float
    
    # Risk analysis
    valuation_volatility: float
    outlier_impact: float


class CCAValuationModel:
    """Advanced Comparable Company Analysis model"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.clustering_model = KMeans(n_clusters=3, random_state=42)
        
    @track_time(ML_INFERENCE_DURATION, {"model_name": "cca"})
    def calculate_valuation(self, inputs: CCAInputs) -> CCAOutputs:
        """
        Calculate CCA valuation with ML-enhanced peer selection
        
        Args:
            inputs: CCA input parameters
        """
        try:
            track_ml_inference("cca", 0, True)
            
            # Step 1: Select comparable companies
            selected_peers = self._select_comparable_companies(inputs)
            
            # Step 2: Calculate multiple statistics
            peer_stats, multiple_stats = self._calculate_statistics(selected_peers)
            
            # Step 3: Apply multiples to target company
            implied_valuations = self._apply_multiples(inputs.target_company, multiple_stats)
            
            # Step 4: Calculate weighted valuation
            weighted_valuation = self._calculate_weighted_valuation(
                implied_valuations, selected_peers, inputs
            )
            
            # Step 5: Determine valuation range
            valuation_range = self._calculate_valuation_range(implied_valuations)
            
            # Step 6: Quality analysis
            correlation_matrix = self._calculate_peer_correlations(selected_peers)
            selection_confidence = self._calculate_selection_confidence(selected_peers)
            data_completeness = self._calculate_data_completeness(selected_peers)
            
            # Step 7: Risk analysis
            valuation_volatility = self._calculate_volatility(implied_valuations)
            outlier_impact = self._analyze_outlier_impact(selected_peers, implied_valuations)
            
            # Step 8: Premium/discount analysis
            premiums_discounts = self._calculate_premiums_discounts(selected_peers, multiple_stats)
            
            outputs = CCAOutputs(
                selected_peers=selected_peers,
                peer_statistics=peer_stats,
                implied_valuations=implied_valuations,
                weighted_valuation=weighted_valuation,
                valuation_range=valuation_range,
                multiple_statistics=multiple_stats,
                peer_premiums_discounts=premiums_discounts,
                peer_correlation_matrix=correlation_matrix,
                selection_confidence=selection_confidence,
                data_completeness=data_completeness,
                valuation_volatility=valuation_volatility,
                outlier_impact=outlier_impact
            )
            
            logger.info(f"CCA valuation completed: ${weighted_valuation:.2f} (Range: ${valuation_range[0]:.2f}-${valuation_range[1]:.2f})")
            return outputs
            
        except Exception as e:
            track_ml_inference("cca", 0, False)
            logger.error(f"CCA calculation error: {str(e)}")
            raise
    
    def _select_comparable_companies(self, inputs: CCAInputs) -> List[ComparableCompany]:
        """Select comparable companies using ML-enhanced scoring"""
        
        # Calculate similarity scores
        for comp in inputs.comparable_companies:
            comp.similarity_score = self._calculate_similarity_score(
                inputs.target_company, comp
            )
        
        # Filter by minimum similarity
        filtered_comps = [
            comp for comp in inputs.comparable_companies 
            if comp.similarity_score >= inputs.min_similarity_score
        ]
        
        # ML-enhanced clustering for peer grouping
        feature_matrix = self._prepare_feature_matrix(filtered_comps)
        if len(feature_matrix) > inputs.max_peers:
            cluster_labels = self._perform_clustering(feature_matrix, filtered_comps)
            # Select best cluster or mix of clusters
            selected_comps = self._select_from_clusters(filtered_comps, cluster_labels)
        else:
            selected_comps = filtered_comps
        
        # Sort by similarity score and limit count
        selected_comps.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Ensure minimum and maximum peer counts
        if len(selected_comps) < inputs.min_peers:
            logger.warning(f"Only {len(selected_comps)} peers found, minimum is {inputs.min_peers}")
        
        return selected_comps[:inputs.max_peers]
    
    def _calculate_similarity_score(self, target: Dict[str, Any], comp: ComparableCompany) -> float:
        """Calculate similarity score between target and comparable company"""
        score = 0.0
        weights = {
            'sector': 0.3,
            'industry': 0.25,
            'geography': 0.15,
            'size': 0.2,
            'business_model': 0.1
        }
        
        # Sector match
        if target.get('sector') == comp.sector:
            score += weights['sector']
        elif target.get('sector', '').split(' ')[0] == comp.sector.split(' ')[0]:
            score += weights['sector'] * 0.5  # Partial match
        
        # Industry match
        if target.get('industry') == comp.industry:
            score += weights['industry']
        
        # Geography match
        if target.get('geography') == comp.geography:
            score += weights['geography']
        elif target.get('geography', '').split(' ')[0] == comp.geography.split(' ')[0]:
            score += weights['geography'] * 0.7
        
        # Size similarity (revenue-based)
        target_revenue = target.get('revenue', 0)
        if target_revenue > 0 and comp.revenue > 0:
            size_ratio = min(target_revenue, comp.revenue) / max(target_revenue, comp.revenue)
            score += weights['size'] * size_ratio
        
        # Business model match
        if target.get('business_model') == comp.business_model:
            score += weights['business_model']
        
        return min(score, 1.0)
    
    def _prepare_feature_matrix(self, companies: List[ComparableCompany]) -> np.ndarray:
        """Prepare feature matrix for ML clustering"""
        features = []
        
        for comp in companies:
            feature_vector = [
                comp.ev_revenue or 0,
                comp.ev_ebitda or 0,
                comp.pe_ratio or 0,
                np.log(comp.market_cap) if comp.market_cap > 0 else 0,
                np.log(comp.revenue) if comp.revenue > 0 else 0,
                comp.similarity_score
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _perform_clustering(self, feature_matrix: np.ndarray, companies: List[ComparableCompany]) -> np.ndarray:
        """Perform clustering to identify peer groups"""
        # Normalize features
        normalized_features = self.scaler.fit_transform(feature_matrix)
        
        # Determine optimal number of clusters (between 2 and min(5, n_companies/3))
        n_companies = len(companies)
        max_clusters = min(5, max(2, n_companies // 3))
        
        if max_clusters > 2:
            # Use elbow method to find optimal clusters
            inertias = []
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(normalized_features)
                inertias.append(kmeans.inertia_)
            
            # Simple elbow detection
            optimal_k = 2
            for i in range(1, len(inertias)):
                if inertias[i-1] - inertias[i] < inertias[0] * 0.1:
                    optimal_k = i + 1
                    break
        else:
            optimal_k = 2
        
        # Perform final clustering
        self.clustering_model = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = self.clustering_model.fit_predict(normalized_features)
        
        return cluster_labels
    
    def _select_from_clusters(self, companies: List[ComparableCompany], cluster_labels: np.ndarray) -> List[ComparableCompany]:
        """Select best companies from each cluster"""
        cluster_groups = {}
        
        # Group companies by cluster
        for i, label in enumerate(cluster_labels):
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append(companies[i])
        
        # Select best companies from each cluster
        selected = []
        for cluster, cluster_companies in cluster_groups.items():
            # Sort by similarity score within cluster
            cluster_companies.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Take top companies from each cluster
            cluster_size = len(cluster_companies)
            take_from_cluster = max(1, min(5, cluster_size // 2))
            selected.extend(cluster_companies[:take_from_cluster])
        
        return selected
    
    def _calculate_statistics(self, companies: List[ComparableCompany]) -> Tuple[Dict, Dict]:
        """Calculate peer statistics and multiple statistics"""
        
        # Extract multiples
        multiples_data = {
            'ev_revenue': [c.ev_revenue for c in companies if c.ev_revenue and c.ev_revenue > 0],
            'ev_ebitda': [c.ev_ebitda for c in companies if c.ev_ebitda and c.ev_ebitda > 0],
            'pe_ratio': [c.pe_ratio for c in companies if c.pe_ratio and c.pe_ratio > 0],
            'pb_ratio': [c.pb_ratio for c in companies if c.pb_ratio and c.pb_ratio > 0],
            'peg_ratio': [c.peg_ratio for c in companies if c.peg_ratio and c.peg_ratio > 0 and c.peg_ratio < 10]
        }
        
        # Calculate statistics for each multiple
        multiple_stats = {}
        for multiple, values in multiples_data.items():
            if values:
                multiple_stats[multiple] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                    '25th_percentile': np.percentile(values, 25),
                    '75th_percentile': np.percentile(values, 75),
                    'count': len(values)
                }
        
        # Peer statistics
        peer_stats = {
            'count': len(companies),
            'avg_similarity_score': np.mean([c.similarity_score for c in companies]),
            'avg_market_cap': np.mean([c.market_cap for c in companies if c.market_cap]),
            'avg_revenue': np.mean([c.revenue for c in companies if c.revenue]),
            'sector_distribution': self._calculate_distribution([c.sector for c in companies]),
            'geography_distribution': self._calculate_distribution([c.geography for c in companies])
        }
        
        return peer_stats, multiple_stats
    
    def _calculate_distribution(self, values: List[str]) -> Dict[str, float]:
        """Calculate distribution of categorical values"""
        from collections import Counter
        counts = Counter(values)
        total = len(values)
        return {k: v/total for k, v in counts.items()}
    
    def _apply_multiples(self, target: Dict[str, Any], multiple_stats: Dict) -> Dict[str, float]:
        """Apply multiples to target company"""
        implied_valuations = {}
        
        # EV/Revenue
        if 'ev_revenue' in multiple_stats and target.get('revenue'):
            ev_revenue_median = multiple_stats['ev_revenue']['median']
            implied_ev = target['revenue'] * ev_revenue_median
            implied_equity_value = implied_ev + target.get('cash', 0) - target.get('debt', 0)
            implied_valuations['ev_revenue'] = implied_equity_value
        
        # EV/EBITDA
        if 'ev_ebitda' in multiple_stats and target.get('ebitda'):
            ev_ebitda_median = multiple_stats['ev_ebitda']['median']
            implied_ev = target['ebitda'] * ev_ebitda_median
            implied_equity_value = implied_ev + target.get('cash', 0) - target.get('debt', 0)
            implied_valuations['ev_ebitda'] = implied_equity_value
        
        # P/E
        if 'pe_ratio' in multiple_stats and target.get('net_income'):
            pe_median = multiple_stats['pe_ratio']['median']
            implied_valuations['pe_ratio'] = target['net_income'] * pe_median
        
        # P/B
        if 'pb_ratio' in multiple_stats and target.get('book_value'):
            pb_median = multiple_stats['pb_ratio']['median']
            implied_valuations['pb_ratio'] = target['book_value'] * pb_median
        
        return implied_valuations
    
    def _calculate_weighted_valuation(self, valuations: Dict[str, float], peers: List[ComparableCompany], inputs: CCAInputs) -> float:
        """Calculate weighted average valuation"""
        if not valuations:
            return 0.0
        
        # Default weights for different multiples
        multiple_weights = {
            'ev_revenue': 0.25,
            'ev_ebitda': 0.35,
            'pe_ratio': 0.25,
            'pb_ratio': 0.15
        }
        
        # Adjust weights based on data quality and availability
        available_multiples = list(valuations.keys())
        total_weight = sum(multiple_weights.get(m, 0) for m in available_multiples)
        
        if total_weight == 0:
            return np.mean(list(valuations.values()))
        
        weighted_sum = 0
        for multiple, valuation in valuations.items():
            weight = multiple_weights.get(multiple, 0) / total_weight
            weighted_sum += valuation * weight
        
        return weighted_sum
    
    def _calculate_valuation_range(self, valuations: Dict[str, float]) -> Tuple[float, float]:
        """Calculate valuation range"""
        if not valuations:
            return (0, 0)
        
        values = list(valuations.values())
        return (min(values), max(values))
    
    def _calculate_peer_correlations(self, companies: List[ComparableCompany]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between peer multiples"""
        multiples = ['ev_revenue', 'ev_ebitda', 'pe_ratio', 'pb_ratio']
        
        # Extract multiple values
        data = {}
        for multiple in multiples:
            values = []
            for comp in companies:
                value = getattr(comp, multiple, None)
                values.append(value if value and value > 0 else np.nan)
            data[multiple] = values
        
        # Calculate correlations
        df = pd.DataFrame(data)
        corr_matrix = df.corr()
        
        return corr_matrix.to_dict()
    
    def _calculate_selection_confidence(self, companies: List[ComparableCompany]) -> float:
        """Calculate confidence in peer selection"""
        if not companies:
            return 0.0
        
        # Base confidence on average similarity score and number of peers
        avg_similarity = np.mean([c.similarity_score for c in companies])
        peer_count_factor = min(1.0, len(companies) / 10)  # Optimal around 10 peers
        
        return avg_similarity * peer_count_factor
    
    def _calculate_data_completeness(self, companies: List[ComparableCompany]) -> float:
        """Calculate data completeness score"""
        if not companies:
            return 0.0
        
        total_fields = 5  # Number of key multiples
        complete_ratios = []
        
        for comp in companies:
            complete_fields = 0
            if comp.ev_revenue and comp.ev_revenue > 0:
                complete_fields += 1
            if comp.ev_ebitda and comp.ev_ebitda > 0:
                complete_fields += 1
            if comp.pe_ratio and comp.pe_ratio > 0:
                complete_fields += 1
            if comp.pb_ratio and comp.pb_ratio > 0:
                complete_fields += 1
            if comp.peg_ratio and comp.peg_ratio > 0:
                complete_fields += 1
            
            complete_ratios.append(complete_fields / total_fields)
        
        return np.mean(complete_ratios)
    
    def _calculate_volatility(self, valuations: Dict[str, float]) -> float:
        """Calculate valuation volatility"""
        if len(valuations) < 2:
            return 0.0
        
        values = list(valuations.values())
        mean_val = np.mean(values)
        
        if mean_val == 0:
            return 0.0
        
        return np.std(values) / mean_val
    
    def _analyze_outlier_impact(self, companies: List[ComparableCompany], valuations: Dict[str, float]) -> float:
        """Analyze impact of outliers on valuation"""
        if len(valuations) < 3:
            return 0.0
        
        values = list(valuations.values())
        
        # Calculate with and without outliers
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        
        # Remove outliers (values beyond 1.5 * IQR)
        filtered_values = [v for v in values if q1 - 1.5 * iqr <= v <= q3 + 1.5 * iqr]
        
        if len(filtered_values) < 2:
            return 0.0
        
        original_mean = np.mean(values)
        filtered_mean = np.mean(filtered_values)
        
        if original_mean == 0:
            return 0.0
        
        return abs(filtered_mean - original_mean) / original_mean
    
    def _calculate_premiums_discounts(self, companies: List[ComparableCompany], multiple_stats: Dict) -> Dict[str, float]:
        """Calculate premiums/discounts for each peer"""
        premiums_discounts = {}
        
        for comp in companies:
            comp_premiums = {}
            
            # EV/Revenue premium/discount
            if 'ev_revenue' in multiple_stats and comp.ev_revenue:
                median_multiple = multiple_stats['ev_revenue']['median']
                premium = (comp.ev_revenue - median_multiple) / median_multiple
                comp_premiums['ev_revenue'] = premium
            
            # EV/EBITDA premium/discount
            if 'ev_ebitda' in multiple_stats and comp.ev_ebitda:
                median_multiple = multiple_stats['ev_ebitda']['median']
                premium = (comp.ev_ebitda - median_multiple) / median_multiple
                comp_premiums['ev_ebitda'] = premium
            
            premiums_discounts[comp.ticker] = comp_premiums
        
        return premiums_discounts


def create_cca_model() -> CCAValuationModel:
    """Create a configured CCA valuation model"""
    return CCAValuationModel()