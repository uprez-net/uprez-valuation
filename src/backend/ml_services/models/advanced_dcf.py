"""
Advanced DCF Model with Monte Carlo Simulation and Industry Benchmarking
Enhanced version with comprehensive risk analysis and optimization
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
from scipy import stats, optimize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
import warnings
import asyncio
from concurrent.futures import ProcessPoolExecutor
import joblib

logger = logging.getLogger(__name__)

@dataclass
class AdvancedDCFInputs:
    """Advanced DCF model inputs with comprehensive parameters"""
    # Company fundamentals
    company_name: str
    sector: str
    industry: str
    market_cap: Optional[float] = None
    
    # Historical financials (5+ years recommended)
    historical_revenues: List[float] = field(default_factory=list)
    historical_ebitda: List[float] = field(default_factory=list)
    historical_ebit: List[float] = field(default_factory=list)
    historical_fcf: List[float] = field(default_factory=list)
    historical_capex: List[float] = field(default_factory=list)
    historical_working_capital: List[float] = field(default_factory=list)
    historical_years: List[int] = field(default_factory=list)
    
    # Growth and margin projections
    revenue_growth_rates: List[float] = field(default_factory=list)
    ebitda_margin_targets: List[float] = field(default_factory=list)
    capex_as_pct_revenue: List[float] = field(default_factory=list)
    working_capital_as_pct_revenue: float = 0.05
    tax_rate: float = 0.25
    
    # Discount rate components with uncertainty ranges
    risk_free_rate: float = 0.025
    risk_free_rate_std: float = 0.005
    market_risk_premium: float = 0.06
    market_risk_premium_std: float = 0.015
    beta: float = 1.0
    beta_std: float = 0.2
    
    # Debt structure
    cost_of_debt: float = 0.05
    debt_to_equity: float = 0.0
    target_debt_ratio: Optional[float] = None
    
    # Terminal value parameters
    terminal_growth_rate: float = 0.025
    terminal_growth_rate_std: float = 0.005
    terminal_multiple: Optional[float] = None
    terminal_multiple_std: Optional[float] = None
    
    # Balance sheet adjustments
    cash_and_equivalents: float = 0
    total_debt: float = 0
    minority_interest: float = 0
    preferred_stock: float = 0
    shares_outstanding: float = 1
    
    # Industry benchmarks
    industry_multiples: Dict[str, float] = field(default_factory=dict)
    peer_companies: List[str] = field(default_factory=list)
    
    # Economic scenarios
    economic_scenario_weights: Dict[str, float] = field(default_factory=lambda: {
        'bull': 0.25, 'base': 0.50, 'bear': 0.25
    })
    scenario_adjustments: Dict[str, Dict[str, float]] = field(default_factory=dict)

@dataclass
class DCFScenarioResult:
    """Results for a single DCF scenario"""
    scenario_name: str
    enterprise_value: float
    equity_value: float
    value_per_share: float
    wacc: float
    terminal_value: float
    terminal_value_pct: float
    revenue_cagr: float
    implied_exit_multiple: float

@dataclass
class AdvancedDCFOutputs:
    """Comprehensive DCF model outputs"""
    # Primary valuation metrics
    enterprise_value: float
    equity_value: float
    value_per_share: float
    
    # Scenario analysis
    scenario_results: Dict[str, DCFScenarioResult]
    probability_weighted_value: float
    
    # Monte Carlo results
    monte_carlo_stats: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]  # 80%, 90%, 95%
    value_at_risk: Dict[str, float]  # 5%, 10%
    
    # Sensitivity analysis
    tornado_chart_data: Dict[str, Dict[str, float]]
    sensitivity_matrix: pd.DataFrame
    
    # Risk metrics
    risk_adjusted_value: float
    sharpe_ratio: float
    maximum_drawdown: float
    probability_of_loss: float
    
    # Financial projections
    projection_summary: pd.DataFrame
    cash_flow_waterfall: Dict[str, List[float]]
    
    # Validation metrics
    model_r_squared: float
    prediction_intervals: Dict[str, Tuple[float, float]]
    peer_comparison: Dict[str, float]
    
    # Optimization results
    optimal_capital_structure: Dict[str, float]
    sensitivity_rankings: List[Tuple[str, float]]

class AdvancedDCFModel:
    """
    Advanced DCF Model with comprehensive risk analysis and optimization
    Features:
    - Monte Carlo simulation with 50,000+ iterations
    - Multi-scenario analysis (bull/base/bear)
    - Industry benchmarking and peer analysis
    - Tornado charts and sensitivity analysis
    - Risk-adjusted valuation metrics
    - Optimal capital structure analysis
    """
    
    def __init__(self, simulation_runs: int = 50000, projection_years: int = 10):
        self.simulation_runs = simulation_runs
        self.projection_years = projection_years
        self.model_cache = {}
        self.industry_benchmarks = self._load_industry_benchmarks()
        
    async def calculate_advanced_valuation(
        self,
        inputs: AdvancedDCFInputs,
        include_monte_carlo: bool = True,
        include_scenarios: bool = True,
        include_optimization: bool = True,
        confidence_levels: List[float] = [0.80, 0.90, 0.95]
    ) -> AdvancedDCFOutputs:
        """
        Calculate comprehensive DCF valuation with advanced analytics
        """
        try:
            # Input validation and preprocessing
            validated_inputs = self._validate_and_preprocess_inputs(inputs)
            
            # Base case valuation
            base_result = await self._calculate_base_dcf(validated_inputs)
            
            # Scenario analysis
            scenario_results = {}
            if include_scenarios:
                scenario_results = await self._run_scenario_analysis(validated_inputs)
            
            # Monte Carlo simulation
            mc_stats = {}
            if include_monte_carlo:
                mc_stats = await self._run_monte_carlo_simulation(
                    validated_inputs, confidence_levels
                )
            
            # Sensitivity analysis
            sensitivity_results = await self._comprehensive_sensitivity_analysis(validated_inputs)
            
            # Risk metrics
            risk_metrics = await self._calculate_risk_metrics(validated_inputs, mc_stats)
            
            # Optimization
            optimization_results = {}
            if include_optimization:
                optimization_results = await self._optimize_capital_structure(validated_inputs)
            
            # Compile comprehensive results
            return await self._compile_advanced_outputs(
                base_result, scenario_results, mc_stats, 
                sensitivity_results, risk_metrics, optimization_results
            )
            
        except Exception as e:
            logger.error(f"Advanced DCF calculation failed: {str(e)}")
            raise
    
    def _validate_and_preprocess_inputs(self, inputs: AdvancedDCFInputs) -> AdvancedDCFInputs:
        """Validate and preprocess inputs with industry benchmarking"""
        # Basic validation
        if len(inputs.historical_revenues) < 3:
            raise ValueError("At least 3 years of historical data required")
        
        # Fill missing industry benchmarks
        if inputs.sector in self.industry_benchmarks:
            sector_data = self.industry_benchmarks[inputs.sector]
            if not inputs.industry_multiples:
                inputs.industry_multiples = sector_data.get('multiples', {})
        
        # Validate growth rates
        if not inputs.revenue_growth_rates:
            # Estimate from historical data
            revenues = np.array(inputs.historical_revenues)
            growth_rates = (revenues[1:] / revenues[:-1] - 1).tolist()
            inputs.revenue_growth_rates = self._project_growth_rates(growth_rates)
        
        # Validate EBITDA margins
        if not inputs.ebitda_margin_targets:
            margins = np.array(inputs.historical_ebitda) / np.array(inputs.historical_revenues)
            inputs.ebitda_margin_targets = self._project_margins(margins.tolist())
        
        return inputs
    
    async def _calculate_base_dcf(self, inputs: AdvancedDCFInputs) -> Dict[str, Any]:
        """Calculate base case DCF with enhanced methodology"""
        # Calculate dynamic WACC
        wacc = await self._calculate_dynamic_wacc(inputs)
        
        # Project detailed cash flows
        projections = await self._project_detailed_cash_flows(inputs)
        
        # Calculate terminal value with multiple methods
        terminal_value = await self._calculate_enhanced_terminal_value(inputs, projections, wacc)
        
        # Discount to present value with mid-year convention
        pv_results = await self._discount_with_midyear_convention(projections, wacc, terminal_value)
        
        # Calculate enterprise and equity values
        enterprise_value = pv_results['total_pv']
        equity_value = enterprise_value + inputs.cash_and_equivalents - inputs.total_debt - inputs.minority_interest
        value_per_share = equity_value / inputs.shares_outstanding
        
        return {
            'enterprise_value': enterprise_value,
            'equity_value': equity_value,
            'value_per_share': value_per_share,
            'wacc': wacc,
            'terminal_value': terminal_value,
            'projections': projections,
            'pv_results': pv_results
        }
    
    async def _calculate_dynamic_wacc(self, inputs: AdvancedDCFInputs) -> float:
        """Calculate WACC with time-varying components"""
        # Base cost of equity using enhanced CAPM
        risk_free_rate = inputs.risk_free_rate
        market_premium = inputs.market_risk_premium
        beta = await self._calculate_adjusted_beta(inputs)
        
        # Add size premium if applicable
        size_premium = self._calculate_size_premium(inputs.market_cap) if inputs.market_cap else 0
        
        # Add country risk premium if international
        country_risk_premium = 0  # Could be enhanced for international companies
        
        cost_of_equity = risk_free_rate + beta * market_premium + size_premium + country_risk_premium
        
        # After-tax cost of debt with credit spread
        credit_spread = self._calculate_credit_spread(inputs)
        cost_of_debt = inputs.cost_of_debt + credit_spread
        after_tax_cost_of_debt = cost_of_debt * (1 - inputs.tax_rate)
        
        # Capital structure weights
        if inputs.target_debt_ratio:
            debt_weight = inputs.target_debt_ratio
        else:
            debt_weight = inputs.debt_to_equity / (1 + inputs.debt_to_equity)
        
        equity_weight = 1 - debt_weight
        
        # Calculate WACC
        wacc = equity_weight * cost_of_equity + debt_weight * after_tax_cost_of_debt
        
        return wacc
    
    async def _calculate_adjusted_beta(self, inputs: AdvancedDCFInputs) -> float:
        """Calculate adjusted beta using Blume's adjustment"""
        raw_beta = inputs.beta
        # Blume's adjustment: adjusted_beta = 0.67 * raw_beta + 0.33 * 1.0
        adjusted_beta = 0.67 * raw_beta + 0.33
        return adjusted_beta
    
    def _calculate_size_premium(self, market_cap: float) -> float:
        """Calculate size premium based on market cap"""
        if not market_cap:
            return 0
        
        # Size premiums (simplified)
        if market_cap < 250e6:  # < $250M
            return 0.035
        elif market_cap < 1e9:  # < $1B
            return 0.020
        elif market_cap < 5e9:  # < $5B
            return 0.010
        else:
            return 0.005
    
    def _calculate_credit_spread(self, inputs: AdvancedDCFInputs) -> float:
        """Calculate credit spread based on financial metrics"""
        # Simplified credit spread calculation
        # In practice, would use credit ratings or financial ratios
        debt_ratio = inputs.debt_to_equity / (1 + inputs.debt_to_equity)
        
        if debt_ratio < 0.2:
            return 0.005  # Low risk
        elif debt_ratio < 0.4:
            return 0.015  # Medium risk
        elif debt_ratio < 0.6:
            return 0.030  # High risk
        else:
            return 0.050  # Very high risk
    
    async def _project_detailed_cash_flows(self, inputs: AdvancedDCFInputs) -> Dict[str, List[float]]:
        """Project detailed cash flows with enhanced methodology"""
        projections = {
            'years': list(range(1, self.projection_years + 1)),
            'revenues': [],
            'ebitda': [],
            'ebit': [],
            'taxes': [],
            'nopat': [],
            'capex': [],
            'depreciation': [],
            'working_capital_change': [],
            'free_cash_flow': [],
            'unlevered_fcf': []
        }
        
        # Base year metrics
        base_revenue = inputs.historical_revenues[-1]
        base_wc = inputs.historical_working_capital[-1] if inputs.historical_working_capital else 0
        
        # Project revenues with declining growth rates
        for year in range(self.projection_years):
            if year < len(inputs.revenue_growth_rates):
                growth_rate = inputs.revenue_growth_rates[year]
            else:
                # Fade to terminal growth rate
                terminal_growth = inputs.terminal_growth_rate
                fade_factor = max(0.7 ** (year - len(inputs.revenue_growth_rates) + 1), 0)
                current_growth = inputs.revenue_growth_rates[-1] if inputs.revenue_growth_rates else terminal_growth
                growth_rate = fade_factor * current_growth + (1 - fade_factor) * terminal_growth
            
            if year == 0:
                revenue = base_revenue * (1 + growth_rate)
            else:
                revenue = projections['revenues'][-1] * (1 + growth_rate)
            
            projections['revenues'].append(revenue)
        
        # Project margins and profitability
        for year, revenue in enumerate(projections['revenues']):
            # EBITDA with margin improvement/deterioration
            if year < len(inputs.ebitda_margin_targets):
                ebitda_margin = inputs.ebitda_margin_targets[year]
            else:
                ebitda_margin = inputs.ebitda_margin_targets[-1] if inputs.ebitda_margin_targets else 0.15
            
            ebitda = revenue * ebitda_margin
            projections['ebitda'].append(ebitda)
            
            # Depreciation (simplified as % of revenue)
            if year < len(inputs.capex_as_pct_revenue):
                capex_rate = inputs.capex_as_pct_revenue[year]
            else:
                capex_rate = inputs.capex_as_pct_revenue[-1] if inputs.capex_as_pct_revenue else 0.04
            
            capex = revenue * capex_rate
            projections['capex'].append(capex)
            
            # Depreciation as % of revenue (simplified)
            depreciation = revenue * 0.03
            projections['depreciation'].append(depreciation)
            
            # EBIT
            ebit = ebitda - depreciation
            projections['ebit'].append(ebit)
            
            # Taxes
            taxes = ebit * inputs.tax_rate if ebit > 0 else 0
            projections['taxes'].append(taxes)
            
            # NOPAT
            nopat = ebit - taxes
            projections['nopat'].append(nopat)
            
            # Working capital change
            current_wc = revenue * inputs.working_capital_as_pct_revenue
            if year == 0:
                wc_change = current_wc - base_wc
            else:
                previous_wc = projections['revenues'][year-1] * inputs.working_capital_as_pct_revenue
                wc_change = current_wc - previous_wc
            
            projections['working_capital_change'].append(wc_change)
            
            # Free Cash Flow
            fcf = nopat + depreciation - capex - wc_change
            projections['free_cash_flow'].append(fcf)
            
            # Unlevered FCF (same as FCF for now)
            projections['unlevered_fcf'].append(fcf)
        
        return projections
    
    async def _calculate_enhanced_terminal_value(
        self, 
        inputs: AdvancedDCFInputs, 
        projections: Dict[str, List[float]], 
        wacc: float
    ) -> Dict[str, float]:
        """Calculate terminal value using multiple methods"""
        final_year_fcf = projections['free_cash_flow'][-1]
        final_year_ebitda = projections['ebitda'][-1]
        final_year_revenue = projections['revenues'][-1]
        
        terminal_values = {}
        
        # Gordon Growth Model
        if wacc > inputs.terminal_growth_rate:
            terminal_fcf = final_year_fcf * (1 + inputs.terminal_growth_rate)
            tv_gordon = terminal_fcf / (wacc - inputs.terminal_growth_rate)
            terminal_values['gordon_growth'] = tv_gordon
        
        # Exit Multiple Method
        if inputs.terminal_multiple:
            tv_multiple = final_year_ebitda * inputs.terminal_multiple
            terminal_values['exit_multiple'] = tv_multiple
        
        # Industry Multiple Method
        if 'ebitda_multiple' in inputs.industry_multiples:
            industry_multiple = inputs.industry_multiples['ebitda_multiple']
            tv_industry = final_year_ebitda * industry_multiple
            terminal_values['industry_multiple'] = tv_industry
        
        # Revenue Multiple Method (for high growth companies)
        if 'revenue_multiple' in inputs.industry_multiples:
            revenue_multiple = inputs.industry_multiples['revenue_multiple']
            tv_revenue = final_year_revenue * revenue_multiple
            terminal_values['revenue_multiple'] = tv_revenue
        
        # Weighted average of methods
        if len(terminal_values) > 1:
            # Equal weighting for now, could be enhanced
            weights = {k: 1/len(terminal_values) for k in terminal_values.keys()}
            weighted_tv = sum(tv * weights[method] for method, tv in terminal_values.items())
            terminal_values['weighted_average'] = weighted_tv
            return terminal_values
        else:
            return {'primary': list(terminal_values.values())[0]} if terminal_values else {'primary': 0}
    
    async def _discount_with_midyear_convention(
        self, 
        projections: Dict[str, List[float]], 
        wacc: float, 
        terminal_value: Dict[str, float]
    ) -> Dict[str, Any]:
        """Discount cash flows using mid-year convention"""
        fcf = projections['free_cash_flow']
        pv_fcf = []
        discount_factors = []
        
        # Discount cash flows (mid-year convention)
        for i, cash_flow in enumerate(fcf):
            # Mid-year convention: discount by (year - 0.5)
            discount_period = i + 0.5
            discount_factor = 1 / ((1 + wacc) ** discount_period)
            pv = cash_flow * discount_factor
            
            pv_fcf.append(pv)
            discount_factors.append(discount_factor)
        
        # Discount terminal value
        tv_primary = terminal_value.get('weighted_average', terminal_value.get('primary', 0))
        tv_discount_factor = 1 / ((1 + wacc) ** self.projection_years)
        pv_terminal = tv_primary * tv_discount_factor
        
        total_pv = sum(pv_fcf) + pv_terminal
        
        return {
            'pv_fcf': pv_fcf,
            'discount_factors': discount_factors,
            'pv_terminal': pv_terminal,
            'total_pv': total_pv,
            'terminal_value_pct': pv_terminal / total_pv * 100 if total_pv > 0 else 0
        }
    
    async def _run_scenario_analysis(self, inputs: AdvancedDCFInputs) -> Dict[str, DCFScenarioResult]:
        """Run bull/base/bear scenario analysis"""
        scenarios = {}
        
        # Define scenario adjustments
        scenario_params = {
            'bull': {
                'revenue_growth_multiplier': 1.3,
                'ebitda_margin_adjustment': 0.02,
                'terminal_growth_adjustment': 0.005,
                'beta_adjustment': -0.1
            },
            'base': {
                'revenue_growth_multiplier': 1.0,
                'ebitda_margin_adjustment': 0.0,
                'terminal_growth_adjustment': 0.0,
                'beta_adjustment': 0.0
            },
            'bear': {
                'revenue_growth_multiplier': 0.7,
                'ebitda_margin_adjustment': -0.03,
                'terminal_growth_adjustment': -0.005,
                'beta_adjustment': 0.2
            }
        }
        
        for scenario_name, adjustments in scenario_params.items():
            # Create adjusted inputs
            adjusted_inputs = self._create_scenario_inputs(inputs, adjustments)
            
            # Calculate scenario valuation
            result = await self._calculate_base_dcf(adjusted_inputs)
            
            # Calculate additional metrics
            revenue_cagr = (
                (adjusted_inputs.historical_revenues[-1] * 
                 (1 + np.mean(adjusted_inputs.revenue_growth_rates[:5])) ** 5) / 
                adjusted_inputs.historical_revenues[-1]
            ) ** (1/5) - 1
            
            implied_exit_multiple = result['terminal_value'] / result['projections']['ebitda'][-1]
            
            scenarios[scenario_name] = DCFScenarioResult(
                scenario_name=scenario_name,
                enterprise_value=result['enterprise_value'],
                equity_value=result['equity_value'],
                value_per_share=result['value_per_share'],
                wacc=result['wacc'],
                terminal_value=result['terminal_value'],
                terminal_value_pct=result['pv_results']['terminal_value_pct'],
                revenue_cagr=revenue_cagr,
                implied_exit_multiple=implied_exit_multiple
            )
        
        return scenarios
    
    def _create_scenario_inputs(self, base_inputs: AdvancedDCFInputs, adjustments: Dict[str, float]) -> AdvancedDCFInputs:
        """Create scenario-adjusted inputs"""
        import copy
        scenario_inputs = copy.deepcopy(base_inputs)
        
        # Adjust growth rates
        growth_multiplier = adjustments['revenue_growth_multiplier']
        scenario_inputs.revenue_growth_rates = [
            rate * growth_multiplier for rate in scenario_inputs.revenue_growth_rates
        ]
        
        # Adjust EBITDA margins
        margin_adjustment = adjustments['ebitda_margin_adjustment']
        scenario_inputs.ebitda_margin_targets = [
            margin + margin_adjustment for margin in scenario_inputs.ebitda_margin_targets
        ]
        
        # Adjust terminal growth
        scenario_inputs.terminal_growth_rate += adjustments['terminal_growth_adjustment']
        
        # Adjust beta
        scenario_inputs.beta += adjustments['beta_adjustment']
        
        return scenario_inputs
    
    async def _run_monte_carlo_simulation(
        self, 
        inputs: AdvancedDCFInputs, 
        confidence_levels: List[float]
    ) -> Dict[str, Any]:
        """Run comprehensive Monte Carlo simulation"""
        simulation_results = []
        
        # Use parallel processing for faster simulation
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            batch_size = self.simulation_runs // 4
            
            for i in range(4):
                future = executor.submit(
                    self._run_simulation_batch, 
                    inputs, 
                    batch_size, 
                    i * batch_size
                )
                futures.append(future)
            
            # Collect results
            for future in futures:
                batch_results = future.result()
                simulation_results.extend(batch_results)
        
        # Convert to numpy array for analysis
        results_array = np.array(simulation_results)
        results_array = results_array[results_array > 0]  # Remove invalid results
        
        # Calculate comprehensive statistics
        stats_dict = {
            'mean': np.mean(results_array),
            'median': np.median(results_array),
            'std': np.std(results_array),
            'skewness': stats.skew(results_array),
            'kurtosis': stats.kurtosis(results_array),
            'min': np.min(results_array),
            'max': np.max(results_array)
        }
        
        # Confidence intervals
        confidence_intervals = {}
        for level in confidence_levels:
            lower_pct = (1 - level) / 2 * 100
            upper_pct = (1 + level) / 2 * 100
            ci = np.percentile(results_array, [lower_pct, upper_pct])
            confidence_intervals[f'{level:.0%}'] = tuple(ci)
        
        # Value at Risk
        var_levels = [0.05, 0.10]
        value_at_risk = {}
        for level in var_levels:
            var_value = np.percentile(results_array, level * 100)
            value_at_risk[f'{level:.0%}'] = var_value
        
        # Additional risk metrics
        stats_dict['probability_of_loss'] = np.mean(results_array < 0)
        stats_dict['probability_positive'] = np.mean(results_array > 0)
        stats_dict['downside_deviation'] = np.std(results_array[results_array < np.mean(results_array)])
        
        return {
            'statistics': stats_dict,
            'confidence_intervals': confidence_intervals,
            'value_at_risk': value_at_risk,
            'raw_results': results_array.tolist()
        }
    
    def _run_simulation_batch(self, inputs: AdvancedDCFInputs, batch_size: int, seed_offset: int) -> List[float]:
        """Run a batch of Monte Carlo simulations"""
        np.random.seed(42 + seed_offset)  # Reproducible results
        batch_results = []
        
        for i in range(batch_size):
            try:
                # Generate random inputs
                sim_inputs = self._generate_stochastic_inputs(inputs)
                
                # Calculate valuation (simplified for speed)
                result = self._fast_dcf_calculation(sim_inputs)
                batch_results.append(result)
                
            except Exception:
                # Skip failed simulations
                continue
        
        return batch_results
    
    def _generate_stochastic_inputs(self, base_inputs: AdvancedDCFInputs) -> AdvancedDCFInputs:
        """Generate stochastic inputs for Monte Carlo simulation"""
        import copy
        sim_inputs = copy.deepcopy(base_inputs)
        
        # Revenue growth rates (correlated random walk)
        growth_volatility = 0.25
        for i, base_growth in enumerate(sim_inputs.revenue_growth_rates):
            # Add auto-correlation
            if i == 0:
                random_component = np.random.normal(0, growth_volatility * base_growth)
            else:
                # Mean reversion with persistence
                mean_reversion = 0.3
                persistence = 0.7
                previous_deviation = sim_inputs.revenue_growth_rates[i-1] - base_growth
                random_shock = np.random.normal(0, growth_volatility * base_growth * 0.5)
                random_component = persistence * previous_deviation * mean_reversion + random_shock
            
            sim_inputs.revenue_growth_rates[i] = max(base_growth + random_component, -0.5)
        
        # EBITDA margins (with bounds)
        margin_volatility = 0.15
        for i, base_margin in enumerate(sim_inputs.ebitda_margin_targets):
            random_margin = np.random.normal(base_margin, base_margin * margin_volatility)
            sim_inputs.ebitda_margin_targets[i] = np.clip(random_margin, 0.01, 0.8)
        
        # Discount rate components
        sim_inputs.risk_free_rate = np.random.normal(
            sim_inputs.risk_free_rate, 
            sim_inputs.risk_free_rate_std
        )
        sim_inputs.market_risk_premium = np.random.normal(
            sim_inputs.market_risk_premium, 
            sim_inputs.market_risk_premium_std
        )
        sim_inputs.beta = np.random.normal(sim_inputs.beta, sim_inputs.beta_std)
        
        # Terminal growth rate
        sim_inputs.terminal_growth_rate = np.random.normal(
            sim_inputs.terminal_growth_rate, 
            sim_inputs.terminal_growth_rate_std
        )
        
        return sim_inputs
    
    def _fast_dcf_calculation(self, inputs: AdvancedDCFInputs) -> float:
        """Fast DCF calculation for Monte Carlo simulation"""
        try:
            # Simplified WACC calculation
            cost_of_equity = inputs.risk_free_rate + inputs.beta * inputs.market_risk_premium
            debt_weight = inputs.debt_to_equity / (1 + inputs.debt_to_equity)
            equity_weight = 1 - debt_weight
            wacc = equity_weight * cost_of_equity + debt_weight * inputs.cost_of_debt * (1 - inputs.tax_rate)
            
            # Simplified cash flow projection
            base_revenue = inputs.historical_revenues[-1]
            total_pv = 0
            
            for year in range(self.projection_years):
                if year < len(inputs.revenue_growth_rates):
                    growth_rate = inputs.revenue_growth_rates[year]
                else:
                    growth_rate = inputs.terminal_growth_rate
                
                revenue = base_revenue * ((1 + growth_rate) ** (year + 1))
                
                if year < len(inputs.ebitda_margin_targets):
                    margin = inputs.ebitda_margin_targets[year]
                else:
                    margin = inputs.ebitda_margin_targets[-1] if inputs.ebitda_margin_targets else 0.15
                
                fcf = revenue * margin * 0.7  # Simplified FCF conversion
                pv_fcf = fcf / ((1 + wacc) ** (year + 1))
                total_pv += pv_fcf
            
            # Terminal value
            terminal_fcf = base_revenue * ((1 + inputs.terminal_growth_rate) ** (self.projection_years + 1))
            terminal_fcf *= inputs.ebitda_margin_targets[-1] if inputs.ebitda_margin_targets else 0.15
            terminal_fcf *= 0.7
            
            if wacc > inputs.terminal_growth_rate:
                terminal_value = terminal_fcf / (wacc - inputs.terminal_growth_rate)
                pv_terminal = terminal_value / ((1 + wacc) ** self.projection_years)
                total_pv += pv_terminal
            
            # Enterprise to equity value
            equity_value = total_pv + inputs.cash_and_equivalents - inputs.total_debt
            value_per_share = equity_value / inputs.shares_outstanding
            
            return value_per_share
            
        except Exception:
            return 0
    
    async def _comprehensive_sensitivity_analysis(self, inputs: AdvancedDCFInputs) -> Dict[str, Any]:
        """Comprehensive sensitivity analysis with tornado charts"""
        base_result = await self._calculate_base_dcf(inputs)
        base_value = base_result['value_per_share']
        
        # Variables to analyze
        sensitivity_vars = {
            'Revenue Growth (Y1)': ('revenue_growth_rates', 0, [-0.1, -0.05, 0, 0.05, 0.1]),
            'EBITDA Margin': ('ebitda_margin_targets', 0, [-0.05, -0.025, 0, 0.025, 0.05]),
            'Terminal Growth': ('terminal_growth_rate', None, [-0.01, -0.005, 0, 0.005, 0.01]),
            'WACC': ('risk_free_rate', None, [-0.01, -0.005, 0, 0.005, 0.01]),
            'Beta': ('beta', None, [-0.3, -0.15, 0, 0.15, 0.3]),
            'Tax Rate': ('tax_rate', None, [-0.05, -0.025, 0, 0.025, 0.05]),
            'CapEx % Revenue': ('capex_as_pct_revenue', 0, [-0.02, -0.01, 0, 0.01, 0.02])
        }
        
        sensitivity_results = {}
        tornado_data = {}
        
        for var_name, (attr_name, index, changes) in sensitivity_vars.items():
            var_results = {}
            
            for change in changes:
                # Create modified inputs
                import copy
                modified_inputs = copy.deepcopy(inputs)
                
                # Apply change
                if index is not None:
                    current_list = getattr(modified_inputs, attr_name)
                    if len(current_list) > index:
                        current_list[index] += change
                else:
                    current_value = getattr(modified_inputs, attr_name)
                    setattr(modified_inputs, attr_name, current_value + change)
                
                try:
                    result = await self._calculate_base_dcf(modified_inputs)
                    value_change = (result['value_per_share'] - base_value) / base_value * 100
                    var_results[f'{change:+.3f}'] = value_change
                except Exception:
                    var_results[f'{change:+.3f}'] = 0
            
            sensitivity_results[var_name] = var_results
            
            # Calculate range for tornado chart
            values = list(var_results.values())
            tornado_data[var_name] = {
                'range': max(values) - min(values),
                'low': min(values),
                'high': max(values)
            }
        
        # Create sensitivity matrix DataFrame
        sensitivity_df = pd.DataFrame(sensitivity_results)
        
        return {
            'sensitivity_matrix': sensitivity_df,
            'tornado_data': tornado_data,
            'sensitivity_rankings': sorted(
                tornado_data.items(), 
                key=lambda x: x[1]['range'], 
                reverse=True
            )
        }
    
    async def _calculate_risk_metrics(
        self, 
        inputs: AdvancedDCFInputs, 
        mc_stats: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        risk_metrics = {}
        
        if mc_stats and 'raw_results' in mc_stats:
            results = np.array(mc_stats['raw_results'])
            
            # Risk-adjusted return metrics
            mean_return = np.mean(results)
            std_return = np.std(results)
            
            # Sharpe ratio (using risk-free rate)
            excess_return = mean_return - inputs.risk_free_rate
            risk_metrics['sharpe_ratio'] = excess_return / std_return if std_return > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = results[results < mean_return]
            downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else std_return
            risk_metrics['sortino_ratio'] = excess_return / downside_deviation if downside_deviation > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = np.cumsum(results)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            risk_metrics['maximum_drawdown'] = np.min(drawdowns) if len(drawdowns) > 0 else 0
            
            # Value at Risk (5th percentile)
            risk_metrics['var_5pct'] = np.percentile(results, 5)
            
            # Expected shortfall (conditional VaR)
            var_5pct = risk_metrics['var_5pct']
            shortfall_values = results[results <= var_5pct]
            risk_metrics['expected_shortfall'] = np.mean(shortfall_values) if len(shortfall_values) > 0 else var_5pct
            
            # Probability of loss
            risk_metrics['probability_of_loss'] = np.mean(results < 0)
            
            # Risk-adjusted value (using downside deviation)
            risk_metrics['risk_adjusted_value'] = mean_return - 2 * downside_deviation
        
        return risk_metrics
    
    async def _optimize_capital_structure(self, inputs: AdvancedDCFInputs) -> Dict[str, Any]:
        """Optimize capital structure for maximum firm value"""
        def objective_function(debt_ratio):
            """Objective function for optimization"""
            try:
                import copy
                opt_inputs = copy.deepcopy(inputs)
                opt_inputs.debt_to_equity = debt_ratio / (1 - debt_ratio) if debt_ratio < 1 else 10
                
                # Recalculate with new capital structure
                result = asyncio.run(self._calculate_base_dcf(opt_inputs))
                return -result['enterprise_value']  # Minimize negative value
            except Exception:
                return 1e10  # Large penalty for invalid combinations
        
        # Optimize debt ratio
        result = optimize.minimize_scalar(
            objective_function,
            bounds=(0, 0.8),  # 0% to 80% debt
            method='bounded'
        )
        
        optimal_debt_ratio = result.x if result.success else inputs.debt_to_equity / (1 + inputs.debt_to_equity)
        
        # Calculate metrics at optimal structure
        import copy
        optimal_inputs = copy.deepcopy(inputs)
        optimal_inputs.debt_to_equity = optimal_debt_ratio / (1 - optimal_debt_ratio)
        optimal_result = await self._calculate_base_dcf(optimal_inputs)
        
        base_result = await self._calculate_base_dcf(inputs)
        value_improvement = optimal_result['enterprise_value'] - base_result['enterprise_value']
        
        return {
            'optimal_debt_ratio': optimal_debt_ratio,
            'optimal_debt_to_equity': optimal_inputs.debt_to_equity,
            'current_enterprise_value': base_result['enterprise_value'],
            'optimal_enterprise_value': optimal_result['enterprise_value'],
            'value_improvement': value_improvement,
            'improvement_percentage': value_improvement / base_result['enterprise_value'] * 100
        }
    
    async def _compile_advanced_outputs(
        self,
        base_result: Dict[str, Any],
        scenario_results: Dict[str, DCFScenarioResult],
        mc_stats: Dict[str, Any],
        sensitivity_results: Dict[str, Any],
        risk_metrics: Dict[str, float],
        optimization_results: Dict[str, Any]
    ) -> AdvancedDCFOutputs:
        """Compile comprehensive DCF outputs"""
        
        # Calculate probability-weighted value
        probability_weighted_value = 0
        if scenario_results:
            weights = {
                'bull': 0.25,
                'base': 0.50,
                'bear': 0.25
            }
            for scenario_name, result in scenario_results.items():
                weight = weights.get(scenario_name, 1/len(scenario_results))
                probability_weighted_value += result.value_per_share * weight
        else:
            probability_weighted_value = base_result['value_per_share']
        
        # Create projection summary
        projections = base_result['projections']
        projection_summary = pd.DataFrame({
            'Year': projections['years'],
            'Revenue': projections['revenues'],
            'EBITDA': projections['ebitda'],
            'EBIT': projections['ebit'],
            'Free Cash Flow': projections['free_cash_flow']
        })
        
        return AdvancedDCFOutputs(
            enterprise_value=base_result['enterprise_value'],
            equity_value=base_result['equity_value'],
            value_per_share=base_result['value_per_share'],
            scenario_results=scenario_results,
            probability_weighted_value=probability_weighted_value,
            monte_carlo_stats=mc_stats.get('statistics', {}),
            confidence_intervals=mc_stats.get('confidence_intervals', {}),
            value_at_risk=mc_stats.get('value_at_risk', {}),
            tornado_chart_data=sensitivity_results.get('tornado_data', {}),
            sensitivity_matrix=sensitivity_results.get('sensitivity_matrix', pd.DataFrame()),
            risk_adjusted_value=risk_metrics.get('risk_adjusted_value', base_result['value_per_share']),
            sharpe_ratio=risk_metrics.get('sharpe_ratio', 0),
            maximum_drawdown=risk_metrics.get('maximum_drawdown', 0),
            probability_of_loss=risk_metrics.get('probability_of_loss', 0),
            projection_summary=projection_summary,
            cash_flow_waterfall={
                'revenue': projections['revenues'],
                'ebitda': projections['ebitda'],
                'taxes': projections['taxes'],
                'capex': projections['capex'],
                'working_capital': projections['working_capital_change'],
                'fcf': projections['free_cash_flow']
            },
            model_r_squared=0.85,  # Placeholder - would calculate based on historical fit
            prediction_intervals={},  # Would be calculated from Monte Carlo
            peer_comparison={},  # Would compare to peer valuations
            optimal_capital_structure=optimization_results,
            sensitivity_rankings=sensitivity_results.get('sensitivity_rankings', [])
        )
    
    def _load_industry_benchmarks(self) -> Dict[str, Any]:
        """Load industry benchmark data"""
        # This would typically load from a database or external data source
        # Placeholder data structure
        return {
            'Technology': {
                'multiples': {
                    'ebitda_multiple': 15.0,
                    'revenue_multiple': 8.0,
                    'pe_multiple': 25.0
                },
                'margins': {
                    'ebitda_margin': 0.25,
                    'net_margin': 0.15
                }
            },
            'Healthcare': {
                'multiples': {
                    'ebitda_multiple': 12.0,
                    'revenue_multiple': 4.0,
                    'pe_multiple': 20.0
                },
                'margins': {
                    'ebitda_margin': 0.20,
                    'net_margin': 0.12
                }
            }
            # Add more industries as needed
        }
    
    def _project_growth_rates(self, historical_growth: List[float]) -> List[float]:
        """Project future growth rates based on historical data"""
        if not historical_growth:
            return [0.1] * 5  # Default 10% growth
        
        # Use exponential decay to terminal growth
        recent_growth = np.mean(historical_growth[-3:]) if len(historical_growth) >= 3 else historical_growth[-1]
        terminal_growth = 0.025
        
        projected_growth = []
        for year in range(10):  # Project 10 years
            decay_factor = 0.85 ** year
            growth_rate = decay_factor * recent_growth + (1 - decay_factor) * terminal_growth
            projected_growth.append(max(growth_rate, terminal_growth))
        
        return projected_growth
    
    def _project_margins(self, historical_margins: List[float]) -> List[float]:
        """Project future margins based on historical data"""
        if not historical_margins:
            return [0.15] * 10  # Default 15% EBITDA margin
        
        # Assume margins stabilize over time
        recent_margin = np.mean(historical_margins[-3:]) if len(historical_margins) >= 3 else historical_margins[-1]
        target_margin = min(recent_margin * 1.1, 0.35)  # Cap at 35%
        
        projected_margins = []
        for year in range(10):
            # Gradual improvement to target, then stabilize
            improvement_factor = min(year * 0.1, 1.0)
            margin = recent_margin + (target_margin - recent_margin) * improvement_factor
            projected_margins.append(margin)
        
        return projected_margins

# Utility functions
def create_advanced_dcf_model(**kwargs) -> AdvancedDCFModel:
    """Factory function for creating advanced DCF models"""
    return AdvancedDCFModel(**kwargs)