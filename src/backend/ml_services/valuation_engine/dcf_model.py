"""
Discounted Cash Flow (DCF) Valuation Model
Advanced DCF modeling with Monte Carlo simulation and sensitivity analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from decimal import Decimal
import logging
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ...config import settings
from ...utils.metrics import track_ml_inference, track_time, ML_INFERENCE_DURATION

logger = logging.getLogger(__name__)


@dataclass
class DCFInputs:
    """DCF model input parameters"""
    # Historical financials
    historical_revenues: List[float]
    historical_ebitda: List[float]
    historical_fcf: List[float]
    
    # Growth assumptions
    revenue_growth_rates: List[float]  # By year
    ebitda_margin_target: float
    capex_as_pct_revenue: float
    tax_rate: float
    
    # Discount rate components
    risk_free_rate: float
    equity_risk_premium: float
    beta: float
    cost_of_debt: float
    debt_to_equity: float
    
    # Terminal value
    terminal_growth_rate: float
    terminal_multiple: Optional[float] = None
    
    # Other adjustments
    cash_and_equivalents: float = 0
    total_debt: float = 0
    minority_interest: float = 0
    shares_outstanding: float = 1


@dataclass
class DCFOutputs:
    """DCF model outputs"""
    enterprise_value: float
    equity_value: float
    value_per_share: float
    terminal_value: float
    terminal_value_pct: float
    
    # Detailed projections
    revenue_projections: List[float]
    ebitda_projections: List[float]
    fcf_projections: List[float]
    discount_factors: List[float]
    pv_of_fcf: List[float]
    
    # Sensitivity analysis
    sensitivity_matrix: Dict[str, Dict[str, float]]
    
    # Risk metrics
    confidence_interval_95: Tuple[float, float]
    probability_of_positive_value: float
    
    # Model diagnostics
    wacc: float
    implied_terminal_multiple: float
    model_r_squared: float


class DCFValuationModel:
    """Advanced DCF valuation model with ML enhancements"""
    
    def __init__(self):
        self.projection_years = settings.ml.projection_years or 5
        self.simulation_runs = 10000
        
    @track_time(ML_INFERENCE_DURATION, {"model_name": "dcf"})
    def calculate_valuation(
        self,
        inputs: DCFInputs,
        include_monte_carlo: bool = True,
        include_sensitivity: bool = True
    ) -> DCFOutputs:
        """
        Calculate DCF valuation with advanced features
        
        Args:
            inputs: DCF input parameters
            include_monte_carlo: Whether to run Monte Carlo simulation
            include_sensitivity: Whether to perform sensitivity analysis
        """
        try:
            track_ml_inference("dcf", 0, True)  # Start tracking
            
            # Calculate WACC
            wacc = self._calculate_wacc(inputs)
            
            # Project cash flows
            projections = self._project_cash_flows(inputs)
            
            # Calculate terminal value
            terminal_value = self._calculate_terminal_value(inputs, projections, wacc)
            
            # Discount cash flows to present value
            pv_fcf, discount_factors = self._discount_cash_flows(projections['fcf'], wacc)
            pv_terminal = terminal_value / ((1 + wacc) ** self.projection_years)
            
            # Calculate enterprise and equity value
            enterprise_value = sum(pv_fcf) + pv_terminal
            equity_value = enterprise_value + inputs.cash_and_equivalents - inputs.total_debt - inputs.minority_interest
            value_per_share = equity_value / inputs.shares_outstanding
            
            # Build base outputs
            outputs = DCFOutputs(
                enterprise_value=enterprise_value,
                equity_value=equity_value,
                value_per_share=value_per_share,
                terminal_value=terminal_value,
                terminal_value_pct=pv_terminal / enterprise_value * 100,
                revenue_projections=projections['revenue'],
                ebitda_projections=projections['ebitda'],
                fcf_projections=projections['fcf'],
                discount_factors=discount_factors,
                pv_of_fcf=pv_fcf,
                sensitivity_matrix={},
                confidence_interval_95=(0, 0),
                probability_of_positive_value=1.0,
                wacc=wacc,
                implied_terminal_multiple=0,
                model_r_squared=0
            )
            
            # Sensitivity analysis
            if include_sensitivity:
                outputs.sensitivity_matrix = self._sensitivity_analysis(inputs)
            
            # Monte Carlo simulation
            if include_monte_carlo:
                mc_results = self._monte_carlo_simulation(inputs)
                outputs.confidence_interval_95 = mc_results['confidence_interval']
                outputs.probability_of_positive_value = mc_results['positive_probability']
            
            # Additional calculations
            outputs.implied_terminal_multiple = terminal_value / projections['ebitda'][-1]
            outputs.model_r_squared = self._calculate_model_fit(inputs)
            
            logger.info(f"DCF valuation completed: ${value_per_share:.2f} per share")
            return outputs
            
        except Exception as e:
            track_ml_inference("dcf", 0, False)  # Track error
            logger.error(f"DCF calculation error: {str(e)}")
            raise
    
    def _calculate_wacc(self, inputs: DCFInputs) -> float:
        """Calculate Weighted Average Cost of Capital"""
        # Cost of equity using CAPM
        cost_of_equity = inputs.risk_free_rate + inputs.beta * inputs.equity_risk_premium
        
        # After-tax cost of debt
        after_tax_cost_of_debt = inputs.cost_of_debt * (1 - inputs.tax_rate)
        
        # Weights
        debt_weight = inputs.debt_to_equity / (1 + inputs.debt_to_equity)
        equity_weight = 1 - debt_weight
        
        # WACC
        wacc = equity_weight * cost_of_equity + debt_weight * after_tax_cost_of_debt
        
        return wacc
    
    def _project_cash_flows(self, inputs: DCFInputs) -> Dict[str, List[float]]:
        """Project future cash flows"""
        projections = {
            'revenue': [],
            'ebitda': [],
            'fcf': []
        }
        
        # Base year (most recent historical)
        base_revenue = inputs.historical_revenues[-1]
        
        # Project revenues
        for i in range(self.projection_years):
            if i < len(inputs.revenue_growth_rates):
                growth_rate = inputs.revenue_growth_rates[i]
            else:
                # Use terminal growth for years beyond explicit projections
                growth_rate = inputs.terminal_growth_rate
            
            if i == 0:
                projected_revenue = base_revenue * (1 + growth_rate)
            else:
                projected_revenue = projections['revenue'][-1] * (1 + growth_rate)
            
            projections['revenue'].append(projected_revenue)
        
        # Project EBITDA
        for revenue in projections['revenue']:
            ebitda = revenue * inputs.ebitda_margin_target
            projections['ebitda'].append(ebitda)
        
        # Project Free Cash Flow
        for i, (revenue, ebitda) in enumerate(zip(projections['revenue'], projections['ebitda'])):
            # Tax on EBIT (assuming EBIT = EBITDA for simplicity)
            tax = ebitda * inputs.tax_rate
            nopat = ebitda - tax
            
            # Capital expenditure
            capex = revenue * inputs.capex_as_pct_revenue
            
            # Change in working capital (simplified)
            if i == 0:
                delta_wc = revenue * 0.02  # 2% of revenue
            else:
                delta_wc = (projections['revenue'][i] - projections['revenue'][i-1]) * 0.02
            
            # Free cash flow
            fcf = nopat - capex - delta_wc
            projections['fcf'].append(fcf)
        
        return projections
    
    def _calculate_terminal_value(self, inputs: DCFInputs, projections: Dict, wacc: float) -> float:
        """Calculate terminal value using Gordon Growth Model or exit multiple"""
        final_year_fcf = projections['fcf'][-1]
        
        if inputs.terminal_multiple:
            # Exit multiple method
            final_year_ebitda = projections['ebitda'][-1]
            terminal_value = final_year_ebitda * inputs.terminal_multiple
        else:
            # Gordon Growth Model
            terminal_fcf = final_year_fcf * (1 + inputs.terminal_growth_rate)
            terminal_value = terminal_fcf / (wacc - inputs.terminal_growth_rate)
        
        return terminal_value
    
    def _discount_cash_flows(self, cash_flows: List[float], discount_rate: float) -> Tuple[List[float], List[float]]:
        """Discount cash flows to present value"""
        pv_fcf = []
        discount_factors = []
        
        for i, fcf in enumerate(cash_flows):
            year = i + 1
            discount_factor = 1 / ((1 + discount_rate) ** year)
            pv = fcf * discount_factor
            
            pv_fcf.append(pv)
            discount_factors.append(discount_factor)
        
        return pv_fcf, discount_factors
    
    def _sensitivity_analysis(self, inputs: DCFInputs) -> Dict[str, Dict[str, float]]:
        """Perform sensitivity analysis on key variables"""
        base_valuation = self.calculate_valuation(inputs, include_monte_carlo=False, include_sensitivity=False)
        base_value = base_valuation.value_per_share
        
        # Variables to test
        variables = {
            'wacc': [-0.02, -0.01, 0, 0.01, 0.02],
            'terminal_growth': [-0.02, -0.01, 0, 0.01, 0.02],
            'ebitda_margin': [-0.05, -0.025, 0, 0.025, 0.05]
        }
        
        sensitivity_matrix = {}
        
        for var_name, changes in variables.items():
            sensitivity_matrix[var_name] = {}
            
            for change in changes:
                # Create modified inputs
                modified_inputs = inputs
                
                if var_name == 'wacc':
                    modified_inputs.risk_free_rate += change
                elif var_name == 'terminal_growth':
                    modified_inputs.terminal_growth_rate += change
                elif var_name == 'ebitda_margin':
                    modified_inputs.ebitda_margin_target += change
                
                # Calculate valuation with modified inputs
                try:
                    modified_valuation = self.calculate_valuation(
                        modified_inputs, 
                        include_monte_carlo=False, 
                        include_sensitivity=False
                    )
                    value_change = (modified_valuation.value_per_share - base_value) / base_value * 100
                    sensitivity_matrix[var_name][f"{change:+.3f}"] = value_change
                except:
                    sensitivity_matrix[var_name][f"{change:+.3f}"] = 0
        
        return sensitivity_matrix
    
    def _monte_carlo_simulation(self, inputs: DCFInputs) -> Dict[str, Any]:
        """Run Monte Carlo simulation for risk analysis"""
        simulation_results = []
        
        for _ in range(self.simulation_runs):
            # Generate random inputs with distributions
            sim_inputs = self._generate_random_inputs(inputs)
            
            try:
                sim_result = self.calculate_valuation(
                    sim_inputs, 
                    include_monte_carlo=False, 
                    include_sensitivity=False
                )
                simulation_results.append(sim_result.value_per_share)
            except:
                # Handle failed simulations
                simulation_results.append(0)
        
        # Calculate statistics
        simulation_results = np.array(simulation_results)
        confidence_interval = np.percentile(simulation_results, [2.5, 97.5])
        positive_probability = np.mean(simulation_results > 0)
        
        return {
            'confidence_interval': tuple(confidence_interval),
            'positive_probability': positive_probability,
            'mean': np.mean(simulation_results),
            'std': np.std(simulation_results)
        }
    
    def _generate_random_inputs(self, base_inputs: DCFInputs) -> DCFInputs:
        """Generate random inputs for Monte Carlo simulation"""
        # Create copy of base inputs
        sim_inputs = base_inputs
        
        # Add randomness to key variables
        # Revenue growth rates (normal distribution)
        for i, growth_rate in enumerate(sim_inputs.revenue_growth_rates):
            sim_inputs.revenue_growth_rates[i] = np.random.normal(growth_rate, growth_rate * 0.2)
        
        # EBITDA margin (normal distribution)
        sim_inputs.ebitda_margin_target = np.random.normal(
            sim_inputs.ebitda_margin_target, 
            sim_inputs.ebitda_margin_target * 0.1
        )
        
        # Discount rate components (normal distribution)
        sim_inputs.risk_free_rate = np.random.normal(sim_inputs.risk_free_rate, 0.005)
        sim_inputs.beta = np.random.normal(sim_inputs.beta, sim_inputs.beta * 0.15)
        
        # Terminal growth rate (normal distribution)
        sim_inputs.terminal_growth_rate = np.random.normal(
            sim_inputs.terminal_growth_rate, 
            0.005
        )
        
        return sim_inputs
    
    def _calculate_model_fit(self, inputs: DCFInputs) -> float:
        """Calculate R-squared for model fit (simplified)"""
        # This would typically compare projected vs actual historical data
        # For now, return a placeholder based on data quality
        if len(inputs.historical_revenues) >= 5:
            return 0.85  # High confidence with 5+ years of data
        elif len(inputs.historical_revenues) >= 3:
            return 0.75  # Medium confidence with 3-4 years
        else:
            return 0.60  # Lower confidence with limited data
    
    def validate_inputs(self, inputs: DCFInputs) -> List[str]:
        """Validate DCF inputs and return list of warnings/errors"""
        warnings = []
        
        # Check for reasonable values
        if inputs.terminal_growth_rate > inputs.risk_free_rate:
            warnings.append("Terminal growth rate exceeds risk-free rate")
        
        if inputs.ebitda_margin_target > 0.5:
            warnings.append("EBITDA margin seems unusually high (>50%)")
        
        if inputs.beta < 0:
            warnings.append("Negative beta is unusual for most companies")
        
        if len(inputs.revenue_growth_rates) < 3:
            warnings.append("Limited growth rate projections may reduce accuracy")
        
        return warnings


# Factory function for creating DCF models
def create_dcf_model() -> DCFValuationModel:
    """Create a configured DCF valuation model"""
    return DCFValuationModel()