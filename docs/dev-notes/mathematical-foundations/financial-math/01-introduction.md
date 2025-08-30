# Financial Mathematics for IPO Valuation

## 💰 Chapter Overview

Financial mathematics provides the quantitative foundation for valuation models, risk assessment, and portfolio optimization in IPO analysis. This chapter covers essential concepts from time value of money to modern portfolio theory, with practical implementations for IPO valuation systems.

## 🎯 Learning Objectives

By the end of this chapter, you will:
- Master time value of money and discounting mathematics
- Calculate present values with varying discount rates
- Understand risk-adjusted return calculations (Sharpe ratio, CAPM)
- Apply option pricing mathematics (Black-Scholes basics)
- Implement portfolio optimization mathematics (Markowitz theory)

## 💼 Why Financial Mathematics Matters in IPO Valuation

### The Challenge
IPO valuation requires:
- **Future Cash Flow Valuation**: Estimating and discounting uncertain future profits
- **Risk Assessment**: Quantifying various types of investment risks
- **Market Comparison**: Comparing across different risk profiles and time horizons
- **Portfolio Context**: Understanding how IPO fits in broader investment portfolio

### The Solution
Financial mathematics provides:
- **Valuation Framework**: Systematic approach to estimate intrinsic value
- **Risk Metrics**: Quantitative measures of investment risk and return
- **Optimization Tools**: Methods to construct optimal investment portfolios
- **Comparative Analysis**: Standardized metrics for cross-company comparisons

## 🏗️ Chapter Structure

### Part I: Time Value of Money
1. **[Present Value Calculations](./02-present-value.md)**
   - Basic present value formulas
   - Discount rate determination
   - Risk-free rates and risk premiums
   - Terminal value calculations

2. **[Discounted Cash Flow Models](./03-dcf-models.md)**
   - Free cash flow modeling
   - WACC (Weighted Average Cost of Capital) calculations
   - Growth rate assumptions
   - Sensitivity analysis

### Part II: Risk and Return Mathematics
3. **[Risk Metrics](./04-risk-metrics.md)**
   - Standard deviation and volatility
   - Value at Risk (VaR) calculations
   - Expected Shortfall (Conditional VaR)
   - Maximum Drawdown analysis

4. **[Risk-Adjusted Returns](./05-risk-adjusted-returns.md)**
   - Sharpe ratio calculations
   - Information ratio
   - Treynor ratio
   - Jensen's alpha

### Part III: Asset Pricing Models
5. **[Capital Asset Pricing Model](./06-capm.md)**
   - Beta calculation methodologies
   - Security Market Line
   - Cost of equity estimation
   - Multi-factor models

6. **[Option Pricing Fundamentals](./07-option-pricing.md)**
   - Black-Scholes model basics
   - Greeks calculation
   - Real options in IPO valuation
   - Monte Carlo simulation methods

### Part IV: Portfolio Theory
7. **[Modern Portfolio Theory](./08-portfolio-theory.md)**
   - Mean-variance optimization
   - Efficient frontier construction
   - Risk budgeting
   - Portfolio rebalancing mathematics

## 💡 Key Concepts with Code Examples

### 1. Present Value Calculations
```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PresentValueCalculator:
    """Present value calculations for IPO valuation"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% risk-free rate
    
    def simple_present_value(self, future_value, discount_rate, periods):
        """Calculate present value with constant discount rate"""
        return future_value / (1 + discount_rate) ** periods
    
    def npv_cash_flows(self, cash_flows, discount_rate):
        """Calculate NPV of a series of cash flows"""
        npv = 0
        for t, cf in enumerate(cash_flows):
            npv += cf / (1 + discount_rate) ** t
        return npv
    
    def dcf_terminal_value(self, final_cash_flow, growth_rate, discount_rate):
        """Calculate terminal value using Gordon Growth Model"""
        if discount_rate <= growth_rate:
            raise ValueError("Discount rate must be greater than growth rate")
        return final_cash_flow * (1 + growth_rate) / (discount_rate - growth_rate)
    
    def dcf_valuation(self, projections, terminal_growth, discount_rate):
        """Complete DCF valuation"""
        # Present value of explicit forecast period
        pv_explicit = self.npv_cash_flows(projections, discount_rate)
        
        # Terminal value
        terminal_cf = projections[-1]
        terminal_value = self.dcf_terminal_value(terminal_cf, terminal_growth, discount_rate)
        pv_terminal = terminal_value / (1 + discount_rate) ** len(projections)
        
        return {
            'pv_explicit': pv_explicit,
            'pv_terminal': pv_terminal,
            'total_value': pv_explicit + pv_terminal,
            'terminal_value': terminal_value
        }
    
    def varying_discount_rates(self, cash_flows, discount_rates):
        """NPV with time-varying discount rates"""
        npv = 0
        cumulative_discount = 1
        
        for t, (cf, rate) in enumerate(zip(cash_flows, discount_rates)):
            if t > 0:
                cumulative_discount *= (1 + discount_rates[t-1])
            pv = cf / cumulative_discount
            npv += pv
            
        return npv
```

### 2. WACC Calculation
```python
class WACCCalculator:
    """Weighted Average Cost of Capital calculations"""
    
    def cost_of_equity_capm(self, risk_free_rate, beta, market_premium):
        """Calculate cost of equity using CAPM"""
        return risk_free_rate + beta * market_premium
    
    def cost_of_debt(self, interest_expense, total_debt, tax_rate):
        """Calculate after-tax cost of debt"""
        pre_tax_cost = interest_expense / total_debt
        return pre_tax_cost * (1 - tax_rate)
    
    def wacc(self, market_value_equity, market_value_debt, 
             cost_of_equity, cost_of_debt, tax_rate):
        """Calculate WACC"""
        total_value = market_value_equity + market_value_debt
        
        equity_weight = market_value_equity / total_value
        debt_weight = market_value_debt / total_value
        
        wacc = (equity_weight * cost_of_equity + 
                debt_weight * cost_of_debt * (1 - tax_rate))
        
        return {
            'wacc': wacc,
            'equity_weight': equity_weight,
            'debt_weight': debt_weight,
            'cost_of_equity': cost_of_equity,
            'after_tax_cost_of_debt': cost_of_debt * (1 - tax_rate)
        }
    
    def sensitivity_analysis(self, base_params, sensitivity_ranges):
        """Perform sensitivity analysis on WACC"""
        results = {}
        
        for param, range_values in sensitivity_ranges.items():
            wacc_values = []
            for value in range_values:
                params = base_params.copy()
                params[param] = value
                wacc_result = self.wacc(**params)
                wacc_values.append(wacc_result['wacc'])
            
            results[param] = {
                'values': range_values,
                'wacc': wacc_values,
                'sensitivity': np.std(wacc_values) / np.mean(wacc_values)  # Coefficient of variation
            }
        
        return results
```

### 3. Risk Metrics Implementation
```python
class RiskMetrics:
    """Comprehensive risk metrics for IPO analysis"""
    
    def volatility_metrics(self, returns):
        """Calculate various volatility measures"""
        return {
            'daily_volatility': np.std(returns),
            'annualized_volatility': np.std(returns) * np.sqrt(252),
            'rolling_volatility': pd.Series(returns).rolling(window=30).std(),
            'garch_volatility': self._garch_volatility(returns)
        }
    
    def _garch_volatility(self, returns, alpha=0.1, beta=0.85):
        """Simple GARCH(1,1) volatility estimation"""
        # Initialize
        long_run_var = np.var(returns)
        conditional_vars = [long_run_var]
        
        for i in range(1, len(returns)):
            # GARCH(1,1): σ²(t) = ω + α*ε²(t-1) + β*σ²(t-1)
            omega = long_run_var * (1 - alpha - beta)
            var_t = (omega + 
                    alpha * returns[i-1]**2 + 
                    beta * conditional_vars[i-1])
            conditional_vars.append(var_t)
        
        return np.sqrt(conditional_vars)
    
    def value_at_risk(self, returns, confidence_level=0.05, method='historical'):
        """Calculate Value at Risk using different methods"""
        if method == 'historical':
            return np.percentile(returns, confidence_level * 100)
        
        elif method == 'parametric':
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            from scipy import stats
            z_score = stats.norm.ppf(confidence_level)
            return mean_return + z_score * std_return
        
        elif method == 'monte_carlo':
            # Monte Carlo simulation
            np.random.seed(42)
            simulated_returns = np.random.normal(
                np.mean(returns), np.std(returns), 10000
            )
            return np.percentile(simulated_returns, confidence_level * 100)
    
    def expected_shortfall(self, returns, confidence_level=0.05):
        """Calculate Expected Shortfall (Conditional VaR)"""
        var = self.value_at_risk(returns, confidence_level, 'historical')
        return np.mean(returns[returns <= var])
    
    def maximum_drawdown(self, price_series):
        """Calculate maximum drawdown"""
        # Convert to numpy array if pandas series
        if isinstance(price_series, pd.Series):
            prices = price_series.values
        else:
            prices = price_series
        
        # Calculate running maximum
        peak = np.maximum.accumulate(prices)
        
        # Calculate drawdown
        drawdown = (prices - peak) / peak
        
        return {
            'max_drawdown': np.min(drawdown),
            'drawdown_series': drawdown,
            'peak_values': peak
        }
```

### 4. Risk-Adjusted Return Metrics
```python
class RiskAdjustedReturns:
    """Risk-adjusted return calculations"""
    
    def sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        excess_returns = np.mean(returns) - risk_free_rate/252  # Daily risk-free rate
        return excess_returns / np.std(returns) * np.sqrt(252)  # Annualized
    
    def information_ratio(self, portfolio_returns, benchmark_returns):
        """Calculate Information ratio"""
        active_returns = portfolio_returns - benchmark_returns
        return np.mean(active_returns) / np.std(active_returns) * np.sqrt(252)
    
    def treynor_ratio(self, returns, beta, risk_free_rate=0.02):
        """Calculate Treynor ratio"""
        excess_return = np.mean(returns) * 252 - risk_free_rate  # Annualized
        return excess_return / beta
    
    def jensens_alpha(self, portfolio_returns, market_returns, beta, risk_free_rate=0.02):
        """Calculate Jensen's Alpha"""
        portfolio_return = np.mean(portfolio_returns) * 252  # Annualized
        market_return = np.mean(market_returns) * 252
        
        expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
        return portfolio_return - expected_return
    
    def sortino_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sortino ratio (focuses on downside risk)"""
        excess_returns = returns - risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        if downside_deviation == 0:
            return np.inf if np.mean(excess_returns) > 0 else 0
        
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)
```

### 5. Black-Scholes Implementation
```python
class BlackScholesModel:
    """Black-Scholes option pricing model"""
    
    def __init__(self):
        from scipy.stats import norm
        self.norm = norm
    
    def black_scholes_call(self, S, K, T, r, sigma):
        """Calculate Black-Scholes call option price"""
        d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * self.norm.cdf(d1) - K * np.exp(-r * T) * self.norm.cdf(d2)
        return call_price
    
    def black_scholes_put(self, S, K, T, r, sigma):
        """Calculate Black-Scholes put option price"""
        d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * self.norm.cdf(-d2) - S * self.norm.cdf(-d1)
        return put_price
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option Greeks"""
        d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta (price sensitivity to underlying)
        if option_type == 'call':
            delta = self.norm.cdf(d1)
        else:
            delta = -self.norm.cdf(-d1)
        
        # Gamma (delta sensitivity to underlying)
        gamma = self.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta (time decay)
        if option_type == 'call':
            theta = (-(S * self.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - 
                    r * K * np.exp(-r * T) * self.norm.cdf(d2)) / 365
        else:
            theta = (-(S * self.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + 
                    r * K * np.exp(-r * T) * self.norm.cdf(-d2)) / 365
        
        # Vega (volatility sensitivity)
        vega = S * self.norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho (interest rate sensitivity)
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * self.norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * self.norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def implied_volatility(self, option_price, S, K, T, r, option_type='call'):
        """Calculate implied volatility using Newton-Raphson method"""
        def objective(sigma):
            if option_type == 'call':
                theoretical_price = self.black_scholes_call(S, K, T, r, sigma)
            else:
                theoretical_price = self.black_scholes_put(S, K, T, r, sigma)
            return theoretical_price - option_price
        
        def vega(sigma):
            d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
            return S * self.norm.pdf(d1) * np.sqrt(T)
        
        # Newton-Raphson iteration
        sigma = 0.2  # Initial guess
        for i in range(100):
            price_diff = objective(sigma)
            if abs(price_diff) < 1e-6:
                break
            
            vega_val = vega(sigma)
            if vega_val == 0:
                break
                
            sigma = sigma - price_diff / vega_val
            
            # Ensure sigma stays positive
            sigma = max(sigma, 1e-6)
        
        return sigma
```

### 6. Portfolio Optimization
```python
class PortfolioOptimizer:
    """Modern Portfolio Theory implementation"""
    
    def __init__(self):
        from scipy.optimize import minimize
        self.minimize = minimize
    
    def calculate_portfolio_metrics(self, weights, returns, cov_matrix):
        """Calculate portfolio return, risk, and Sharpe ratio"""
        portfolio_return = np.sum(returns * weights)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        
        return portfolio_return, portfolio_std
    
    def efficient_frontier(self, returns, cov_matrix, n_portfolios=100):
        """Generate efficient frontier"""
        n_assets = len(returns)
        results = np.zeros((4, n_portfolios))  # return, volatility, sharpe, weights
        
        # Target returns for efficient frontier
        target_returns = np.linspace(returns.min(), returns.max(), n_portfolios)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        ]
        
        bounds = tuple((0, 1) for _ in range(n_assets))  # Long-only
        
        for i, target in enumerate(target_returns):
            # Add return constraint
            cons = constraints + [{'type': 'eq', 'fun': lambda x, target=target: 
                                 np.sum(returns * x) - target}]
            
            # Minimize variance
            def objective(x):
                return np.dot(x.T, np.dot(cov_matrix, x))
            
            # Optimize
            result = self.minimize(objective, 
                                 x0=np.array([1/n_assets] * n_assets),
                                 method='SLSQP',
                                 bounds=bounds,
                                 constraints=cons)
            
            if result.success:
                weights = result.x
                port_return, port_std = self.calculate_portfolio_metrics(
                    weights, returns, cov_matrix)
                
                results[0, i] = port_return
                results[1, i] = port_std
                results[2, i] = (port_return - 0.02) / port_std  # Sharpe ratio
                results[3, i] = i  # Index for weights storage
        
        return results
    
    def maximum_sharpe_portfolio(self, returns, cov_matrix, risk_free_rate=0.02):
        """Find maximum Sharpe ratio portfolio"""
        n_assets = len(returns)
        
        def negative_sharpe(weights):
            port_return, port_std = self.calculate_portfolio_metrics(
                weights, returns, cov_matrix)
            return -(port_return - risk_free_rate) / port_std
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        result = self.minimize(negative_sharpe,
                             x0=np.array([1/n_assets] * n_assets),
                             method='SLSQP',
                             bounds=bounds,
                             constraints=constraints)
        
        optimal_weights = result.x
        port_return, port_std = self.calculate_portfolio_metrics(
            optimal_weights, returns, cov_matrix)
        sharpe_ratio = (port_return - risk_free_rate) / port_std
        
        return {
            'weights': optimal_weights,
            'expected_return': port_return,
            'volatility': port_std,
            'sharpe_ratio': sharpe_ratio
        }
    
    def minimum_variance_portfolio(self, cov_matrix):
        """Find minimum variance portfolio"""
        n_assets = cov_matrix.shape[0]
        
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        result = self.minimize(objective,
                             x0=np.array([1/n_assets] * n_assets),
                             method='SLSQP',
                             bounds=bounds,
                             constraints=constraints)
        
        return result.x
```

## 🎯 IPO-Specific Applications

### 1. IPO Valuation Model
```python
def comprehensive_ipo_valuation():
    """Comprehensive IPO valuation combining multiple approaches"""
    
    class IPOValuationModel:
        def __init__(self, company_data, market_data, comparable_companies):
            self.company = company_data
            self.market = market_data
            self.comparables = comparable_companies
            
        def dcf_valuation(self, projection_years=5):
            """DCF valuation for IPO"""
            # Revenue projections
            revenue_growth_rates = self.estimate_growth_rates()
            revenue_projections = self.project_revenues(revenue_growth_rates, projection_years)
            
            # Cash flow projections
            cash_flow_projections = self.project_cash_flows(revenue_projections)
            
            # Terminal value
            terminal_growth = self.estimate_terminal_growth()
            wacc = self.calculate_wacc()
            
            # DCF calculation
            pv_calculator = PresentValueCalculator()
            valuation = pv_calculator.dcf_valuation(
                cash_flow_projections, terminal_growth, wacc
            )
            
            return valuation
        
        def comparable_company_analysis(self):
            """Valuation using comparable companies"""
            # Calculate multiples for comparable companies
            multiples = {}
            for metric in ['P/E', 'EV/Revenue', 'EV/EBITDA', 'P/B']:
                comp_multiples = [comp[metric] for comp in self.comparables 
                                if comp[metric] is not None]
                multiples[metric] = {
                    'median': np.median(comp_multiples),
                    'mean': np.mean(comp_multiples),
                    'std': np.std(comp_multiples)
                }
            
            # Apply to target company
            valuations = {}
            for metric, stats in multiples.items():
                if metric in self.company:
                    valuations[f'{metric}_median'] = (
                        self.company[metric.split('/')[1]] * stats['median']
                    )
                    valuations[f'{metric}_mean'] = (
                        self.company[metric.split('/')[1]] * stats['mean']
                    )
            
            return valuations
        
        def risk_adjusted_valuation(self):
            """Risk-adjusted valuation using real options"""
            # Base case DCF
            base_dcf = self.dcf_valuation()
            
            # Risk adjustments
            market_risk_adjustment = self.calculate_market_risk_premium()
            company_specific_risk = self.assess_company_risk()
            
            # Real options value (expansion, abandonment, etc.)
            real_options_value = self.estimate_real_options_value()
            
            adjusted_value = (base_dcf['total_value'] * 
                            (1 - market_risk_adjustment - company_specific_risk) + 
                            real_options_value)
            
            return {
                'base_dcf': base_dcf['total_value'],
                'risk_adjustments': market_risk_adjustment + company_specific_risk,
                'real_options': real_options_value,
                'adjusted_value': adjusted_value
            }
        
        def monte_carlo_valuation(self, n_simulations=10000):
            """Monte Carlo simulation for valuation uncertainty"""
            np.random.seed(42)
            valuations = []
            
            for _ in range(n_simulations):
                # Sample uncertain parameters
                revenue_growth = np.random.normal(0.15, 0.05)  # 15% ± 5%
                margin_improvement = np.random.normal(0.02, 0.01)  # 2% ± 1%
                terminal_growth = np.random.normal(0.03, 0.01)  # 3% ± 1%
                risk_premium = np.random.normal(0.06, 0.02)  # 6% ± 2%
                
                # Calculate valuation with sampled parameters
                simulated_valuation = self._simulate_valuation(
                    revenue_growth, margin_improvement, terminal_growth, risk_premium
                )
                valuations.append(simulated_valuation)
            
            valuations = np.array(valuations)
            
            return {
                'mean': np.mean(valuations),
                'median': np.median(valuations),
                'std': np.std(valuations),
                'var_95': np.percentile(valuations, 5),
                'var_5': np.percentile(valuations, 95),
                'distribution': valuations
            }
    
    return IPOValuationModel
```

## 🚨 Common Pitfalls and Solutions

### 1. Terminal Value Sensitivity
```python
def terminal_value_sensitivity_analysis():
    """Analyze sensitivity to terminal value assumptions"""
    
    def sensitivity_to_growth_and_discount(base_cf, growth_range, discount_range):
        """2D sensitivity analysis"""
        results = np.zeros((len(growth_range), len(discount_range)))
        
        for i, g in enumerate(growth_range):
            for j, r in enumerate(discount_range):
                if r > g:  # Ensure discount rate > growth rate
                    terminal_value = base_cf * (1 + g) / (r - g)
                    results[i, j] = terminal_value
                else:
                    results[i, j] = np.nan
        
        return results
    
    return sensitivity_to_growth_and_discount
```

### 2. Circular Reference in WACC
```python
def iterative_wacc_calculation():
    """Solve circular reference in WACC calculation"""
    
    def solve_wacc_iteratively(debt_value, equity_shares, share_price_initial,
                              cost_of_debt, risk_free_rate, market_premium, beta, 
                              tax_rate, tolerance=1e-6, max_iterations=100):
        """Iteratively solve for WACC and firm value"""
        
        share_price = share_price_initial
        
        for iteration in range(max_iterations):
            # Current market values
            equity_value = equity_shares * share_price
            total_value = equity_value + debt_value
            
            # Weights
            equity_weight = equity_value / total_value
            debt_weight = debt_value / total_value
            
            # Cost of equity (CAPM)
            cost_of_equity = risk_free_rate + beta * market_premium
            
            # WACC
            wacc = (equity_weight * cost_of_equity + 
                   debt_weight * cost_of_debt * (1 - tax_rate))
            
            # New firm value (from DCF)
            # This would come from your DCF model
            new_firm_value = dcf_value_function(wacc)  # Placeholder
            new_equity_value = new_firm_value - debt_value
            new_share_price = new_equity_value / equity_shares
            
            # Check convergence
            if abs(new_share_price - share_price) < tolerance:
                break
                
            share_price = new_share_price
        
        return {
            'wacc': wacc,
            'firm_value': new_firm_value,
            'equity_value': new_equity_value,
            'share_price': new_share_price,
            'iterations': iteration + 1
        }
    
    return solve_wacc_iteratively
```

## 📚 Prerequisites for Next Chapters

Before proceeding to Advanced ML Mathematics, ensure you understand:
- Time value of money calculations
- Risk and return relationships
- Portfolio optimization theory
- Option pricing fundamentals
- Valuation model integration

---

**Next**: [Advanced ML Mathematics](../advanced-ml/01-introduction.md)