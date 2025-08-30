# Calculus for ML/NLP in Financial Valuation

This section covers essential calculus concepts for implementing and optimizing ML/NLP models in financial valuation.

## 📊 Table of Contents

1. [Partial Derivatives for Gradient Descent](#partial-derivatives)
2. [Chain Rule for Backpropagation](#chain-rule)
3. [Optimization Techniques (SGD, Adam)](#optimization-techniques)
4. [Loss Function Derivatives](#loss-function-derivatives)

---

## 🟢 Partial Derivatives for Gradient Descent {#partial-derivatives}

### Intuitive Understanding

Partial derivatives tell us how a function changes when we change one variable while keeping others constant:
- **Gradient**: Direction of steepest increase
- **Optimization**: Find minimum by moving in negative gradient direction
- **Financial applications**: Minimize portfolio risk, maximize Sharpe ratio, optimize model parameters

Think of it like adjusting portfolio weights:
- How does portfolio risk change if I increase Apple stock by 1%?
- How does expected return change if I decrease bond allocation?

### 🟡 Practical Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from typing import Callable, Tuple, Dict, Any, List
import warnings

class FinancialCalculus:
    """
    Calculus operations for financial modeling and optimization
    """
    
    @staticmethod
    def partial_derivative(f: Callable, x: np.ndarray, var_index: int, 
                          h: float = 1e-8) -> float:
        """
        Compute partial derivative using finite differences
        
        Args:
            f: Function to differentiate
            x: Point at which to compute derivative
            var_index: Index of variable to differentiate with respect to
            h: Step size
        
        Returns:
            Partial derivative value
        """
        x_forward = x.copy()
        x_backward = x.copy()
        
        x_forward[var_index] += h
        x_backward[var_index] -= h
        
        # Central difference formula
        return (f(x_forward) - f(x_backward)) / (2 * h)
    
    @staticmethod
    def gradient_vector(f: Callable, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        """
        Compute full gradient vector
        
        Args:
            f: Function to differentiate  
            x: Point at which to compute gradient
            h: Step size
        
        Returns:
            Gradient vector
        """
        gradient = np.zeros_like(x)
        
        for i in range(len(x)):
            gradient[i] = FinancialCalculus.partial_derivative(f, x, i, h)
        
        return gradient
    
    @staticmethod
    def portfolio_risk_derivatives(weights: np.ndarray, 
                                 cov_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analytical derivatives for portfolio risk metrics
        
        Args:
            weights: Portfolio weights
            cov_matrix: Asset covariance matrix
        
        Returns:
            Dictionary with various risk derivatives
        """
        # Portfolio variance: σ²_p = w^T Σ w
        portfolio_variance = weights.T @ cov_matrix @ weights
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Derivative of variance: ∂σ²_p/∂w = 2Σw
        variance_gradient = 2 * cov_matrix @ weights
        
        # Derivative of volatility: ∂σ_p/∂w = (Σw)/σ_p
        if portfolio_std > 0:
            volatility_gradient = (cov_matrix @ weights) / portfolio_std
        else:
            volatility_gradient = np.zeros_like(weights)
        
        # Marginal contributions to risk (used in risk budgeting)
        if portfolio_std > 0:
            marginal_contributions = volatility_gradient
            component_contributions = weights * marginal_contributions
            percentage_contributions = component_contributions / portfolio_variance * 100
        else:
            marginal_contributions = np.zeros_like(weights)
            component_contributions = np.zeros_like(weights)
            percentage_contributions = np.zeros_like(weights)
        
        return {
            'portfolio_variance': portfolio_variance,
            'portfolio_volatility': portfolio_std,
            'variance_gradient': variance_gradient,
            'volatility_gradient': volatility_gradient,
            'marginal_contributions': marginal_contributions,
            'component_contributions': component_contributions,
            'percentage_contributions': percentage_contributions
        }
    
    @staticmethod
    def sharpe_ratio_derivatives(weights: np.ndarray,
                               mean_returns: np.ndarray,
                               cov_matrix: np.ndarray,
                               risk_free_rate: float = 0.0) -> Dict[str, Any]:
        """
        Analytical derivatives for Sharpe ratio
        
        Sharpe ratio: S = (μ_p - r_f) / σ_p
        where μ_p = w^T μ and σ_p = sqrt(w^T Σ w)
        """
        excess_returns = mean_returns - risk_free_rate
        portfolio_return = weights @ excess_returns
        portfolio_variance = weights @ cov_matrix @ weights
        portfolio_std = np.sqrt(portfolio_variance)
        
        if portfolio_std == 0:
            return {
                'sharpe_ratio': 0,
                'sharpe_gradient': np.zeros_like(weights),
                'return_component': np.zeros_like(weights),
                'risk_component': np.zeros_like(weights)
            }
        
        sharpe_ratio = portfolio_return / portfolio_std
        
        # Derivative using quotient rule: d/dx(f/g) = (g*f' - f*g')/g²
        # f = portfolio_return, g = portfolio_std
        f_prime = excess_returns  # ∂μ_p/∂w
        g_prime = (cov_matrix @ weights) / portfolio_std  # ∂σ_p/∂w
        
        sharpe_gradient = (portfolio_std * f_prime - portfolio_return * g_prime) / portfolio_variance
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sharpe_gradient': sharpe_gradient,
            'return_component': f_prime,  # Contribution from return
            'risk_component': g_prime,    # Contribution from risk
            'portfolio_return': portfolio_return,
            'portfolio_std': portfolio_std
        }
    
    @staticmethod
    def option_greeks(S: float, K: float, T: float, r: float, sigma: float,
                     option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate option Greeks (derivatives of option price)
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
        
        Returns:
            Dictionary with Greeks
        """
        from scipy.stats import norm
        import math
        
        # Black-Scholes parameters
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        
        # Standard normal CDF and PDF
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)
        
        if option_type == 'call':
            # Call option price
            option_price = S * N_d1 - K * math.exp(-r*T) * N_d2
            
            # Delta: ∂V/∂S
            delta = N_d1
            
            # Theta: ∂V/∂T (time decay)
            theta = -(S * n_d1 * sigma)/(2*math.sqrt(T)) - r*K*math.exp(-r*T)*N_d2
            
        else:  # put option
            # Put option price
            option_price = K * math.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            # Delta: ∂V/∂S
            delta = N_d1 - 1
            
            # Theta: ∂V/∂T
            theta = -(S * n_d1 * sigma)/(2*math.sqrt(T)) + r*K*math.exp(-r*T)*norm.cdf(-d2)
        
        # Gamma: ∂²V/∂S² (same for calls and puts)
        gamma = n_d1 / (S * sigma * math.sqrt(T))
        
        # Vega: ∂V/∂σ (same for calls and puts)
        vega = S * n_d1 * math.sqrt(T)
        
        # Rho: ∂V/∂r
        if option_type == 'call':
            rho = K * T * math.exp(-r*T) * N_d2
        else:
            rho = -K * T * math.exp(-r*T) * norm.cdf(-d2)
        
        return {
            'option_price': option_price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365,  # Per day
            'vega': vega / 100,    # Per 1% volatility
            'rho': rho / 100       # Per 1% interest rate
        }
    
    @staticmethod
    def value_at_risk_derivative(returns: np.ndarray, 
                               weights: np.ndarray,
                               confidence_level: float = 0.05) -> Dict[str, Any]:
        """
        Compute derivative of Value at Risk with respect to portfolio weights
        
        This is more complex as VaR is not differentiable everywhere
        """
        # Portfolio returns
        portfolio_returns = returns @ weights
        
        # Historical VaR
        var_value = np.percentile(portfolio_returns, confidence_level * 100)
        
        # For normal distribution approximation
        portfolio_mean = np.mean(portfolio_returns)
        portfolio_std = np.std(portfolio_returns, ddof=1)
        
        # Parametric VaR (assuming normal distribution)
        z_score = stats.norm.ppf(confidence_level)
        parametric_var = portfolio_mean + z_score * portfolio_std
        
        # Gradient of parametric VaR
        mean_gradient = np.mean(returns, axis=0)
        
        # Gradient of portfolio standard deviation
        portfolio_variance = weights @ np.cov(returns.T) @ weights
        if portfolio_variance > 0:
            std_gradient = (np.cov(returns.T) @ weights) / np.sqrt(portfolio_variance)
        else:
            std_gradient = np.zeros_like(weights)
        
        var_gradient = mean_gradient + z_score * std_gradient
        
        return {
            'historical_var': var_value,
            'parametric_var': parametric_var,
            'var_gradient': var_gradient,
            'portfolio_mean': portfolio_mean,
            'portfolio_std': portfolio_std,
            'confidence_level': confidence_level
        }
    
    @staticmethod
    def visualize_gradient_descent(objective_func: Callable,
                                 gradient_func: Callable,
                                 x_range: Tuple[float, float],
                                 y_range: Tuple[float, float],
                                 start_point: np.ndarray,
                                 learning_rate: float = 0.1,
                                 n_iterations: int = 50) -> Dict[str, Any]:
        """
        Visualize gradient descent optimization process
        """
        # Create mesh for contour plot
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate function on grid
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = objective_func(np.array([X[i, j], Y[i, j]]))
        
        # Run gradient descent
        current_point = start_point.copy()
        path = [current_point.copy()]
        function_values = [objective_func(current_point)]
        
        for i in range(n_iterations):
            grad = gradient_func(current_point)
            current_point = current_point - learning_rate * grad
            path.append(current_point.copy())
            function_values.append(objective_func(current_point))
        
        path = np.array(path)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Contour plot with gradient descent path
        contour = ax1.contour(X, Y, Z, levels=20, alpha=0.6)
        ax1.clabel(contour, inline=True, fontsize=8)
        
        # Plot gradient descent path
        ax1.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=4, alpha=0.8)
        ax1.scatter(path[0, 0], path[0, 1], color='green', s=100, label='Start', zorder=5)
        ax1.scatter(path[-1, 0], path[-1, 1], color='red', s=100, label='End', zorder=5)
        
        ax1.set_title('Gradient Descent Path')
        ax1.set_xlabel('x₁')
        ax1.set_ylabel('x₂')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Convergence plot
        ax2.plot(function_values, 'b-', linewidth=2)
        ax2.set_title('Objective Function Convergence')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Function Value')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'path': path,
            'function_values': function_values,
            'final_point': path[-1],
            'final_value': function_values[-1],
            'n_iterations': len(path) - 1
        }

def demonstrate_partial_derivatives():
    """Demonstrate partial derivatives in financial context"""
    
    print("=== Partial Derivatives in Financial Optimization ===\\n")
    
    # Example: Portfolio variance optimization
    # Portfolio variance: σ²_p = w^T Σ w
    
    # Sample covariance matrix (3 assets)
    n_assets = 3
    np.random.seed(42)
    asset_names = ['Stock A', 'Stock B', 'Stock C']
    
    # Create correlation matrix
    correlation = np.array([
        [1.0, 0.3, 0.1],
        [0.3, 1.0, 0.4], 
        [0.1, 0.4, 1.0]
    ])
    
    # Volatilities
    volatilities = np.array([0.15, 0.20, 0.12])  # 15%, 20%, 12% annual vol
    
    # Covariance matrix
    cov_matrix = np.outer(volatilities, volatilities) * correlation
    
    print("1. Portfolio Variance Derivatives:")
    print("   Covariance Matrix:")
    print(cov_matrix)
    print()
    
    # Test portfolio weights
    weights = np.array([0.4, 0.3, 0.3])
    
    # Analytical derivatives
    calc = FinancialCalculus()
    risk_derivatives = calc.portfolio_risk_derivatives(weights, cov_matrix)
    
    print(f"   Portfolio weights: {weights}")
    print(f"   Portfolio variance: {risk_derivatives['portfolio_variance']:.6f}")
    print(f"   Portfolio volatility: {risk_derivatives['portfolio_volatility']:.4f}")
    print()
    print("   Variance gradient (∂σ²/∂w):")
    for i, asset in enumerate(asset_names):
        print(f"     {asset}: {risk_derivatives['variance_gradient'][i]:.6f}")
    print()
    print("   Volatility gradient (∂σ/∂w):")
    for i, asset in enumerate(asset_names):
        print(f"     {asset}: {risk_derivatives['volatility_gradient'][i]:.6f}")
    print()
    
    # Numerical verification
    def portfolio_variance_func(w):
        return w @ cov_matrix @ w
    
    numerical_grad = calc.gradient_vector(portfolio_variance_func, weights)
    analytical_grad = risk_derivatives['variance_gradient']
    
    gradient_error = np.linalg.norm(numerical_grad - analytical_grad)
    print(f"   Gradient verification error: {gradient_error:.2e}")
    print()
    
    # 2. Risk budgeting application
    print("2. Risk Budgeting (Marginal Contributions):")
    print("   Risk contributions by asset:")
    for i, asset in enumerate(asset_names):
        print(f"     {asset}: {risk_derivatives['percentage_contributions'][i]:.1f}%")
    
    # Verify risk contributions sum to 100%
    total_contribution = np.sum(risk_derivatives['percentage_contributions'])
    print(f"   Total contributions: {total_contribution:.1f}%")
    print()
    
    # 3. Visualize risk surface
    print("3. Risk Surface Visualization:")
    
    # Create 2D visualization (fixing third weight)
    w1_range = np.linspace(0.1, 0.8, 50)
    w2_range = np.linspace(0.1, 0.8, 50)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    
    # Calculate risk for each combination (w3 = 1 - w1 - w2)
    Risk = np.zeros_like(W1)
    
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w1, w2 = W1[i, j], W2[i, j]
            w3 = 1 - w1 - w2
            
            if w3 >= 0:  # Valid weight combination
                w = np.array([w1, w2, w3])
                Risk[i, j] = np.sqrt(w @ cov_matrix @ w)
            else:
                Risk[i, j] = np.nan
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 5))
    
    # 2D contour plot
    ax1 = fig.add_subplot(121)
    contour = ax1.contour(W1, W2, Risk, levels=15)
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.scatter(weights[0], weights[1], color='red', s=100, label='Current Portfolio', zorder=5)
    ax1.set_xlabel('Weight Stock A')
    ax1.set_ylabel('Weight Stock B')
    ax1.set_title('Portfolio Risk Contours\\n(Stock C weight = 1 - A - B)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(W1, W2, Risk, alpha=0.6, cmap='viridis')
    ax2.scatter(weights[0], weights[1], risk_derivatives['portfolio_volatility'], 
               color='red', s=100, label='Current Portfolio')
    ax2.set_xlabel('Weight Stock A')
    ax2.set_ylabel('Weight Stock B') 
    ax2.set_zlabel('Portfolio Risk')
    ax2.set_title('Portfolio Risk Surface')
    
    plt.tight_layout()
    plt.show()
    
    return risk_derivatives

if __name__ == "__main__":
    results = demonstrate_partial_derivatives()
```

### 🔴 Theoretical Foundation

**Partial Derivative Definition:**
$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}$$

**Gradient Vector:**
$$\nabla f(x) = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)$$

**Portfolio Risk Derivatives:**
For portfolio variance $\sigma_p^2 = w^T\Sigma w$:
$$\frac{\partial \sigma_p^2}{\partial w_i} = 2(\Sigma w)_i$$

For portfolio volatility $\sigma_p = \sqrt{w^T\Sigma w}$:
$$\frac{\partial \sigma_p}{\partial w_i} = \frac{(\Sigma w)_i}{\sigma_p}$$

---

## 🟢 Chain Rule for Backpropagation {#chain-rule}

### Intuitive Understanding

The chain rule lets us compute derivatives of composite functions:
- **Neural networks**: Backpropagate errors through layers
- **Complex models**: Break down complicated derivatives
- **Financial models**: Derivatives of nested functions (e.g., option on portfolio)

Think of it like tracking how a change propagates:
- Change in interest rates → Change in bond prices → Change in portfolio value

### 🟡 Practical Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any, Tuple

class ChainRuleCalculus:
    """
    Chain rule applications for financial neural networks and complex models
    """
    
    @staticmethod
    def simple_chain_rule(f: Callable, g: Callable, 
                         df_dx: Callable, dg_dx: Callable,
                         x: float) -> float:
        """
        Apply chain rule for composition f(g(x))
        d/dx[f(g(x))] = f'(g(x)) * g'(x)
        """
        return df_dx(g(x)) * dg_dx(x)
    
    @staticmethod
    def multi_variable_chain_rule(f: Callable, 
                                g_functions: List[Callable],
                                df_dg: List[Callable],
                                dg_dx: List[List[Callable]],
                                x: np.ndarray) -> np.ndarray:
        """
        Multi-variable chain rule: f(g₁(x), g₂(x), ..., gₘ(x))
        
        ∂f/∂xᵢ = Σⱼ (∂f/∂gⱼ) * (∂gⱼ/∂xᵢ)
        """
        n_vars = len(x)
        gradient = np.zeros(n_vars)
        
        # Evaluate intermediate functions
        g_values = [g_func(x) for g_func in g_functions]
        
        for i in range(n_vars):
            partial_sum = 0
            for j in range(len(g_functions)):
                # ∂f/∂gⱼ evaluated at g(x)
                df_dgj = df_dg[j](g_values)
                
                # ∂gⱼ/∂xᵢ
                dgj_dxi = dg_dx[j][i](x)
                
                partial_sum += df_dgj * dgj_dxi
            
            gradient[i] = partial_sum
        
        return gradient
    
    @staticmethod
    def neural_network_backprop(inputs: np.ndarray,
                              weights: List[np.ndarray],
                              biases: List[np.ndarray],
                              target: np.ndarray,
                              activation: str = 'relu') -> Dict[str, Any]:
        """
        Simplified neural network backpropagation for financial prediction
        
        Args:
            inputs: Input features (batch_size × input_dim)
            weights: List of weight matrices for each layer
            biases: List of bias vectors for each layer
            target: Target values (batch_size × output_dim)
            activation: Activation function ('relu', 'sigmoid', 'tanh')
        
        Returns:
            Gradients and forward pass results
        """
        # Define activation functions and their derivatives
        if activation == 'relu':
            def activate(x):
                return np.maximum(0, x)
            def activate_derivative(x):
                return (x > 0).astype(float)
        elif activation == 'sigmoid':
            def activate(x):
                return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            def activate_derivative(x):
                s = activate(x)
                return s * (1 - s)
        elif activation == 'tanh':
            def activate(x):
                return np.tanh(x)
            def activate_derivative(x):
                return 1 - np.tanh(x)**2
        else:
            raise ValueError(f"Activation {activation} not supported")
        
        # Forward pass
        layer_inputs = [inputs]
        layer_outputs = [inputs]
        
        for i, (W, b) in enumerate(zip(weights, biases)):
            # Linear transformation
            z = layer_outputs[-1] @ W + b
            layer_inputs.append(z)
            
            # Activation (except for output layer)
            if i < len(weights) - 1:
                a = activate(z)
            else:
                a = z  # Linear output for regression
            
            layer_outputs.append(a)
        
        # Calculate loss (Mean Squared Error)
        predictions = layer_outputs[-1]
        loss = 0.5 * np.mean((predictions - target)**2)
        
        # Backward pass
        n_layers = len(weights)
        weight_gradients = [np.zeros_like(W) for W in weights]
        bias_gradients = [np.zeros_like(b) for b in biases]
        
        # Output layer error
        output_error = (predictions - target) / len(target)  # Average over batch
        
        # Backpropagate through layers
        current_error = output_error
        
        for i in range(n_layers - 1, -1, -1):
            # Gradient w.r.t. weights: ∂L/∂W = (input)^T * error
            weight_gradients[i] = layer_outputs[i].T @ current_error
            
            # Gradient w.r.t. biases: ∂L/∂b = sum(error, axis=0)
            bias_gradients[i] = np.sum(current_error, axis=0)
            
            if i > 0:  # Not input layer
                # Error for previous layer: ∂L/∂input = error * W^T * activation'
                weighted_error = current_error @ weights[i].T
                
                # Apply activation derivative
                if i > 0:  # Hidden layers use activation
                    activation_derivative = activate_derivative(layer_inputs[i])
                    current_error = weighted_error * activation_derivative
                else:
                    current_error = weighted_error
        
        return {
            'loss': loss,
            'predictions': predictions,
            'weight_gradients': weight_gradients,
            'bias_gradients': bias_gradients,
            'layer_outputs': layer_outputs,
            'layer_inputs': layer_inputs
        }
    
    @staticmethod
    def option_pricing_chain_rule(S: float, K: float, T: float, r: float, sigma: float,
                                underlying_weights: np.ndarray,
                                underlying_cov: np.ndarray) -> Dict[str, Any]:
        """
        Chain rule for option on portfolio (complex derivative)
        
        Option price = f(Portfolio value)
        Portfolio value = g(Asset weights, Asset prices)
        
        ∂Option/∂weights = (∂Option/∂Portfolio) * (∂Portfolio/∂weights)
        """
        # Current portfolio value
        portfolio_value = S  # Assume S is current portfolio value
        
        # Option price (using Black-Scholes as example)
        from scipy.stats import norm
        import math
        
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        
        call_price = S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)
        
        # Option delta (∂Option/∂Portfolio)
        delta = norm.cdf(d1)
        
        # Portfolio derivatives (∂Portfolio/∂weights)
        # Assuming portfolio value is weighted sum: P = Σ wᵢ * Sᵢ
        # ∂P/∂wᵢ = Sᵢ (assuming current asset prices = 1 for simplicity)
        portfolio_gradient = np.ones_like(underlying_weights)  # Simplified
        
        # Chain rule: ∂Option/∂weights = delta * (∂Portfolio/∂weights)
        option_weight_gradient = delta * portfolio_gradient
        
        # Risk impact: How option value changes with portfolio risk
        # ∂Option/∂σ_portfolio (vega effect)
        vega = S * norm.pdf(d1) * math.sqrt(T)
        
        # Portfolio volatility derivatives (from previous function)
        portfolio_variance = underlying_weights @ underlying_cov @ underlying_weights
        portfolio_vol = np.sqrt(portfolio_variance)
        
        if portfolio_vol > 0:
            vol_weight_gradient = (underlying_cov @ underlying_weights) / portfolio_vol
        else:
            vol_weight_gradient = np.zeros_like(underlying_weights)
        
        # Total option sensitivity to weights
        total_option_gradient = option_weight_gradient + vega * vol_weight_gradient
        
        return {
            'call_price': call_price,
            'delta': delta,
            'vega': vega,
            'portfolio_gradient': portfolio_gradient,
            'vol_weight_gradient': vol_weight_gradient,
            'option_weight_gradient': option_weight_gradient,
            'total_option_gradient': total_option_gradient
        }

def demonstrate_chain_rule():
    """Demonstrate chain rule applications in finance"""
    
    print("=== Chain Rule Applications in Finance ===\\n")
    
    # 1. Simple neural network for financial prediction
    print("1. Neural Network Backpropagation:")
    
    # Generate synthetic financial data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    # Features: P/E ratio, ROE, Debt/Equity, Market Cap, Beta
    features = np.random.randn(n_samples, n_features)
    
    # Target: Stock returns (synthetic relationship)
    true_weights = np.array([0.1, 0.15, -0.08, 0.05, -0.12])
    noise = np.random.normal(0, 0.02, n_samples)
    targets = features @ true_weights + noise
    targets = targets.reshape(-1, 1)
    
    # Simple 2-layer network
    input_dim = n_features
    hidden_dim = 10
    output_dim = 1
    
    # Initialize weights
    W1 = np.random.randn(input_dim, hidden_dim) * 0.1
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, output_dim) * 0.1
    b2 = np.zeros(output_dim)
    
    weights = [W1, W2]
    biases = [b1, b2]
    
    calc = ChainRuleCalculus()
    
    # Forward and backward pass
    backprop_result = calc.neural_network_backprop(
        features, weights, biases, targets, activation='relu'
    )
    
    print(f"   Initial loss: {backprop_result['loss']:.6f}")
    print(f"   Weight gradient shapes: {[grad.shape for grad in backprop_result['weight_gradients']]}")
    print()
    
    # 2. Training loop demonstration
    print("2. Neural Network Training (Gradient Descent):")
    
    # Simple training loop
    learning_rate = 0.01
    n_epochs = 100
    loss_history = []
    
    current_weights = [W.copy() for W in weights]
    current_biases = [b.copy() for b in biases]
    
    for epoch in range(n_epochs):
        # Forward and backward pass
        result = calc.neural_network_backprop(
            features, current_weights, current_biases, targets
        )
        
        loss_history.append(result['loss'])
        
        # Update weights using gradients
        for i in range(len(current_weights)):
            current_weights[i] -= learning_rate * result['weight_gradients'][i]
            current_biases[i] -= learning_rate * result['bias_gradients'][i]
        
        if epoch % 20 == 0:
            print(f"     Epoch {epoch}: Loss = {result['loss']:.6f}")
    
    print(f"   Final loss: {loss_history[-1]:.6f}")
    print(f"   Loss reduction: {(loss_history[0] - loss_history[-1]) / loss_history[0]:.1%}")
    print()
    
    # 3. Option on portfolio chain rule
    print("3. Option on Portfolio (Chain Rule):")
    
    # Portfolio parameters
    n_assets = 3
    portfolio_weights = np.array([0.4, 0.35, 0.25])
    portfolio_cov = np.array([
        [0.04, 0.01, 0.005],
        [0.01, 0.06, 0.008],
        [0.005, 0.008, 0.03]
    ]) * 0.01  # Daily covariance
    
    # Option parameters
    S = 100  # Current portfolio value
    K = 105  # Strike price
    T = 0.25  # 3 months
    r = 0.05  # Risk-free rate
    sigma = 0.20  # Implied volatility
    
    option_result = calc.option_pricing_chain_rule(
        S, K, T, r, sigma, portfolio_weights, portfolio_cov
    )
    
    print(f"   Call option price: ${option_result['call_price']:.2f}")
    print(f"   Delta (∂Option/∂Portfolio): {option_result['delta']:.4f}")
    print(f"   Vega (∂Option/∂σ): ${option_result['vega']:.2f}")
    print("   Option sensitivity to portfolio weights:")
    for i in range(n_assets):
        print(f"     Asset {i+1}: {option_result['total_option_gradient'][i]:.4f}")
    print()
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Neural network loss convergence
    axes[0, 0].plot(loss_history, linewidth=2)
    axes[0, 0].set_title('Neural Network Training\\n(Chain Rule Backpropagation)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Predicted vs actual returns
    final_predictions = result['predictions'].flatten()
    actual_returns = targets.flatten()
    
    axes[0, 1].scatter(actual_returns, final_predictions, alpha=0.6)
    min_val = min(actual_returns.min(), final_predictions.min())
    max_val = max(actual_returns.max(), final_predictions.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[0, 1].set_title('Predicted vs Actual Returns\\n(After Training)')
    axes[0, 1].set_xlabel('Actual Returns')
    axes[0, 1].set_ylabel('Predicted Returns')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Option Greeks
    greeks_names = ['Delta', 'Vega']
    greeks_values = [option_result['delta'], option_result['vega']]
    
    axes[1, 0].bar(greeks_names, greeks_values)
    axes[1, 0].set_title('Option Greeks\\n(First-order Sensitivities)')
    axes[1, 0].set_ylabel('Greek Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Option sensitivity to portfolio weights
    asset_labels = ['Asset 1', 'Asset 2', 'Asset 3']
    sensitivities = option_result['total_option_gradient']
    
    colors = ['green' if s > 0 else 'red' for s in sensitivities]
    axes[1, 1].bar(asset_labels, sensitivities, color=colors, alpha=0.7)
    axes[1, 1].set_title('Option Sensitivity to Portfolio Weights\\n(Chain Rule Application)')
    axes[1, 1].set_ylabel('∂Option/∂Weight')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'neural_network_result': result,
        'loss_history': loss_history,
        'option_result': option_result,
        'trained_weights': current_weights,
        'trained_biases': current_biases
    }

if __name__ == "__main__":
    results = demonstrate_chain_rule()
```

### 🔴 Theoretical Foundation

**Chain Rule (Single Variable):**
For $h(x) = f(g(x))$:
$$\frac{dh}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

**Chain Rule (Multivariable):**
For $z = f(u_1, u_2, \ldots, u_m)$ where $u_i = g_i(x_1, x_2, \ldots, x_n)$:
$$\frac{\partial z}{\partial x_j} = \sum_{i=1}^m \frac{\partial f}{\partial u_i} \frac{\partial u_i}{\partial x_j}$$

**Backpropagation Algorithm:**
For neural network with layers $L_1, L_2, \ldots, L_k$:

1. **Forward pass**: Compute activations for each layer
2. **Backward pass**: Propagate error gradients from output to input

**Error propagation**:
$$\delta^{(l)} = \delta^{(l+1)} W^{(l+1)T} \odot \sigma'(z^{(l)})$$

where $\odot$ denotes element-wise multiplication.

---

## 🟢 Optimization Techniques (SGD, Adam) {#optimization-techniques}

### Intuitive Understanding

Different optimization algorithms have different strengths:
- **Gradient Descent**: Simple but can be slow
- **Stochastic Gradient Descent (SGD)**: Faster but noisier
- **Adam**: Adaptive learning rates, good for most problems
- **L-BFGS**: Good for smooth problems with accurate gradients

In finance:
- **Portfolio optimization**: Usually smooth, use L-BFGS or Adam
- **Neural network training**: Use Adam or SGD with momentum
- **Risk model fitting**: Depends on complexity and data size

### 🟡 Practical Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    x: np.ndarray
    fun: float
    grad_norm: float
    n_iter: int
    converged: bool
    time_elapsed: float
    history: Dict[str, List]

class AdvancedOptimizers:
    """
    Advanced optimization algorithms for financial problems
    """
    
    @staticmethod
    def sgd_with_momentum(f: Callable, grad_f: Callable, x0: np.ndarray,
                         learning_rate: float = 0.01,
                         momentum: float = 0.9,
                         max_iter: int = 1000,
                         tolerance: float = 1e-6,
                         batch_size: Optional[int] = None) -> OptimizationResult:
        """
        Stochastic Gradient Descent with momentum
        
        Args:
            f: Objective function
            grad_f: Gradient function (can handle batch indices)
            x0: Starting point
            learning_rate: Learning rate
            momentum: Momentum parameter (0-1)
            max_iter: Maximum iterations
            tolerance: Convergence tolerance
            batch_size: Batch size for stochastic updates
        
        Returns:
            OptimizationResult object
        """
        start_time = time.time()
        x = x0.copy()
        velocity = np.zeros_like(x)
        
        history = {
            'x': [x.copy()],
            'f': [f(x)],
            'grad_norm': [],
            'learning_rate': []
        }
        
        for i in range(max_iter):
            # Compute gradient (full or stochastic)
            if batch_size is not None:
                # Stochastic gradient (simplified - assume grad_f handles batching)
                grad = grad_f(x, batch_indices=None)  # Would implement proper batching
            else:
                grad = grad_f(x)
            
            grad_norm = np.linalg.norm(grad)
            history['grad_norm'].append(grad_norm)
            history['learning_rate'].append(learning_rate)
            
            # Check convergence
            if grad_norm < tolerance:
                break
            
            # Momentum update
            velocity = momentum * velocity - learning_rate * grad
            x = x + velocity
            
            # Store history
            history['x'].append(x.copy())
            history['f'].append(f(x))
        
        time_elapsed = time.time() - start_time
        
        return OptimizationResult(
            x=x,
            fun=f(x),
            grad_norm=grad_norm,
            n_iter=i + 1,
            converged=grad_norm < tolerance,
            time_elapsed=time_elapsed,
            history=history
        )
    
    @staticmethod
    def adam_optimizer(f: Callable, grad_f: Callable, x0: np.ndarray,
                      learning_rate: float = 0.001,
                      beta1: float = 0.9,
                      beta2: float = 0.999,
                      epsilon: float = 1e-8,
                      max_iter: int = 1000,
                      tolerance: float = 1e-6,
                      weight_decay: float = 0.0) -> OptimizationResult:
        """
        Adam optimizer with optional weight decay (AdamW)
        """
        start_time = time.time()
        x = x0.copy()
        m = np.zeros_like(x)  # First moment
        v = np.zeros_like(x)  # Second moment
        
        history = {
            'x': [x.copy()],
            'f': [f(x)],
            'grad_norm': [],
            'learning_rate': [],
            'm_norm': [],
            'v_norm': []
        }
        
        for t in range(1, max_iter + 1):
            grad = grad_f(x)
            
            # Add weight decay if specified
            if weight_decay > 0:
                grad += weight_decay * x
            
            grad_norm = np.linalg.norm(grad)
            history['grad_norm'].append(grad_norm)
            
            # Check convergence
            if grad_norm < tolerance:
                break
            
            # Update biased first and second moment estimates
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            
            # Bias correction
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            # Update parameters
            x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Store history
            history['x'].append(x.copy())
            history['f'].append(f(x))
            history['learning_rate'].append(learning_rate)
            history['m_norm'].append(np.linalg.norm(m))
            history['v_norm'].append(np.linalg.norm(v))
        
        time_elapsed = time.time() - start_time
        
        return OptimizationResult(
            x=x,
            fun=f(x),
            grad_norm=grad_norm,
            n_iter=t,
            converged=grad_norm < tolerance,
            time_elapsed=time_elapsed,
            history=history
        )
    
    @staticmethod
    def rmsprop_optimizer(f: Callable, grad_f: Callable, x0: np.ndarray,
                         learning_rate: float = 0.001,
                         decay_rate: float = 0.9,
                         epsilon: float = 1e-8,
                         max_iter: int = 1000,
                         tolerance: float = 1e-6) -> OptimizationResult:
        """
        RMSprop optimizer
        """
        start_time = time.time()
        x = x0.copy()
        v = np.zeros_like(x)  # Running average of squared gradients
        
        history = {
            'x': [x.copy()],
            'f': [f(x)],
            'grad_norm': [],
            'learning_rate': []
        }
        
        for i in range(max_iter):
            grad = grad_f(x)
            grad_norm = np.linalg.norm(grad)
            history['grad_norm'].append(grad_norm)
            
            # Check convergence
            if grad_norm < tolerance:
                break
            
            # Update running average of squared gradients
            v = decay_rate * v + (1 - decay_rate) * grad**2
            
            # Update parameters
            x = x - learning_rate * grad / (np.sqrt(v) + epsilon)
            
            # Store history
            history['x'].append(x.copy())
            history['f'].append(f(x))
            history['learning_rate'].append(learning_rate)
        
        time_elapsed = time.time() - start_time
        
        return OptimizationResult(
            x=x,
            fun=f(x),
            grad_norm=grad_norm,
            n_iter=i + 1,
            converged=grad_norm < tolerance,
            time_elapsed=time_elapsed,
            history=history
        )
    
    @staticmethod
    def compare_optimizers(f: Callable, grad_f: Callable, x0: np.ndarray,
                          max_iter: int = 1000) -> Dict[str, OptimizationResult]:
        """
        Compare different optimization algorithms on the same problem
        """
        optimizers = {
            'SGD_Momentum': lambda: AdvancedOptimizers.sgd_with_momentum(
                f, grad_f, x0, learning_rate=0.01, momentum=0.9, max_iter=max_iter
            ),
            'Adam': lambda: AdvancedOptimizers.adam_optimizer(
                f, grad_f, x0, learning_rate=0.01, max_iter=max_iter
            ),
            'RMSprop': lambda: AdvancedOptimizers.rmsprop_optimizer(
                f, grad_f, x0, learning_rate=0.01, max_iter=max_iter
            )
        }
        
        results = {}
        for name, optimizer_func in optimizers.items():
            try:
                results[name] = optimizer_func()
            except Exception as e:
                print(f"Optimizer {name} failed: {str(e)}")
        
        return results

def demonstrate_optimization_techniques():
    """Demonstrate different optimization techniques on financial problems"""
    
    print("=== Advanced Optimization Techniques ===\\n")
    
    # Problem 1: Portfolio optimization with transaction costs
    print("1. Portfolio Optimization with Transaction Costs:")
    
    # Set up problem
    np.random.seed(42)
    n_assets = 5
    
    # Expected returns and covariance
    mean_returns = np.array([0.08, 0.12, 0.10, 0.15, 0.09])
    volatilities = np.array([0.15, 0.20, 0.18, 0.25, 0.16])
    correlation = np.random.uniform(0.1, 0.6, (n_assets, n_assets))
    correlation = (correlation + correlation.T) / 2  # Make symmetric
    np.fill_diagonal(correlation, 1.0)
    cov_matrix = np.outer(volatilities, volatilities) * correlation
    
    # Current portfolio (starting point)
    current_weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
    
    # Transaction cost rate
    transaction_cost_rate = 0.001  # 0.1% per trade
    
    # Define objective function with transaction costs
    def portfolio_objective(weights):
        # Portfolio return and risk
        portfolio_return = weights @ mean_returns
        portfolio_variance = weights @ cov_matrix @ weights
        
        # Transaction costs (proportional to weight changes)
        transaction_costs = transaction_cost_rate * np.sum(np.abs(weights - current_weights))
        
        # Combined objective (negative Sharpe ratio + transaction costs)
        portfolio_std = np.sqrt(portfolio_variance)
        if portfolio_std > 0:
            sharpe_ratio = portfolio_return / portfolio_std
        else:
            sharpe_ratio = 0
        
        return -sharpe_ratio + transaction_costs * 100  # Scale transaction costs
    
    def portfolio_gradient(weights):
        # Numerical gradient (for simplicity)
        h = 1e-8
        grad = np.zeros_like(weights)
        for i in range(len(weights)):
            weights_forward = weights.copy()
            weights_backward = weights.copy()
            weights_forward[i] += h
            weights_backward[i] -= h
            grad[i] = (portfolio_objective(weights_forward) - portfolio_objective(weights_backward)) / (2 * h)
        return grad
    
    # Compare optimizers
    optimizer = AdvancedOptimizers()
    comparison_results = optimizer.compare_optimizers(
        portfolio_objective, portfolio_gradient, current_weights, max_iter=500
    )
    
    print("   Optimizer Performance Comparison:")
    for name, result in comparison_results.items():
        print(f"     {name}:")
        print(f"       Final objective: {result.fun:.6f}")
        print(f"       Iterations: {result.n_iter}")
        print(f"       Converged: {result.converged}")
        print(f"       Time: {result.time_elapsed:.3f}s")
        
        # Calculate portfolio metrics
        final_weights = result.x
        portfolio_return = final_weights @ mean_returns
        portfolio_std = np.sqrt(final_weights @ cov_matrix @ final_weights)
        sharpe = portfolio_return / portfolio_std if portfolio_std > 0 else 0
        print(f"       Sharpe ratio: {sharpe:.3f}")
        print(f"       Turnover: {np.sum(np.abs(final_weights - current_weights)):.1%}")
        print()
    
    # Problem 2: Risk factor model fitting
    print("2. Risk Factor Model Optimization:")
    
    # Generate synthetic factor model data
    n_periods = 252
    n_factors = 4
    
    # True factor loadings
    true_loadings = np.random.uniform(-1, 1, (n_assets, n_factors))
    
    # Factor returns
    factor_returns = np.random.multivariate_normal(
        mean=np.zeros(n_factors),
        cov=np.eye(n_factors) * 0.01,
        size=n_periods
    )
    
    # Asset returns with factor structure
    idiosyncratic_risk = np.random.normal(0, 0.005, (n_periods, n_assets))
    asset_returns = factor_returns @ true_loadings.T + idiosyncratic_risk
    
    # Objective: Minimize sum of squared residuals
    def factor_model_objective(loadings_flat):
        loadings = loadings_flat.reshape(n_assets, n_factors)
        predicted_returns = factor_returns @ loadings.T
        residuals = asset_returns - predicted_returns
        return 0.5 * np.sum(residuals**2)  # Mean squared error
    
    def factor_model_gradient(loadings_flat):
        loadings = loadings_flat.reshape(n_assets, n_factors)
        predicted_returns = factor_returns @ loadings.T
        residuals = asset_returns - predicted_returns
        
        # Gradient w.r.t. loadings
        grad_loadings = -factor_returns.T @ residuals  # (n_factors × n_assets)
        return grad_loadings.T.flatten()  # Flatten to match input shape
    
    # Starting point
    x0_factor = np.random.normal(0, 0.1, n_assets * n_factors)
    
    # Compare optimizers
    factor_results = optimizer.compare_optimizers(
        factor_model_objective, factor_model_gradient, x0_factor, max_iter=1000
    )
    
    print("   Factor Model Fitting Results:")
    for name, result in factor_results.items():
        estimated_loadings = result.x.reshape(n_assets, n_factors)
        
        # Calculate R-squared
        predicted = factor_returns @ estimated_loadings.T
        tss = np.sum((asset_returns - np.mean(asset_returns, axis=0))**2)
        rss = np.sum((asset_returns - predicted)**2)
        r_squared = 1 - rss / tss
        
        print(f"     {name}:")
        print(f"       R-squared: {r_squared:.3f}")
        print(f"       RMSE: {np.sqrt(result.fun / (n_periods * n_assets)):.6f}")
        print(f"       Iterations: {result.n_iter}")
        print(f"       Time: {result.time_elapsed:.3f}s")
    print()
    
    # 3. Learning rate scheduling
    print("3. Learning Rate Scheduling:")
    
    # Define learning rate schedules
    def exponential_decay(epoch, initial_lr=0.1, decay_rate=0.95):
        return initial_lr * (decay_rate ** epoch)
    
    def step_decay(epoch, initial_lr=0.1, drop_rate=0.5, epochs_drop=100):
        return initial_lr * (drop_rate ** (epoch // epochs_drop))
    
    def cosine_annealing(epoch, initial_lr=0.1, min_lr=0.001, max_epochs=1000):
        return min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))
    
    # Demonstrate different schedules
    epochs = range(500)
    exp_schedule = [exponential_decay(e) for e in epochs]
    step_schedule = [step_decay(e) for e in epochs]
    cosine_schedule = [cosine_annealing(e) for e in epochs]
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Optimizer convergence comparison
    for name, result in comparison_results.items():
        if len(result.history['f']) > 1:
            axes[0, 0].plot(result.history['f'][:200], label=name, linewidth=2)
    
    axes[0, 0].set_title('Portfolio Optimization\\nConvergence Comparison')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Objective Value')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Gradient norm convergence
    for name, result in comparison_results.items():
        if len(result.history['grad_norm']) > 1:
            axes[0, 1].plot(result.history['grad_norm'][:200], label=name, linewidth=2)
    
    axes[0, 1].set_title('Gradient Norm Convergence')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Gradient Norm')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Learning rate schedules
    axes[0, 2].plot(epochs, exp_schedule, label='Exponential Decay', linewidth=2)
    axes[0, 2].plot(epochs, step_schedule, label='Step Decay', linewidth=2)
    axes[0, 2].plot(epochs, cosine_schedule, label='Cosine Annealing', linewidth=2)
    axes[0, 2].set_title('Learning Rate Schedules')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Factor model convergence
    for name, result in factor_results.items():
        if len(result.history['f']) > 1:
            axes[1, 0].plot(result.history['f'][:500], label=name, linewidth=2)
    
    axes[1, 0].set_title('Factor Model Fitting\\nConvergence')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Sum of Squared Residuals')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Portfolio weights evolution (Adam)
    if 'Adam' in comparison_results:
        adam_result = comparison_results['Adam']
        weight_history = np.array(adam_result.history['x'])
        
        for i in range(min(n_assets, weight_history.shape[1])):
            axes[1, 1].plot(weight_history[:100, i], label=f'Asset {i+1}', linewidth=2)
        
        axes[1, 1].set_title('Portfolio Weight Evolution\\n(Adam Optimizer)')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Weight')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Performance summary
    optimizer_names = list(comparison_results.keys())
    final_objectives = [result.fun for result in comparison_results.values()]
    convergence_times = [result.time_elapsed for result in comparison_results.values()]
    
    x_pos = np.arange(len(optimizer_names))
    
    # Dual y-axis plot
    ax6a = axes[1, 2]
    ax6b = ax6a.twinx()
    
    bars1 = ax6a.bar(x_pos - 0.2, final_objectives, 0.4, label='Final Objective', alpha=0.7, color='blue')
    bars2 = ax6b.bar(x_pos + 0.2, convergence_times, 0.4, label='Time (s)', alpha=0.7, color='red')
    
    ax6a.set_xlabel('Optimizer')
    ax6a.set_ylabel('Final Objective Value', color='blue')
    ax6b.set_ylabel('Convergence Time (s)', color='red')
    ax6a.set_title('Optimizer Performance Summary')
    ax6a.set_xticks(x_pos)
    ax6a.set_xticklabels(optimizer_names, rotation=45)
    ax6a.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (obj, time_val) in enumerate(zip(final_objectives, convergence_times)):
        ax6a.text(i - 0.2, obj, f'{obj:.3f}', ha='center', va='bottom', fontsize=8)
        ax6b.text(i + 0.2, time_val, f'{time_val:.2f}s', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'portfolio_optimization': comparison_results,
        'factor_model_fitting': factor_results,
        'learning_schedules': {
            'exponential': exp_schedule,
            'step': step_schedule,
            'cosine': cosine_schedule
        }
    }

if __name__ == "__main__":
    results = demonstrate_optimization_techniques()
```

### 🔴 Theoretical Foundation

**Gradient Descent Variants:**

1. **Vanilla Gradient Descent**:
   $$x_{t+1} = x_t - \alpha \nabla f(x_t)$$

2. **SGD with Momentum**:
   $$v_{t+1} = \gamma v_t + \alpha \nabla f(x_t)$$
   $$x_{t+1} = x_t - v_{t+1}$$

3. **Adam Optimizer**:
   $$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
   $$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
   $$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
   $$x_{t+1} = x_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

4. **RMSprop**:
   $$v_t = \gamma v_{t-1} + (1-\gamma) g_t^2$$
   $$x_{t+1} = x_t - \frac{\alpha}{\sqrt{v_t} + \epsilon} g_t$$

**Convergence Analysis:**
- **Learning rate too large**: Oscillation or divergence
- **Learning rate too small**: Slow convergence
- **Adaptive methods**: Automatically adjust learning rates

---

## 🟢 Loss Function Derivatives {#loss-function-derivatives}

### Intuitive Understanding

Loss functions measure how wrong our predictions are:
- **Mean Squared Error**: Penalizes large errors heavily
- **Mean Absolute Error**: Robust to outliers
- **Huber loss**: Combines benefits of both
- **Financial loss functions**: Often asymmetric (losses hurt more than gains)

### 🟡 Practical Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, Any, Tuple
from scipy.optimize import minimize

class FinancialLossFunctions:
    """
    Loss functions and their derivatives for financial modeling
    """
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Mean Squared Error and its derivative
        
        MSE = (1/n) * Σ(y_true - y_pred)²
        ∂MSE/∂y_pred = -(2/n) * (y_true - y_pred)
        """
        residuals = y_true - y_pred
        loss = np.mean(residuals**2)
        gradient = -2 * residuals / len(residuals)
        
        return loss, gradient
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Mean Absolute Error and its derivative
        
        MAE = (1/n) * Σ|y_true - y_pred|
        ∂MAE/∂y_pred = -(1/n) * sign(y_true - y_pred)
        """
        residuals = y_true - y_pred
        loss = np.mean(np.abs(residuals))
        gradient = -np.sign(residuals) / len(residuals)
        
        return loss, gradient
    
    @staticmethod
    def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, 
                  delta: float = 1.0) -> Tuple[float, np.ndarray]:
        """
        Huber loss (robust to outliers)
        
        Huber(r) = { 0.5*r²        if |r| ≤ δ
                   { δ|r| - 0.5*δ²  if |r| > δ
        """
        residuals = y_true - y_pred
        abs_residuals = np.abs(residuals)
        
        # Piecewise loss calculation
        quadratic_region = abs_residuals <= delta
        linear_region = abs_residuals > delta
        
        loss_values = np.zeros_like(residuals)
        loss_values[quadratic_region] = 0.5 * residuals[quadratic_region]**2
        loss_values[linear_region] = delta * abs_residuals[linear_region] - 0.5 * delta**2
        
        loss = np.mean(loss_values)
        
        # Gradient calculation
        gradient = np.zeros_like(residuals)
        gradient[quadratic_region] = -residuals[quadratic_region]
        gradient[linear_region] = -delta * np.sign(residuals[linear_region])
        gradient = gradient / len(residuals)
        
        return loss, gradient
    
    @staticmethod
    def asymmetric_loss(y_true: np.ndarray, y_pred: np.ndarray,
                       alpha: float = 0.7) -> Tuple[float, np.ndarray]:
        """
        Asymmetric loss function for financial modeling
        Penalizes underestimation more than overestimation (or vice versa)
        
        Args:
            alpha: Asymmetry parameter (0.5 = symmetric, >0.5 = penalize underestimation more)
        """
        residuals = y_true - y_pred
        
        # Asymmetric weighting
        loss_values = np.where(residuals >= 0, 
                              alpha * residuals**2,        # Underestimation penalty
                              (1 - alpha) * residuals**2)  # Overestimation penalty
        
        loss = np.mean(loss_values)
        
        # Gradient
        gradient = np.where(residuals >= 0,
                           -2 * alpha * residuals,
                           -2 * (1 - alpha) * residuals) / len(residuals)
        
        return loss, gradient
    
    @staticmethod
    def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, 
                     tau: float = 0.5) -> Tuple[float, np.ndarray]:
        """
        Quantile loss (pinball loss) for quantile regression
        
        Used for VaR estimation and risk modeling
        
        Args:
            tau: Quantile level (0 < tau < 1)
        """
        residuals = y_true - y_pred
        
        loss_values = np.where(residuals >= 0,
                              tau * residuals,
                              (tau - 1) * residuals)
        
        loss = np.mean(loss_values)
        
        # Gradient (subgradient at zero)
        gradient = np.where(residuals > 0, -tau,
                           np.where(residuals < 0, -(tau - 1), 0))
        gradient = gradient / len(residuals)
        
        return loss, gradient
    
    @staticmethod
    def expected_shortfall_loss(y_true: np.ndarray, y_pred: np.ndarray,
                               alpha: float = 0.05) -> Tuple[float, np.ndarray]:
        """
        Expected Shortfall (Conditional VaR) loss function
        
        Minimizes expected loss beyond VaR threshold
        """
        residuals = y_true - y_pred
        
        # Estimate VaR threshold
        var_threshold = np.percentile(residuals, alpha * 100)
        
        # Expected shortfall: mean of residuals below VaR
        tail_residuals = residuals[residuals <= var_threshold]
        
        if len(tail_residuals) > 0:
            expected_shortfall = np.mean(tail_residuals)
        else:
            expected_shortfall = var_threshold
        
        # Loss is negative expected shortfall (we want to minimize negative returns)
        loss = -expected_shortfall
        
        # Gradient (simplified - proper ES gradient is more complex)
        gradient = np.where(residuals <= var_threshold, 
                           -1 / (len(residuals) * alpha),
                           0)
        
        return loss, gradient
    
    @staticmethod
    def directional_accuracy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Loss function that penalizes wrong directional predictions
        Important for trading strategies
        """
        # Calculate actual and predicted directions
        true_directions = np.sign(y_true)
        pred_directions = np.sign(y_pred)
        
        # Directional accuracy
        correct_directions = (true_directions == pred_directions).astype(float)
        directional_accuracy = np.mean(correct_directions)
        
        # Loss: 1 - directional accuracy
        loss = 1 - directional_accuracy
        
        # Gradient (simplified - true gradient would be discontinuous)
        # Use smooth approximation: penalize when signs don't match
        gradient = -true_directions * np.exp(-true_directions * y_pred) / (1 + np.exp(-true_directions * y_pred))**2
        gradient = gradient / len(y_pred)
        
        return loss, gradient

def demonstrate_loss_functions():
    """Demonstrate various loss functions in financial context"""
    
    print("=== Loss Functions for Financial Modeling ===\\n")
    
    # Generate synthetic prediction data
    np.random.seed(42)
    n_samples = 1000
    
    # True values (e.g., actual stock returns)
    y_true = np.random.normal(0.001, 0.02, n_samples)  # Daily returns with 2% volatility
    
    # Predictions with different error characteristics
    base_predictions = y_true + np.random.normal(0, 0.005, n_samples)  # Small error
    
    # Add systematic bias and outliers
    y_pred_biased = base_predictions + 0.002  # Overestimation bias
    y_pred_outliers = base_predictions.copy()
    outlier_mask = np.random.choice([False, True], n_samples, p=[0.95, 0.05])
    y_pred_outliers[outlier_mask] += np.random.normal(0, 0.05, np.sum(outlier_mask))
    
    loss_calculator = FinancialLossFunctions()
    
    print("1. Loss Function Comparison:")
    
    # Calculate different losses
    mse_loss, mse_grad = loss_calculator.mean_squared_error(y_true, base_predictions)
    mae_loss, mae_grad = loss_calculator.mean_absolute_error(y_true, base_predictions)
    huber_loss, huber_grad = loss_calculator.huber_loss(y_true, base_predictions, delta=0.01)
    asymmetric_loss, asym_grad = loss_calculator.asymmetric_loss(y_true, base_predictions, alpha=0.7)
    
    print(f"   MSE Loss: {mse_loss:.8f}")
    print(f"   MAE Loss: {mae_loss:.8f}")
    print(f"   Huber Loss: {huber_loss:.8f}")
    print(f"   Asymmetric Loss: {asymmetric_loss:.8f}")
    print()
    
    # 2. Robustness to outliers
    print("2. Robustness to Outliers:")
    
    mse_outlier, _ = loss_calculator.mean_squared_error(y_true, y_pred_outliers)
    mae_outlier, _ = loss_calculator.mean_absolute_error(y_true, y_pred_outliers)
    huber_outlier, _ = loss_calculator.huber_loss(y_true, y_pred_outliers, delta=0.01)
    
    print("   Loss values with outliers:")
    print(f"     MSE: {mse_outlier:.8f} (increase: {(mse_outlier/mse_loss - 1)*100:.1f}%)")
    print(f"     MAE: {mae_outlier:.8f} (increase: {(mae_outlier/mae_loss - 1)*100:.1f}%)")
    print(f"     Huber: {huber_outlier:.8f} (increase: {(huber_outlier/huber_loss - 1)*100:.1f}%)")
    print()
    
    # 3. Quantile regression for VaR
    print("3. Quantile Regression for VaR Estimation:")
    
    # Estimate different quantiles
    quantiles = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
    estimated_quantiles = []
    
    for tau in quantiles:
        # Simple quantile regression using optimization
        def quantile_objective(q):
            return loss_calculator.quantile_loss(y_true, np.full_like(y_true, q), tau)[0]
        
        result = minimize(quantile_objective, np.median(y_true), method='Nelder-Mead')
        estimated_quantiles.append(result.x[0])
    
    print("   Estimated quantiles:")
    for tau, q_est in zip(quantiles, estimated_quantiles):
        q_true = np.percentile(y_true, tau * 100)
        error = abs(q_est - q_true)
        print(f"     {tau:.0%} quantile: {q_est:.6f} (true: {q_true:.6f}, error: {error:.6f})")
    print()
    
    # 4. Expected Shortfall estimation
    print("4. Expected Shortfall Loss:")
    
    es_loss, es_grad = loss_calculator.expected_shortfall_loss(y_true, base_predictions)
    print(f"   Expected Shortfall loss: {es_loss:.6f}")
    
    # True Expected Shortfall
    var_5 = np.percentile(y_true, 5)
    true_es = np.mean(y_true[y_true <= var_5])
    print(f"   True 5% Expected Shortfall: {true_es:.6f}")
    print()
    
    # 5. Directional accuracy
    print("5. Directional Accuracy:")
    
    dir_loss, dir_grad = loss_calculator.directional_accuracy_loss(y_true, base_predictions)
    
    # Calculate actual directional accuracy
    correct_directions = np.sign(y_true) == np.sign(base_predictions)
    actual_accuracy = np.mean(correct_directions)
    
    print(f"   Directional accuracy: {actual_accuracy:.1%}")
    print(f"   Directional loss: {dir_loss:.6f}")
    print()
    
    # Create comprehensive visualizations
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # Plot 1: Loss function shapes
    x = np.linspace(-0.05, 0.05, 1000)
    y_true_single = 0.01  # Single true value for visualization
    
    mse_values = [(x_val - y_true_single)**2 for x_val in x]
    mae_values = [abs(x_val - y_true_single) for x_val in x]
    huber_values = []
    delta = 0.01
    
    for x_val in x:
        residual = abs(x_val - y_true_single)
        if residual <= delta:
            huber_values.append(0.5 * residual**2)
        else:
            huber_values.append(delta * residual - 0.5 * delta**2)
    
    axes[0, 0].plot(x, mse_values, label='MSE', linewidth=2)
    axes[0, 0].plot(x, mae_values, label='MAE', linewidth=2)
    axes[0, 0].plot(x, huber_values, label='Huber', linewidth=2)
    axes[0, 0].axvline(y_true_single, color='red', linestyle='--', alpha=0.7, label='True Value')
    axes[0, 0].set_title('Loss Function Shapes')
    axes[0, 0].set_xlabel('Predicted Value')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Loss function derivatives
    mse_derivs = [-2 * (y_true_single - x_val) for x_val in x]
    mae_derivs = [-np.sign(y_true_single - x_val) for x_val in x]
    huber_derivs = []
    
    for x_val in x:
        residual = y_true_single - x_val
        if abs(residual) <= delta:
            huber_derivs.append(-residual)
        else:
            huber_derivs.append(-delta * np.sign(residual))
    
    axes[0, 1].plot(x, mse_derivs, label='MSE Derivative', linewidth=2)
    axes[0, 1].plot(x, mae_derivs, label='MAE Derivative', linewidth=2)
    axes[0, 1].plot(x, huber_derivs, label='Huber Derivative', linewidth=2)
    axes[0, 1].axvline(y_true_single, color='red', linestyle='--', alpha=0.7, label='True Value')
    axes[0, 1].set_title('Loss Function Derivatives')
    axes[0, 1].set_xlabel('Predicted Value')
    axes[0, 1].set_ylabel('Derivative')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Prediction vs actual scatter
    axes[1, 0].scatter(y_true, base_predictions, alpha=0.6, s=20)
    min_val = min(y_true.min(), base_predictions.min())
    max_val = max(y_true.max(), base_predictions.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[1, 0].set_title('Predictions vs Actual\\n(Base Case)')
    axes[1, 0].set_xlabel('Actual Returns')
    axes[1, 0].set_ylabel('Predicted Returns')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Residuals distribution
    residuals = y_true - base_predictions
    
    axes[1, 1].hist(residuals, bins=50, alpha=0.7, density=True)
    axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.7, label='Zero Error')
    axes[1, 1].set_title('Prediction Residuals Distribution')
    axes[1, 1].set_xlabel('Residual (Actual - Predicted)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Quantile estimates
    axes[2, 0].bar(range(len(quantiles)), estimated_quantiles, alpha=0.7, label='Estimated')
    true_quantiles = [np.percentile(y_true, q * 100) for q in quantiles]
    axes[2, 0].bar(range(len(quantiles)), true_quantiles, alpha=0.5, label='True', width=0.5)
    axes[2, 0].set_title('Quantile Regression Results')
    axes[2, 0].set_xlabel('Quantile')
    axes[2, 0].set_ylabel('Value')
    axes[2, 0].set_xticks(range(len(quantiles)))
    axes[2, 0].set_xticklabels([f'{q:.0%}' for q in quantiles])
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Gradient magnitudes comparison
    gradient_norms = {
        'MSE': np.linalg.norm(mse_grad),
        'MAE': np.linalg.norm(mae_grad),
        'Huber': np.linalg.norm(huber_grad),
        'Asymmetric': np.linalg.norm(asym_grad)
    }
    
    axes[2, 1].bar(gradient_norms.keys(), gradient_norms.values(), alpha=0.7)
    axes[2, 1].set_title('Gradient Magnitudes\\n(Different Loss Functions)')
    axes[2, 1].set_ylabel('Gradient Norm')
    axes[2, 1].tick_params(axis='x', rotation=45)
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'loss_values': {
            'mse': mse_loss,
            'mae': mae_loss,
            'huber': huber_loss,
            'asymmetric': asymmetric_loss
        },
        'gradients': {
            'mse': mse_grad,
            'mae': mae_grad,
            'huber': huber_grad,
            'asymmetric': asym_grad
        },
        'quantile_estimates': dict(zip(quantiles, estimated_quantiles)),
        'directional_accuracy': actual_accuracy
    }

if __name__ == "__main__":
    results = demonstrate_loss_functions()
```

### 🔴 Theoretical Foundation

**Common Loss Functions and Derivatives:**

1. **Mean Squared Error**:
   $$L(y, \hat{y}) = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$
   $$\frac{\partial L}{\partial \hat{y}_i} = -\frac{2}{n}(y_i - \hat{y}_i)$$

2. **Mean Absolute Error**:
   $$L(y, \hat{y}) = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$
   $$\frac{\partial L}{\partial \hat{y}_i} = -\frac{1}{n}\text{sign}(y_i - \hat{y}_i)$$

3. **Huber Loss**:
   $$L_\delta(r) = \begin{cases}
   \frac{1}{2}r^2 & \text{if } |r| \leq \delta \\
   \delta|r| - \frac{1}{2}\delta^2 & \text{if } |r| > \delta
   \end{cases}$$

4. **Quantile Loss** (for $\tau$-quantile):
   $$L_\tau(y, \hat{y}) = \sum_{i=1}^n \rho_\tau(y_i - \hat{y}_i)$$
   where $\rho_\tau(u) = u(\tau - \mathbf{1}_{u < 0})$

**Properties:**
- **MSE**: Differentiable, convex, sensitive to outliers
- **MAE**: Non-differentiable at zero, robust to outliers
- **Huber**: Combines benefits of MSE and MAE
- **Quantile**: Asymmetric, useful for risk measures

---

## 📝 Summary

This section covered essential calculus concepts for financial ML/NLP:

1. **Partial Derivatives**: Foundation for gradient-based optimization, with applications to portfolio risk and sensitivity analysis

2. **Chain Rule**: Critical for backpropagation in neural networks and complex derivative calculations

3. **Optimization Techniques**: Various algorithms (SGD, Adam, RMSprop) with their strengths and weaknesses

4. **Loss Function Derivatives**: Different loss functions for financial applications, from MSE to quantile loss

**Key Takeaways:**
- Partial derivatives enable gradient-based optimization for portfolio and model optimization
- Chain rule is essential for training neural networks and complex financial models
- Different optimizers have different strengths - Adam is often a good default choice
- Loss function choice depends on the specific financial application and error characteristics
- Understanding derivatives helps with debugging and improving model performance
- Financial applications often require specialized loss functions (asymmetric, quantile-based)

**Practical Tips:**
- Always verify analytical gradients with numerical approximations
- Use appropriate learning rates and schedules for different problems
- Choose loss functions based on the financial interpretation of errors
- Consider robustness to outliers in financial data
- Monitor convergence and adjust hyperparameters as needed