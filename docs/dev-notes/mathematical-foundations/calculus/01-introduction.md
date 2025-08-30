# Calculus for Deep Learning in IPO Valuation

## 📈 Chapter Overview

Calculus forms the mathematical backbone of deep learning optimization. This chapter covers essential derivative concepts, gradient calculations, and optimization mathematics that power neural networks and advanced ML models in IPO valuation systems.

## 🎯 Learning Objectives

By the end of this chapter, you will:
- Master partial derivatives and the chain rule for backpropagation
- Understand gradient descent mathematics and convergence analysis
- Implement loss function derivatives and optimization landscapes
- Apply Lagrange multipliers for constrained optimization
- Use Taylor series approximations in numerical methods

## 🧠 Why Calculus Matters in Deep Learning

### The Challenge
Deep learning for IPO valuation involves:
- **Complex Models**: Neural networks with millions of parameters
- **Non-Linear Functions**: Multiple layers of transformations
- **Optimization**: Finding optimal parameters in high-dimensional space
- **Constraints**: Regulatory and business constraints on predictions

### The Solution
Calculus provides:
- **Gradient Information**: Direction of steepest ascent/descent
- **Optimization Theory**: Mathematical foundation for learning algorithms
- **Error Propagation**: Backpropagation through network layers
- **Convergence Analysis**: Understanding when and why algorithms converge

## 🏗️ Chapter Structure

### Part I: Fundamental Derivatives
1. **[Single Variable Calculus](./02-single-variable.md)**
   - Derivatives of common functions
   - Activation function derivatives
   - Loss function derivatives
   - Numerical differentiation

2. **[Multivariable Calculus](./03-multivariable.md)**
   - Partial derivatives
   - Gradient vectors
   - Directional derivatives
   - Chain rule for composition

### Part II: Optimization Mathematics
3. **[Gradient Descent](./04-gradient-descent.md)**
   - Gradient descent algorithm
   - Learning rates and convergence
   - Momentum and acceleration
   - Adaptive learning rates (Adam, RMSprop)

4. **[Second-Order Methods](./05-second-order.md)**
   - Hessian matrices
   - Newton's method
   - Quasi-Newton methods (BFGS)
   - Natural gradients

### Part III: Advanced Topics
5. **[Backpropagation Mathematics](./06-backpropagation.md)**
   - Chain rule in neural networks
   - Forward and backward passes
   - Computational graphs
   - Automatic differentiation

6. **[Constrained Optimization](./07-constrained-optimization.md)**
   - Lagrange multipliers
   - KKT conditions
   - Inequality constraints
   - Portfolio optimization applications

## 💡 Key Concepts with Code Examples

### 1. Activation Function Derivatives
```python
import numpy as np
import matplotlib.pyplot as plt

def activation_functions_and_derivatives():
    """Common activation functions and their derivatives"""
    
    # Sigmoid and its derivative
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)
    
    # ReLU and its derivative
    def relu(x):
        return np.maximum(0, x)
    
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    # Tanh and its derivative
    def tanh(x):
        return np.tanh(x)
    
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2
    
    # Leaky ReLU and its derivative
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    
    return {
        'sigmoid': (sigmoid, sigmoid_derivative),
        'relu': (relu, relu_derivative),
        'tanh': (tanh, tanh_derivative),
        'leaky_relu': (leaky_relu, leaky_relu_derivative)
    }
```

### 2. Loss Function Derivatives
```python
def loss_functions_and_gradients():
    """Common loss functions and their gradients for IPO prediction"""
    
    # Mean Squared Error for regression
    def mse_loss(y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    
    def mse_gradient(y_true, y_pred):
        return 2 * (y_pred - y_true) / len(y_true)
    
    # Cross-Entropy Loss for classification
    def cross_entropy_loss(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def cross_entropy_gradient(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred)) / len(y_true)
    
    # Huber Loss (robust to outliers)
    def huber_loss(y_true, y_pred, delta=1.0):
        error = np.abs(y_true - y_pred)
        return np.where(error <= delta,
                       0.5 * error**2,
                       delta * error - 0.5 * delta**2).mean()
    
    def huber_gradient(y_true, y_pred, delta=1.0):
        error = y_pred - y_true
        return np.where(np.abs(error) <= delta,
                       error,
                       delta * np.sign(error)) / len(y_true)
    
    return {
        'mse': (mse_loss, mse_gradient),
        'cross_entropy': (cross_entropy_loss, cross_entropy_gradient),
        'huber': (huber_loss, huber_gradient)
    }
```

### 3. Gradient Descent Implementation
```python
class GradientDescentOptimizer:
    """Gradient descent optimizer with various variants"""
    
    def __init__(self, learning_rate=0.01, momentum=0.0, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta1 = beta1  # For Adam
        self.beta2 = beta2  # For Adam
        self.velocity = None
        self.m = None  # First moment (Adam)
        self.v = None  # Second moment (Adam)
        self.t = 0     # Time step (Adam)
    
    def sgd_with_momentum(self, params, gradients):
        """SGD with momentum"""
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, gradients)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grad
            updated_param = param + self.velocity[i]
            updated_params.append(updated_param)
        
        return updated_params
    
    def adam(self, params, gradients, eps=1e-8):
        """Adam optimizer"""
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        updated_params = []
        
        for i, (param, grad) in enumerate(zip(params, gradients)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Update parameters
            updated_param = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            updated_params.append(updated_param)
        
        return updated_params
```

### 4. Backpropagation Implementation
```python
class SimpleNeuralNetwork:
    """Simple neural network with backpropagation for IPO prediction"""
    
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights randomly
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        """Forward pass"""
        # Layer 1
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        
        # Layer 2
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)  # Output layer
        
        return self.a2
    
    def backward(self, X, y, output):
        """Backward pass (backpropagation)"""
        m = X.shape[0]  # Number of samples
        
        # Output layer gradients
        dz2 = output - y  # For sigmoid + cross-entropy
        dW2 = (1/m) * self.a1.T @ dz2
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (1/m) * X.T @ dz1
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        return {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2
        }
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
```

## 🎯 IPO Valuation Applications

### 1. Neural Network for IPO Success Prediction
```python
def ipo_success_prediction_network():
    """Neural network specifically designed for IPO success prediction"""
    
    class IPOPredictionNetwork:
        def __init__(self, financial_features, market_features, text_features):
            self.financial_dim = financial_features
            self.market_dim = market_features
            self.text_dim = text_features
            
            # Multi-branch architecture
            self.financial_branch = self._build_financial_branch()
            self.market_branch = self._build_market_branch()
            self.text_branch = self._build_text_branch()
            self.fusion_layer = self._build_fusion_layer()
        
        def _build_financial_branch(self):
            """Branch for processing financial metrics"""
            return {
                'W1': np.random.randn(self.financial_dim, 64) * 0.1,
                'b1': np.zeros((1, 64)),
                'W2': np.random.randn(64, 32) * 0.1,
                'b2': np.zeros((1, 32))
            }
        
        def forward_financial(self, financial_data):
            """Forward pass through financial branch"""
            z1 = financial_data @ self.financial_branch['W1'] + self.financial_branch['b1']
            a1 = np.maximum(0, z1)  # ReLU
            
            z2 = a1 @ self.financial_branch['W2'] + self.financial_branch['b2']
            a2 = np.maximum(0, z2)  # ReLU
            
            return a2, (z1, a1, z2)  # Return activations and intermediate values
        
        def calculate_gradients_financial(self, financial_data, intermediate_values, upstream_grad):
            """Calculate gradients for financial branch"""
            z1, a1, z2 = intermediate_values
            
            # Gradients for second layer
            dz2 = upstream_grad * (z2 > 0)  # ReLU derivative
            dW2 = a1.T @ dz2 / len(financial_data)
            db2 = np.mean(dz2, axis=0, keepdims=True)
            
            # Gradients for first layer
            da1 = dz2 @ self.financial_branch['W2'].T
            dz1 = da1 * (z1 > 0)  # ReLU derivative
            dW1 = financial_data.T @ dz1 / len(financial_data)
            db1 = np.mean(dz1, axis=0, keepdims=True)
            
            return {
                'dW1': dW1, 'db1': db1,
                'dW2': dW2, 'db2': db2
            }
    
    return IPOPredictionNetwork
```

### 2. Constrained Portfolio Optimization
```python
def constrained_portfolio_optimization(expected_returns, cov_matrix, budget=1.0):
    """Portfolio optimization with constraints using Lagrange multipliers"""
    
    def lagrangian(weights, lambda_1, lambda_2):
        """Lagrangian function for constrained optimization"""
        n = len(weights)
        
        # Objective: maximize expected return - 0.5 * risk_aversion * variance
        expected_return = expected_returns.T @ weights
        portfolio_variance = weights.T @ cov_matrix @ weights
        objective = expected_return - 0.5 * 0.5 * portfolio_variance  # risk_aversion = 0.5
        
        # Constraints
        budget_constraint = np.sum(weights) - budget  # Sum of weights = budget
        
        # Lagrangian
        L = objective - lambda_1 * budget_constraint
        
        return L
    
    def lagrangian_gradients(weights, lambda_1):
        """Calculate gradients of Lagrangian"""
        # Gradient w.r.t. weights
        dL_dw = expected_returns - 0.5 * 0.5 * 2 * (cov_matrix @ weights) - lambda_1 * np.ones_like(weights)
        
        # Gradient w.r.t. lambda (constraint)
        dL_dl1 = -(np.sum(weights) - budget)
        
        return dL_dw, dL_dl1
    
    # Solve using gradient-based method
    n_assets = len(expected_returns)
    weights = np.ones(n_assets) / n_assets  # Initial guess
    lambda_1 = 0.0  # Initial Lagrange multiplier
    
    learning_rate = 0.01
    for iteration in range(1000):
        dL_dw, dL_dl1 = lagrangian_gradients(weights, lambda_1)
        
        # Update weights and multiplier
        weights += learning_rate * dL_dw
        lambda_1 += learning_rate * dL_dl1
        
        # Check convergence
        if np.linalg.norm(dL_dw) < 1e-6 and abs(dL_dl1) < 1e-6:
            break
    
    return weights, lambda_1
```

### 3. Taylor Series for Risk Approximation
```python
def taylor_series_risk_approximation():
    """Use Taylor series to approximate portfolio risk changes"""
    
    def portfolio_variance(weights, cov_matrix):
        """Calculate portfolio variance"""
        return weights.T @ cov_matrix @ weights
    
    def portfolio_variance_gradient(weights, cov_matrix):
        """Gradient of portfolio variance"""
        return 2 * (cov_matrix @ weights)
    
    def portfolio_variance_hessian(cov_matrix):
        """Hessian of portfolio variance"""
        return 2 * cov_matrix
    
    def taylor_approximation(weights_0, delta_weights, cov_matrix, order=2):
        """Taylor series approximation of variance change"""
        # Current variance
        var_0 = portfolio_variance(weights_0, cov_matrix)
        
        if order >= 1:
            # First-order term
            grad = portfolio_variance_gradient(weights_0, cov_matrix)
            first_order = grad.T @ delta_weights
        else:
            first_order = 0
        
        if order >= 2:
            # Second-order term
            hessian = portfolio_variance_hessian(cov_matrix)
            second_order = 0.5 * delta_weights.T @ hessian @ delta_weights
        else:
            second_order = 0
        
        # Taylor approximation
        var_approx = var_0 + first_order + second_order
        
        # Actual variance for comparison
        new_weights = weights_0 + delta_weights
        var_actual = portfolio_variance(new_weights, cov_matrix)
        
        return {
            'approximation': var_approx,
            'actual': var_actual,
            'error': abs(var_actual - var_approx),
            'first_order_contribution': first_order,
            'second_order_contribution': second_order
        }
    
    return taylor_approximation
```

## 🚨 Common Pitfalls and Solutions

### 1. Vanishing/Exploding Gradients
```python
def gradient_clipping_and_monitoring():
    """Monitor and handle gradient problems"""
    
    def clip_gradients(gradients, max_norm=1.0):
        """Clip gradients by norm"""
        total_norm = 0
        for grad in gradients:
            total_norm += np.sum(grad**2)
        total_norm = np.sqrt(total_norm)
        
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            return [grad * clip_coef for grad in gradients]
        return gradients
    
    def gradient_monitoring(gradients, layer_names):
        """Monitor gradient statistics"""
        stats = {}
        for grad, name in zip(gradients, layer_names):
            stats[name] = {
                'mean': np.mean(grad),
                'std': np.std(grad),
                'max': np.max(grad),
                'min': np.min(grad),
                'norm': np.linalg.norm(grad)
            }
        return stats
    
    return clip_gradients, gradient_monitoring
```

### 2. Numerical Stability
```python
def numerically_stable_functions():
    """Numerically stable implementations of common functions"""
    
    def stable_softmax(x):
        """Numerically stable softmax"""
        # Subtract max for stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def stable_log_sum_exp(x):
        """Numerically stable log-sum-exp"""
        max_x = np.max(x, axis=-1, keepdims=True)
        return max_x + np.log(np.sum(np.exp(x - max_x), axis=-1, keepdims=True))
    
    def stable_sigmoid(x):
        """Numerically stable sigmoid"""
        return np.where(x >= 0,
                       1 / (1 + np.exp(-x)),
                       np.exp(x) / (1 + np.exp(x)))
    
    return stable_softmax, stable_log_sum_exp, stable_sigmoid
```

## 📊 Optimization Diagnostics

### 1. Convergence Analysis
```python
def convergence_diagnostics():
    """Tools for analyzing optimization convergence"""
    
    def plot_loss_landscape(loss_fn, param_ranges, resolution=50):
        """Visualize loss landscape for 2D parameter space"""
        x_range, y_range = param_ranges
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = loss_fn([X[i, j], Y[i, j]])
        
        return X, Y, Z
    
    def check_convergence_criteria(loss_history, tolerance=1e-6, patience=10):
        """Check various convergence criteria"""
        if len(loss_history) < patience + 1:
            return False
        
        # Relative improvement
        recent_losses = loss_history[-patience:]
        relative_improvement = (recent_losses[0] - recent_losses[-1]) / abs(recent_losses[0])
        
        # Gradient-based convergence (if available)
        # Standard deviation of recent losses
        loss_std = np.std(recent_losses)
        
        convergence_checks = {
            'relative_improvement': relative_improvement < tolerance,
            'loss_stability': loss_std < tolerance,
            'converged': relative_improvement < tolerance and loss_std < tolerance
        }
        
        return convergence_checks
    
    return plot_loss_landscape, check_convergence_criteria
```

## 📚 Prerequisites for Next Chapters

Before proceeding to Financial Mathematics, ensure you understand:
- Partial derivatives and chain rule
- Gradient descent convergence theory
- Backpropagation mechanics
- Constrained optimization basics
- Numerical stability considerations

---

**Next**: [Financial Mathematics for Valuation Models](../financial-math/01-introduction.md)