# Advanced ML Mathematics for IPO Valuation

## 🤖 Chapter Overview

Advanced machine learning mathematics provides the theoretical foundation for sophisticated modeling techniques in IPO valuation. This chapter covers statistical learning theory, regularization methods, ensemble techniques, and neural network mathematics that power state-of-the-art IPO prediction systems.

## 🎯 Learning Objectives

By the end of this chapter, you will:
- Master maximum likelihood estimation principles
- Understand regularization mathematics (L1/L2 penalty terms)
- Apply cross-validation theory and statistical significance
- Implement ensemble method mathematics (bagging, boosting)
- Design neural network architectures with mathematical rigor

## 🧠 Why Advanced ML Mathematics Matters

### The Challenge
Modern IPO valuation requires:
- **Complex Pattern Recognition**: Non-linear relationships in financial data
- **Overfitting Prevention**: Models that generalize to unseen IPOs
- **Uncertainty Quantification**: Confidence intervals and prediction reliability
- **Multi-Modal Data**: Text, numerical, and categorical features combined
- **Real-Time Inference**: Efficient algorithms for live predictions

### The Solution
Advanced ML mathematics provides:
- **Theoretical Guarantees**: Understanding when and why algorithms work
- **Optimization Frameworks**: Systematic approach to model improvement
- **Regularization Theory**: Methods to prevent overfitting
- **Statistical Inference**: Rigorous testing and validation
- **Ensemble Methods**: Combining multiple models for better performance

## 🏗️ Chapter Structure

### Part I: Statistical Learning Theory
1. **[Maximum Likelihood Estimation](./02-maximum-likelihood.md)**
   - MLE principles and derivations
   - Log-likelihood optimization
   - Fisher information and Cramér-Rao bounds
   - Applications to financial modeling

2. **[Bayesian Machine Learning](./03-bayesian-ml.md)**
   - Prior distributions and posterior inference
   - Variational Bayes
   - Markov Chain Monte Carlo methods
   - Gaussian Processes for uncertainty quantification

### Part II: Regularization and Model Selection
3. **[Regularization Theory](./04-regularization.md)**
   - L1 (Lasso) and L2 (Ridge) regularization mathematics
   - Elastic Net and hybrid approaches
   - Group regularization and structured sparsity
   - Regularization paths and parameter selection

4. **[Cross-Validation Mathematics](./05-cross-validation.md)**
   - Statistical theory of cross-validation
   - Bias-variance decomposition
   - Model selection criteria (AIC, BIC, MDL)
   - Time series cross-validation for financial data

### Part III: Ensemble Methods
5. **[Bagging and Random Forests](./06-bagging.md)**
   - Bootstrap aggregating theory
   - Random Forest mathematics
   - Out-of-bag error estimation
   - Variable importance measures

6. **[Boosting Mathematics](./07-boosting.md)**
   - AdaBoost algorithm and theory
   - Gradient boosting mathematics
   - XGBoost and LightGBM optimizations
   - Regularized boosting

### Part IV: Neural Networks
7. **[Deep Learning Mathematics](./08-deep-learning.md)**
   - Universal approximation theorem
   - Backpropagation derivations
   - Activation function theory
   - Optimization landscapes

8. **[Advanced Architectures](./09-advanced-architectures.md)**
   - Attention mechanisms mathematics
   - Transformer architectures
   - Graph neural networks
   - Multi-modal fusion networks

## 💡 Core Mathematical Concepts

### 1. Maximum Likelihood Estimation
```python
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class MaximumLikelihoodEstimator:
    """Maximum Likelihood Estimation for various distributions"""
    
    def __init__(self):
        self.fitted_params = None
        self.log_likelihood = None
    
    def normal_mle(self, data):
        """MLE for normal distribution parameters"""
        n = len(data)
        
        # Analytical solution for normal distribution
        mu_mle = np.mean(data)
        sigma2_mle = np.mean((data - mu_mle)**2)  # Biased estimator
        sigma_mle = np.sqrt(sigma2_mle)
        
        # Log-likelihood
        log_likelihood = -n/2 * np.log(2 * np.pi * sigma2_mle) - n/2
        
        return {
            'mu': mu_mle,
            'sigma': sigma_mle,
            'log_likelihood': log_likelihood,
            'aic': 2 * 2 - 2 * log_likelihood,  # 2 parameters
            'bic': 2 * np.log(n) - 2 * log_likelihood
        }
    
    def logistic_regression_mle(self, X, y, max_iter=1000):
        """MLE for logistic regression"""
        n_features = X.shape[1]
        
        def negative_log_likelihood(beta):
            """Negative log-likelihood for minimization"""
            linear_pred = X @ beta
            # Numerically stable sigmoid
            sigmoid_pred = 1 / (1 + np.exp(-np.clip(linear_pred, -250, 250)))
            
            # Avoid log(0) by clipping
            sigmoid_pred = np.clip(sigmoid_pred, 1e-15, 1 - 1e-15)
            
            # Log-likelihood
            ll = np.sum(y * np.log(sigmoid_pred) + (1 - y) * np.log(1 - sigmoid_pred))
            return -ll
        
        def gradient(beta):
            """Gradient of negative log-likelihood"""
            linear_pred = X @ beta
            sigmoid_pred = 1 / (1 + np.exp(-np.clip(linear_pred, -250, 250)))
            return -X.T @ (y - sigmoid_pred)
        
        def hessian(beta):
            """Hessian matrix for Newton's method"""
            linear_pred = X @ beta
            sigmoid_pred = 1 / (1 + np.exp(-np.clip(linear_pred, -250, 250)))
            weights = sigmoid_pred * (1 - sigmoid_pred)
            return X.T @ np.diag(weights) @ X
        
        # Initial guess
        beta_init = np.zeros(n_features)
        
        # Optimization using L-BFGS-B
        result = minimize(negative_log_likelihood, beta_init, 
                         method='L-BFGS-B', jac=gradient,
                         options={'maxiter': max_iter})
        
        beta_mle = result.x
        log_likelihood = -result.fun
        
        # Standard errors (inverse of Fisher information)
        try:
            fisher_info = hessian(beta_mle)
            cov_matrix = np.linalg.inv(fisher_info)
            standard_errors = np.sqrt(np.diag(cov_matrix))
        except:
            standard_errors = np.full(n_features, np.nan)
        
        return {
            'coefficients': beta_mle,
            'standard_errors': standard_errors,
            'log_likelihood': log_likelihood,
            'aic': 2 * n_features - 2 * log_likelihood,
            'bic': n_features * np.log(len(y)) - 2 * log_likelihood,
            'convergence': result.success
        }
    
    def beta_distribution_mle(self, data):
        """MLE for Beta distribution (useful for success probabilities)"""
        # Beta distribution is bounded [0,1], check data
        if np.any(data <= 0) or np.any(data >= 1):
            raise ValueError("Beta distribution requires data in (0,1)")
        
        def negative_log_likelihood(params):
            alpha, beta = params
            if alpha <= 0 or beta <= 0:
                return np.inf
            
            from scipy.stats import beta as beta_dist
            ll = np.sum(beta_dist.logpdf(data, alpha, beta))
            return -ll
        
        # Method of moments for initial guess
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)
        
        # Method of moments estimators
        alpha_mom = sample_mean * ((sample_mean * (1 - sample_mean)) / sample_var - 1)
        beta_mom = (1 - sample_mean) * ((sample_mean * (1 - sample_mean)) / sample_var - 1)
        
        # Ensure positive initial values
        alpha_mom = max(alpha_mom, 0.1)
        beta_mom = max(beta_mom, 0.1)
        
        result = minimize(negative_log_likelihood, [alpha_mom, beta_mom],
                         method='L-BFGS-B',
                         bounds=[(0.01, None), (0.01, None)])
        
        alpha_mle, beta_mle = result.x
        log_likelihood = -result.fun
        
        return {
            'alpha': alpha_mle,
            'beta': beta_mle,
            'log_likelihood': log_likelihood,
            'aic': 2 * 2 - 2 * log_likelihood,
            'bic': 2 * np.log(len(data)) - 2 * log_likelihood
        }
```

### 2. Regularization Mathematics
```python
class RegularizedRegression:
    """Regularized regression with L1, L2, and Elastic Net penalties"""
    
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        self.alpha = alpha  # Overall regularization strength
        self.l1_ratio = l1_ratio  # Balance between L1 and L2 (0=Ridge, 1=Lasso)
        self.coefficients = None
        self.regularization_path = None
    
    def ridge_regression(self, X, y, alpha):
        """Ridge regression with L2 penalty"""
        # Analytical solution: (X'X + αI)^-1 X'y
        n_features = X.shape[1]
        identity = np.eye(n_features)
        
        # Ridge solution
        XtX_regularized = X.T @ X + alpha * identity
        coefficients = np.linalg.solve(XtX_regularized, X.T @ y)
        
        return coefficients
    
    def lasso_coordinate_descent(self, X, y, alpha, max_iter=1000, tol=1e-6):
        """Lasso regression using coordinate descent"""
        n_samples, n_features = X.shape
        
        # Standardize features for coordinate descent
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / X_std
        
        # Initialize coefficients
        beta = np.zeros(n_features)
        
        # Precompute X'X diagonal for efficiency
        XtX_diag = np.sum(X_normalized**2, axis=0)
        
        for iteration in range(max_iter):
            beta_old = beta.copy()
            
            for j in range(n_features):
                # Compute residual without feature j
                residual = y - X_normalized @ beta + X_normalized[:, j] * beta[j]
                
                # Coordinate update with soft thresholding
                rho = X_normalized[:, j] @ residual
                
                if rho > alpha:
                    beta[j] = (rho - alpha) / XtX_diag[j]
                elif rho < -alpha:
                    beta[j] = (rho + alpha) / XtX_diag[j]
                else:
                    beta[j] = 0.0
            
            # Check convergence
            if np.max(np.abs(beta - beta_old)) < tol:
                break
        
        # Scale back coefficients
        beta_scaled = beta / X_std
        
        return beta_scaled, iteration + 1
    
    def elastic_net(self, X, y, alpha, l1_ratio, max_iter=1000, tol=1e-6):
        """Elastic Net regression combining L1 and L2 penalties"""
        n_samples, n_features = X.shape
        
        # Penalty components
        alpha_l1 = alpha * l1_ratio
        alpha_l2 = alpha * (1 - l1_ratio)
        
        # Standardize
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / X_std
        
        beta = np.zeros(n_features)
        XtX_diag = np.sum(X_normalized**2, axis=0)
        
        for iteration in range(max_iter):
            beta_old = beta.copy()
            
            for j in range(n_features):
                # Partial residual
                residual = y - X_normalized @ beta + X_normalized[:, j] * beta[j]
                rho = X_normalized[:, j] @ residual
                
                # Elastic Net soft thresholding
                if rho > alpha_l1:
                    beta[j] = (rho - alpha_l1) / (XtX_diag[j] + alpha_l2)
                elif rho < -alpha_l1:
                    beta[j] = (rho + alpha_l1) / (XtX_diag[j] + alpha_l2)
                else:
                    beta[j] = 0.0
            
            if np.max(np.abs(beta - beta_old)) < tol:
                break
        
        return beta / X_std, iteration + 1
    
    def regularization_path(self, X, y, alphas=None, l1_ratio=0.5):
        """Compute regularization path for different alpha values"""
        if alphas is None:
            # Logarithmically spaced alphas
            alpha_max = np.max(np.abs(X.T @ y)) / len(y)
            alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_max * 0.001), 100)
        
        coefficients_path = np.zeros((len(alphas), X.shape[1]))
        
        for i, alpha in enumerate(alphas):
            coef, _ = self.elastic_net(X, y, alpha, l1_ratio)
            coefficients_path[i] = coef
        
        self.regularization_path = {
            'alphas': alphas,
            'coefficients': coefficients_path,
            'n_nonzero': np.sum(coefficients_path != 0, axis=1)
        }
        
        return self.regularization_path
    
    def cross_validation_score(self, X, y, alpha, l1_ratio=0.5, cv_folds=5):
        """Cross-validation score for hyperparameter selection"""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit model
            coef, _ = self.elastic_net(X_train, y_train, alpha, l1_ratio)
            
            # Predict and score
            y_pred = X_val @ coef
            mse = np.mean((y_val - y_pred)**2)
            cv_scores.append(mse)
        
        return np.mean(cv_scores), np.std(cv_scores)
```

### 3. Ensemble Methods Implementation
```python
class EnsembleMethods:
    """Implementation of bagging and boosting algorithms"""
    
    def __init__(self):
        self.models = []
        self.weights = []
    
    def bootstrap_sample(self, X, y, random_state=None):
        """Generate bootstrap sample"""
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples = len(X)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices], indices
    
    def random_forest_regressor(self, X, y, n_trees=100, max_features='sqrt', 
                               max_depth=None, min_samples_split=2):
        """Random Forest implementation"""
        from sklearn.tree import DecisionTreeRegressor
        
        n_samples, n_features = X.shape
        
        if max_features == 'sqrt':
            max_feat = int(np.sqrt(n_features))
        elif max_features == 'log2':
            max_feat = int(np.log2(n_features))
        else:
            max_feat = max_features
        
        trees = []
        feature_indices = []
        
        for i in range(n_trees):
            # Bootstrap sample
            X_boot, y_boot, _ = self.bootstrap_sample(X, y, random_state=i)
            
            # Random feature selection
            feature_idx = np.random.choice(n_features, size=max_feat, replace=False)
            X_boot_subset = X_boot[:, feature_idx]
            
            # Train tree
            tree = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=i
            )
            tree.fit(X_boot_subset, y_boot)
            
            trees.append(tree)
            feature_indices.append(feature_idx)
        
        self.trees = trees
        self.feature_indices = feature_indices
        
        return self
    
    def random_forest_predict(self, X):
        """Predict using Random Forest"""
        predictions = np.zeros((X.shape[0], len(self.trees)))
        
        for i, (tree, feature_idx) in enumerate(zip(self.trees, self.feature_indices)):
            X_subset = X[:, feature_idx]
            predictions[:, i] = tree.predict(X_subset)
        
        return np.mean(predictions, axis=1)
    
    def adaboost_regressor(self, X, y, n_estimators=50, learning_rate=1.0):
        """AdaBoost for regression"""
        from sklearn.tree import DecisionTreeRegressor
        
        n_samples = len(X)
        
        # Initialize weights
        sample_weights = np.ones(n_samples) / n_samples
        
        estimators = []
        estimator_weights = []
        estimator_errors = []
        
        for iboost in range(n_estimators):
            # Train weak learner with weighted samples
            estimator = DecisionTreeRegressor(max_depth=1, random_state=iboost)
            estimator.fit(X, y, sample_weight=sample_weights)
            
            # Predictions
            y_pred = estimator.predict(X)
            
            # Calculate error
            error_vect = np.abs(y_pred - y)
            error_max = error_vect.max()
            
            if error_max != 0:
                error_vect /= error_max
            
            # Average error weighted by sample weights
            estimator_error = np.average(error_vect, weights=sample_weights)
            
            # If perfect prediction, stop
            if estimator_error <= 0:
                break
            
            # If worse than random, don't use this estimator
            if estimator_error >= 0.5:
                if len(estimators) == 0:
                    raise ValueError("First estimator error is >= 0.5")
                break
            
            # Calculate alpha (estimator weight)
            alpha = learning_rate * np.log((1 - estimator_error) / estimator_error)
            
            # Update sample weights
            sample_weights *= np.exp(alpha * error_vect)
            sample_weights /= np.sum(sample_weights)  # Normalize
            
            estimators.append(estimator)
            estimator_weights.append(alpha)
            estimator_errors.append(estimator_error)
        
        self.estimators = estimators
        self.estimator_weights = np.array(estimator_weights)
        self.estimator_errors = np.array(estimator_errors)
        
        return self
    
    def adaboost_predict(self, X):
        """Predict using AdaBoost"""
        predictions = np.zeros((X.shape[0], len(self.estimators)))
        
        for i, estimator in enumerate(self.estimators):
            predictions[:, i] = estimator.predict(X)
        
        # Weighted average of predictions
        return np.average(predictions, axis=1, weights=self.estimator_weights)
    
    def gradient_boosting_regressor(self, X, y, n_estimators=100, learning_rate=0.1, 
                                   max_depth=3, subsample=1.0):
        """Gradient Boosting implementation"""
        from sklearn.tree import DecisionTreeRegressor
        
        n_samples = X.shape[0]
        
        # Initialize with mean prediction
        initial_prediction = np.mean(y)
        predictions = np.full(n_samples, initial_prediction)
        
        estimators = []
        
        for i in range(n_estimators):
            # Calculate negative gradients (residuals for MSE)
            residuals = y - predictions
            
            # Subsample for stochastic gradient boosting
            if subsample < 1.0:
                subsample_size = int(n_samples * subsample)
                indices = np.random.choice(n_samples, size=subsample_size, replace=False)
                X_subsample = X[indices]
                residuals_subsample = residuals[indices]
            else:
                X_subsample = X
                residuals_subsample = residuals
            
            # Fit tree to residuals
            tree = DecisionTreeRegressor(
                max_depth=max_depth,
                random_state=i
            )
            tree.fit(X_subsample, residuals_subsample)
            
            # Update predictions
            tree_predictions = tree.predict(X)
            predictions += learning_rate * tree_predictions
            
            estimators.append(tree)
        
        self.initial_prediction = initial_prediction
        self.estimators = estimators
        self.learning_rate = learning_rate
        
        return self
    
    def gradient_boosting_predict(self, X):
        """Predict using Gradient Boosting"""
        predictions = np.full(X.shape[0], self.initial_prediction)
        
        for estimator in self.estimators:
            predictions += self.learning_rate * estimator.predict(X)
        
        return predictions
```

### 4. Neural Network Mathematics
```python
class NeuralNetworkMath:
    """Mathematical implementations of neural network components"""
    
    def __init__(self):
        self.layers = []
        self.history = {'loss': [], 'accuracy': []}
    
    def xavier_initialization(self, fan_in, fan_out):
        """Xavier/Glorot initialization for weights"""
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, (fan_in, fan_out))
    
    def he_initialization(self, fan_in, fan_out):
        """He initialization for ReLU networks"""
        std = np.sqrt(2 / fan_in)
        return np.random.normal(0, std, (fan_in, fan_out))
    
    def batch_normalization_forward(self, X, gamma, beta, eps=1e-8):
        """Batch normalization forward pass"""
        # Calculate statistics
        mean = np.mean(X, axis=0)
        variance = np.var(X, axis=0)
        
        # Normalize
        X_normalized = (X - mean) / np.sqrt(variance + eps)
        
        # Scale and shift
        output = gamma * X_normalized + beta
        
        # Cache for backward pass
        cache = {
            'X': X,
            'mean': mean,
            'variance': variance,
            'X_normalized': X_normalized,
            'gamma': gamma,
            'eps': eps
        }
        
        return output, cache
    
    def batch_normalization_backward(self, dout, cache):
        """Batch normalization backward pass"""
        X = cache['X']
        mean = cache['mean']
        variance = cache['variance']
        X_normalized = cache['X_normalized']
        gamma = cache['gamma']
        eps = cache['eps']
        
        N = X.shape[0]
        
        # Gradients w.r.t. gamma and beta
        dgamma = np.sum(dout * X_normalized, axis=0)
        dbeta = np.sum(dout, axis=0)
        
        # Gradient w.r.t. normalized input
        dX_normalized = dout * gamma
        
        # Gradient w.r.t. variance
        dvariance = np.sum(dX_normalized * (X - mean) * -0.5 * 
                          (variance + eps)**(-3/2), axis=0)
        
        # Gradient w.r.t. mean
        dmean = (np.sum(dX_normalized * -1 / np.sqrt(variance + eps), axis=0) +
                dvariance * np.sum(-2 * (X - mean), axis=0) / N)
        
        # Gradient w.r.t. input
        dX = (dX_normalized / np.sqrt(variance + eps) +
              dvariance * 2 * (X - mean) / N +
              dmean / N)
        
        return dX, dgamma, dbeta
    
    def dropout_forward(self, X, p=0.5, training=True):
        """Dropout forward pass"""
        if training:
            mask = np.random.binomial(1, 1-p, size=X.shape) / (1-p)
            output = X * mask
            cache = mask
        else:
            output = X
            cache = None
        
        return output, cache
    
    def dropout_backward(self, dout, cache):
        """Dropout backward pass"""
        if cache is not None:
            dX = dout * cache
        else:
            dX = dout
        
        return dX
    
    def attention_mechanism(self, query, key, value, mask=None):
        """Scaled dot-product attention"""
        d_k = query.shape[-1]
        
        # Compute attention scores
        scores = np.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        
        # Softmax
        attention_weights = self.stable_softmax(scores)
        
        # Apply attention to values
        output = np.matmul(attention_weights, value)
        
        return output, attention_weights
    
    def stable_softmax(self, x, axis=-1):
        """Numerically stable softmax"""
        # Subtract max for numerical stability
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def layer_normalization(self, X, gamma, beta, eps=1e-6):
        """Layer normalization (alternative to batch norm)"""
        # Normalize along feature dimension
        mean = np.mean(X, axis=-1, keepdims=True)
        variance = np.var(X, axis=-1, keepdims=True)
        
        X_normalized = (X - mean) / np.sqrt(variance + eps)
        output = gamma * X_normalized + beta
        
        return output
    
    def residual_connection(self, X, layer_output):
        """Residual connection (skip connection)"""
        # Ensure dimensions match
        if X.shape != layer_output.shape:
            raise ValueError("Shapes must match for residual connection")
        
        return X + layer_output
    
    def positional_encoding(self, seq_length, d_model):
        """Positional encoding for transformer models"""
        pe = np.zeros((seq_length, d_model))
        
        position = np.arange(seq_length).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * 
                         -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
```

## 🎯 IPO-Specific Applications

### 1. Multi-Modal IPO Prediction
```python
def multi_modal_ipo_prediction():
    """Multi-modal neural network for IPO success prediction"""
    
    class MultiModalIPOPredictor:
        def __init__(self, financial_dim, text_dim, market_dim):
            self.financial_dim = financial_dim
            self.text_dim = text_dim
            self.market_dim = market_dim
            
            # Initialize networks for each modality
            self.financial_net = self._build_financial_branch()
            self.text_net = self._build_text_branch()
            self.market_net = self._build_market_branch()
            self.fusion_net = self._build_fusion_layer()
        
        def _build_financial_branch(self):
            """Deep network for financial metrics"""
            return {
                'W1': np.random.randn(self.financial_dim, 128) * 0.1,
                'b1': np.zeros((1, 128)),
                'W2': np.random.randn(128, 64) * 0.1,
                'b2': np.zeros((1, 64)),
                'dropout_p': 0.3
            }
        
        def _build_text_branch(self):
            """Transformer-based text processing"""
            return {
                'embedding_dim': 256,
                'num_heads': 8,
                'num_layers': 4,
                'W_embed': np.random.randn(10000, 256) * 0.1,  # Vocab size
                'positional_encoding': self._generate_positional_encoding(512, 256)
            }
        
        def _build_market_branch(self):
            """Market condition processing"""
            return {
                'W1': np.random.randn(self.market_dim, 32) * 0.1,
                'b1': np.zeros((1, 32)),
                'W2': np.random.randn(32, 16) * 0.1,
                'b2': np.zeros((1, 16))
            }
        
        def _build_fusion_layer(self):
            """Fusion layer combining all modalities"""
            fusion_input_dim = 64 + 256 + 16  # Output dims from branches
            return {
                'W_attention': np.random.randn(fusion_input_dim, 128) * 0.1,
                'W_final': np.random.randn(128, 1) * 0.1,
                'b_final': np.zeros((1, 1))
            }
        
        def forward_pass(self, financial_data, text_data, market_data):
            """Forward pass through multi-modal network"""
            # Financial branch
            f1 = np.maximum(0, financial_data @ self.financial_net['W1'] + 
                           self.financial_net['b1'])  # ReLU
            f1_dropout = f1 * (np.random.binomial(1, 0.7, f1.shape) / 0.7)  # Dropout
            financial_output = np.maximum(0, f1_dropout @ self.financial_net['W2'] + 
                                        self.financial_net['b2'])
            
            # Text branch (simplified transformer)
            text_embedded = self._embed_text(text_data)
            text_output = self._transformer_forward(text_embedded)
            
            # Market branch
            m1 = np.maximum(0, market_data @ self.market_net['W1'] + 
                           self.market_net['b1'])
            market_output = np.maximum(0, m1 @ self.market_net['W2'] + 
                                     self.market_net['b2'])
            
            # Fusion
            combined = np.concatenate([financial_output, text_output, market_output], 
                                    axis=1)
            
            # Attention-based fusion
            attention_weights = self._compute_attention(combined)
            attended_features = combined * attention_weights
            
            # Final prediction
            final_features = np.maximum(0, attended_features @ 
                                      self.fusion_net['W_attention'])
            prediction = self._sigmoid(final_features @ self.fusion_net['W_final'] + 
                                     self.fusion_net['b_final'])
            
            return prediction
        
        def _embed_text(self, text_indices):
            """Simple text embedding"""
            return self.text_net['W_embed'][text_indices]
        
        def _transformer_forward(self, embedded_text):
            """Simplified transformer forward pass"""
            # Add positional encoding
            seq_len = embedded_text.shape[1]
            pos_enc = self.text_net['positional_encoding'][:seq_len]
            
            x = embedded_text + pos_enc
            
            # Multi-head self-attention (simplified)
            attention_output = self._multi_head_attention(x, x, x)
            
            # Global average pooling
            return np.mean(attention_output, axis=1)
        
        def _multi_head_attention(self, query, key, value):
            """Multi-head attention mechanism"""
            batch_size, seq_len, d_model = query.shape
            num_heads = self.text_net['num_heads']
            d_k = d_model // num_heads
            
            # Split into heads (simplified)
            attention_output = np.zeros_like(query)
            
            for head in range(num_heads):
                start_idx = head * d_k
                end_idx = (head + 1) * d_k
                
                q_head = query[:, :, start_idx:end_idx]
                k_head = key[:, :, start_idx:end_idx]
                v_head = value[:, :, start_idx:end_idx]
                
                # Scaled dot-product attention
                scores = np.matmul(q_head, k_head.transpose(0, 2, 1)) / np.sqrt(d_k)
                attention_weights = self._softmax(scores, axis=-1)
                head_output = np.matmul(attention_weights, v_head)
                
                attention_output[:, :, start_idx:end_idx] = head_output
            
            return attention_output
        
        def _compute_attention(self, features):
            """Compute attention weights for feature fusion"""
            # Simple attention mechanism
            scores = np.tanh(features @ self.fusion_net['W_attention'])
            attention_weights = self._softmax(np.sum(scores, axis=-1, keepdims=True), 
                                            axis=1)
            return attention_weights
        
        def _softmax(self, x, axis=-1):
            """Stable softmax implementation"""
            x_max = np.max(x, axis=axis, keepdims=True)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        
        def _sigmoid(self, x):
            """Stable sigmoid implementation"""
            return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    return MultiModalIPOPredictor
```

### 2. Uncertainty Quantification with Bayesian Neural Networks
```python
class BayesianNeuralNetwork:
    """Bayesian Neural Network for uncertainty quantification in IPO prediction"""
    
    def __init__(self, layer_sizes, prior_std=1.0):
        self.layer_sizes = layer_sizes
        self.prior_std = prior_std
        self.posterior_samples = []
        
    def log_prior(self, weights):
        """Log prior probability of weights (Gaussian prior)"""
        return -0.5 * np.sum(weights**2) / (self.prior_std**2)
    
    def log_likelihood(self, weights, X, y):
        """Log likelihood of data given weights"""
        predictions = self.forward_pass(X, weights)
        
        # Assume Gaussian likelihood for regression
        mse = np.mean((y - predictions)**2)
        log_likelihood = -0.5 * len(y) * np.log(2 * np.pi * mse) - len(y) * mse / (2 * mse)
        
        return log_likelihood
    
    def log_posterior(self, weights, X, y):
        """Log posterior probability"""
        return self.log_prior(weights) + self.log_likelihood(weights, X, y)
    
    def hamiltonian_monte_carlo(self, X, y, n_samples=1000, step_size=0.01, n_steps=10):
        """Hamiltonian Monte Carlo sampling"""
        n_weights = sum(l1 * l2 for l1, l2 in zip(self.layer_sizes[:-1], self.layer_sizes[1:]))
        
        # Initialize
        current_weights = np.random.normal(0, self.prior_std, n_weights)
        samples = []
        
        for sample in range(n_samples):
            # Current state
            current_log_prob = self.log_posterior(current_weights, X, y)
            
            # Initialize momentum
            momentum = np.random.normal(0, 1, n_weights)
            current_kinetic = 0.5 * np.sum(momentum**2)
            
            # Make copy of current state
            proposed_weights = current_weights.copy()
            proposed_momentum = momentum.copy()
            
            # Leapfrog steps
            for step in range(n_steps):
                # Half step for momentum
                grad = self.gradient_log_posterior(proposed_weights, X, y)
                proposed_momentum += 0.5 * step_size * grad
                
                # Full step for position
                proposed_weights += step_size * proposed_momentum
                
                # Half step for momentum
                grad = self.gradient_log_posterior(proposed_weights, X, y)
                proposed_momentum += 0.5 * step_size * grad
            
            # Evaluate proposed state
            proposed_log_prob = self.log_posterior(proposed_weights, X, y)
            proposed_kinetic = 0.5 * np.sum(proposed_momentum**2)
            
            # Acceptance probability
            log_alpha = (proposed_log_prob - current_log_prob + 
                        current_kinetic - proposed_kinetic)
            alpha = min(1, np.exp(log_alpha))
            
            # Accept or reject
            if np.random.uniform() < alpha:
                current_weights = proposed_weights
                current_log_prob = proposed_log_prob
            
            samples.append(current_weights.copy())
        
        self.posterior_samples = samples
        return samples
    
    def gradient_log_posterior(self, weights, X, y):
        """Gradient of log posterior (for HMC)"""
        # Gradient of log prior
        grad_prior = -weights / (self.prior_std**2)
        
        # Gradient of log likelihood (numerical approximation)
        eps = 1e-5
        grad_likelihood = np.zeros_like(weights)
        
        for i in range(len(weights)):
            weights_plus = weights.copy()
            weights_minus = weights.copy()
            weights_plus[i] += eps
            weights_minus[i] -= eps
            
            ll_plus = self.log_likelihood(weights_plus, X, y)
            ll_minus = self.log_likelihood(weights_minus, X, y)
            
            grad_likelihood[i] = (ll_plus - ll_minus) / (2 * eps)
        
        return grad_prior + grad_likelihood
    
    def predict_with_uncertainty(self, X, n_samples=100):
        """Make predictions with uncertainty estimates"""
        if not self.posterior_samples:
            raise ValueError("No posterior samples available. Run sampling first.")
        
        # Use last n_samples from posterior
        recent_samples = self.posterior_samples[-n_samples:]
        
        predictions = []
        for weights in recent_samples:
            pred = self.forward_pass(X, weights)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Confidence intervals
        lower_ci = np.percentile(predictions, 2.5, axis=0)
        upper_ci = np.percentile(predictions, 97.5, axis=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'samples': predictions
        }
    
    def forward_pass(self, X, weights):
        """Forward pass with given weights"""
        # Reshape weights into layers
        weight_idx = 0
        layer_weights = []
        
        for i in range(len(self.layer_sizes) - 1):
            layer_size = self.layer_sizes[i] * self.layer_sizes[i+1]
            layer_w = weights[weight_idx:weight_idx + layer_size]
            layer_w = layer_w.reshape(self.layer_sizes[i], self.layer_sizes[i+1])
            layer_weights.append(layer_w)
            weight_idx += layer_size
        
        # Forward propagation
        activation = X
        for i, W in enumerate(layer_weights):
            z = activation @ W
            if i < len(layer_weights) - 1:  # Hidden layers
                activation = np.maximum(0, z)  # ReLU
            else:  # Output layer
                activation = 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # Sigmoid
        
        return activation.flatten()
```

## 🚨 Common Pitfalls and Advanced Solutions

### 1. Overfitting Detection and Prevention
```python
class OverfittingDetection:
    """Advanced overfitting detection and prevention techniques"""
    
    def __init__(self):
        self.validation_history = []
        self.training_history = []
    
    def early_stopping_monitor(self, train_loss, val_loss, patience=10, min_delta=1e-6):
        """Advanced early stopping with trend analysis"""
        self.training_history.append(train_loss)
        self.validation_history.append(val_loss)
        
        if len(self.validation_history) < patience + 1:
            return False, None
        
        # Check for overfitting patterns
        recent_val = self.validation_history[-patience:]
        recent_train = self.training_history[-patience:]
        
        # Validation loss increasing while training decreases
        val_trend = np.polyfit(range(len(recent_val)), recent_val, 1)[0]
        train_trend = np.polyfit(range(len(recent_train)), recent_train, 1)[0]
        
        # Gap between train and validation increasing
        gap_history = [v - t for v, t in zip(recent_val, recent_train)]
        gap_trend = np.polyfit(range(len(gap_history)), gap_history, 1)[0]
        
        should_stop = (val_trend > min_delta and 
                      train_trend < -min_delta and 
                      gap_trend > min_delta)
        
        diagnostics = {
            'val_trend': val_trend,
            'train_trend': train_trend,
            'gap_trend': gap_trend,
            'current_gap': recent_val[-1] - recent_train[-1]
        }
        
        return should_stop, diagnostics
    
    def learning_curve_analysis(self, train_sizes, train_scores, val_scores):
        """Analyze learning curves to diagnose overfitting"""
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Calculate bias-variance decomposition
        final_gap = val_mean[-1] - train_mean[-1]
        final_val_score = val_mean[-1]
        
        # Diagnose issues
        diagnosis = []
        
        if final_gap > 0.1:  # Significant gap
            diagnosis.append("High variance (overfitting)")
        
        if final_val_score < 0.7:  # Poor performance
            diagnosis.append("High bias (underfitting)")
        
        if train_std[-1] > 0.05:  # High variance in training
            diagnosis.append("Unstable training")
        
        return {
            'train_scores': {'mean': train_mean, 'std': train_std},
            'val_scores': {'mean': val_mean, 'std': val_std},
            'bias_variance_gap': final_gap,
            'diagnosis': diagnosis
        }
    
    def regularization_path_analysis(self, alphas, train_errors, val_errors):
        """Find optimal regularization using validation curve"""
        # Find minimum validation error
        optimal_idx = np.argmin(val_errors)
        optimal_alpha = alphas[optimal_idx]
        
        # One standard error rule
        min_val_error = val_errors[optimal_idx]
        val_std = np.std(val_errors)
        threshold = min_val_error + val_std
        
        # Find simplest model within one standard error
        simple_idx = np.where(val_errors <= threshold)[0][0]
        simple_alpha = alphas[simple_idx]
        
        return {
            'optimal_alpha': optimal_alpha,
            'optimal_error': min_val_error,
            'simple_alpha': simple_alpha,
            'simple_error': val_errors[simple_idx],
            'regularization_effect': train_errors[-1] - train_errors[0]
        }
```

## 📚 Prerequisites Summary

This advanced mathematics foundation prepares you for:
- Implementing sophisticated ML models for IPO prediction
- Understanding theoretical guarantees and limitations
- Debugging mathematical issues in production systems
- Communicating technical concepts to stakeholders
- Advancing to cutting-edge research applications

---

**Next**: [Practical Code Examples](../examples/01-introduction.md)