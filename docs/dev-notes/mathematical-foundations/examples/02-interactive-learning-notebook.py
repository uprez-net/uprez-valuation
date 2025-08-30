#!/usr/bin/env python3
"""
Interactive Learning Notebook - Mathematical Foundations
=======================================================

This interactive notebook provides hands-on examples for each mathematical concept
covered in the IPO valuation ML/NLP documentation. Run each section individually
to build understanding progressively.

Usage:
    python 02-interactive-learning-notebook.py --section [stats|algebra|calculus|finance|ml]
    
Or run all sections:
    python 02-interactive-learning-notebook.py --section all

Author: Uprez Team
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize, linalg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import argparse
import sys

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class InteractiveLearningModule:
    """
    Interactive learning module for mathematical foundations
    """
    
    def __init__(self):
        self.data = None
        self.results = {}
        print("🎓 Interactive Mathematical Foundations Learning Module")
        print("=" * 60)
    
    def generate_sample_data(self):
        """Generate sample IPO data for learning exercises"""
        print("📊 Generating sample data for learning exercises...")
        
        np.random.seed(123)  # For consistency in learning
        n = 500
        
        # Create correlated financial features
        # Revenue (log-normal)
        revenue = np.random.lognormal(mean=14, sigma=1.2, size=n)
        
        # Growth rate (correlated with revenue size, but with noise)
        revenue_growth = 0.3 - 0.1 * np.log(revenue/1e6) + np.random.normal(0, 0.1, n)
        revenue_growth = np.clip(revenue_growth, -0.5, 1.0)
        
        # Profit margin (beta distribution)
        profit_margin = np.random.beta(2, 5, n) * 0.4
        
        # Debt ratio (with some companies having high leverage)
        debt_ratio = np.random.exponential(0.3, n)
        debt_ratio = np.clip(debt_ratio, 0, 2.0)
        
        # Market conditions
        market_sentiment = np.random.normal(0.1, 0.15, n)
        volatility = np.random.gamma(2, 0.05, n)
        
        # Text-derived features
        management_quality = np.random.beta(3, 2, n)  # Generally positive
        media_buzz = np.random.poisson(8, n)
        
        # Create IPO success based on multiple factors
        success_logit = (
            2.0 +  # Base probability
            0.5 * np.log(revenue/1e6) +  # Larger companies more likely to succeed
            1.5 * revenue_growth +        # Growth is key
            2.0 * profit_margin * 10 +    # Profitability matters
            -0.8 * debt_ratio +           # Debt hurts
            1.0 * market_sentiment * 10 + # Market conditions
            -1.5 * volatility * 20 +      # Volatility hurts
            1.0 * management_quality +     # Management quality
            0.3 * np.log(media_buzz + 1) + # Media attention helps
            np.random.normal(0, 0.8, n)   # Random noise
        )
        
        success_prob = 1 / (1 + np.exp(-success_logit))
        ipo_success = np.random.binomial(1, success_prob, n)
        
        # First-day returns
        first_day_return = np.where(
            ipo_success == 1,
            np.random.normal(0.2, 0.3, n),  # Successful IPOs: 20% ± 30%
            np.random.normal(-0.1, 0.2, n)  # Failed IPOs: -10% ± 20%
        )
        
        self.data = pd.DataFrame({
            'revenue': revenue,
            'revenue_growth': revenue_growth,
            'profit_margin': profit_margin,
            'debt_ratio': debt_ratio,
            'market_sentiment': market_sentiment,
            'volatility': volatility,
            'management_quality': management_quality,
            'media_buzz': media_buzz,
            'success_prob': success_prob,
            'ipo_success': ipo_success,
            'first_day_return': first_day_return
        })
        
        print(f"✅ Generated {n} companies with {len(self.data.columns)} features")
        print(f"   Success rate: {np.mean(ipo_success):.1%}")
        return self.data
    
    def statistics_module(self):
        """
        Interactive Statistics and Probability Module
        """
        print("\n" + "="*60)
        print("📊 STATISTICS AND PROBABILITY MODULE")
        print("="*60)
        
        if self.data is None:
            self.generate_sample_data()
        
        print("\n🎯 Learning Objective: Master statistical analysis for financial data")
        
        # 1. Descriptive Statistics
        print("\n1️⃣ DESCRIPTIVE STATISTICS")
        print("-" * 40)
        
        revenue = self.data['revenue']
        returns = self.data['first_day_return']
        
        def describe_distribution(data, name):
            """Comprehensive distribution description"""
            print(f"\n{name} Distribution Analysis:")
            print(f"  Mean (μ):      {np.mean(data):10.3f}")
            print(f"  Median:        {np.median(data):10.3f}")
            print(f"  Std Dev (σ):   {np.std(data):10.3f}")
            print(f"  Skewness:      {stats.skew(data):10.3f}")
            print(f"  Kurtosis:      {stats.kurtosis(data):10.3f}")
            print(f"  Min:           {np.min(data):10.3f}")
            print(f"  Max:           {np.max(data):10.3f}")
            
            # Interpretation
            if abs(stats.skew(data)) > 1:
                skew_desc = "highly skewed"
            elif abs(stats.skew(data)) > 0.5:
                skew_desc = "moderately skewed"
            else:
                skew_desc = "approximately symmetric"
            
            print(f"  Interpretation: {skew_desc}")
            
            return {
                'mean': np.mean(data),
                'std': np.std(data),
                'skew': stats.skew(data),
                'kurtosis': stats.kurtosis(data)
            }
        
        # Analyze key distributions
        revenue_stats = describe_distribution(revenue, "Revenue")
        returns_stats = describe_distribution(returns, "First-Day Returns")
        
        # 2. Hypothesis Testing
        print("\n2️⃣ HYPOTHESIS TESTING")
        print("-" * 40)
        
        # Test 1: Do successful IPOs have higher revenue?
        successful_revenue = self.data[self.data['ipo_success'] == 1]['revenue']
        failed_revenue = self.data[self.data['ipo_success'] == 0]['revenue']
        
        # Use log-transform for revenue (more normal)
        log_successful = np.log(successful_revenue)
        log_failed = np.log(failed_revenue)
        
        t_stat, p_value = stats.ttest_ind(log_successful, log_failed)
        
        print(f"Hypothesis Test: Successful vs Failed IPO Revenue")
        print(f"  H₀: μ_successful = μ_failed")
        print(f"  H₁: μ_successful ≠ μ_failed")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value:     {p_value:.6f}")
        
        if p_value < 0.05:
            print(f"  Result: Reject H₀ (p < 0.05) - Significant difference!")
        else:
            print(f"  Result: Fail to reject H₀ (p ≥ 0.05)")
        
        effect_size = (np.mean(log_successful) - np.mean(log_failed)) / \
                     np.sqrt(((len(log_successful)-1)*np.var(log_successful) + 
                             (len(log_failed)-1)*np.var(log_failed)) / 
                            (len(log_successful)+len(log_failed)-2))
        
        print(f"  Cohen's d:   {effect_size:.4f} (effect size)")
        
        # 3. Distribution Fitting
        print("\n3️⃣ DISTRIBUTION FITTING")
        print("-" * 40)
        
        # Test if revenue follows log-normal distribution
        log_revenue = np.log(revenue)
        
        # Normality tests on log-revenue
        shapiro_stat, shapiro_p = stats.shapiro(log_revenue[:1000])  # Limit sample size
        jb_stat, jb_p = stats.jarque_bera(log_revenue)
        
        print(f"Testing if log(revenue) is normally distributed:")
        print(f"  Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_p:.6f}")
        print(f"  Jarque-Bera:  JB={jb_stat:.4f}, p={jb_p:.6f}")
        
        if shapiro_p > 0.05 and jb_p > 0.05:
            print(f"  Result: Revenue appears to be log-normally distributed!")
        else:
            print(f"  Result: Revenue may not be perfectly log-normal")
        
        # 4. Correlation Analysis
        print("\n4️⃣ CORRELATION ANALYSIS")
        print("-" * 40)
        
        # Calculate correlation matrix
        numeric_cols = ['revenue_growth', 'profit_margin', 'debt_ratio', 
                       'management_quality', 'ipo_success']
        corr_matrix = self.data[numeric_cols].corr()
        
        print("Correlation with IPO Success:")
        success_corrs = corr_matrix['ipo_success'].drop('ipo_success').sort_values(key=abs, ascending=False)
        
        for var, corr in success_corrs.items():
            stars = "***" if abs(corr) > 0.3 else "**" if abs(corr) > 0.2 else "*" if abs(corr) > 0.1 else ""
            print(f"  {var:20s}: {corr:7.4f} {stars}")
        
        print("\n  Legend: *** strong, ** moderate, * weak correlation")
        
        # Statistical significance of correlations
        def correlation_significance(x, y):
            """Test significance of correlation"""
            r, p = stats.pearsonr(x, y)
            n = len(x)
            t_stat = r * np.sqrt((n-2)/(1-r**2))
            return r, p, t_stat
        
        # Test most important correlation
        best_predictor = success_corrs.abs().idxmax()
        r, p, t_stat = correlation_significance(
            self.data[best_predictor], self.data['ipo_success']
        )
        
        print(f"\nSignificance test for strongest predictor ({best_predictor}):")
        print(f"  Correlation: {r:.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value:     {p:.6f}")
        
        # 5. Interactive Exercise
        print("\n5️⃣ INTERACTIVE EXERCISE")
        print("-" * 40)
        
        print("Exercise: Analyze the relationship between profit margin and success")
        print("Try to predict: What should be the correlation?")
        
        actual_corr = stats.pearsonr(self.data['profit_margin'], self.data['ipo_success'])[0]
        print(f"\nActual correlation: {actual_corr:.4f}")
        
        # Create bins for analysis
        self.data['margin_quartile'] = pd.qcut(self.data['profit_margin'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        success_by_quartile = self.data.groupby('margin_quartile')['ipo_success'].agg(['mean', 'count'])
        
        print("\nSuccess rate by profit margin quartile:")
        for quartile, row in success_by_quartile.iterrows():
            print(f"  {quartile}: {row['mean']:.1%} (n={row['count']})")
        
        # Visualization
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(revenue, bins=30, alpha=0.7, color='blue')
        plt.xlabel('Revenue')
        plt.ylabel('Frequency')
        plt.title('Revenue Distribution')
        plt.yscale('log')
        
        plt.subplot(2, 2, 2)
        plt.hist(returns, bins=30, alpha=0.7, color='green')
        plt.xlabel('First-Day Return')
        plt.ylabel('Frequency')
        plt.title('First-Day Returns Distribution')
        plt.axvline(0, color='red', linestyle='--', alpha=0.7)
        
        plt.subplot(2, 2, 3)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, cbar_kws={'label': 'Correlation'})
        plt.title('Correlation Matrix')
        
        plt.subplot(2, 2, 4)
        success_by_quartile['mean'].plot(kind='bar')
        plt.title('Success Rate by Profit Margin Quartile')
        plt.ylabel('Success Rate')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('statistics_module_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Store results
        self.results['statistics'] = {
            'revenue_stats': revenue_stats,
            'returns_stats': returns_stats,
            'hypothesis_test': {'t_stat': t_stat, 'p_value': p_value, 'effect_size': effect_size},
            'correlations': success_corrs.to_dict(),
            'normality_tests': {'shapiro_p': shapiro_p, 'jb_p': jb_p}
        }
        
        print("\n✅ Statistics module completed!")
        print("📚 Key takeaways:")
        print("  • Financial data often follows non-normal distributions")
        print("  • Hypothesis testing helps validate business assumptions")
        print("  • Correlation analysis reveals important relationships")
        print("  • Always check statistical significance and effect sizes")
        
        return self.results['statistics']
    
    def linear_algebra_module(self):
        """
        Interactive Linear Algebra Module
        """
        print("\n" + "="*60)
        print("🧮 LINEAR ALGEBRA MODULE")
        print("="*60)
        
        if self.data is None:
            self.generate_sample_data()
        
        print("\n🎯 Learning Objective: Apply matrix operations for ML in finance")
        
        # Prepare feature matrix
        feature_cols = ['revenue_growth', 'profit_margin', 'debt_ratio', 
                       'market_sentiment', 'volatility', 'management_quality']
        X = self.data[feature_cols].values
        
        print(f"\nWorking with feature matrix: {X.shape}")
        print(f"Features: {feature_cols}")
        
        # 1. Matrix Operations
        print("\n1️⃣ FUNDAMENTAL MATRIX OPERATIONS")
        print("-" * 40)
        
        # Standardize the matrix
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        
        print("Matrix standardization:")
        print(f"  Original mean: {np.mean(X, axis=0)}")
        print(f"  Standardized mean: {np.mean(X_std, axis=0)}")
        print(f"  Standardized std: {np.std(X_std, axis=0)}")
        
        # Covariance matrix
        cov_matrix = np.cov(X_std.T)
        print(f"\nCovariance matrix shape: {cov_matrix.shape}")
        print(f"Diagonal elements (variances): {np.diag(cov_matrix)}")
        
        # Matrix properties
        print(f"\nMatrix properties:")
        print(f"  Determinant: {np.linalg.det(cov_matrix):.6f}")
        print(f"  Trace: {np.trace(cov_matrix):.6f}")
        print(f"  Condition number: {np.linalg.cond(cov_matrix):.2e}")
        
        # 2. Eigendecomposition
        print("\n2️⃣ EIGENDECOMPOSITION AND PCA")
        print("-" * 40)
        
        # Manual eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Explained variance
        explained_var_ratio = eigenvals / np.sum(eigenvals)
        cumulative_var = np.cumsum(explained_var_ratio)
        
        print("Principal Components Analysis:")
        for i in range(len(eigenvals)):
            print(f"  PC{i+1}: λ={eigenvals[i]:.4f}, "
                  f"explained={explained_var_ratio[i]:.3f}, "
                  f"cumulative={cumulative_var[i]:.3f}")
        
        # Find components for 90% variance
        n_components_90 = np.argmax(cumulative_var >= 0.9) + 1
        print(f"\nComponents needed for 90% variance: {n_components_90}")
        
        # Transform data
        X_pca = X_std @ eigenvecs[:, :n_components_90]
        print(f"Reduced dimensionality: {X.shape[1]} → {X_pca.shape[1]}")
        
        # 3. SVD Analysis
        print("\n3️⃣ SINGULAR VALUE DECOMPOSITION")
        print("-" * 40)
        
        # Perform SVD
        U, s, Vt = np.linalg.svd(X_std, full_matrices=False)
        
        print(f"SVD decomposition:")
        print(f"  U matrix: {U.shape}")
        print(f"  Singular values: {s.shape}")
        print(f"  V^T matrix: {Vt.shape}")
        
        # Singular values are related to eigenvalues
        print(f"\nSingular values: {s}")
        print(f"s² / (n-1): {s**2 / (len(X_std)-1)}")
        print(f"Eigenvalues: {eigenvals}")
        
        # Reconstruction with different ranks
        print(f"\nReconstruction analysis:")
        for k in [1, 2, 3, len(s)]:
            X_reconstructed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
            reconstruction_error = np.mean((X_std - X_reconstructed)**2)
            print(f"  Rank {k}: MSE = {reconstruction_error:.6f}")
        
        # 4. Matrix Norms and Distances
        print("\n4️⃣ MATRIX NORMS AND DISTANCES")
        print("-" * 40)
        
        # Different matrix norms
        frobenius_norm = np.linalg.norm(X_std, 'fro')
        nuclear_norm = np.sum(s)  # Sum of singular values
        spectral_norm = np.max(s)  # Largest singular value
        
        print(f"Matrix norms:")
        print(f"  Frobenius norm: {frobenius_norm:.4f}")
        print(f"  Nuclear norm: {nuclear_norm:.4f}")
        print(f"  Spectral norm: {spectral_norm:.4f}")
        
        # Pairwise distances between companies
        def compute_distance_matrix(X, metric='euclidean'):
            \"\"\"Compute pairwise distance matrix\"\"\"
            n = X.shape[0]
            distances = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i+1, n):
                    if metric == 'euclidean':
                        dist = np.linalg.norm(X[i] - X[j])
                    elif metric == 'cosine':
                        dist = 1 - np.dot(X[i], X[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(X[j]))
                    distances[i, j] = distances[j, i] = dist
            
            return distances
        
        # Compute distance matrix for first 10 companies (for speed)
        sample_distances = compute_distance_matrix(X_std[:10])
        print(f"\nDistance matrix (10x10 sample):")
        print(f"  Mean distance: {np.mean(sample_distances[sample_distances > 0]):.4f}")
        print(f"  Min distance: {np.min(sample_distances[sample_distances > 0]):.4f}")
        print(f"  Max distance: {np.max(sample_distances):.4f}")
        
        # 5. Interactive Exercise
        print("\n5️⃣ INTERACTIVE EXERCISE")
        print("-" * 40)
        
        print("Exercise: Compare PCA with manual eigendecomposition")
        
        # Using sklearn PCA
        pca = PCA()
        X_pca_sklearn = pca.fit_transform(X_std)
        
        print(f"\nComparison:")
        print(f"  Manual explained variance: {explained_var_ratio[:3]}")
        print(f"  Sklearn explained variance: {pca.explained_variance_ratio_[:3]}")
        print(f"  Difference: {np.abs(explained_var_ratio[:3] - pca.explained_variance_ratio_[:3])}")
        
        # Test orthogonality of eigenvectors
        orthogonality_test = eigenvecs.T @ eigenvecs
        print(f"\nOrthogonality test (should be identity matrix):")
        print(f"  Max off-diagonal: {np.max(np.abs(orthogonality_test - np.eye(len(orthogonality_test)))):.8f}")
        
        # Visualization
        plt.figure(figsize=(15, 10))
        
        # Eigenvalues
        plt.subplot(2, 3, 1)
        plt.bar(range(1, len(eigenvals)+1), eigenvals)
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalues (Scree Plot)')
        
        # Explained variance
        plt.subplot(2, 3, 2)
        plt.plot(range(1, len(cumulative_var)+1), cumulative_var, 'bo-')
        plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.7)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance')
        
        # PC1 vs PC2 scatter
        plt.subplot(2, 3, 3)
        colors = ['red' if success else 'blue' for success in self.data['ipo_success']]
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('PCA Projection')
        
        # Covariance heatmap
        plt.subplot(2, 3, 4)
        sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', center=0, 
                   xticklabels=feature_cols, yticklabels=feature_cols)
        plt.title('Covariance Matrix')
        
        # Singular values
        plt.subplot(2, 3, 5)
        plt.semilogy(range(1, len(s)+1), s, 'ro-')
        plt.xlabel('Singular Value Index')
        plt.ylabel('Singular Value (log scale)')
        plt.title('Singular Values')
        
        # Reconstruction error
        plt.subplot(2, 3, 6)
        reconstruction_errors = []
        for k in range(1, len(s)+1):
            X_rec = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
            error = np.mean((X_std - X_rec)**2)
            reconstruction_errors.append(error)
        
        plt.semilogy(range(1, len(s)+1), reconstruction_errors, 'go-')
        plt.xlabel('Number of Components')
        plt.ylabel('Reconstruction Error (log scale)')
        plt.title('Reconstruction Error')
        
        plt.tight_layout()
        plt.savefig('linear_algebra_module_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Store results
        self.results['linear_algebra'] = {
            'eigenvalues': eigenvals,
            'explained_variance_ratio': explained_var_ratio,
            'singular_values': s,
            'n_components_90': n_components_90,
            'reconstruction_errors': reconstruction_errors,
            'condition_number': np.linalg.cond(cov_matrix)
        }
        
        print("\n✅ Linear Algebra module completed!")
        print("📚 Key takeaways:")
        print("  • PCA reduces dimensionality while preserving variance")
        print("  • Eigendecomposition reveals data structure")
        print("  • SVD provides robust matrix approximation")
        print("  • Matrix condition number indicates numerical stability")
        
        return self.results['linear_algebra']
    
    def calculus_module(self):
        """
        Interactive Calculus and Optimization Module
        """
        print("\n" + "="*60)
        print("📈 CALCULUS AND OPTIMIZATION MODULE")
        print("="*60)
        
        if self.data is None:
            self.generate_sample_data()
        
        print("\n🎯 Learning Objective: Master optimization for ML algorithms")
        
        # Prepare data
        X = self.data[['revenue_growth', 'profit_margin', 'management_quality']].values
        y = self.data['ipo_success'].values
        
        # Standardize
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        X_std = np.column_stack([np.ones(len(X_std)), X_std])  # Add intercept
        
        print(f"Working with {X_std.shape[0]} samples, {X_std.shape[1]-1} features + intercept")
        
        # 1. Gradient Descent Implementation
        print("\n1️⃣ GRADIENT DESCENT FOR LOGISTIC REGRESSION")
        print("-" * 40)
        
        def sigmoid(z):
            \"\"\"Numerically stable sigmoid function\"\"\"
            return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
        
        def cost_function(theta, X, y):
            \"\"\"Logistic regression cost function\"\"\"
            z = X @ theta
            h = sigmoid(z)
            # Add small epsilon to prevent log(0)
            epsilon = 1e-15
            h = np.clip(h, epsilon, 1 - epsilon)
            cost = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
            return cost
        
        def gradient(theta, X, y):
            \"\"\"Gradient of cost function\"\"\"
            z = X @ theta
            h = sigmoid(z)
            grad = X.T @ (h - y) / len(y)
            return grad
        
        def hessian(theta, X, y):
            \"\"\"Hessian matrix for Newton's method\"\"\"
            z = X @ theta
            h = sigmoid(z)
            W = np.diag(h * (1 - h))
            H = X.T @ W @ X / len(y)
            return H
        
        # Initialize parameters
        theta = np.random.normal(0, 0.01, X_std.shape[1])
        
        # Gradient descent parameters
        learning_rates = [0.01, 0.1, 1.0]
        max_iterations = 1000
        tolerance = 1e-6
        
        print("Comparing different learning rates:")
        
        gd_results = {}
        
        for lr in learning_rates:
            theta_gd = theta.copy()
            costs = []
            
            print(f"\nLearning rate: {lr}")
            
            for i in range(max_iterations):
                cost = cost_function(theta_gd, X_std, y)
                grad = gradient(theta_gd, X_std, y)
                
                # Update parameters
                theta_new = theta_gd - lr * grad
                
                # Check convergence
                if np.linalg.norm(theta_new - theta_gd) < tolerance:
                    print(f"  Converged at iteration {i}")
                    break
                
                theta_gd = theta_new
                costs.append(cost)
                
                if i % 100 == 0:
                    print(f"  Iteration {i}: Cost = {cost:.6f}")
            
            gd_results[lr] = {
                'theta': theta_gd,
                'costs': costs,
                'converged': i < max_iterations - 1
            }
        
        # 2. Newton's Method
        print("\n2️⃣ NEWTON'S METHOD")
        print("-" * 40)
        
        theta_newton = theta.copy()
        newton_costs = []
        
        print("Newton's method optimization:")
        
        for i in range(20):  # Newton's method usually converges faster
            cost = cost_function(theta_newton, X_std, y)
            grad = gradient(theta_newton, X_std, y)
            H = hessian(theta_newton, X_std, y)
            
            try:
                # Newton update: θ_new = θ_old - H^(-1) * g
                theta_newton = theta_newton - np.linalg.solve(H, grad)
                newton_costs.append(cost)
                
                print(f"  Iteration {i+1}: Cost = {cost:.6f}")
                
                if np.linalg.norm(grad) < tolerance:
                    print(f"  Converged at iteration {i+1}")
                    break
                    
            except np.linalg.LinAlgError:
                print(f"  Singular Hessian at iteration {i+1}")
                break
        
        # 3. Momentum and Adaptive Methods
        print("\n3️⃣ ADVANCED OPTIMIZATION METHODS")
        print("-" * 40)
        
        def momentum_gradient_descent(theta_init, X, y, lr=0.01, momentum=0.9, max_iter=1000):
            \"\"\"Gradient descent with momentum\"\"\"
            theta = theta_init.copy()
            velocity = np.zeros_like(theta)
            costs = []
            
            for i in range(max_iter):
                cost = cost_function(theta, X, y)
                grad = gradient(theta, X, y)
                
                # Update velocity and parameters
                velocity = momentum * velocity - lr * grad
                theta = theta + velocity
                
                costs.append(cost)
                
                if np.linalg.norm(grad) < tolerance:
                    break
            
            return theta, costs
        
        def adam_optimizer(theta_init, X, y, lr=0.001, beta1=0.9, beta2=0.999, max_iter=1000):
            \"\"\"Adam optimizer\"\"\"
            theta = theta_init.copy()
            m = np.zeros_like(theta)  # First moment
            v = np.zeros_like(theta)  # Second moment
            costs = []
            epsilon = 1e-8
            
            for t in range(1, max_iter + 1):
                cost = cost_function(theta, X, y)
                grad = gradient(theta, X, y)
                
                # Update moments
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                
                # Bias correction
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                
                # Update parameters
                theta = theta - lr * m_hat / (np.sqrt(v_hat) + epsilon)
                
                costs.append(cost)
                
                if np.linalg.norm(grad) < tolerance:
                    break
            
            return theta, costs
        
        # Run advanced optimizers
        theta_momentum, momentum_costs = momentum_gradient_descent(theta, X_std, y)
        theta_adam, adam_costs = adam_optimizer(theta, X_std, y)
        
        print(f"Momentum GD final cost: {momentum_costs[-1]:.6f}")
        print(f"Adam final cost: {adam_costs[-1]:.6f}")
        
        # 4. Constrained Optimization
        print("\n4️⃣ CONSTRAINED OPTIMIZATION")
        print("-" * 40)
        
        print("Portfolio optimization example:")
        
        # Simulate expected returns and covariance matrix
        np.random.seed(42)
        n_assets = 4
        expected_returns = np.random.uniform(0.05, 0.15, n_assets)
        
        # Generate positive definite covariance matrix
        A = np.random.randn(n_assets, n_assets)
        cov_matrix = A @ A.T / n_assets  # Make it positive definite
        
        print(f"Expected returns: {expected_returns}")
        print(f"Risk matrix diagonal: {np.diag(cov_matrix)}")
        
        def portfolio_objective(weights):
            \"\"\"Negative Sharpe ratio (for minimization)\"\"\"
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
            risk_free_rate = 0.02
            
            if portfolio_risk == 0:
                return -np.inf
            
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
            return -sharpe_ratio  # Negative for minimization
        
        def weight_constraint(weights):
            \"\"\"Weights must sum to 1\"\"\"
            return np.sum(weights) - 1
        
        # Constraints and bounds
        constraints = {'type': 'eq', 'fun': weight_constraint}
        bounds = [(0, 1) for _ in range(n_assets)]  # Long-only
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize using scipy
        from scipy.optimize import minimize
        
        result = optimize.minimize(portfolio_objective, x0, method='SLSQP',
                                 bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            optimal_return = np.dot(optimal_weights, expected_returns)
            optimal_risk = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
            optimal_sharpe = (optimal_return - 0.02) / optimal_risk
            
            print(f"Optimal portfolio:")
            for i, weight in enumerate(optimal_weights):
                print(f"  Asset {i+1}: {weight:.3f}")
            print(f"Expected return: {optimal_return:.4f}")
            print(f"Risk: {optimal_risk:.4f}")
            print(f"Sharpe ratio: {optimal_sharpe:.4f}")
        else:
            print(f"Optimization failed: {result.message}")
        
        # 5. Interactive Exercise
        print("\n5️⃣ INTERACTIVE EXERCISE")
        print("-" * 40)
        
        print("Exercise: Compare convergence rates")
        
        # Compare final parameters
        print(f"\\nFinal parameters comparison:")
        print(f"Gradient Descent (lr=0.1): {gd_results[0.1]['theta']}")
        print(f"Newton's Method:            {theta_newton}")
        print(f"Momentum GD:                {theta_momentum}")
        print(f"Adam:                       {theta_adam}")
        
        # Use sklearn for comparison
        lr_sklearn = LogisticRegression(fit_intercept=False, max_iter=1000)
        lr_sklearn.fit(X_std, y)
        sklearn_coef = np.concatenate([lr_sklearn.intercept_, lr_sklearn.coef_[0]])
        print(f"Sklearn LogisticRegression: {sklearn_coef}")
        
        # Visualization
        plt.figure(figsize=(15, 10))
        
        # Convergence comparison
        plt.subplot(2, 3, 1)
        for lr in learning_rates:
            if gd_results[lr]['converged']:
                plt.plot(gd_results[lr]['costs'], label=f'GD lr={lr}')
        plt.plot(newton_costs, label='Newton\\'s Method', linewidth=2)
        plt.plot(momentum_costs[:len(newton_costs)], label='Momentum')
        plt.plot(adam_costs[:len(newton_costs)], label='Adam')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Optimization Convergence')
        plt.legend()
        plt.yscale('log')
        
        # Cost landscape (2D slice)
        plt.subplot(2, 3, 2)
        theta1_range = np.linspace(-2, 2, 50)
        theta2_range = np.linspace(-2, 2, 50)
        T1, T2 = np.meshgrid(theta1_range, theta2_range)
        
        costs_landscape = np.zeros_like(T1)
        for i in range(len(theta1_range)):
            for j in range(len(theta2_range)):
                theta_temp = np.array([T1[i,j], T2[i,j], 0, 0])  # Fix other parameters
                costs_landscape[i,j] = cost_function(theta_temp, X_std, y)
        
        contour = plt.contour(T1, T2, costs_landscape, levels=20)
        plt.colorbar(contour)
        plt.xlabel('θ₁')
        plt.ylabel('θ₂')
        plt.title('Cost Function Landscape')
        
        # Gradient norms during optimization
        plt.subplot(2, 3, 3)
        # Calculate gradient norms for GD
        grad_norms = []
        theta_temp = theta.copy()
        for cost in gd_results[0.1]['costs'][:50]:  # First 50 iterations
            grad = gradient(theta_temp, X_std, y)
            grad_norms.append(np.linalg.norm(grad))
            theta_temp = theta_temp - 0.1 * grad
        
        plt.plot(grad_norms, label='Gradient Norm')
        plt.xlabel('Iteration')
        plt.ylabel('||∇θ||')
        plt.title('Gradient Norm During Optimization')
        plt.yscale('log')
        
        # Portfolio weights
        plt.subplot(2, 3, 4)
        if 'optimal_weights' in locals():
            plt.bar(range(1, n_assets+1), optimal_weights)
            plt.xlabel('Asset')
            plt.ylabel('Weight')
            plt.title('Optimal Portfolio Weights')
        
        # Learning rate sensitivity
        plt.subplot(2, 3, 5)
        final_costs = []
        learning_rates_test = np.logspace(-3, 0, 20)
        
        for lr in learning_rates_test:
            theta_temp = theta.copy()
            for _ in range(100):  # Fixed number of iterations
                grad = gradient(theta_temp, X_std, y)
                theta_temp = theta_temp - lr * grad
            final_cost = cost_function(theta_temp, X_std, y)
            final_costs.append(final_cost)
        
        plt.semilogx(learning_rates_test, final_costs, 'bo-')
        plt.xlabel('Learning Rate')
        plt.ylabel('Final Cost')
        plt.title('Learning Rate Sensitivity')
        
        # Parameter trajectory (2D projection)
        plt.subplot(2, 3, 6)
        # Show trajectory for first two parameters
        if len(gd_results[0.1]['costs']) > 1:
            theta_trajectory = [theta]
            theta_temp = theta.copy()
            for _ in range(min(50, len(gd_results[0.1]['costs']))):
                grad = gradient(theta_temp, X_std, y)
                theta_temp = theta_temp - 0.1 * grad
                theta_trajectory.append(theta_temp.copy())
            
            theta_trajectory = np.array(theta_trajectory)
            plt.plot(theta_trajectory[:, 0], theta_trajectory[:, 1], 'ro-', alpha=0.7)
            plt.xlabel('θ₀ (intercept)')
            plt.ylabel('θ₁')
            plt.title('Parameter Trajectory')
        
        plt.tight_layout()
        plt.savefig('calculus_module_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Store results
        self.results['calculus'] = {
            'gradient_descent_results': gd_results,
            'newton_theta': theta_newton,
            'momentum_theta': theta_momentum,
            'adam_theta': theta_adam,
            'portfolio_weights': optimal_weights if 'optimal_weights' in locals() else None
        }
        
        print("\n✅ Calculus module completed!")
        print("📚 Key takeaways:")
        print("  • Gradient descent requires careful learning rate tuning")
        print("  • Newton's method converges faster but requires Hessian")
        print("  • Momentum and Adam help with optimization challenges")
        print("  • Constrained optimization solves real portfolio problems")
        
        return self.results['calculus']
    
    def run_all_modules(self):
        \"\"\"Run all learning modules sequentially\"\"\"
        print("🚀 Running all mathematical foundation modules...")
        
        try:
            self.generate_sample_data()
            self.statistics_module()
            self.linear_algebra_module()
            self.calculus_module()
            
            print("\n" + "="*60)
            print("🎉 ALL MODULES COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            print("\\n📊 Summary of Results:")
            print("-" * 30)
            
            if 'statistics' in self.results:
                print(f"Statistics: {len(self.results['statistics'])} analyses completed")
            
            if 'linear_algebra' in self.results:
                print(f"Linear Algebra: {self.results['linear_algebra']['n_components_90']} components for 90% variance")
            
            if 'calculus' in self.results:
                converged = sum(1 for lr_result in self.results['calculus']['gradient_descent_results'].values() 
                               if lr_result['converged'])
                print(f"Calculus: {converged}/{len(self.results['calculus']['gradient_descent_results'])} optimizers converged")
            
            return self.results
            
        except Exception as e:
            print(f"❌ Error during module execution: {str(e)}")
            raise


def main():
    \"\"\"Main function with command-line interface\"\"\"
    parser = argparse.ArgumentParser(description='Interactive Mathematical Foundations Learning')
    parser.add_argument('--section', choices=['stats', 'algebra', 'calculus', 'finance', 'ml', 'all'],
                       default='all', help='Which section to run')
    
    args = parser.parse_args()
    
    # Create learning module
    learner = InteractiveLearningModule()
    
    try:
        if args.section == 'all':
            results = learner.run_all_modules()
        elif args.section == 'stats':
            learner.generate_sample_data()
            results = learner.statistics_module()
        elif args.section == 'algebra':
            learner.generate_sample_data()
            results = learner.linear_algebra_module()
        elif args.section == 'calculus':
            learner.generate_sample_data()
            results = learner.calculus_module()
        else:
            print(f"Section {args.section} not yet implemented")
            return
        
        print(f"\n✅ Learning session completed!")
        print(f"📁 Results saved in learner.results")
        
        return learner, results
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Learning session interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    learner, results = main()