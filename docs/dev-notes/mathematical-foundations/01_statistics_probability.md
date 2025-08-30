# Statistics and Probability for Financial Valuation

This section covers essential statistical and probability concepts for implementing ML/NLP models in financial valuation.

## 📊 Table of Contents

1. [Descriptive Statistics for Financial Data](#descriptive-statistics)
2. [Probability Distributions in Finance](#probability-distributions)  
3. [Hypothesis Testing and Confidence Intervals](#hypothesis-testing)
4. [Bayesian Inference Applications](#bayesian-inference)
5. [Correlation vs Causation](#correlation-causation)

---

## 🟢 Descriptive Statistics for Financial Data {#descriptive-statistics}

### Intuitive Understanding

Financial data has unique characteristics that require special statistical treatment:
- **Skewness**: Returns are often not normally distributed
- **Fat tails**: Extreme events occur more frequently than normal distribution predicts
- **Time dependency**: Financial data exhibits temporal patterns
- **Volatility clustering**: Periods of high volatility followed by high volatility

### 🟡 Practical Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import skew, kurtosis, jarque_bera
import seaborn as sns

class FinancialDescriptiveStats:
    """
    Comprehensive descriptive statistics for financial data
    """
    
    def __init__(self, data: np.ndarray, data_type: str = "returns"):
        """
        Args:
            data: Financial time series data
            data_type: "returns", "prices", or "ratios"
        """
        self.data = np.array(data)
        self.data_type = data_type
        self.clean_data = self._clean_data()
        
    def _clean_data(self) -> np.ndarray:
        """Remove outliers and handle missing values"""
        # Remove NaN and infinite values
        clean = self.data[np.isfinite(self.data)]
        
        # Remove extreme outliers (beyond 5 standard deviations)
        if len(clean) > 10:
            z_scores = np.abs(stats.zscore(clean))
            clean = clean[z_scores < 5]
            
        return clean
    
    def basic_statistics(self) -> dict:
        """Calculate basic descriptive statistics"""
        data = self.clean_data
        
        return {
            'count': len(data),
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data, ddof=1),
            'variance': np.var(data, ddof=1),
            'min': np.min(data),
            'max': np.max(data),
            'range': np.max(data) - np.min(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25)
        }
    
    def advanced_statistics(self) -> dict:
        """Calculate advanced statistics relevant to finance"""
        data = self.clean_data
        
        # Skewness (asymmetry measure)
        skewness = skew(data)
        
        # Kurtosis (tail fatness measure)
        kurt = kurtosis(data, fisher=True)  # Excess kurtosis (normal = 0)
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = jarque_bera(data)
        
        # Coefficient of variation
        cv = np.std(data, ddof=1) / np.mean(data) if np.mean(data) != 0 else np.inf
        
        # Value at Risk (VaR) at different confidence levels
        var_95 = np.percentile(data, 5)  # 5% VaR
        var_99 = np.percentile(data, 1)  # 1% VaR
        
        # Expected Shortfall (Conditional VaR)
        es_95 = np.mean(data[data <= var_95]) if np.any(data <= var_95) else var_95
        es_99 = np.mean(data[data <= var_99]) if np.any(data <= var_99) else var_99
        
        return {
            'skewness': skewness,
            'kurtosis': kurt,
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'is_normal': jb_pvalue > 0.05,
            'coefficient_of_variation': cv,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'expected_shortfall_99': es_99,
            'volatility_annual': np.std(data, ddof=1) * np.sqrt(252) if self.data_type == "returns" else None
        }
    
    def time_series_statistics(self, data_with_dates: pd.Series = None) -> dict:
        """Time series specific statistics"""
        if data_with_dates is None:
            data_series = pd.Series(self.clean_data)
        else:
            data_series = data_with_dates.dropna()
        
        # Autocorrelation at different lags
        autocorr_1 = data_series.autocorr(lag=1)
        autocorr_5 = data_series.autocorr(lag=5)
        autocorr_22 = data_series.autocorr(lag=22)  # Monthly for daily data
        
        # Rolling statistics
        rolling_mean = data_series.rolling(window=30).mean()
        rolling_std = data_series.rolling(window=30).std()
        
        return {
            'autocorr_lag1': autocorr_1,
            'autocorr_lag5': autocorr_5,
            'autocorr_lag22': autocorr_22,
            'rolling_mean_stability': rolling_mean.std() / rolling_mean.mean() if rolling_mean.mean() != 0 else np.inf,
            'rolling_std_mean': rolling_std.mean(),
            'volatility_clustering': self._detect_volatility_clustering(data_series)
        }
    
    def _detect_volatility_clustering(self, data_series: pd.Series) -> float:
        """Detect volatility clustering using squared returns autocorrelation"""
        if self.data_type == "returns":
            squared_returns = data_series ** 2
            return squared_returns.autocorr(lag=1)
        return 0.0
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report"""
        basic = self.basic_statistics()
        advanced = self.advanced_statistics()
        
        report = f"""
Financial Data Summary Report
============================

Basic Statistics:
- Sample Size: {basic['count']:,}
- Mean: {basic['mean']:.6f}
- Median: {basic['median']:.6f}
- Standard Deviation: {basic['std']:.6f}
- Range: {basic['range']:.6f}

Distribution Characteristics:
- Skewness: {advanced['skewness']:.4f} {"(Left-skewed)" if advanced['skewness'] < -0.5 else "(Right-skewed)" if advanced['skewness'] > 0.5 else "(Approximately symmetric)"}
- Kurtosis: {advanced['kurtosis']:.4f} {"(Heavy tails)" if advanced['kurtosis'] > 1 else "(Light tails)" if advanced['kurtosis'] < -1 else "(Normal tails)"}
- Normality Test: {"Passed" if advanced['is_normal'] else "Failed"} (p-value: {advanced['jarque_bera_pvalue']:.4f})

Risk Metrics:
- 95% VaR: {advanced['var_95']:.6f}
- 99% VaR: {advanced['var_99']:.6f}
- 95% Expected Shortfall: {advanced['expected_shortfall_95']:.6f}
- Coefficient of Variation: {advanced['coefficient_of_variation']:.4f}
"""
        
        if self.data_type == "returns" and advanced['volatility_annual']:
            report += f"- Annualized Volatility: {advanced['volatility_annual']:.2%}\n"
        
        return report

# Example usage with real financial data
def demonstrate_financial_statistics():
    """Demonstrate statistical analysis of financial data"""
    
    # Simulate stock returns
    np.random.seed(42)
    n_days = 1000
    
    # Generate returns with realistic characteristics
    base_returns = np.random.normal(0.0008, 0.02, n_days)  # ~20% annual vol, positive drift
    
    # Add volatility clustering
    vol_process = np.random.normal(0, 0.01, n_days)
    for i in range(1, n_days):
        vol_process[i] += 0.1 * vol_process[i-1]  # Persistence in volatility
    
    # Combine to create realistic returns
    returns = base_returns + vol_process * np.random.normal(0, 1, n_days)
    
    # Add some fat tails with occasional large moves
    extreme_moves = np.random.choice([0, 1], n_days, p=[0.95, 0.05])
    extreme_returns = np.random.normal(0, 0.08, n_days) * extreme_moves
    returns += extreme_returns
    
    # Create dates
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    returns_series = pd.Series(returns, index=dates)
    
    # Analyze the data
    stats_analyzer = FinancialDescriptiveStats(returns, "returns")
    
    print(stats_analyzer.generate_summary_report())
    
    # Advanced analysis
    advanced_stats = stats_analyzer.advanced_statistics()
    ts_stats = stats_analyzer.time_series_statistics(returns_series)
    
    print("\nTime Series Properties:")
    print(f"- Lag-1 Autocorrelation: {ts_stats['autocorr_lag1']:.4f}")
    print(f"- Volatility Clustering: {ts_stats['volatility_clustering']:.4f}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time series plot
    axes[0, 0].plot(dates, returns)
    axes[0, 0].set_title('Return Time Series')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Return')
    
    # Distribution histogram
    axes[0, 1].hist(returns, bins=50, alpha=0.7, density=True)
    # Overlay normal distribution
    x = np.linspace(returns.min(), returns.max(), 100)
    normal_pdf = stats.norm.pdf(x, np.mean(returns), np.std(returns))
    axes[0, 1].plot(x, normal_pdf, 'r-', label='Normal Distribution')
    axes[0, 1].set_title('Return Distribution vs Normal')
    axes[0, 1].legend()
    
    # Q-Q plot for normality check
    stats.probplot(returns, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot vs Normal Distribution')
    
    # Autocorrelation plot
    lags = range(1, 21)
    autocorrs = [returns_series.autocorr(lag=lag) for lag in lags]
    axes[1, 1].bar(lags, autocorrs)
    axes[1, 1].set_title('Autocorrelation Function')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('Autocorrelation')
    
    plt.tight_layout()
    plt.show()
    
    return stats_analyzer

# Run the demonstration
if __name__ == "__main__":
    analyzer = demonstrate_financial_statistics()
```

### 🔴 Theoretical Foundation

**Central Limit Theorem in Finance:**
For a sequence of returns $R_1, R_2, \ldots, R_n$ that are independent and identically distributed with mean $\mu$ and variance $\sigma^2$:

$$\frac{\bar{R}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0, 1) \text{ as } n \to \infty$$

However, financial returns often violate the i.i.d. assumption due to:
- **Serial correlation** in volatility
- **Time-varying volatility** (heteroskedasticity)
- **Fat-tailed distributions** (non-normal)

**Moments and Their Financial Interpretation:**

1. **First Moment (Mean)**: $E[R] = \mu$ - Expected return
2. **Second Moment (Variance)**: $Var[R] = E[(R - \mu)^2] = \sigma^2$ - Risk measure
3. **Third Moment (Skewness)**: $S = \frac{E[(R - \mu)^3]}{\sigma^3}$ - Asymmetry
4. **Fourth Moment (Kurtosis)**: $K = \frac{E[(R - \mu)^4]}{\sigma^4}$ - Tail thickness

---

## 🟢 Probability Distributions in Finance {#probability-distributions}

### Intuitive Understanding

Different financial variables follow different probability distributions:
- **Stock returns**: Often exhibit fat tails and skewness
- **Interest rates**: Mean-reverting behavior
- **Option prices**: Log-normal characteristics
- **Default probabilities**: Exponential or Weibull distributions

### 🟡 Practical Implementation

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

class FinancialDistributions:
    """
    Common probability distributions used in financial modeling
    """
    
    @staticmethod
    def normal_distribution(mu: float = 0, sigma: float = 1) -> Dict[str, Any]:
        """
        Normal distribution: N(μ, σ²)
        Used for: Basic return modeling, risk-neutral assumptions
        """
        dist = stats.norm(loc=mu, scale=sigma)
        
        return {
            'distribution': dist,
            'pdf': lambda x: dist.pdf(x),
            'cdf': lambda x: dist.cdf(x),
            'ppf': lambda p: dist.ppf(p),  # Inverse CDF
            'mean': mu,
            'variance': sigma**2,
            'skewness': 0,
            'kurtosis': 0,
            'use_cases': ['Basic return modeling', 'Risk-neutral pricing', 'Portfolio optimization']
        }
    
    @staticmethod
    def log_normal_distribution(mu: float = 0, sigma: float = 1) -> Dict[str, Any]:
        """
        Log-normal distribution: ln(X) ~ N(μ, σ²)
        Used for: Stock prices, asset values (always positive)
        """
        dist = stats.lognorm(s=sigma, scale=np.exp(mu))
        
        return {
            'distribution': dist,
            'pdf': lambda x: dist.pdf(x),
            'cdf': lambda x: dist.cdf(x),
            'ppf': lambda p: dist.ppf(p),
            'mean': np.exp(mu + sigma**2/2),
            'variance': (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2),
            'skewness': (np.exp(sigma**2) + 2) * np.sqrt(np.exp(sigma**2) - 1),
            'kurtosis': np.exp(4*sigma**2) + 2*np.exp(3*sigma**2) + 3*np.exp(2*sigma**2) - 6,
            'use_cases': ['Stock prices', 'Asset valuations', 'Option pricing models']
        }
    
    @staticmethod
    def student_t_distribution(df: float = 3) -> Dict[str, Any]:
        """
        Student's t-distribution
        Used for: Heavy-tailed return distributions, robust estimation
        """
        dist = stats.t(df=df)
        
        mean = 0 if df > 1 else np.nan
        variance = df/(df-2) if df > 2 else np.inf if df > 1 else np.nan
        
        return {
            'distribution': dist,
            'pdf': lambda x: dist.pdf(x),
            'cdf': lambda x: dist.cdf(x),
            'ppf': lambda p: dist.ppf(p),
            'mean': mean,
            'variance': variance,
            'skewness': 0 if df > 3 else np.nan,
            'kurtosis': 6/(df-4) if df > 4 else np.inf if df > 2 else np.nan,
            'use_cases': ['Heavy-tailed returns', 'Robust portfolio optimization', 'Crisis modeling']
        }
    
    @staticmethod
    def exponential_distribution(lam: float = 1) -> Dict[str, Any]:
        """
        Exponential distribution
        Used for: Time between events, default times
        """
        dist = stats.expon(scale=1/lam)
        
        return {
            'distribution': dist,
            'pdf': lambda x: dist.pdf(x),
            'cdf': lambda x: dist.cdf(x),
            'ppf': lambda p: dist.ppf(p),
            'mean': 1/lam,
            'variance': 1/lam**2,
            'skewness': 2,
            'kurtosis': 6,
            'use_cases': ['Time to default', 'Inter-arrival times', 'Survival analysis']
        }
    
    @staticmethod
    def beta_distribution(alpha: float = 2, beta: float = 2) -> Dict[str, Any]:
        """
        Beta distribution: Beta(α, β)
        Used for: Probabilities, correlation coefficients
        """
        dist = stats.beta(a=alpha, b=beta)
        
        mean = alpha / (alpha + beta)
        variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        
        return {
            'distribution': dist,
            'pdf': lambda x: dist.pdf(x),
            'cdf': lambda x: dist.cdf(x),
            'ppf': lambda p: dist.ppf(p),
            'mean': mean,
            'variance': variance,
            'skewness': (2 * (beta - alpha) * np.sqrt(alpha + beta + 1)) / ((alpha + beta + 2) * np.sqrt(alpha * beta)),
            'use_cases': ['Probability modeling', 'Correlation coefficients', 'Recovery rates']
        }
    
    @staticmethod
    def fit_distribution_to_data(data: np.ndarray, dist_name: str = 'auto') -> Dict[str, Any]:
        """
        Fit a probability distribution to financial data
        """
        data = data[np.isfinite(data)]  # Remove NaN and infinite values
        
        if dist_name == 'auto':
            # Test multiple distributions and select best fit
            distributions = ['norm', 'lognorm', 't', 'skewnorm']
            best_dist = None
            best_aic = np.inf
            
            for dist_name in distributions:
                try:
                    dist = getattr(stats, dist_name)
                    params = dist.fit(data)
                    
                    # Calculate AIC (Akaike Information Criterion)
                    log_likelihood = np.sum(dist.logpdf(data, *params))
                    aic = 2 * len(params) - 2 * log_likelihood
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_dist = (dist_name, params, aic)
                        
                except Exception:
                    continue
            
            if best_dist is None:
                raise ValueError("Could not fit any distribution to the data")
            
            dist_name, params, aic = best_dist
        else:
            dist = getattr(stats, dist_name)
            params = dist.fit(data)
            log_likelihood = np.sum(dist.logpdf(data, *params))
            aic = 2 * len(params) - 2 * log_likelihood
        
        fitted_dist = getattr(stats, dist_name)(*params)
        
        # Goodness of fit tests
        ks_stat, ks_pvalue = stats.kstest(data, fitted_dist.cdf)
        
        return {
            'distribution_name': dist_name,
            'parameters': params,
            'fitted_distribution': fitted_dist,
            'aic': aic,
            'log_likelihood': log_likelihood,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'goodness_of_fit': 'Good' if ks_pvalue > 0.05 else 'Poor'
        }
    
    @staticmethod
    def visualize_distributions():
        """Visualize common financial distributions"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        x = np.linspace(-5, 5, 1000)
        
        # 1. Normal vs t-distribution (fat tails)
        normal = FinancialDistributions.normal_distribution(0, 1)
        t_dist = FinancialDistributions.student_t_distribution(3)
        
        axes[0].plot(x, normal['pdf'](x), label='Normal(0,1)', linewidth=2)
        axes[0].plot(x, t_dist['pdf'](x), label='t(df=3)', linewidth=2)
        axes[0].set_title('Normal vs t-distribution\n(Fat Tails in Finance)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Log-normal for asset prices
        x_pos = np.linspace(0.1, 10, 1000)
        lognorm = FinancialDistributions.log_normal_distribution(0, 0.5)
        
        axes[1].plot(x_pos, lognorm['pdf'](x_pos), label='Log-Normal(0, 0.5)', linewidth=2, color='green')
        axes[1].set_title('Log-Normal Distribution\n(Asset Prices)')
        axes[1].set_xlabel('Asset Price')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Exponential for time to events
        x_exp = np.linspace(0, 5, 1000)
        exp_dist = FinancialDistributions.exponential_distribution(1)
        
        axes[2].plot(x_exp, exp_dist['pdf'](x_exp), label='Exponential(λ=1)', linewidth=2, color='red')
        axes[2].set_title('Exponential Distribution\n(Time to Default)')
        axes[2].set_xlabel('Time')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. Beta for probabilities
        x_beta = np.linspace(0, 1, 1000)
        beta_dist = FinancialDistributions.beta_distribution(2, 5)
        
        axes[3].plot(x_beta, beta_dist['pdf'](x_beta), label='Beta(2, 5)', linewidth=2, color='purple')
        axes[3].set_title('Beta Distribution\n(Probabilities, Recovery Rates)')
        axes[3].set_xlabel('Probability')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # 5. Comparison of different t-distributions
        for df in [1, 3, 10]:
            t_dist = FinancialDistributions.student_t_distribution(df)
            axes[4].plot(x, t_dist['pdf'](x), label=f't(df={df})', linewidth=2)
        
        axes[4].plot(x, normal['pdf'](x), '--', label='Normal', linewidth=2, alpha=0.7)
        axes[4].set_title('t-distributions with Different\nDegrees of Freedom')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
        
        # 6. VaR illustration
        normal = FinancialDistributions.normal_distribution(-0.001, 0.02)  # Daily returns
        x_var = np.linspace(-0.1, 0.05, 1000)
        pdf_vals = normal['pdf'](x_var)
        
        axes[5].fill_between(x_var, pdf_vals, alpha=0.3, color='blue')
        
        # 5% VaR
        var_5 = normal['ppf'](0.05)
        axes[5].axvline(var_5, color='red', linestyle='--', linewidth=2, label=f'5% VaR = {var_5:.3f}')
        axes[5].fill_between(x_var[x_var <= var_5], pdf_vals[x_var <= var_5], alpha=0.7, color='red')
        
        axes[5].set_title('Value at Risk (VaR) Illustration')
        axes[5].set_xlabel('Daily Return')
        axes[5].set_ylabel('Probability Density')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Demonstrate distribution fitting
def demonstrate_distribution_fitting():
    """Demonstrate fitting distributions to financial data"""
    
    # Generate synthetic financial data with different characteristics
    np.random.seed(42)
    
    # Normal returns
    normal_returns = np.random.normal(0.001, 0.02, 1000)
    
    # Fat-tailed returns (t-distribution)
    t_returns = stats.t.rvs(df=3, size=1000) * 0.02 + 0.001
    
    # Skewed returns
    skewed_returns = stats.skewnorm.rvs(a=-2, size=1000) * 0.02 + 0.001
    
    datasets = {
        'Normal Returns': normal_returns,
        'Fat-tailed Returns': t_returns,
        'Skewed Returns': skewed_returns
    }
    
    results = {}
    
    for name, data in datasets.items():
        print(f"\n--- {name} ---")
        result = FinancialDistributions.fit_distribution_to_data(data)
        results[name] = result
        
        print(f"Best fitting distribution: {result['distribution_name']}")
        print(f"Parameters: {result['parameters']}")
        print(f"AIC: {result['aic']:.2f}")
        print(f"Goodness of fit: {result['goodness_of_fit']} (p-value: {result['ks_pvalue']:.4f})")
    
    return results

if __name__ == "__main__":
    # Visualize distributions
    FinancialDistributions.visualize_distributions()
    
    # Demonstrate distribution fitting
    fitting_results = demonstrate_distribution_fitting()
```

### 🔴 Theoretical Foundation

**Key Financial Distributions:**

1. **Normal Distribution**: $X \sim N(\mu, \sigma^2)$
   - PDF: $f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
   - Used in Black-Scholes model assumptions

2. **Log-Normal Distribution**: $\ln(X) \sim N(\mu, \sigma^2)$
   - PDF: $f(x) = \frac{1}{x\sigma\sqrt{2\pi}} e^{-\frac{(\ln x-\mu)^2}{2\sigma^2}}$
   - Stock prices follow geometric Brownian motion

3. **Student's t-Distribution**: Degrees of freedom $\nu$
   - PDF: $f(x) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu\pi}\Gamma(\frac{\nu}{2})} \left(1+\frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}$
   - Heavy-tailed alternative to normal distribution

---

## 🟢 Hypothesis Testing and Confidence Intervals {#hypothesis-testing}

### Intuitive Understanding

Hypothesis testing helps us make decisions about financial models:
- **Is a strategy statistically significant?**
- **Are returns normally distributed?**
- **Is there a structural break in the data?**
- **Do risk factors explain returns?**

### 🟡 Practical Implementation

```python
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, Dict, Any

class FinancialHypothesisTests:
    """
    Hypothesis tests commonly used in financial modeling
    """
    
    @staticmethod
    def test_mean_return(returns: np.ndarray, 
                        null_mean: float = 0, 
                        alpha: float = 0.05) -> Dict[str, Any]:
        """
        Test if mean return is significantly different from null hypothesis
        H0: μ = null_mean vs H1: μ ≠ null_mean
        """
        returns = returns[np.isfinite(returns)]
        n = len(returns)
        sample_mean = np.mean(returns)
        sample_std = np.std(returns, ddof=1)
        
        # t-test statistic
        t_stat = (sample_mean - null_mean) / (sample_std / np.sqrt(n))
        
        # p-value (two-tailed)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
        
        # Confidence interval for mean
        t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
        margin_error = t_critical * (sample_std / np.sqrt(n))
        ci_lower = sample_mean - margin_error
        ci_upper = sample_mean + margin_error
        
        return {
            'test_name': 'One-sample t-test for mean return',
            'null_hypothesis': f'Mean return = {null_mean}',
            'sample_mean': sample_mean,
            'sample_std': sample_std,
            't_statistic': t_stat,
            'p_value': p_value,
            'reject_null': p_value < alpha,
            'confidence_interval': (ci_lower, ci_upper),
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0 at {alpha:.0%} significance level"
        }
    
    @staticmethod
    def test_normality(data: np.ndarray, 
                      alpha: float = 0.05) -> Dict[str, Any]:
        """
        Comprehensive normality tests for financial data
        """
        data = data[np.isfinite(data)]
        
        # Shapiro-Wilk test (good for small samples)
        shapiro_stat, shapiro_p = stats.shapiro(data) if len(data) <= 5000 else (np.nan, np.nan)
        
        # Jarque-Bera test (based on skewness and kurtosis)
        jb_stat, jb_p = stats.jarque_bera(data)
        
        # Anderson-Darling test
        ad_stat, ad_crit_vals, ad_sig_levels = stats.anderson(data, dist='norm')
        ad_p_approx = None
        for i, sig_level in enumerate(ad_sig_levels):
            if ad_stat < ad_crit_vals[i]:
                ad_p_approx = 1 - sig_level/100
                break
        if ad_p_approx is None:
            ad_p_approx = 0.001  # Very low p-value
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
        
        tests_results = {
            'Jarque-Bera': {'statistic': jb_stat, 'p_value': jb_p, 'reject_null': jb_p < alpha},
            'Kolmogorov-Smirnov': {'statistic': ks_stat, 'p_value': ks_p, 'reject_null': ks_p < alpha},
            'Anderson-Darling': {'statistic': ad_stat, 'p_value': ad_p_approx, 'reject_null': ad_p_approx < alpha}
        }
        
        if not np.isnan(shapiro_stat):
            tests_results['Shapiro-Wilk'] = {
                'statistic': shapiro_stat, 
                'p_value': shapiro_p, 
                'reject_null': shapiro_p < alpha
            }
        
        # Overall assessment
        rejections = sum(test['reject_null'] for test in tests_results.values())
        total_tests = len(tests_results)
        
        return {
            'test_name': 'Normality Tests',
            'null_hypothesis': 'Data follows normal distribution',
            'individual_tests': tests_results,
            'overall_rejection_rate': rejections / total_tests,
            'likely_normal': rejections <= total_tests / 2,
            'interpretation': f"Data {'likely follows' if rejections <= total_tests / 2 else 'likely does not follow'} normal distribution"
        }
    
    @staticmethod
    def test_stationarity(data: np.ndarray, 
                         alpha: float = 0.05) -> Dict[str, Any]:
        """
        Test for stationarity using Augmented Dickey-Fuller test
        """
        data = data[np.isfinite(data)]
        
        # ADF test
        adf_stat, adf_p, n_lags, n_obs, crit_vals, _ = adfuller(data, autolag='AIC')
        
        return {
            'test_name': 'Augmented Dickey-Fuller Test',
            'null_hypothesis': 'Data has unit root (non-stationary)',
            'adf_statistic': adf_stat,
            'p_value': adf_p,
            'n_lags': n_lags,
            'n_observations': n_obs,
            'critical_values': crit_vals,
            'reject_null': adf_p < alpha,
            'is_stationary': adf_p < alpha,
            'interpretation': f"Data {'is' if adf_p < alpha else 'is not'} stationary at {alpha:.0%} significance level"
        }
    
    @staticmethod
    def test_equal_means(returns1: np.ndarray, 
                        returns2: np.ndarray,
                        alpha: float = 0.05,
                        equal_var: bool = False) -> Dict[str, Any]:
        """
        Test if two return series have equal means (comparing strategies)
        """
        returns1 = returns1[np.isfinite(returns1)]
        returns2 = returns2[np.isfinite(returns2)]
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(returns1, returns2, equal_var=equal_var)
        
        # Descriptive statistics
        mean1, mean2 = np.mean(returns1), np.mean(returns2)
        std1, std2 = np.std(returns1, ddof=1), np.std(returns2, ddof=1)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(returns1) - 1) * std1**2 + (len(returns2) - 1) * std2**2) / 
                            (len(returns1) + len(returns2) - 2))
        cohens_d = (mean1 - mean2) / pooled_std
        
        return {
            'test_name': 'Two-sample t-test for equal means',
            'null_hypothesis': 'Mean returns are equal',
            'sample1_mean': mean1,
            'sample2_mean': mean2,
            'mean_difference': mean1 - mean2,
            't_statistic': t_stat,
            'p_value': p_value,
            'reject_null': p_value < alpha,
            'cohens_d': cohens_d,
            'effect_size': ('Small' if abs(cohens_d) < 0.2 else 
                          'Medium' if abs(cohens_d) < 0.8 else 'Large'),
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0: strategies have equal mean returns"
        }
    
    @staticmethod
    def test_arch_effects(returns: np.ndarray, 
                         lags: int = 5,
                         alpha: float = 0.05) -> Dict[str, Any]:
        """
        Test for ARCH effects (volatility clustering) in returns
        """
        returns = returns[np.isfinite(returns)]
        
        # Calculate squared returns
        squared_returns = returns**2
        
        # Ljung-Box test on squared returns
        lb_stat, lb_p = stats.diagnostic.acorr_ljungbox(squared_returns, lags=lags, return_df=False)
        
        # ARCH-LM test using regression approach
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant
        
        # Create lagged squared returns
        n = len(squared_returns)
        X = np.column_stack([squared_returns[i:n-lags+i] for i in range(lags)])
        y = squared_returns[lags:]
        
        # Add constant
        X = add_constant(X)
        
        # Run regression
        model = OLS(y, X).fit()
        lm_stat = model.rsquared * len(y)
        lm_p = 1 - stats.chi2.cdf(lm_stat, df=lags)
        
        return {
            'test_name': 'ARCH Effects Test',
            'null_hypothesis': 'No ARCH effects (constant volatility)',
            'ljung_box_stat': lb_stat[-1] if isinstance(lb_stat, np.ndarray) else lb_stat,
            'ljung_box_p': lb_p[-1] if isinstance(lb_p, np.ndarray) else lb_p,
            'arch_lm_stat': lm_stat,
            'arch_lm_p': lm_p,
            'reject_null_lb': (lb_p[-1] if isinstance(lb_p, np.ndarray) else lb_p) < alpha,
            'reject_null_lm': lm_p < alpha,
            'has_arch_effects': lm_p < alpha,
            'interpretation': f"{'Evidence of' if lm_p < alpha else 'No evidence of'} ARCH effects (volatility clustering)"
        }

def demonstrate_hypothesis_testing():
    """Demonstrate hypothesis testing on financial data"""
    
    np.random.seed(42)
    
    # Generate different types of financial data
    n = 1000
    
    # Normal returns with small positive mean
    normal_returns = np.random.normal(0.0008, 0.02, n)
    
    # Returns with ARCH effects (volatility clustering)
    arch_returns = []
    sigma = 0.02
    for i in range(n):
        if i > 0:
            sigma = 0.01 + 0.1 * (arch_returns[i-1]**2)  # GARCH(1,1) like
        ret = np.random.normal(0.0008, sigma)
        arch_returns.append(ret)
    arch_returns = np.array(arch_returns)
    
    # Non-stationary price series
    prices = 100 * np.exp(np.cumsum(normal_returns))
    
    # Strategy comparison data
    strategy_a = np.random.normal(0.001, 0.02, n)
    strategy_b = np.random.normal(0.0015, 0.022, n)  # Slightly higher mean and vol
    
    print("=== Financial Hypothesis Testing Examples ===\\n")
    
    # 1. Test mean return
    print("1. Testing Mean Return:")
    mean_test = FinancialHypothesisTests.test_mean_return(normal_returns)
    print(f"   {mean_test['interpretation']}")
    print(f"   Sample mean: {mean_test['sample_mean']:.6f}")
    print(f"   95% CI: [{mean_test['confidence_interval'][0]:.6f}, {mean_test['confidence_interval'][1]:.6f}]")
    print(f"   p-value: {mean_test['p_value']:.4f}\\n")
    
    # 2. Test normality
    print("2. Testing Normality:")
    normality_test = FinancialHypothesisTests.test_normality(normal_returns)
    print(f"   {normality_test['interpretation']}")
    for test_name, result in normality_test['individual_tests'].items():
        print(f"   {test_name}: p-value = {result['p_value']:.4f}")
    print()
    
    # 3. Test stationarity
    print("3. Testing Stationarity:")
    # Returns should be stationary
    returns_stationarity = FinancialHypothesisTests.test_stationarity(normal_returns)
    print(f"   Returns: {returns_stationarity['interpretation']}")
    
    # Prices should be non-stationary
    prices_stationarity = FinancialHypothesisTests.test_stationarity(prices)
    print(f"   Prices: {prices_stationarity['interpretation']}")
    print()
    
    # 4. Test equal means (strategy comparison)
    print("4. Strategy Comparison:")
    strategy_test = FinancialHypothesisTests.test_equal_means(strategy_a, strategy_b)
    print(f"   {strategy_test['interpretation']}")
    print(f"   Strategy A mean: {strategy_test['sample1_mean']:.6f}")
    print(f"   Strategy B mean: {strategy_test['sample2_mean']:.6f}")
    print(f"   Difference: {strategy_test['mean_difference']:.6f}")
    print(f"   Effect size: {strategy_test['effect_size']} (Cohen's d = {strategy_test['cohens_d']:.3f})")
    print()
    
    # 5. Test ARCH effects
    print("5. Testing for Volatility Clustering (ARCH effects):")
    arch_test_normal = FinancialHypothesisTests.test_arch_effects(normal_returns)
    print(f"   Normal returns: {arch_test_normal['interpretation']}")
    
    arch_test_cluster = FinancialHypothesisTests.test_arch_effects(arch_returns)
    print(f"   Clustered volatility returns: {arch_test_cluster['interpretation']}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Normal vs ARCH returns
    axes[0, 0].plot(normal_returns[:200], label='Normal Returns', alpha=0.7)
    axes[0, 0].plot(arch_returns[:200], label='ARCH Returns', alpha=0.7)
    axes[0, 0].set_title('Returns: Normal vs ARCH Effects')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Strategy comparison
    cumulative_a = np.cumprod(1 + strategy_a)
    cumulative_b = np.cumprod(1 + strategy_b)
    
    axes[0, 1].plot(cumulative_a, label='Strategy A', linewidth=2)
    axes[0, 1].plot(cumulative_b, label='Strategy B', linewidth=2)
    axes[0, 1].set_title('Strategy Performance Comparison')
    axes[0, 1].set_ylabel('Cumulative Return')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Returns vs Prices (stationarity)
    axes[1, 0].plot(normal_returns[:200], label='Returns (Stationary)')
    axes[1, 0].set_title('Stationary Returns')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(prices[:200], label='Prices (Non-stationary)', color='red')
    axes[1, 1].set_title('Non-Stationary Prices')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demonstrate_hypothesis_testing()
```

### 🔴 Theoretical Foundation

**General Hypothesis Testing Framework:**

1. **Null Hypothesis** ($H_0$): Status quo assumption
2. **Alternative Hypothesis** ($H_1$): What we want to prove
3. **Test Statistic**: Measures evidence against $H_0$
4. **p-value**: Probability of observing data as extreme as observed, given $H_0$ is true
5. **Significance Level** ($\alpha$): Threshold for rejecting $H_0$

**Type I and Type II Errors:**
- **Type I Error**: Rejecting true $H_0$ (False Positive) - Probability = $\alpha$
- **Type II Error**: Accepting false $H_0$ (False Negative) - Probability = $\beta$
- **Power**: Probability of correctly rejecting false $H_0$ = $1 - \beta$

**Confidence Interval for Mean:**
For sample mean $\bar{X}$ with unknown population variance:
$$CI = \bar{X} \pm t_{\alpha/2,n-1} \cdot \frac{s}{\sqrt{n}}$$

---

## 🟢 Bayesian Inference Applications {#bayesian-inference}

### Intuitive Understanding

Bayesian inference updates beliefs based on new evidence:
- **Prior belief** + **New evidence** → **Updated belief**
- Particularly useful in finance for:
  - Portfolio optimization with uncertain parameters
  - Risk model updating
  - Incorporating expert opinions

### 🟡 Practical Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from typing import Tuple, Dict, Any, Optional

class BayesianFinance:
    """
    Bayesian methods for financial modeling
    """
    
    @staticmethod
    def bayesian_return_estimation(returns: np.ndarray,
                                 prior_mean: float = 0.0,
                                 prior_variance: float = 1.0,
                                 confidence: float = 0.95) -> Dict[str, Any]:
        """
        Bayesian estimation of expected return with normal prior
        
        Prior: μ ~ N(prior_mean, prior_variance)
        Likelihood: returns ~ N(μ, σ²)
        Posterior: μ|data ~ N(posterior_mean, posterior_variance)
        """
        returns = returns[np.isfinite(returns)]
        n = len(returns)
        
        # Sample statistics
        sample_mean = np.mean(returns)
        sample_var = np.var(returns, ddof=1)
        
        # Bayesian update (conjugate prior)
        precision_prior = 1 / prior_variance
        precision_likelihood = n / sample_var
        
        posterior_precision = precision_prior + precision_likelihood
        posterior_variance = 1 / posterior_precision
        posterior_mean = (precision_prior * prior_mean + precision_likelihood * sample_mean) / posterior_precision
        
        # Credible interval
        posterior_std = np.sqrt(posterior_variance)
        alpha = 1 - confidence
        z_score = stats.norm.ppf(1 - alpha/2)
        
        credible_interval = (
            posterior_mean - z_score * posterior_std,
            posterior_mean + z_score * posterior_std
        )
        
        return {
            'prior_mean': prior_mean,
            'prior_variance': prior_variance,
            'sample_mean': sample_mean,
            'sample_size': n,
            'posterior_mean': posterior_mean,
            'posterior_variance': posterior_variance,
            'posterior_std': posterior_std,
            'credible_interval': credible_interval,
            'shrinkage_factor': precision_prior / posterior_precision,
            'interpretation': f"Posterior mean return: {posterior_mean:.4f} with {confidence:.0%} credible interval {credible_interval}"
        }
    
    @staticmethod
    def bayesian_portfolio_optimization(returns: np.ndarray,
                                      risk_aversion: float = 3.0,
                                      prior_returns: Optional[np.ndarray] = None,
                                      prior_confidence: float = 0.1) -> Dict[str, Any]:
        """
        Bayesian portfolio optimization incorporating parameter uncertainty
        """
        returns = returns[~np.isnan(returns).any(axis=1)]  # Remove rows with NaN
        n_obs, n_assets = returns.shape
        
        # Sample statistics
        sample_mean = np.mean(returns, axis=0)
        sample_cov = np.cov(returns.T)
        
        # Set priors
        if prior_returns is None:
            prior_returns = np.zeros(n_assets)
        
        prior_cov = np.eye(n_assets) / prior_confidence  # Higher confidence = smaller variance
        
        # Bayesian update for mean returns (conjugate prior)
        precision_prior = np.linalg.inv(prior_cov)
        precision_likelihood = n_obs * np.linalg.inv(sample_cov)
        
        try:
            precision_posterior = precision_prior + precision_likelihood
            cov_posterior = np.linalg.inv(precision_posterior)
            mean_posterior = cov_posterior @ (precision_prior @ prior_returns + precision_likelihood @ sample_mean)
        except np.linalg.LinAlgError:
            # Fallback to sample estimates if matrix inversion fails
            mean_posterior = sample_mean
            cov_posterior = sample_cov
        
        # Mean-variance optimization with posterior estimates
        def objective(weights):
            portfolio_return = weights @ mean_posterior
            portfolio_variance = weights @ cov_posterior @ weights
            return -portfolio_return + 0.5 * risk_aversion * portfolio_variance
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_assets)]  # Long-only
        
        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            portfolio_return = optimal_weights @ mean_posterior
            portfolio_variance = optimal_weights @ cov_posterior @ optimal_weights
            portfolio_std = np.sqrt(portfolio_variance)
            sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0
        else:
            optimal_weights = x0  # Fallback to equal weights
            portfolio_return = optimal_weights @ mean_posterior
            portfolio_variance = optimal_weights @ cov_posterior @ optimal_weights
            portfolio_std = np.sqrt(portfolio_variance)
            sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0
        
        return {
            'optimal_weights': optimal_weights,
            'portfolio_return': portfolio_return,
            'portfolio_std': portfolio_std,
            'portfolio_variance': portfolio_variance,
            'sharpe_ratio': sharpe_ratio,
            'posterior_mean': mean_posterior,
            'posterior_cov': cov_posterior,
            'sample_mean': sample_mean,
            'sample_cov': sample_cov,
            'shrinkage_intensity': np.trace(precision_prior) / np.trace(precision_posterior)
        }
    
    @staticmethod
    def bayesian_value_at_risk(returns: np.ndarray,
                             confidence_level: float = 0.05,
                             prior_df: float = 10.0) -> Dict[str, Any]:
        """
        Bayesian Value at Risk estimation using t-distribution
        """
        returns = returns[np.isfinite(returns)]
        n = len(returns)
        
        # Sample statistics
        sample_mean = np.mean(returns)
        sample_var = np.var(returns, ddof=1)
        
        # Posterior parameters for t-distribution
        # Using conjugate prior for t-distribution (scaled inverse chi-square for variance)
        posterior_df = prior_df + n
        posterior_scale = (prior_df + (n-1) * sample_var + n * sample_mean**2) / posterior_df
        posterior_location = sample_mean
        
        # Posterior predictive distribution is t-distribution
        posterior_t = stats.t(df=posterior_df, loc=posterior_location, scale=np.sqrt(posterior_scale))
        
        # VaR calculation
        var_estimate = posterior_t.ppf(confidence_level)
        
        # Expected Shortfall (Conditional VaR)
        # E[X | X ≤ VaR] for t-distribution
        es_estimate = posterior_location - np.sqrt(posterior_scale) * \
                     (posterior_df + var_estimate**2) / (posterior_df - 1) * \
                     stats.t.pdf(var_estimate, df=posterior_df) / confidence_level
        
        # Credible interval for VaR
        var_samples = []
        n_samples = 10000
        
        # Monte Carlo sampling from posterior predictive
        for _ in range(n_samples):
            sample_t = posterior_t.rvs(size=100)  # Sample future returns
            var_samples.append(np.percentile(sample_t, confidence_level * 100))
        
        var_credible_interval = (np.percentile(var_samples, 2.5), np.percentile(var_samples, 97.5))
        
        return {
            'var_estimate': var_estimate,
            'expected_shortfall': es_estimate,
            'var_credible_interval': var_credible_interval,
            'posterior_df': posterior_df,
            'posterior_location': posterior_location,
            'posterior_scale': posterior_scale,
            'confidence_level': confidence_level,
            'interpretation': f"{confidence_level:.0%} VaR: {var_estimate:.4f} with 95% credible interval {var_credible_interval}"
        }

def demonstrate_bayesian_inference():
    """Demonstrate Bayesian inference in finance"""
    
    np.random.seed(42)
    
    # Generate synthetic financial data
    true_mean = 0.001  # Daily return
    true_vol = 0.02
    n_days = 250
    
    returns_1d = np.random.normal(true_mean, true_vol, n_days)
    
    # Multi-asset returns
    n_assets = 3
    mean_returns = np.array([0.0008, 0.0012, 0.0006])
    cov_matrix = np.array([[0.04, 0.01, 0.005],
                          [0.01, 0.06, 0.008],
                          [0.005, 0.008, 0.03]]) * 0.01
    
    returns_multivariate = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
    
    print("=== Bayesian Inference in Finance ===\\n")
    
    # 1. Bayesian return estimation
    print("1. Bayesian Expected Return Estimation:")
    
    # Weak prior (high uncertainty)
    weak_prior = BayesianFinance.bayesian_return_estimation(
        returns_1d, prior_mean=0.0, prior_variance=0.01
    )
    print(f"   Weak prior: {weak_prior['interpretation']}")
    print(f"   Shrinkage factor: {weak_prior['shrinkage_factor']:.3f}")
    
    # Strong prior (low uncertainty) 
    strong_prior = BayesianFinance.bayesian_return_estimation(
        returns_1d, prior_mean=0.002, prior_variance=0.0001
    )
    print(f"   Strong prior: {strong_prior['interpretation']}")
    print(f"   Shrinkage factor: {strong_prior['shrinkage_factor']:.3f}")
    print()
    
    # 2. Bayesian portfolio optimization
    print("2. Bayesian Portfolio Optimization:")
    portfolio_result = BayesianFinance.bayesian_portfolio_optimization(
        returns_multivariate, risk_aversion=3.0
    )
    print(f"   Optimal weights: {portfolio_result['optimal_weights']}")
    print(f"   Expected return: {portfolio_result['portfolio_return']:.4f}")
    print(f"   Expected volatility: {portfolio_result['portfolio_std']:.4f}")
    print(f"   Sharpe ratio: {portfolio_result['sharpe_ratio']:.3f}")
    print()
    
    # 3. Bayesian VaR estimation
    print("3. Bayesian Value at Risk:")
    var_result = BayesianFinance.bayesian_value_at_risk(returns_1d)
    print(f"   {var_result['interpretation']}")
    print(f"   Expected Shortfall: {var_result['expected_shortfall']:.4f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Prior vs Posterior distributions
    x = np.linspace(-0.01, 0.01, 1000)
    
    # Weak prior case
    prior_weak = stats.norm(0.0, np.sqrt(0.01))
    posterior_weak = stats.norm(weak_prior['posterior_mean'], weak_prior['posterior_std'])
    likelihood_weak = stats.norm(np.mean(returns_1d), true_vol/np.sqrt(n_days))
    
    axes[0, 0].plot(x, prior_weak.pdf(x), label='Prior', linewidth=2)
    axes[0, 0].plot(x, likelihood_weak.pdf(x), label='Likelihood', linewidth=2)
    axes[0, 0].plot(x, posterior_weak.pdf(x), label='Posterior', linewidth=2)
    axes[0, 0].set_title('Bayesian Updating: Weak Prior')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Strong prior case  
    prior_strong = stats.norm(0.002, np.sqrt(0.0001))
    posterior_strong = stats.norm(strong_prior['posterior_mean'], strong_prior['posterior_std'])
    
    axes[0, 1].plot(x, prior_strong.pdf(x), label='Prior', linewidth=2)
    axes[0, 1].plot(x, likelihood_weak.pdf(x), label='Likelihood', linewidth=2)
    axes[0, 1].plot(x, posterior_strong.pdf(x), label='Posterior', linewidth=2)
    axes[0, 1].set_title('Bayesian Updating: Strong Prior')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 2: Portfolio weights
    asset_names = ['Asset A', 'Asset B', 'Asset C']
    axes[1, 0].bar(asset_names, portfolio_result['optimal_weights'])
    axes[1, 0].set_title('Bayesian Optimal Portfolio Weights')
    axes[1, 0].set_ylabel('Weight')
    
    # Plot 3: VaR visualization
    axes[1, 1].hist(returns_1d, bins=50, alpha=0.7, density=True)
    axes[1, 1].axvline(var_result['var_estimate'], color='red', linestyle='--', 
                      linewidth=2, label=f"5% VaR: {var_result['var_estimate']:.4f}")
    axes[1, 1].set_title('Bayesian VaR Estimation')
    axes[1, 1].set_xlabel('Daily Return')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demonstrate_bayesian_inference()
```

### 🔴 Theoretical Foundation

**Bayes' Theorem:**
$$P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}$$

Where:
- $P(\theta|D)$: Posterior (updated belief)
- $P(D|\theta)$: Likelihood (evidence)
- $P(\theta)$: Prior (initial belief)
- $P(D)$: Marginal likelihood (normalization)

**Conjugate Priors:**
For computational efficiency, choose priors that result in posterior distributions of the same family:

- **Normal-Normal**: If $X|\mu \sim N(\mu, \sigma^2)$ and $\mu \sim N(\mu_0, \tau^2)$, then $\mu|X \sim N(\mu_n, \tau_n^2)$

**Bayesian Portfolio Theory:**
Incorporates parameter uncertainty into optimization:
$$\max_w E[U(w^T R)] = \max_w \int U(w^T r) p(r|\theta) p(\theta|D) dr d\theta$$

---

## 🟢 Correlation vs Causation in Financial Modeling {#correlation-causation}

### Intuitive Understanding

**Correlation ≠ Causation** is critical in finance:
- High correlation between variables doesn't imply one causes the other
- **Spurious correlations** can lead to false strategies
- **Causal inference** requires proper experimental design or instrumental variables

### 🟡 Practical Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from typing import Dict, Any, Tuple, List

class CorrelationCausationAnalysis:
    """
    Tools for analyzing correlation vs causation in financial data
    """
    
    @staticmethod
    def correlation_analysis(x: np.ndarray, y: np.ndarray, method: str = 'all') -> Dict[str, Any]:
        """
        Comprehensive correlation analysis
        """
        # Remove pairs with missing values
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean, y_clean = x[mask], y[mask]
        
        if len(x_clean) < 3:
            raise ValueError("Insufficient data for correlation analysis")
        
        results = {}
        
        if method in ['all', 'pearson']:
            # Pearson correlation (linear relationship)
            pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
            results['pearson'] = {
                'correlation': pearson_r,
                'p_value': pearson_p,
                'significant': pearson_p < 0.05,
                'interpretation': f"{'Strong' if abs(pearson_r) > 0.7 else 'Moderate' if abs(pearson_r) > 0.3 else 'Weak'} linear correlation"
            }
        
        if method in ['all', 'spearman']:
            # Spearman correlation (monotonic relationship)
            spearman_r, spearman_p = spearmanr(x_clean, y_clean)
            results['spearman'] = {
                'correlation': spearman_r,
                'p_value': spearman_p,
                'significant': spearman_p < 0.05,
                'interpretation': f"{'Strong' if abs(spearman_r) > 0.7 else 'Moderate' if abs(spearman_r) > 0.3 else 'Weak'} monotonic correlation"
            }
        
        if method in ['all', 'kendall']:
            # Kendall's Tau (rank-based correlation)
            kendall_tau, kendall_p = stats.kendalltau(x_clean, y_clean)
            results['kendall'] = {
                'correlation': kendall_tau,
                'p_value': kendall_p,
                'significant': kendall_p < 0.05,
                'interpretation': f"{'Strong' if abs(kendall_tau) > 0.7 else 'Moderate' if abs(kendall_tau) > 0.3 else 'Weak'} rank correlation"
            }
        
        # Linear regression R-squared
        reg = LinearRegression().fit(x_clean.reshape(-1, 1), y_clean)
        r_squared = reg.score(x_clean.reshape(-1, 1), y_clean)
        
        results['regression'] = {
            'r_squared': r_squared,
            'slope': reg.coef_[0],
            'intercept': reg.intercept_,
            'explained_variance': f"{r_squared:.1%}"
        }
        
        return results
    
    @staticmethod
    def spurious_correlation_test(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict[str, Any]:
        """
        Test for spurious correlation due to common factor z
        """
        # Original correlation
        original_corr, _ = stats.pearsonr(x, y)
        
        # Partial correlation controlling for z
        # Formula: r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1-r_xz²)(1-r_yz²))
        r_xy = original_corr
        r_xz, _ = stats.pearsonr(x, z)
        r_yz, _ = stats.pearsonr(y, z)
        
        numerator = r_xy - r_xz * r_yz
        denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        
        if abs(denominator) < 1e-10:
            partial_corr = 0  # Avoid division by zero
        else:
            partial_corr = numerator / denominator
        
        # Test significance of partial correlation
        n = len(x)
        t_stat = partial_corr * np.sqrt((n - 3) / (1 - partial_corr**2)) if abs(partial_corr) < 1 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-3))
        
        # Interpretation
        correlation_drop = abs(original_corr) - abs(partial_corr)
        is_spurious = (abs(original_corr) > 0.3) and (abs(partial_corr) < 0.1)
        
        return {
            'original_correlation': original_corr,
            'partial_correlation': partial_corr,
            'correlation_drop': correlation_drop,
            'drop_percentage': (correlation_drop / abs(original_corr)) * 100 if abs(original_corr) > 0 else 0,
            'partial_p_value': p_value,
            'is_spurious': is_spurious,
            'common_factor_x_corr': r_xz,
            'common_factor_y_corr': r_yz,
            'interpretation': 'Likely spurious correlation' if is_spurious else 'Correlation remains after controlling for common factor'
        }
    
    @staticmethod
    def granger_causality_test(x: np.ndarray, y: np.ndarray, max_lag: int = 5) -> Dict[str, Any]:
        """
        Granger causality test: Does X help predict Y beyond Y's own lags?
        """
        from statsmodels.tsa.stattools import grangercausalitytests
        
        # Prepare data
        data = pd.DataFrame({'y': y, 'x': x}).dropna()
        
        if len(data) < max_lag * 3:
            raise ValueError("Insufficient data for Granger causality test")
        
        try:
            # Test if X Granger-causes Y
            result_xy = grangercausalitytests(data[['y', 'x']], max_lag, verbose=False)
            
            # Test if Y Granger-causes X  
            result_yx = grangercausalitytests(data[['x', 'y']], max_lag, verbose=False)
            
            # Extract results for optimal lag
            best_lag = 1
            best_p_xy = 1.0
            best_p_yx = 1.0
            
            for lag in range(1, max_lag + 1):
                if lag in result_xy:
                    p_xy = result_xy[lag][0]['ssr_ftest'][1]  # F-test p-value
                    p_yx = result_yx[lag][0]['ssr_ftest'][1]
                    
                    if p_xy < best_p_xy:
                        best_p_xy = p_xy
                        best_lag = lag
                    
                    if p_yx < best_p_yx:
                        best_p_yx = p_yx
            
            # Determine causality direction
            alpha = 0.05
            x_causes_y = best_p_xy < alpha
            y_causes_x = best_p_yx < alpha
            
            if x_causes_y and y_causes_x:
                causality_direction = "Bidirectional"
            elif x_causes_y:
                causality_direction = "X → Y"
            elif y_causes_x:
                causality_direction = "Y → X"
            else:
                causality_direction = "No significant causality"
            
            return {
                'x_granger_causes_y': x_causes_y,
                'y_granger_causes_x': y_causes_x,
                'x_to_y_pvalue': best_p_xy,
                'y_to_x_pvalue': best_p_yx,
                'optimal_lag': best_lag,
                'causality_direction': causality_direction,
                'interpretation': f"At {alpha:.0%} significance: {causality_direction}"
            }
            
        except Exception as e:
            return {
                'error': f"Granger causality test failed: {str(e)}",
                'causality_direction': "Unable to determine"
            }
    
    @staticmethod
    def confounding_variable_analysis(treatment: np.ndarray, 
                                    outcome: np.ndarray,
                                    confounders: np.ndarray) -> Dict[str, Any]:
        """
        Analysis of confounding variables in causal relationships
        """
        # Simple approach: compare coefficients before/after controlling for confounders
        
        # Regression without confounders
        model_simple = LinearRegression().fit(treatment.reshape(-1, 1), outcome)
        coef_simple = model_simple.coef_[0]
        r2_simple = model_simple.score(treatment.reshape(-1, 1), outcome)
        
        # Regression with confounders
        X_full = np.column_stack([treatment, confounders])
        model_full = LinearRegression().fit(X_full, outcome)
        coef_controlled = model_full.coef_[0]  # Coefficient for treatment variable
        r2_controlled = model_full.score(X_full, outcome)
        
        # Calculate confounding bias
        confounding_bias = coef_simple - coef_controlled
        bias_percentage = (confounding_bias / coef_simple) * 100 if abs(coef_simple) > 1e-10 else 0
        
        # Simpson's paradox check
        simpsons_paradox = (np.sign(coef_simple) != np.sign(coef_controlled)) and (abs(coef_controlled) > 1e-10)
        
        return {
            'coefficient_simple': coef_simple,
            'coefficient_controlled': coef_controlled,
            'confounding_bias': confounding_bias,
            'bias_percentage': bias_percentage,
            'r_squared_simple': r2_simple,
            'r_squared_controlled': r2_controlled,
            'simpsons_paradox': simpsons_paradox,
            'interpretation': f"Confounding bias: {bias_percentage:.1f}%" + 
                           (" - Simpson's Paradox detected!" if simpsons_paradox else "")
        }

def demonstrate_correlation_causation():
    """Demonstrate correlation vs causation analysis"""
    
    np.random.seed(42)
    n = 500
    
    # Scenario 1: True causal relationship (X causes Y)
    print("=== Correlation vs Causation Analysis ===\\n")
    
    x_causal = np.random.normal(0, 1, n)
    y_causal = 0.8 * x_causal + np.random.normal(0, 0.5, n)  # Y depends on X
    
    print("1. True Causal Relationship (X → Y):")
    causal_corr = CorrelationCausationAnalysis.correlation_analysis(x_causal, y_causal)
    print(f"   Pearson correlation: {causal_corr['pearson']['correlation']:.3f}")
    print(f"   R-squared: {causal_corr['regression']['r_squared']:.3f}")
    
    # Granger causality test
    granger_result = CorrelationCausationAnalysis.granger_causality_test(x_causal, y_causal)
    print(f"   Granger causality: {granger_result['interpretation']}")
    print()
    
    # Scenario 2: Spurious correlation due to common factor
    print("2. Spurious Correlation (Common Factor Z):")
    z_common = np.random.normal(0, 1, n)  # Common factor
    x_spurious = 0.7 * z_common + np.random.normal(0, 0.7, n)
    y_spurious = 0.6 * z_common + np.random.normal(0, 0.8, n)
    
    spurious_result = CorrelationCausationAnalysis.spurious_correlation_test(
        x_spurious, y_spurious, z_common
    )
    print(f"   Original correlation: {spurious_result['original_correlation']:.3f}")
    print(f"   Partial correlation (controlling for Z): {spurious_result['partial_correlation']:.3f}")
    print(f"   {spurious_result['interpretation']}")
    print()
    
    # Scenario 3: Confounding variable analysis
    print("3. Confounding Variable Analysis:")
    treatment = np.random.normal(0, 1, n)
    confounder = np.random.normal(0, 1, n)
    outcome = 0.5 * treatment + 0.8 * confounder + np.random.normal(0, 0.5, n)
    
    confounding_result = CorrelationCausationAnalysis.confounding_variable_analysis(
        treatment, outcome, confounder.reshape(-1, 1)
    )
    print(f"   Treatment effect (simple): {confounding_result['coefficient_simple']:.3f}")
    print(f"   Treatment effect (controlled): {confounding_result['coefficient_controlled']:.3f}")
    print(f"   {confounding_result['interpretation']}")
    print()
    
    # Scenario 4: Financial example - Stock returns and market sentiment
    print("4. Financial Example: Stock Returns vs Market Sentiment")
    
    # Market factor (common driver)
    market_factor = np.random.normal(0, 1, n)
    
    # Stock returns influenced by market
    stock_returns = 0.6 * market_factor + np.random.normal(0, 0.8, n)
    
    # Market sentiment also influenced by market (with lag)
    sentiment_noise = np.random.normal(0, 0.5, n)
    market_sentiment = np.zeros(n)
    market_sentiment[0] = sentiment_noise[0]
    for i in range(1, n):
        market_sentiment[i] = 0.4 * market_factor[i-1] + 0.3 * market_sentiment[i-1] + sentiment_noise[i]
    
    # Analyze relationship
    financial_corr = CorrelationCausationAnalysis.correlation_analysis(market_sentiment, stock_returns)
    financial_spurious = CorrelationCausationAnalysis.spurious_correlation_test(
        market_sentiment, stock_returns, market_factor
    )
    
    print(f"   Correlation (sentiment vs returns): {financial_corr['pearson']['correlation']:.3f}")
    print(f"   Partial correlation (controlling for market): {financial_spurious['partial_correlation']:.3f}")
    print(f"   Likely spurious: {financial_spurious['is_spurious']}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Causal relationship
    axes[0, 0].scatter(x_causal, y_causal, alpha=0.6)
    axes[0, 0].set_title(f'True Causal Relationship\\nr = {causal_corr["pearson"]["correlation"]:.3f}')
    axes[0, 0].set_xlabel('X (Cause)')
    axes[0, 0].set_ylabel('Y (Effect)')
    
    # Add regression line
    z = np.polyfit(x_causal, y_causal, 1)
    p = np.poly1d(z)
    axes[0, 0].plot(x_causal, p(x_causal), "r--", alpha=0.8)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Spurious correlation
    axes[0, 1].scatter(x_spurious, y_spurious, alpha=0.6, color='orange')
    axes[0, 1].set_title(f'Spurious Correlation\\nr = {spurious_result["original_correlation"]:.3f} → {spurious_result["partial_correlation"]:.3f}')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    
    z_spur = np.polyfit(x_spurious, y_spurious, 1)
    p_spur = np.poly1d(z_spur)
    axes[0, 1].plot(x_spurious, p_spur(x_spurious), "r--", alpha=0.8)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Time series for Granger causality
    time = np.arange(len(x_causal[:100]))
    axes[1, 0].plot(time, x_causal[:100], label='X (Cause)', alpha=0.8)
    axes[1, 0].plot(time, y_causal[:100], label='Y (Effect)', alpha=0.8)
    axes[1, 0].set_title('Time Series: Granger Causality')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Financial example
    axes[1, 1].scatter(market_sentiment, stock_returns, alpha=0.6, color='green')
    axes[1, 1].set_title(f'Financial Example\\nCorrelation: {financial_corr["pearson"]["correlation"]:.3f}')
    axes[1, 1].set_xlabel('Market Sentiment')
    axes[1, 1].set_ylabel('Stock Returns')
    
    z_fin = np.polyfit(market_sentiment, stock_returns, 1)
    p_fin = np.poly1d(z_fin)
    axes[1, 1].plot(market_sentiment, p_fin(market_sentiment), "r--", alpha=0.8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demonstrate_correlation_causation()
```

### 🔴 Theoretical Foundation

**Key Concepts:**

1. **Correlation Coefficient**: $r = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$

2. **Partial Correlation**: Controls for confounding variables
   $$r_{XY \cdot Z} = \frac{r_{XY} - r_{XZ} r_{YZ}}{\sqrt{(1-r_{XZ}^2)(1-r_{YZ}^2)}}$$

3. **Granger Causality**: X Granger-causes Y if past values of X contain information that helps predict Y beyond Y's own past values.

4. **Causal Inference Requirements**:
   - **Temporal precedence**: Cause precedes effect
   - **Covariation**: Changes in cause associate with changes in effect
   - **Non-spuriousness**: Relationship not due to confounding variables

**Bradford Hill Criteria for Causation** (adapted for finance):
- Strength of association
- Consistency across studies
- Temporal sequence
- Biological gradient (dose-response)
- Plausibility
- Coherence with existing knowledge

---

## 📝 Summary

This section covered essential statistical and probability concepts for financial ML/NLP:

1. **Descriptive Statistics**: Understanding financial data characteristics including fat tails, skewness, and volatility clustering

2. **Probability Distributions**: Key distributions used in finance (normal, log-normal, t-distribution, etc.) and their applications

3. **Hypothesis Testing**: Statistical tests for financial modeling including normality tests, stationarity tests, and strategy comparisons

4. **Bayesian Inference**: Incorporating uncertainty and prior knowledge into financial models

5. **Correlation vs Causation**: Critical distinction for avoiding spurious relationships in financial modeling

**Key Takeaways:**
- Financial data rarely follows normal distributions
- Proper statistical testing is essential for model validation
- Bayesian methods help incorporate uncertainty
- Correlation does not imply causation - be careful with spurious relationships
- Understanding these concepts is crucial for building robust ML/NLP models in finance