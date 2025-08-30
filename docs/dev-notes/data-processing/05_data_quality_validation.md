# Data Quality and Validation Framework

This document provides comprehensive guidance for implementing automated data validation frameworks, statistical integrity tests, bias detection, and data lineage tracking for the IPO valuation platform.

## 📊 Overview

Data quality is critical for accurate IPO valuations. This framework ensures that all data entering the valuation models meets strict quality standards through automated validation, statistical testing, and continuous monitoring.

## 🎯 Key Components

- **Automated Validation**: Real-time data quality checks
- **Statistical Integrity**: Advanced statistical tests for data reliability
- **Bias Detection**: Identify and mitigate data bias
- **Data Lineage**: Track data transformations and dependencies
- **Performance Monitoring**: Detect data drift and quality degradation

## 1. Automated Data Validation Framework

### 1.1 Comprehensive Data Validator

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationRule:
    """Definition of a data validation rule."""
    name: str
    description: str
    columns: List[str]
    rule_function: Callable
    severity: ValidationSeverity = ValidationSeverity.ERROR
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

@dataclass
class ValidationResult:
    """Result of a validation check."""
    rule_name: str
    severity: ValidationSeverity
    passed: bool
    message: str
    affected_rows: List[int] = field(default_factory=list)
    affected_columns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class FinancialDataValidator:
    """
    Comprehensive data validation framework for financial data.
    
    Implements rule-based validation with configurable severity levels
    and automated remediation suggestions.
    """
    
    def __init__(self, config_path: str = None):
        self.validation_rules = []
        self.validation_results = []
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config_path:
            self.load_validation_config(config_path)
        else:
            self._setup_default_rules()
        
        # Statistics for ongoing monitoring
        self.validation_history = []
        
    def _setup_default_rules(self):
        """Setup default validation rules for financial data."""
        
        # 1. Completeness Rules
        self.add_rule(ValidationRule(
            name="required_fields_check",
            description="Check that required fields are not null",
            columns=["revenue", "total_assets", "shareholders_equity"],
            rule_function=self._check_required_fields,
            severity=ValidationSeverity.CRITICAL
        ))
        
        # 2. Range and Boundary Rules
        self.add_rule(ValidationRule(
            name="financial_amounts_positive",
            description="Financial amounts should be positive (except profit/loss)",
            columns=["revenue", "total_assets", "cash_and_equivalents"],
            rule_function=self._check_positive_amounts,
            severity=ValidationSeverity.ERROR
        ))
        
        # 3. Relationship Rules
        self.add_rule(ValidationRule(
            name="balance_sheet_equation",
            description="Assets = Liabilities + Equity",
            columns=["total_assets", "total_liabilities", "shareholders_equity"],
            rule_function=self._check_balance_sheet_equation,
            severity=ValidationSeverity.CRITICAL,
            parameters={"tolerance": 0.01}  # 1% tolerance
        ))
        
        # 4. Consistency Rules
        self.add_rule(ValidationRule(
            name="price_consistency",
            description="High >= Low, and both contain Open and Close",
            columns=["open", "high", "low", "close"],
            rule_function=self._check_price_consistency,
            severity=ValidationSeverity.ERROR
        ))
        
        # 5. Statistical Outlier Rules
        self.add_rule(ValidationRule(
            name="statistical_outliers",
            description="Detect statistical outliers in financial ratios",
            columns=["roe", "roa", "debt_to_equity", "current_ratio"],
            rule_function=self._check_statistical_outliers,
            severity=ValidationSeverity.WARNING,
            parameters={"method": "iqr", "multiplier": 3.0}
        ))
        
        # 6. Temporal Consistency Rules
        self.add_rule(ValidationRule(
            name="temporal_consistency",
            description="Check for unrealistic period-over-period changes",
            columns=["revenue", "net_income", "total_assets"],
            rule_function=self._check_temporal_consistency,
            severity=ValidationSeverity.WARNING,
            parameters={"max_change_rate": 10.0}  # 1000% max change
        ))
        
        # 7. Data Type and Format Rules
        self.add_rule(ValidationRule(
            name="data_type_validation",
            description="Validate data types and formats",
            columns=["date", "company_code", "industry"],
            rule_function=self._check_data_types,
            severity=ValidationSeverity.ERROR
        ))
    
    def add_rule(self, rule: ValidationRule):
        """Add a validation rule to the framework."""
        self.validation_rules.append(rule)
    
    def validate_dataframe(self, df: pd.DataFrame,
                          rules_subset: List[str] = None) -> Dict[str, Any]:
        """
        Validate a DataFrame against all applicable rules.
        
        Args:
            df: DataFrame to validate
            rules_subset: Specific rules to run (None = all rules)
            
        Returns:
            Dictionary with validation results and summary
        """
        validation_start = datetime.now()
        results = []
        
        # Filter rules if subset specified
        rules_to_run = self.validation_rules
        if rules_subset:
            rules_to_run = [r for r in self.validation_rules if r.name in rules_subset]
        
        for rule in rules_to_run:
            if not rule.enabled:
                continue
                
            try:
                # Check if required columns exist
                available_cols = [col for col in rule.columns if col in df.columns]
                if not available_cols:
                    continue  # Skip if no relevant columns
                
                # Run validation rule
                result = rule.rule_function(df, available_cols, rule.parameters)
                result.rule_name = rule.name
                result.severity = rule.severity
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error running validation rule '{rule.name}': {e}")
                error_result = ValidationResult(
                    rule_name=rule.name,
                    severity=ValidationSeverity.CRITICAL,
                    passed=False,
                    message=f"Validation rule failed with error: {str(e)}"
                )
                results.append(error_result)
        
        # Store results
        self.validation_results = results
        
        # Generate summary
        summary = self._generate_validation_summary(results, validation_start)
        
        # Store in history
        self.validation_history.append({
            'timestamp': validation_start,
            'summary': summary,
            'results': results
        })
        
        return {
            'summary': summary,
            'results': results,
            'passed_count': sum(1 for r in results if r.passed),
            'failed_count': sum(1 for r in results if not r.passed),
            'recommendations': self._generate_recommendations(results)
        }
    
    def _check_required_fields(self, df: pd.DataFrame, columns: List[str], 
                              params: Dict) -> ValidationResult:
        """Check that required fields are not null."""
        null_counts = df[columns].isnull().sum()
        total_nulls = null_counts.sum()
        
        if total_nulls == 0:
            return ValidationResult(
                rule_name="required_fields_check",
                severity=ValidationSeverity.CRITICAL,
                passed=True,
                message="All required fields are complete"
            )
        else:
            null_details = null_counts[null_counts > 0].to_dict()
            return ValidationResult(
                rule_name="required_fields_check",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Missing values found in required fields: {null_details}",
                metadata={"null_counts": null_details}
            )
    
    def _check_positive_amounts(self, df: pd.DataFrame, columns: List[str], 
                               params: Dict) -> ValidationResult:
        """Check that specified financial amounts are positive."""
        negative_counts = {}
        affected_rows = []
        
        for col in columns:
            if col in df.columns:
                negative_mask = df[col] < 0
                negative_count = negative_mask.sum()
                
                if negative_count > 0:
                    negative_counts[col] = negative_count
                    affected_rows.extend(df[negative_mask].index.tolist())
        
        if not negative_counts:
            return ValidationResult(
                rule_name="financial_amounts_positive",
                severity=ValidationSeverity.ERROR,
                passed=True,
                message="All financial amounts are positive"
            )
        else:
            return ValidationResult(
                rule_name="financial_amounts_positive",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Negative values found: {negative_counts}",
                affected_rows=list(set(affected_rows)),
                metadata={"negative_counts": negative_counts}
            )
    
    def _check_balance_sheet_equation(self, df: pd.DataFrame, columns: List[str], 
                                    params: Dict) -> ValidationResult:
        """Check balance sheet equation: Assets = Liabilities + Equity."""
        required_cols = ["total_assets", "total_liabilities", "shareholders_equity"]
        
        if not all(col in columns for col in required_cols):
            return ValidationResult(
                rule_name="balance_sheet_equation",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message="Missing required columns for balance sheet equation"
            )
        
        tolerance = params.get("tolerance", 0.01)
        
        # Calculate equation difference
        assets = df["total_assets"]
        liabilities_equity = df["total_liabilities"] + df["shareholders_equity"]
        
        # Relative difference
        relative_diff = abs(assets - liabilities_equity) / assets.abs()
        violations = relative_diff > tolerance
        
        violation_count = violations.sum()
        
        if violation_count == 0:
            return ValidationResult(
                rule_name="balance_sheet_equation",
                severity=ValidationSeverity.CRITICAL,
                passed=True,
                message="Balance sheet equation holds within tolerance"
            )
        else:
            affected_rows = df[violations].index.tolist()
            max_violation = relative_diff.max()
            
            return ValidationResult(
                rule_name="balance_sheet_equation",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Balance sheet equation violated in {violation_count} rows (max violation: {max_violation:.2%})",
                affected_rows=affected_rows,
                metadata={"violation_count": violation_count, "max_violation": max_violation}
            )
    
    def _check_price_consistency(self, df: pd.DataFrame, columns: List[str], 
                                params: Dict) -> ValidationResult:
        """Check price data consistency (High >= Low, etc.)."""
        price_cols = ["open", "high", "low", "close"]
        available_price_cols = [col for col in price_cols if col in columns]
        
        if len(available_price_cols) < 4:
            return ValidationResult(
                rule_name="price_consistency",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Insufficient price columns available: {available_price_cols}"
            )
        
        violations = []
        affected_rows = []
        
        # Check High >= Low
        high_low_violations = df["high"] < df["low"]
        if high_low_violations.any():
            violations.append("High < Low")
            affected_rows.extend(df[high_low_violations].index.tolist())
        
        # Check High >= max(Open, Close)
        max_open_close = df[["open", "close"]].max(axis=1)
        high_max_violations = df["high"] < max_open_close
        if high_max_violations.any():
            violations.append("High < max(Open, Close)")
            affected_rows.extend(df[high_max_violations].index.tolist())
        
        # Check Low <= min(Open, Close)
        min_open_close = df[["open", "close"]].min(axis=1)
        low_min_violations = df["low"] > min_open_close
        if low_min_violations.any():
            violations.append("Low > min(Open, Close)")
            affected_rows.extend(df[low_min_violations].index.tolist())
        
        if not violations:
            return ValidationResult(
                rule_name="price_consistency",
                severity=ValidationSeverity.ERROR,
                passed=True,
                message="Price data is consistent"
            )
        else:
            return ValidationResult(
                rule_name="price_consistency",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Price consistency violations: {', '.join(violations)}",
                affected_rows=list(set(affected_rows)),
                metadata={"violation_types": violations}
            )
    
    def _check_statistical_outliers(self, df: pd.DataFrame, columns: List[str], 
                                   params: Dict) -> ValidationResult:
        """Detect statistical outliers in financial data."""
        method = params.get("method", "iqr")
        multiplier = params.get("multiplier", 3.0)
        
        outlier_summary = {}
        all_outlier_rows = []
        
        for col in columns:
            if col not in df.columns:
                continue
                
            data = df[col].dropna()
            if len(data) < 10:  # Need sufficient data
                continue
            
            if method == "iqr":
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                outliers = (data < lower_bound) | (data > upper_bound)
                
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(data))
                outliers = z_scores > multiplier
                
            elif method == "isolation_forest":
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(data.values.reshape(-1, 1))
                outliers = pd.Series(outlier_labels == -1, index=data.index)
            
            outlier_count = outliers.sum()
            if outlier_count > 0:
                outlier_summary[col] = outlier_count
                all_outlier_rows.extend(data[outliers].index.tolist())
        
        total_outliers = sum(outlier_summary.values())
        outlier_percentage = total_outliers / len(df) * 100
        
        if total_outliers == 0:
            return ValidationResult(
                rule_name="statistical_outliers",
                severity=ValidationSeverity.WARNING,
                passed=True,
                message="No statistical outliers detected"
            )
        else:
            return ValidationResult(
                rule_name="statistical_outliers",
                severity=ValidationSeverity.WARNING,
                passed=outlier_percentage < 5.0,  # Pass if < 5% outliers
                message=f"Statistical outliers detected: {outlier_summary} ({outlier_percentage:.1f}% of data)",
                affected_rows=list(set(all_outlier_rows)),
                metadata={
                    "outlier_summary": outlier_summary,
                    "outlier_percentage": outlier_percentage,
                    "method": method
                }
            )
    
    def _check_temporal_consistency(self, df: pd.DataFrame, columns: List[str], 
                                   params: Dict) -> ValidationResult:
        """Check for unrealistic temporal changes."""
        max_change_rate = params.get("max_change_rate", 10.0)  # 1000% max change
        
        if not isinstance(df.index, pd.DatetimeIndex):
            return ValidationResult(
                rule_name="temporal_consistency",
                severity=ValidationSeverity.WARNING,
                passed=False,
                message="DataFrame must have DatetimeIndex for temporal validation"
            )
        
        violation_summary = {}
        affected_rows = []
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Calculate period-over-period changes
            pct_change = df[col].pct_change().abs()
            
            # Find extreme changes
            extreme_changes = pct_change > max_change_rate
            extreme_count = extreme_changes.sum()
            
            if extreme_count > 0:
                violation_summary[col] = extreme_count
                affected_rows.extend(df[extreme_changes].index.tolist())
        
        if not violation_summary:
            return ValidationResult(
                rule_name="temporal_consistency",
                severity=ValidationSeverity.WARNING,
                passed=True,
                message="No unrealistic temporal changes detected"
            )
        else:
            return ValidationResult(
                rule_name="temporal_consistency",
                severity=ValidationSeverity.WARNING,
                passed=False,
                message=f"Unrealistic temporal changes detected: {violation_summary}",
                affected_rows=list(set(affected_rows)),
                metadata={"violation_summary": violation_summary}
            )
    
    def _check_data_types(self, df: pd.DataFrame, columns: List[str], 
                         params: Dict) -> ValidationResult:
        """Validate data types and formats."""
        type_issues = {}
        
        # Expected data types
        expected_types = {
            'date': 'datetime64',
            'company_code': 'object',
            'industry': 'object'
        }
        
        for col in columns:
            if col not in df.columns:
                continue
                
            expected_type = expected_types.get(col)
            actual_type = str(df[col].dtype)
            
            if expected_type and expected_type not in actual_type:
                type_issues[col] = {
                    'expected': expected_type,
                    'actual': actual_type
                }
        
        # Additional format checks
        if 'company_code' in columns and 'company_code' in df.columns:
            # ASX codes should be 3 letters
            invalid_codes = df['company_code'].str.len() != 3
            if invalid_codes.any():
                type_issues['company_code_format'] = f"{invalid_codes.sum()} invalid ASX codes"
        
        if not type_issues:
            return ValidationResult(
                rule_name="data_type_validation",
                severity=ValidationSeverity.ERROR,
                passed=True,
                message="All data types are correct"
            )
        else:
            return ValidationResult(
                rule_name="data_type_validation",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Data type issues found: {type_issues}",
                metadata={"type_issues": type_issues}
            )
    
    def _generate_validation_summary(self, results: List[ValidationResult], 
                                   start_time: datetime) -> Dict:
        """Generate summary of validation results."""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        summary = {
            'total_rules': len(results),
            'passed_rules': sum(1 for r in results if r.passed),
            'failed_rules': sum(1 for r in results if not r.passed),
            'execution_time_seconds': duration,
            'severity_breakdown': {},
            'overall_status': 'PASS'
        }
        
        # Count by severity
        for severity in ValidationSeverity:
            count = sum(1 for r in results if r.severity == severity)
            summary['severity_breakdown'][severity.value] = count
        
        # Determine overall status
        critical_failures = sum(1 for r in results 
                              if r.severity == ValidationSeverity.CRITICAL and not r.passed)
        error_failures = sum(1 for r in results 
                           if r.severity == ValidationSeverity.ERROR and not r.passed)
        
        if critical_failures > 0:
            summary['overall_status'] = 'CRITICAL'
        elif error_failures > 0:
            summary['overall_status'] = 'FAILED'
        elif summary['failed_rules'] > 0:
            summary['overall_status'] = 'WARNING'
        
        return summary
    
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[Dict]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for result in results:
            if not result.passed:
                recommendation = {
                    'rule': result.rule_name,
                    'severity': result.severity.value,
                    'issue': result.message,
                    'suggested_actions': []
                }
                
                # Generate specific recommendations based on rule type
                if result.rule_name == "required_fields_check":
                    recommendation['suggested_actions'] = [
                        "Investigate data source for completeness",
                        "Implement data imputation strategies",
                        "Set up alerts for missing critical data"
                    ]
                elif result.rule_name == "balance_sheet_equation":
                    recommendation['suggested_actions'] = [
                        "Review data extraction and transformation logic",
                        "Check for rounding errors in calculations",
                        "Verify chart of accounts mapping"
                    ]
                elif result.rule_name == "statistical_outliers":
                    recommendation['suggested_actions'] = [
                        "Investigate outlier causes (data errors vs. legitimate values)",
                        "Consider winsorization for extreme values",
                        "Review data collection processes"
                    ]
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def export_validation_report(self, output_path: str, format: str = 'json'):
        """Export validation results to file."""
        if not self.validation_results:
            raise ValueError("No validation results to export")
        
        if format == 'json':
            report_data = {
                'validation_summary': self.validation_history[-1]['summary'],
                'results': [
                    {
                        'rule_name': r.rule_name,
                        'severity': r.severity.value,
                        'passed': r.passed,
                        'message': r.message,
                        'affected_rows_count': len(r.affected_rows),
                        'metadata': r.metadata,
                        'timestamp': r.timestamp.isoformat()
                    }
                    for r in self.validation_results
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
        
        elif format == 'csv':
            # Convert results to DataFrame for CSV export
            results_data = []
            for r in self.validation_results:
                results_data.append({
                    'rule_name': r.rule_name,
                    'severity': r.severity.value,
                    'passed': r.passed,
                    'message': r.message,
                    'affected_rows_count': len(r.affected_rows),
                    'timestamp': r.timestamp
                })
            
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(output_path, index=False)
```

## 2. Statistical Integrity Tests

### 2.1 Advanced Statistical Testing

```python
from scipy import stats
from scipy.stats import ks_2samp, anderson, shapiro, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class StatisticalIntegrityTester:
    """
    Advanced statistical tests for data integrity and quality assessment.
    
    Includes distribution tests, stationarity tests, and financial-specific validations.
    """
    
    def __init__(self):
        self.test_results = {}
        self.significance_level = 0.05
        
    def run_comprehensive_tests(self, df: pd.DataFrame, 
                              target_columns: List[str] = None) -> Dict:
        """
        Run comprehensive statistical integrity tests.
        
        Args:
            df: DataFrame to test
            target_columns: Specific columns to test (None = all numeric)
            
        Returns:
            Dictionary with all test results
        """
        if target_columns is None:
            target_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {
            'distribution_tests': {},
            'normality_tests': {},
            'stationarity_tests': {},
            'autocorrelation_tests': {},
            'homoscedasticity_tests': {},
            'independence_tests': {},
            'summary': {}
        }
        
        for column in target_columns:
            if column not in df.columns:
                continue
                
            series = df[column].dropna()
            if len(series) < 10:  # Need sufficient data
                continue
            
            # Distribution and normality tests
            results['normality_tests'][column] = self._test_normality(series)
            results['distribution_tests'][column] = self._test_distributions(series)
            
            # Time series tests (if datetime index)
            if isinstance(df.index, pd.DatetimeIndex):
                results['stationarity_tests'][column] = self._test_stationarity(series)
                results['autocorrelation_tests'][column] = self._test_autocorrelation(series)
            
            # Heteroscedasticity tests (if sufficient data)
            if len(series) > 20:
                results['homoscedasticity_tests'][column] = self._test_heteroscedasticity(df, column)
        
        # Cross-column tests
        if len(target_columns) > 1:
            results['independence_tests'] = self._test_independence(df[target_columns])
        
        # Generate summary
        results['summary'] = self._generate_statistical_summary(results)
        
        return results
    
    def _test_normality(self, series: pd.Series) -> Dict:
        """Test for normality using multiple methods."""
        results = {}
        
        try:
            # Shapiro-Wilk test (good for small samples)
            if len(series) <= 5000:  # Shapiro-Wilk limitation
                shapiro_stat, shapiro_p = shapiro(series)
                results['shapiro_wilk'] = {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > self.significance_level,
                    'interpretation': 'Normal' if shapiro_p > self.significance_level else 'Not normal'
                }
            
            # Jarque-Bera test
            jb_stat, jb_p, skewness, kurtosis = jarque_bera(series)
            results['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_p,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'is_normal': jb_p > self.significance_level,
                'interpretation': 'Normal' if jb_p > self.significance_level else 'Not normal'
            }
            
            # Anderson-Darling test
            ad_result = anderson(series, dist='norm')
            ad_critical_value = ad_result.critical_values[2]  # 5% significance level
            results['anderson_darling'] = {
                'statistic': ad_result.statistic,
                'critical_value': ad_critical_value,
                'is_normal': ad_result.statistic < ad_critical_value,
                'interpretation': 'Normal' if ad_result.statistic < ad_critical_value else 'Not normal'
            }
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _test_distributions(self, series: pd.Series) -> Dict:
        """Test fit to various distributions."""
        results = {}
        
        # Test common financial distributions
        distributions = {
            'normal': stats.norm,
            'lognormal': stats.lognorm,
            'exponential': stats.expon,
            'gamma': stats.gamma,
            't': stats.t,
            'laplace': stats.laplace
        }
        
        best_fit = {'distribution': None, 'p_value': 0, 'ks_statistic': float('inf')}
        
        for dist_name, distribution in distributions.items():
            try:
                # Fit distribution parameters
                if dist_name == 'normal':
                    params = (series.mean(), series.std())
                elif dist_name == 'exponential':
                    params = (1/series.mean(),)
                else:
                    params = distribution.fit(series)
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.kstest(series, lambda x: distribution.cdf(x, *params))
                
                results[dist_name] = {
                    'parameters': params,
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p,
                    'fits_well': ks_p > self.significance_level
                }
                
                # Track best fit
                if ks_p > best_fit['p_value']:
                    best_fit = {
                        'distribution': dist_name,
                        'p_value': ks_p,
                        'ks_statistic': ks_stat,
                        'parameters': params
                    }
                    
            except Exception as e:
                results[dist_name] = {'error': str(e)}
        
        results['best_fit'] = best_fit
        return results
    
    def _test_stationarity(self, series: pd.Series) -> Dict:
        """Test for stationarity in time series."""
        results = {}
        
        try:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(series, autolag='AIC')
            results['adf'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < self.significance_level,
                'interpretation': 'Stationary' if adf_result[1] < self.significance_level else 'Non-stationary'
            }
            
            # KPSS test
            kpss_result = kpss(series, regression='c')
            results['kpss'] = {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > self.significance_level,
                'interpretation': 'Stationary' if kpss_result[1] > self.significance_level else 'Non-stationary'
            }
            
            # Combined interpretation
            adf_stationary = results['adf']['is_stationary']
            kpss_stationary = results['kpss']['is_stationary']
            
            if adf_stationary and kpss_stationary:
                combined_result = 'Stationary'
            elif not adf_stationary and not kpss_stationary:
                combined_result = 'Non-stationary'
            else:
                combined_result = 'Inconclusive'
            
            results['combined_interpretation'] = combined_result
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _test_autocorrelation(self, series: pd.Series) -> Dict:
        """Test for autocorrelation in time series."""
        results = {}
        
        try:
            # Durbin-Watson test (for first-order autocorrelation)
            dw_stat = durbin_watson(series)
            results['durbin_watson'] = {
                'statistic': dw_stat,
                'interpretation': self._interpret_durbin_watson(dw_stat)
            }
            
            # Ljung-Box test for autocorrelation
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(series, lags=min(10, len(series)//4), return_df=True)
            
            results['ljung_box'] = {
                'statistics': lb_result['lb_stat'].to_dict(),
                'p_values': lb_result['lb_pvalue'].to_dict(),
                'has_autocorrelation': (lb_result['lb_pvalue'] < self.significance_level).any(),
                'interpretation': 'Autocorrelation detected' if (lb_result['lb_pvalue'] < self.significance_level).any() else 'No significant autocorrelation'
            }
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _interpret_durbin_watson(self, dw_stat: float) -> str:
        """Interpret Durbin-Watson statistic."""
        if dw_stat < 1.5:
            return 'Strong positive autocorrelation'
        elif dw_stat < 2.5:
            return 'No autocorrelation'
        else:
            return 'Negative autocorrelation'
    
    def _test_heteroscedasticity(self, df: pd.DataFrame, target_column: str) -> Dict:
        """Test for heteroscedasticity (non-constant variance)."""
        results = {}
        
        try:
            # Need another variable to test against - use index as proxy for time
            if isinstance(df.index, pd.DatetimeIndex):
                time_var = np.arange(len(df))
                y = df[target_column].dropna()
                x = time_var[:len(y)].reshape(-1, 1)
                
                # Fit simple linear regression
                from sklearn.linear_model import LinearRegression
                model = LinearRegression().fit(x, y)
                residuals = y - model.predict(x)
                
                # Breusch-Pagan test
                from statsmodels.stats.diagnostic import het_breuschpagan
                bp_stat, bp_p, f_stat, f_p = het_breuschpagan(residuals, x)
                
                results['breusch_pagan'] = {
                    'statistic': bp_stat,
                    'p_value': bp_p,
                    'f_statistic': f_stat,
                    'f_p_value': f_p,
                    'is_heteroscedastic': bp_p < self.significance_level,
                    'interpretation': 'Heteroscedastic' if bp_p < self.significance_level else 'Homoscedastic'
                }
                
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _test_independence(self, df: pd.DataFrame) -> Dict:
        """Test independence between variables."""
        results = {}
        
        try:
            # Correlation matrix
            corr_matrix = df.corr()
            
            # Find high correlations (> 0.7)
            high_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        high_correlations.append({
                            'variable1': corr_matrix.columns[i],
                            'variable2': corr_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            results['high_correlations'] = high_correlations
            results['max_correlation'] = corr_matrix.abs().max().max()
            
            # Test for multicollinearity using VIF (if multiple variables)
            if len(df.columns) > 2:
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                
                vif_data = pd.DataFrame()
                vif_data["Variable"] = df.columns
                vif_data["VIF"] = [variance_inflation_factor(df.values, i) 
                                  for i in range(len(df.columns))]
                
                results['vif'] = vif_data.to_dict('records')
                results['high_vif_variables'] = vif_data[vif_data['VIF'] > 5]['Variable'].tolist()
                
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _generate_statistical_summary(self, results: Dict) -> Dict:
        """Generate summary of statistical test results."""
        summary = {
            'data_quality_score': 0.0,
            'issues_found': [],
            'recommendations': []
        }
        
        total_tests = 0
        passed_tests = 0
        
        # Analyze normality test results
        for column, normality_result in results.get('normality_tests', {}).items():
            if 'error' not in normality_result:
                total_tests += 1
                if any(test.get('is_normal', False) for test in normality_result.values() if isinstance(test, dict)):
                    passed_tests += 1
                else:
                    summary['issues_found'].append(f"Non-normal distribution in {column}")
                    summary['recommendations'].append(f"Consider transformation for {column}")
        
        # Analyze stationarity test results
        for column, stationarity_result in results.get('stationarity_tests', {}).items():
            if 'error' not in stationarity_result:
                total_tests += 1
                if stationarity_result.get('combined_interpretation') == 'Stationary':
                    passed_tests += 1
                else:
                    summary['issues_found'].append(f"Non-stationary series: {column}")
                    summary['recommendations'].append(f"Consider differencing {column}")
        
        # Calculate overall score
        if total_tests > 0:
            summary['data_quality_score'] = passed_tests / total_tests
        
        return summary
```

This comprehensive data quality and validation framework provides robust testing and monitoring capabilities for the IPO valuation platform. The next sections will cover integration patterns and best practices guides to complete the documentation.