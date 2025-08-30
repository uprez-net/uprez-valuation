# Cost Optimization Guide for GCP AI/ML Services

## Overview

This guide provides comprehensive strategies for optimizing costs across Google Cloud AI/ML services in the IPO valuation platform. We'll cover monitoring, budgeting, resource optimization, and cost-effective architectural patterns.

## Cost Analysis Framework

### 1. Service Cost Breakdown

```python
# cost_analysis/cost_analyzer.py
from google.cloud import billing_v1
from google.cloud import bigquery
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime, timedelta

class GCPCostAnalyzer:
    """Analyze and optimize GCP AI/ML service costs"""
    
    def __init__(self, project_id: str, billing_account_id: str):
        self.project_id = project_id
        self.billing_account_id = billing_account_id
        self.billing_client = billing_v1.CloudBillingClient()
        self.bq_client = bigquery.Client()
    
    async def analyze_ai_ml_costs(
        self,
        analysis_period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze AI/ML service costs over specified period
        
        Args:
            analysis_period_days: Number of days to analyze
            
        Returns:
            Detailed cost breakdown and optimization recommendations
        """
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=analysis_period_days)
        
        # Query billing data
        cost_data = await self._query_billing_data(start_date, end_date)
        
        # Analyze by service
        service_analysis = self._analyze_by_service(cost_data)
        
        # Analyze by resource
        resource_analysis = self._analyze_by_resource(cost_data)
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(
            service_analysis, resource_analysis
        )
        
        return {
            'analysis_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'days': analysis_period_days
            },
            'total_cost': sum(item['cost'] for item in cost_data),
            'service_breakdown': service_analysis,
            'resource_breakdown': resource_analysis,
            'optimization_recommendations': recommendations,
            'cost_trends': await self._analyze_cost_trends(cost_data)
        }
    
    async def _query_billing_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Query billing data from BigQuery export"""
        
        query = f"""
        SELECT
          service.description as service_name,
          sku.description as sku_description,
          usage_start_time,
          usage_end_time,
          location.location as location,
          resource.name as resource_name,
          SUM(cost) as cost,
          SUM(usage.amount) as usage_amount,
          usage.unit as usage_unit,
          
          -- AI/ML specific labels
          labels.key as label_key,
          labels.value as label_value
          
        FROM `{self.project_id}.billing_export.gcp_billing_export_v1_{self.billing_account_id.replace('-', '_')}`
        CROSS JOIN UNNEST(labels) as labels
        WHERE DATE(usage_start_time) BETWEEN '{start_date.date()}' AND '{end_date.date()}'
          AND project.id = '{self.project_id}'
          AND (
            service.description LIKE '%AI Platform%' OR
            service.description LIKE '%Vertex AI%' OR
            service.description LIKE '%Document AI%' OR
            service.description LIKE '%Natural Language%' OR
            service.description LIKE '%BigQuery%' OR
            service.description LIKE '%Cloud Storage%'
          )
        GROUP BY 1,2,3,4,5,6,8,9,10,11
        ORDER BY cost DESC
        """
        
        df = self.bq_client.query(query).to_dataframe()
        return df.to_dict('records')
    
    def _analyze_by_service(self, cost_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze costs by AI/ML service"""
        
        service_costs = {}
        
        for item in cost_data:
            service = item['service_name']
            cost = item['cost']
            
            if service not in service_costs:
                service_costs[service] = {
                    'total_cost': 0,
                    'usage_details': [],
                    'optimization_potential': 0
                }
            
            service_costs[service]['total_cost'] += cost
            service_costs[service]['usage_details'].append({
                'sku': item['sku_description'],
                'cost': cost,
                'usage_amount': item['usage_amount'],
                'usage_unit': item['usage_unit'],
                'location': item['location']
            })
        
        # Calculate optimization potential for each service
        for service, data in service_costs.items():
            data['optimization_potential'] = self._calculate_optimization_potential(
                service, data['usage_details']
            )
        
        return service_costs
    
    def _calculate_optimization_potential(
        self,
        service_name: str,
        usage_details: List[Dict[str, Any]]
    ) -> float:
        """Calculate potential cost savings for a service"""
        
        optimization_strategies = {
            'Vertex AI': {
                'auto_scaling': 0.25,  # 25% potential savings
                'resource_optimization': 0.15,
                'batch_processing': 0.20
            },
            'Document AI': {
                'batch_processing': 0.30,
                'caching': 0.15,
                'preprocessing': 0.10
            },
            'BigQuery': {
                'query_optimization': 0.40,
                'partitioning': 0.25,
                'clustering': 0.15
            },
            'Cloud Storage': {
                'lifecycle_management': 0.30,
                'compression': 0.10,
                'storage_class_optimization': 0.20
            }
        }
        
        if service_name in optimization_strategies:
            strategies = optimization_strategies[service_name]
            max_savings_rate = max(strategies.values())
            
            # Calculate potential savings based on usage patterns
            total_cost = sum(item['cost'] for item in usage_details)
            potential_savings = total_cost * max_savings_rate
            
            return potential_savings
        
        return 0.0
```

### 2. Budget Management

```python
# budgets/budget_manager.py
from google.cloud import billing_budgets_v1
from typing import Dict, Any, List
import json

class BudgetManager:
    """Manage budgets and cost alerts for AI/ML services"""
    
    def __init__(self, billing_account_id: str, project_id: str):
        self.billing_account_id = billing_account_id
        self.project_id = project_id
        self.client = billing_budgets_v1.BudgetServiceClient()
    
    async def create_ai_ml_budget(
        self,
        budget_name: str,
        monthly_amount: float,
        alert_thresholds: List[float],
        notification_channel: str
    ) -> str:
        """
        Create budget for AI/ML services
        
        Args:
            budget_name: Name for the budget
            monthly_amount: Monthly budget amount in USD
            alert_thresholds: List of threshold percentages (e.g., [0.5, 0.8, 1.0])
            notification_channel: Pub/Sub topic for notifications
            
        Returns:
            Budget resource name
        """
        
        # Define budget filter for AI/ML services
        budget_filter = billing_budgets_v1.Filter(
            projects=[f"projects/{self.project_id}"],
            services=[
                "services/2062F6B3-2E7E-4ACB-B000-B5D4E4AA45C3",  # Vertex AI
                "services/3F8F1F7A-7B8F-4B5A-8A3E-5C5B5A3B4C2D",  # Document AI
                "services/24E6F5A3-B3A1-4B6F-9D95-BF9A1C4C8F2E",  # Natural Language AI
                "services/24E6F5A3-B3A1-4B6F-9D95-BF9F1C3C3E2E"   # BigQuery
            ]
        )
        
        # Create budget amount
        budget_amount = billing_budgets_v1.BudgetAmount(
            specified_amount=billing_budgets_v1.Money(
                currency_code="USD",
                units=int(monthly_amount)
            )
        )
        
        # Create threshold rules
        threshold_rules = []
        for threshold in alert_thresholds:
            threshold_rules.append(
                billing_budgets_v1.ThresholdRule(
                    threshold_percent=threshold,
                    spend_basis=billing_budgets_v1.ThresholdRule.Basis.CURRENT_SPEND
                )
            )
        
        # Create notification channels
        notifications = billing_budgets_v1.NotificationsRule(
            pubsub_topic=notification_channel,
            monitoring_notification_channels=[],
            disable_default_iam_recipients=False
        )
        
        # Create budget
        budget = billing_budgets_v1.Budget(
            display_name=budget_name,
            budget_filter=budget_filter,
            amount=budget_amount,
            threshold_rules=threshold_rules,
            notifications_rule=notifications
        )
        
        # Submit budget creation request
        parent = f"billingAccounts/{self.billing_account_id}"
        request = billing_budgets_v1.CreateBudgetRequest(
            parent=parent,
            budget=budget
        )
        
        response = self.client.create_budget(request=request)
        
        return response.name
    
    async def setup_cost_anomaly_detection(
        self,
        anomaly_config: Dict[str, Any]
    ) -> str:
        """Setup cost anomaly detection for AI/ML services"""
        
        # Create monitoring for unusual spending patterns
        anomaly_query = f"""
        WITH daily_costs AS (
          SELECT
            DATE(usage_start_time) as usage_date,
            service.description as service_name,
            SUM(cost) as daily_cost
          FROM `{self.project_id}.billing_export.gcp_billing_export_v1_{self.billing_account_id.replace('-', '_')}`
          WHERE project.id = '{self.project_id}'
            AND DATE(usage_start_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
            AND (
              service.description LIKE '%AI Platform%' OR
              service.description LIKE '%Vertex AI%' OR
              service.description LIKE '%Document AI%'
            )
          GROUP BY usage_date, service_name
        ),
        
        cost_stats AS (
          SELECT
            service_name,
            AVG(daily_cost) as avg_daily_cost,
            STDDEV(daily_cost) as cost_stddev
          FROM daily_costs
          WHERE usage_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY)
          GROUP BY service_name
        ),
        
        anomalies AS (
          SELECT
            dc.usage_date,
            dc.service_name,
            dc.daily_cost,
            cs.avg_daily_cost,
            cs.cost_stddev,
            ABS(dc.daily_cost - cs.avg_daily_cost) / cs.cost_stddev as z_score
          FROM daily_costs dc
          JOIN cost_stats cs ON dc.service_name = cs.service_name
          WHERE dc.usage_date = CURRENT_DATE()
            AND ABS(dc.daily_cost - cs.avg_daily_cost) / cs.cost_stddev > 2
        )
        
        SELECT
          service_name,
          daily_cost,
          avg_daily_cost,
          z_score,
          CASE 
            WHEN z_score > 3 THEN 'HIGH'
            WHEN z_score > 2 THEN 'MEDIUM'
            ELSE 'LOW'
          END as anomaly_severity
        FROM anomalies
        ORDER BY z_score DESC
        """
        
        # Execute anomaly detection query
        df = self.bq_client.query(anomaly_query).to_dataframe()
        
        return f"Anomaly detection setup complete. Found {len(df)} potential anomalies."
```

## Resource Optimization Strategies

### 1. Vertex AI Cost Optimization

```python
# optimization/vertex_ai_optimizer.py
from google.cloud import aiplatform
from typing import Dict, Any, List
import numpy as np

class VertexAIOptimizer:
    """Optimize Vertex AI resource usage and costs"""
    
    def __init__(self, project_id: str, region: str):
        self.project_id = project_id
        self.region = region
        aiplatform.init(project=project_id, location=region)
    
    async def optimize_training_costs(
        self,
        training_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize training job costs"""
        
        optimization_strategies = {
            'machine_type_optimization': await self._optimize_machine_types(training_config),
            'preemptible_instances': await self._evaluate_preemptible_usage(training_config),
            'batch_size_optimization': await self._optimize_batch_sizes(training_config),
            'hyperparameter_efficiency': await self._optimize_hyperparameter_tuning(training_config)
        }
        
        return optimization_strategies
    
    async def _optimize_machine_types(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend optimal machine types for training workloads"""
        
        # Machine type cost analysis
        machine_costs = {
            'n1-standard-4': {'cpu': 4, 'memory': 15, 'hourly_cost': 0.19},
            'n1-standard-8': {'cpu': 8, 'memory': 30, 'hourly_cost': 0.38},
            'n1-highmem-2': {'cpu': 2, 'memory': 13, 'hourly_cost': 0.16},
            'n1-highmem-4': {'cpu': 4, 'memory': 26, 'hourly_cost': 0.32},
            'c2-standard-4': {'cpu': 4, 'memory': 16, 'hourly_cost': 0.21},
            'c2-standard-8': {'cpu': 8, 'memory': 32, 'hourly_cost': 0.42}
        }
        
        # GPU options
        gpu_costs = {
            'NVIDIA_TESLA_T4': {'hourly_cost': 0.35, 'memory_gb': 16},
            'NVIDIA_TESLA_V100': {'hourly_cost': 2.48, 'memory_gb': 32},
            'NVIDIA_TESLA_P4': {'hourly_cost': 0.60, 'memory_gb': 8}
        }
        
        # Analyze workload requirements
        workload_analysis = {
            'model_size_gb': config.get('model_size_gb', 1.0),
            'training_data_gb': config.get('training_data_gb', 10.0),
            'expected_training_hours': config.get('expected_training_hours', 4.0),
            'requires_gpu': config.get('requires_gpu', False),
            'memory_intensive': config.get('memory_intensive', False)
        }
        
        # Recommend optimal configuration
        recommended_config = self._select_optimal_machine_type(
            workload_analysis, machine_costs, gpu_costs
        )
        
        return {
            'current_config': config,
            'recommended_config': recommended_config,
            'cost_savings': self._calculate_cost_savings(config, recommended_config),
            'reasoning': self._explain_machine_type_selection(recommended_config)
        }
    
    def _select_optimal_machine_type(
        self,
        workload: Dict[str, Any],
        machine_costs: Dict[str, Any],
        gpu_costs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Select optimal machine type based on workload requirements"""
        
        # Memory requirement calculation
        memory_needed = max(
            workload['model_size_gb'] * 2,  # 2x model size for training
            workload['training_data_gb'] * 0.5,  # 50% of data size in memory
            4  # Minimum 4GB
        )
        
        # Filter machines by memory requirement
        suitable_machines = {
            name: specs for name, specs in machine_costs.items()
            if specs['memory'] >= memory_needed
        }
        
        if not suitable_machines:
            # Use highest memory machine if none meet requirements
            suitable_machines = {
                max(machine_costs.keys(), key=lambda x: machine_costs[x]['memory']): 
                machine_costs[max(machine_costs.keys(), key=lambda x: machine_costs[x]['memory'])]
            }
        
        # Select most cost-effective option
        optimal_machine = min(
            suitable_machines.keys(),
            key=lambda x: suitable_machines[x]['hourly_cost']
        )
        
        config = {
            'machine_type': optimal_machine,
            'accelerator_type': None,
            'accelerator_count': 0
        }
        
        # Add GPU if needed and cost-effective
        if workload['requires_gpu']:
            # Select most cost-effective GPU
            optimal_gpu = min(gpu_costs.keys(), key=lambda x: gpu_costs[x]['hourly_cost'])
            config.update({
                'accelerator_type': optimal_gpu,
                'accelerator_count': 1
            })
        
        # Calculate total cost
        base_cost = suitable_machines[optimal_machine]['hourly_cost']
        gpu_cost = gpu_costs.get(config['accelerator_type'], {}).get('hourly_cost', 0) * config['accelerator_count']
        total_hourly_cost = base_cost + gpu_cost
        
        config['estimated_hourly_cost'] = total_hourly_cost
        config['estimated_training_cost'] = total_hourly_cost * workload['expected_training_hours']
        
        return config
    
    async def _evaluate_preemptible_usage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate cost savings from preemptible instances"""
        
        preemptible_savings = 0.80  # 80% cost reduction
        
        # Analyze job characteristics
        job_duration_hours = config.get('expected_training_hours', 4.0)
        checkpoint_frequency = config.get('checkpoint_frequency_minutes', 30)
        fault_tolerance = config.get('fault_tolerance', 'medium')
        
        # Determine preemptible suitability
        suitability_score = 0
        
        # Shorter jobs are better for preemptible
        if job_duration_hours <= 2:
            suitability_score += 0.4
        elif job_duration_hours <= 6:
            suitability_score += 0.2
        
        # Frequent checkpointing helps
        if checkpoint_frequency <= 15:
            suitability_score += 0.3
        elif checkpoint_frequency <= 30:
            suitability_score += 0.2
        
        # Fault tolerance
        if fault_tolerance == 'high':
            suitability_score += 0.3
        elif fault_tolerance == 'medium':
            suitability_score += 0.1
        
        recommendation = {
            'use_preemptible': suitability_score >= 0.6,
            'suitability_score': suitability_score,
            'potential_savings_percent': preemptible_savings * 100 if suitability_score >= 0.6 else 0,
            'estimated_interruption_risk': self._calculate_interruption_risk(job_duration_hours),
            'recommendations': []
        }
        
        if suitability_score >= 0.6:
            recommendation['recommendations'].append("Use preemptible instances for training")
            if checkpoint_frequency > 15:
                recommendation['recommendations'].append("Increase checkpoint frequency to 10-15 minutes")
        else:
            recommendation['recommendations'].append("Standard instances recommended due to job characteristics")
            if checkpoint_frequency > 30:
                recommendation['recommendations'].append("Improve fault tolerance with more frequent checkpointing")
        
        return recommendation
    
    def _calculate_interruption_risk(self, job_duration_hours: float) -> float:
        """Calculate risk of preemptible instance interruption"""
        
        # Preemptible instances have ~5% hourly interruption rate
        hourly_survival_rate = 0.95
        job_survival_probability = hourly_survival_rate ** job_duration_hours
        interruption_risk = 1 - job_survival_probability
        
        return interruption_risk
```

### 2. BigQuery Cost Optimization

```python
# optimization/bigquery_optimizer.py
from google.cloud import bigquery
from typing import Dict, List, Any
import re

class BigQueryOptimizer:
    """Optimize BigQuery costs for ML workloads"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = bigquery.Client()
    
    async def analyze_query_costs(
        self,
        analysis_days: int = 7
    ) -> Dict[str, Any]:
        """Analyze BigQuery query costs and optimization opportunities"""
        
        # Query job history
        query = f"""
        SELECT
          query,
          total_bytes_processed,
          total_bytes_billed,
          total_slot_ms,
          creation_time,
          user_email,
          job_id,
          
          -- Cost estimation (approximate)
          (total_bytes_billed / 1024 / 1024 / 1024 / 1024) * 5 as estimated_cost_usd,
          
          -- Efficiency metrics
          SAFE_DIVIDE(total_bytes_processed, total_slot_ms) as bytes_per_slot_ms
          
        FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
        WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {analysis_days} DAY)
          AND state = 'DONE'
          AND job_type = 'QUERY'
          AND total_bytes_billed > 0
        ORDER BY total_bytes_billed DESC
        LIMIT 100
        """
        
        df = self.client.query(query).to_dataframe()
        
        # Analyze expensive queries
        expensive_queries = df.nlargest(20, 'estimated_cost_usd')
        
        # Generate optimization recommendations
        optimizations = []
        for _, row in expensive_queries.iterrows():
            query_text = row['query']
            recommendations = self._analyze_query_for_optimization(query_text)
            
            optimizations.append({
                'job_id': row['job_id'],
                'estimated_cost': row['estimated_cost_usd'],
                'bytes_processed': row['total_bytes_processed'],
                'recommendations': recommendations
            })
        
        return {
            'total_queries_analyzed': len(df),
            'total_estimated_cost': df['estimated_cost_usd'].sum(),
            'average_query_cost': df['estimated_cost_usd'].mean(),
            'expensive_queries': optimizations,
            'optimization_summary': self._generate_optimization_summary(optimizations)
        }
    
    def _analyze_query_for_optimization(self, query: str) -> List[str]:
        """Analyze individual query for optimization opportunities"""
        
        recommendations = []
        query_lower = query.lower()
        
        # Check for SELECT * usage
        if re.search(r'select\s+\*', query_lower):
            recommendations.append("Avoid SELECT *, specify only required columns")
        
        # Check for missing WHERE clauses on large tables
        if 'where' not in query_lower and any(table in query_lower for table in ['financial_data', 'market_data']):
            recommendations.append("Add WHERE clause to filter data and reduce scanning")
        
        # Check for unpartitioned table scans
        if not re.search(r'where.*date.*between|where.*timestamp.*>', query_lower):
            recommendations.append("Use partitioning predicates in WHERE clause")
        
        # Check for ORDER BY without LIMIT
        if 'order by' in query_lower and 'limit' not in query_lower:
            recommendations.append("Add LIMIT clause when using ORDER BY")
        
        # Check for inefficient JOINs
        if re.search(r'join\s+(?!.*on)', query_lower):
            recommendations.append("Ensure JOINs have proper ON conditions")
        
        # Check for window functions without partitioning
        if 'over (' in query_lower and 'partition by' not in query_lower:
            recommendations.append("Add PARTITION BY to window functions for better performance")
        
        # Check for unnecessary GROUP BY
        if 'group by' in query_lower and 'sum(' not in query_lower and 'count(' not in query_lower:
            recommendations.append("Review if GROUP BY is necessary without aggregations")
        
        return recommendations
    
    async def implement_cost_optimization(
        self,
        table_name: str,
        optimization_type: str
    ) -> Dict[str, Any]:
        """Implement specific cost optimization"""
        
        if optimization_type == 'partitioning':
            return await self._implement_table_partitioning(table_name)
        elif optimization_type == 'clustering':
            return await self._implement_table_clustering(table_name)
        elif optimization_type == 'lifecycle':
            return await self._implement_lifecycle_management(table_name)
        else:
            return {'error': f'Unknown optimization type: {optimization_type}'}
    
    async def _implement_table_partitioning(self, table_name: str) -> Dict[str, Any]:
        """Implement table partitioning for cost optimization"""
        
        # Analyze table structure
        table_ref = self.client.get_table(table_name)
        
        # Suggest partitioning strategy
        partitioning_recommendations = []
        
        # Look for date/timestamp columns
        date_columns = [field.name for field in table_ref.schema 
                       if field.field_type in ['DATE', 'TIMESTAMP', 'DATETIME']]
        
        if date_columns:
            partitioning_recommendations.append({
                'strategy': 'time_partitioning',
                'column': date_columns[0],
                'type': 'DAY',
                'benefits': [
                    'Automatic partition pruning',
                    'Reduced scan costs',
                    'Improved query performance'
                ]
            })
        
        # Look for high-cardinality columns for range partitioning
        # This would require additional analysis of column statistics
        
        return {
            'table_name': table_name,
            'current_partitioning': table_ref.time_partitioning,
            'recommendations': partitioning_recommendations,
            'estimated_cost_savings': '30-70% for time-filtered queries'
        }
```

### 3. Document AI Cost Optimization

```python
# optimization/document_ai_optimizer.py
from google.cloud import documentai
from typing import Dict, Any, List
import asyncio

class DocumentAIOptimizer:
    """Optimize Document AI processing costs"""
    
    def __init__(self, project_id: str, location: str):
        self.project_id = project_id
        self.location = location
        self.client = documentai.DocumentProcessorServiceClient()
    
    async def optimize_document_processing(
        self,
        processing_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize document processing costs"""
        
        optimizations = {
            'batch_processing': await self._analyze_batch_opportunities(processing_config),
            'caching_strategy': await self._analyze_caching_opportunities(processing_config),
            'preprocessing': await self._analyze_preprocessing_benefits(processing_config),
            'processor_selection': await self._optimize_processor_selection(processing_config)
        }
        
        return optimizations
    
    async def _analyze_batch_opportunities(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze opportunities for batch processing"""
        
        current_volume = config.get('daily_document_count', 100)
        avg_document_size = config.get('avg_document_size_pages', 5)
        
        # Batch processing pricing tiers
        batch_tiers = {
            'tier_1': {'min_pages': 1, 'max_pages': 5, 'cost_per_page': 0.0015},
            'tier_2': {'min_pages': 6, 'max_pages': 20, 'cost_per_page': 0.0010},
            'tier_3': {'min_pages': 21, 'max_pages': float('inf'), 'cost_per_page': 0.0005}
        }
        
        # Calculate current costs (online processing)
        online_cost_per_page = 0.002  # Online processing cost
        current_monthly_cost = current_volume * avg_document_size * online_cost_per_page * 30
        
        # Calculate batch processing costs
        batch_cost_per_page = self._get_batch_cost_per_page(avg_document_size, batch_tiers)
        batch_monthly_cost = current_volume * avg_document_size * batch_cost_per_page * 30
        
        savings = current_monthly_cost - batch_monthly_cost
        savings_percentage = (savings / current_monthly_cost) * 100 if current_monthly_cost > 0 else 0
        
        return {
            'current_monthly_cost': current_monthly_cost,
            'batch_monthly_cost': batch_monthly_cost,
            'monthly_savings': savings,
            'savings_percentage': savings_percentage,
            'recommendation': 'Use batch processing' if savings > 0 else 'Continue with online processing',
            'batch_configuration': {
                'batch_size': min(50, current_volume),  # Optimize batch size
                'processing_frequency': 'hourly' if current_volume > 500 else 'daily'
            }
        }
    
    def _get_batch_cost_per_page(
        self,
        avg_pages: int,
        batch_tiers: Dict[str, Any]
    ) -> float:
        """Get cost per page for batch processing"""
        
        for tier_name, tier_config in batch_tiers.items():
            if tier_config['min_pages'] <= avg_pages <= tier_config['max_pages']:
                return tier_config['cost_per_page']
        
        # Default to highest tier
        return batch_tiers['tier_3']['cost_per_page']
    
    async def _analyze_caching_opportunities(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document processing caching opportunities"""
        
        # Estimate document similarity for caching potential
        duplicate_rate = config.get('estimated_duplicate_rate', 0.15)  # 15% duplicates
        similar_document_rate = config.get('similar_document_rate', 0.25)  # 25% similar
        
        daily_volume = config.get('daily_document_count', 100)
        processing_cost_per_doc = config.get('processing_cost_per_doc', 0.01)
        
        # Calculate potential savings
        cache_hit_rate = duplicate_rate + (similar_document_rate * 0.5)  # 50% of similar docs cacheable
        daily_cacheable_docs = daily_volume * cache_hit_rate
        daily_savings = daily_cacheable_docs * processing_cost_per_doc
        monthly_savings = daily_savings * 30
        
        # Caching infrastructure costs
        cache_storage_cost = 0.020 * 30  # $0.02/GB/month for Redis
        cache_maintenance_cost = 50  # Monthly maintenance
        total_cache_cost = cache_storage_cost + cache_maintenance_cost
        
        net_savings = monthly_savings - total_cache_cost
        
        return {
            'estimated_cache_hit_rate': cache_hit_rate,
            'monthly_processing_savings': monthly_savings,
            'cache_infrastructure_cost': total_cache_cost,
            'net_monthly_savings': net_savings,
            'roi': net_savings / total_cache_cost if total_cache_cost > 0 else 0,
            'recommendation': 'Implement caching' if net_savings > 0 else 'Caching not cost-effective',
            'cache_strategy': {
                'cache_duration_hours': 24,
                'cache_key_strategy': 'document_hash',
                'storage_type': 'Redis'
            }
        }
```

## Budget Monitoring and Alerts

### 1. Real-time Cost Monitoring

```python
# monitoring/cost_monitor.py
from google.cloud import monitoring_v3, pubsub_v1
import json
from typing import Dict, Any

class CostMonitor:
    """Monitor AI/ML service costs in real-time"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        self.subscriber = pubsub_v1.SubscriberClient()
    
    async def setup_cost_alerts(
        self,
        alert_config: Dict[str, Any]
    ) -> List[str]:
        """Setup cost monitoring alerts"""
        
        alert_policies = []
        
        # Daily cost threshold alert
        daily_policy = await self._create_daily_cost_alert(
            threshold=alert_config.get('daily_threshold', 100),
            notification_channels=alert_config.get('notification_channels', [])
        )
        alert_policies.append(daily_policy)
        
        # Service-specific alerts
        for service, threshold in alert_config.get('service_thresholds', {}).items():
            service_policy = await self._create_service_cost_alert(
                service, threshold, alert_config.get('notification_channels', [])
            )
            alert_policies.append(service_policy)
        
        # Anomaly detection alert
        anomaly_policy = await self._create_cost_anomaly_alert(
            alert_config.get('anomaly_sensitivity', 2.0),
            alert_config.get('notification_channels', [])
        )
        alert_policies.append(anomaly_policy)
        
        return alert_policies
    
    async def _create_daily_cost_alert(
        self,
        threshold: float,
        notification_channels: List[str]
    ) -> str:
        """Create daily cost threshold alert"""
        
        # Define alert policy
        alert_policy = monitoring_v3.AlertPolicy(
            display_name="Daily AI/ML Cost Threshold",
            documentation=monitoring_v3.AlertPolicy.Documentation(
                content=f"Alert when daily AI/ML costs exceed ${threshold}",
                mime_type="text/markdown"
            ),
            conditions=[
                monitoring_v3.AlertPolicy.Condition(
                    display_name="Daily cost exceeds threshold",
                    condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                        filter='resource.type="billing_account" AND '
                               f'metric.type="billing.googleapis.com/billing/amount" AND '
                               f'metric.labels.currency="USD"',
                        comparison=monitoring_v3.ComparisonType.COMPARISON_GREATER_THAN,
                        threshold_value=threshold,
                        duration={"seconds": 300},  # 5 minutes
                        aggregations=[
                            monitoring_v3.Aggregation(
                                alignment_period={"seconds": 86400},  # 1 day
                                per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_SUM
                            )
                        ]
                    )
                )
            ],
            notification_channels=notification_channels,
            alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
                auto_close={"seconds": 86400}  # Auto-close after 1 day
            )
        )
        
        # Create alert policy
        project_name = f"projects/{self.project_id}"
        created_policy = self.monitoring_client.create_alert_policy(
            name=project_name,
            alert_policy=alert_policy
        )
        
        return created_policy.name
    
    def start_cost_monitoring(self, subscription_name: str):
        """Start real-time cost monitoring"""
        
        subscription_path = self.subscriber.subscription_path(
            self.project_id, subscription_name
        )
        
        def callback(message):
            try:
                # Parse budget alert message
                budget_data = json.loads(message.data.decode('utf-8'))
                
                # Process cost alert
                asyncio.run(self._process_cost_alert(budget_data))
                
                message.ack()
                
            except Exception as e:
                print(f"Error processing cost alert: {e}")
                message.nack()
        
        # Start listening for budget alerts
        streaming_pull_future = self.subscriber.subscribe(
            subscription_path,
            callback=callback
        )
        
        print(f"Listening for cost alerts on {subscription_path}")
        
        try:
            streaming_pull_future.result()
        except KeyboardInterrupt:
            streaming_pull_future.cancel()
    
    async def _process_cost_alert(self, budget_data: Dict[str, Any]):
        """Process budget alert and take action"""
        
        alert_threshold = budget_data.get('alertThresholdExceeded', {})
        cost_amount = budget_data.get('costAmount', 0)
        budget_amount = budget_data.get('budgetAmount', 0)
        
        if alert_threshold:
            threshold_percent = alert_threshold.get('thresholdPercent', 0)
            
            # Take action based on threshold
            if threshold_percent >= 0.9:  # 90% threshold
                await self._emergency_cost_mitigation()
            elif threshold_percent >= 0.8:  # 80% threshold
                await self._aggressive_cost_optimization()
            elif threshold_percent >= 0.5:  # 50% threshold
                await self._standard_cost_review()
    
    async def _emergency_cost_mitigation(self):
        """Emergency cost mitigation actions"""
        
        actions_taken = []
        
        # Pause non-critical training jobs
        training_jobs = aiplatform.CustomJob.list(
            filter='state=JOB_STATE_RUNNING'
        )
        
        for job in training_jobs:
            if 'non_critical' in job.labels:
                job.cancel()
                actions_taken.append(f"Cancelled non-critical training job: {job.display_name}")
        
        # Scale down endpoints
        endpoints = aiplatform.Endpoint.list()
        for endpoint in endpoints:
            if 'auto_scale' in endpoint.labels:
                # Reduce to minimum replicas
                deployed_models = endpoint.list_models()
                for model in deployed_models:
                    endpoint.update(
                        deployed_models=[{
                            'id': model.id,
                            'min_replica_count': 1,
                            'max_replica_count': 1
                        }]
                    )
                actions_taken.append(f"Scaled down endpoint: {endpoint.display_name}")
        
        # Log emergency actions
        print("Emergency cost mitigation actions taken:")
        for action in actions_taken:
            print(f"  - {action}")
```

## Cost Optimization Implementation Templates

### 1. Auto-scaling Configuration

```yaml
# config/cost-optimization/auto-scaling.yaml
vertex_ai_endpoints:
  production:
    ipo-valuation-endpoint:
      min_replicas: 1
      max_replicas: 5
      cpu_utilization_target: 70
      scale_down_delay: 300  # 5 minutes
      scale_up_delay: 60     # 1 minute
      
  development:
    ipo-valuation-dev-endpoint:
      min_replicas: 0  # Scale to zero when not in use
      max_replicas: 2
      cpu_utilization_target: 80
      
document_ai_processing:
  batch_processing:
    enable: true
    batch_size: 20
    max_wait_time_seconds: 300
    cost_savings_target: 30  # 30% cost reduction
    
  caching:
    enable: true
    cache_duration_hours: 24
    cache_hit_rate_target: 40  # 40% cache hit rate
    
bigquery_optimization:
  query_optimization:
    enable_query_cache: true
    require_partition_filter: true
    max_bytes_billed: 1073741824  # 1GB limit
    
  storage_optimization:
    enable_table_expiration: true
    default_expiration_days: 365
    archive_to_coldline_days: 90
```

### 2. Cost Monitoring Dashboard

```python
# dashboards/cost_dashboard.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from cost_analysis.cost_analyzer import GCPCostAnalyzer

class CostOptimizationDashboard:
    """Interactive dashboard for cost monitoring and optimization"""
    
    def __init__(self, project_id: str, billing_account_id: str):
        self.cost_analyzer = GCPCostAnalyzer(project_id, billing_account_id)
    
    def render_dashboard(self):
        """Render the cost optimization dashboard"""
        
        st.title("GCP AI/ML Cost Optimization Dashboard")
        
        # Sidebar controls
        st.sidebar.header("Analysis Configuration")
        analysis_days = st.sidebar.slider("Analysis Period (days)", 7, 90, 30)
        
        # Load cost data
        with st.spinner("Loading cost data..."):
            cost_analysis = asyncio.run(
                self.cost_analyzer.analyze_ai_ml_costs(analysis_days)
            )
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Cost",
                f"${cost_analysis['total_cost']:.2f}",
                delta=f"{cost_analysis.get('cost_change_percent', 0):.1f}%"
            )
        
        with col2:
            st.metric(
                "Daily Average",
                f"${cost_analysis['total_cost'] / analysis_days:.2f}"
            )
        
        with col3:
            potential_savings = sum(
                rec.get('estimated_savings', 0) 
                for rec in cost_analysis['optimization_recommendations']
            )
            st.metric(
                "Potential Savings",
                f"${potential_savings:.2f}",
                delta=f"{(potential_savings / cost_analysis['total_cost']) * 100:.1f}%"
            )
        
        with col4:
            st.metric(
                "Optimization Score",
                f"{cost_analysis.get('optimization_score', 75)}/100"
            )
        
        # Service cost breakdown
        st.subheader("Cost by Service")
        service_data = cost_analysis['service_breakdown']
        
        fig_pie = px.pie(
            values=[data['total_cost'] for data in service_data.values()],
            names=list(service_data.keys()),
            title="Cost Distribution by Service"
        )
        st.plotly_chart(fig_pie)
        
        # Cost trends
        st.subheader("Cost Trends")
        if 'cost_trends' in cost_analysis:
            trend_data = cost_analysis['cost_trends']
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=trend_data['dates'],
                y=trend_data['daily_costs'],
                mode='lines+markers',
                name='Daily Cost'
            ))
            fig_trend.update_layout(
                title="Daily Cost Trend",
                xaxis_title="Date",
                yaxis_title="Cost ($USD)"
            )
            st.plotly_chart(fig_trend)
        
        # Optimization recommendations
        st.subheader("Optimization Recommendations")
        recommendations = cost_analysis['optimization_recommendations']
        
        for i, rec in enumerate(recommendations):
            with st.expander(f"{rec['title']} - ${rec.get('estimated_savings', 0):.2f} potential savings"):
                st.write(rec['description'])
                st.write(f"**Implementation effort:** {rec.get('implementation_effort', 'Medium')}")
                st.write(f"**Expected timeline:** {rec.get('timeline', '1-2 weeks')}")
                
                if rec.get('implementation_steps'):
                    st.write("**Implementation steps:**")
                    for step in rec['implementation_steps']:
                        st.write(f"  - {step}")

if __name__ == "__main__":
    # Run dashboard
    dashboard = CostOptimizationDashboard(
        project_id="your-project-id",
        billing_account_id="your-billing-account"
    )
    dashboard.render_dashboard()
```

## Implementation Roadmap

### Phase 1: Basic Cost Monitoring (Week 1-2)
1. Set up billing export to BigQuery
2. Create basic cost analysis queries
3. Implement service account cost tracking
4. Set up budget alerts

### Phase 2: Optimization Implementation (Week 3-4)
1. Implement batch processing for Document AI
2. Optimize BigQuery queries and partitioning
3. Configure auto-scaling for Vertex AI endpoints
4. Set up caching for repeated operations

### Phase 3: Advanced Monitoring (Week 5-6)
1. Implement cost anomaly detection
2. Create cost optimization dashboard
3. Set up automated optimization actions
4. Implement cost allocation by feature/team

### Phase 4: Continuous Optimization (Ongoing)
1. Regular cost analysis and reporting
2. Optimization strategy refinement
3. New service integration cost planning
4. Cost-aware development practices

## Best Practices Summary

### 1. Cost Awareness
- Monitor costs daily, not just monthly
- Implement cost-aware development practices
- Use cost calculators for capacity planning
- Regular cost optimization reviews

### 2. Resource Optimization
- Use appropriate machine types for workloads
- Implement auto-scaling where applicable
- Use preemptible instances for fault-tolerant workloads
- Optimize data transfer and storage

### 3. Service-Specific Optimizations
- **Vertex AI**: Use batch processing, optimize machine types, implement model caching
- **Document AI**: Batch processing, result caching, preprocessing optimization
- **BigQuery**: Query optimization, partitioning, clustering, slot optimization
- **Storage**: Lifecycle management, compression, appropriate storage classes

### 4. Monitoring and Alerting
- Set up proactive cost monitoring
- Implement automated cost controls
- Regular cost anomaly detection
- Budget variance analysis

## Related Documentation

- [Authentication Setup](../authentication/README.md)
- [Vertex AI Integration](../vertex-ai/README.md)
- [BigQuery ML Integration](../bigquery-ml/README.md)
- [AI Platform Setup](../ai-platform/README.md)