# Auto-scaling Configuration for IPO Valuation Platform

## Overview

This document outlines comprehensive auto-scaling configurations for the Uprez IPO valuation platform, including horizontal and vertical scaling strategies, load balancing, resource optimization, and performance-based scaling policies.

## Architecture Overview

```
Auto-scaling Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                        Load Balancers                          │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Application   │   Global LB     │     CDN                     │
│   Load Balancer │   (Traffic Mgr) │     (Static Assets)         │
└─────────────────┴─────────────────┴─────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                          │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   HPA           │   VPA           │     Cluster Autoscaler      │
│   (Horizontal)  │   (Vertical)    │     (Node Scaling)          │
└─────────────────┴─────────────────┴─────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                     Application Pods                           │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Frontend      │   API Service   │     ML Service              │
│   (Auto-scale)  │   (Auto-scale)  │     (Auto-scale)           │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## 1. Horizontal Pod Autoscaler (HPA) Configuration

### API Service HPA

```yaml
# k8s/autoscaling/api-service-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-service-hpa
  namespace: uprez-production
  labels:
    app: api-service
    component: autoscaling
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-service
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  
  metrics:
  # CPU utilization
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  
  # Memory utilization
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  
  # Custom metric: Request rate
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  
  # Custom metric: Response time
  - type: Pods
    pods:
      metric:
        name: http_request_duration_p95
      target:
        type: AverageValue
        averageValue: "500m"  # 500ms
  
  # Custom metric: Queue depth
  - type: Pods
    pods:
      metric:
        name: request_queue_depth
      target:
        type: AverageValue
        averageValue: "10"

  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # 5 minutes
      policies:
      - type: Percent
        value: 10    # Scale down max 10% of replicas
        periodSeconds: 60
      - type: Pods
        value: 2     # Scale down max 2 pods
        periodSeconds: 60
      selectPolicy: Min  # Use the most conservative policy
      
    scaleUp:
      stabilizationWindowSeconds: 60   # 1 minute
      policies:
      - type: Percent
        value: 50    # Scale up max 50% of replicas
        periodSeconds: 60
      - type: Pods
        value: 4     # Scale up max 4 pods
        periodSeconds: 60
      selectPolicy: Max  # Use the most aggressive policy

---
# ML Service HPA with specialized metrics
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-service-hpa
  namespace: uprez-production
  labels:
    app: ml-service
    component: autoscaling
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-service
  minReplicas: 2
  maxReplicas: 15
  
  metrics:
  # CPU utilization (ML workloads are CPU intensive)
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
  
  # Memory utilization (models can be memory intensive)
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 70
  
  # GPU utilization (if using GPU instances)
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "75"
  
  # Model inference queue length
  - type: Pods
    pods:
      metric:
        name: model_inference_queue_length
      target:
        type: AverageValue
        averageValue: "20"
  
  # Inference latency (scale up if latency increases)
  - type: Pods
    pods:
      metric:
        name: model_inference_duration_p95
      target:
        type: AverageValue
        averageValue: "2000m"  # 2 seconds

  behavior:
    scaleDown:
      stabilizationWindowSeconds: 600  # 10 minutes (models take time to warm up)
      policies:
      - type: Percent
        value: 15
        periodSeconds: 120
      - type: Pods
        value: 1
        periodSeconds: 120
      selectPolicy: Min
      
    scaleUp:
      stabilizationWindowSeconds: 120  # 2 minutes
      policies:
      - type: Percent
        value: 100   # Can double replicas if needed
        periodSeconds: 60
      - type: Pods
        value: 3
        periodSeconds: 60
      selectPolicy: Max

---
# Frontend HPA (simpler scaling needs)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: frontend-service-hpa
  namespace: uprez-production
  labels:
    app: frontend-service
    component: autoscaling
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: frontend-service
  minReplicas: 2
  maxReplicas: 10
  
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
  
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 70
  
  # Request rate per pod
  - type: Pods
    pods:
      metric:
        name: nginx_requests_per_second
      target:
        type: AverageValue
        averageValue: "50"

  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 20
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### Custom Metrics Server Configuration

```yaml
# k8s/autoscaling/custom-metrics-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: custom-metrics-config
  namespace: uprez-production
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
      - "custom_rules.yml"
    
    scrape_configs:
      # API Service metrics
      - job_name: 'api-service'
        kubernetes_sd_configs:
        - role: pod
        relabel_configs:
        - source_labels: [__meta_kubernetes_pod_label_app]
          action: keep
          regex: api-service
        - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
          action: replace
          target_label: __address__
          regex: (.+)
          replacement: ${1}:9090
    
      # ML Service metrics  
      - job_name: 'ml-service'
        kubernetes_sd_configs:
        - role: pod
        relabel_configs:
        - source_labels: [__meta_kubernetes_pod_label_app]
          action: keep
          regex: ml-service
        - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
          action: replace
          target_label: __address__
          regex: (.+)
          replacement: ${1}:9090

  custom_rules.yml: |
    groups:
    - name: custom.rules
      rules:
      # HTTP requests per second per pod
      - record: http_requests_per_second
        expr: rate(http_requests_total[1m])
      
      # 95th percentile response time
      - record: http_request_duration_p95
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[1m]))
      
      # Request queue depth
      - record: request_queue_depth
        expr: sum(rate(http_requests_total[1m])) by (pod) - sum(rate(http_responses_total[1m])) by (pod)
      
      # ML model inference queue length
      - record: model_inference_queue_length
        expr: model_pending_requests
      
      # GPU utilization (if applicable)
      - record: gpu_utilization
        expr: nvidia_gpu_utilization_gpu
      
      # Model inference 95th percentile duration
      - record: model_inference_duration_p95
        expr: histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket[1m]))

---
# Prometheus Adapter for custom metrics
apiVersion: v1
kind: ConfigMap
metadata:
  name: adapter-config
  namespace: custom-metrics
data:
  config.yaml: |
    rules:
    # API Service metrics
    - seriesQuery: 'http_requests_per_second{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        matches: "^(.*)$"
        as: "http_requests_per_second"
      metricsQuery: 'sum(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
    
    - seriesQuery: 'http_request_duration_p95{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        matches: "^(.*)$"
        as: "http_request_duration_p95"
      metricsQuery: 'avg(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
    
    # ML Service metrics
    - seriesQuery: 'model_inference_queue_length{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        matches: "^(.*)$"
        as: "model_inference_queue_length"
      metricsQuery: 'sum(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
    
    - seriesQuery: 'gpu_utilization{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        matches: "^(.*)$"
        as: "gpu_utilization"
      metricsQuery: 'avg(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
```

## 2. Vertical Pod Autoscaler (VPA) Configuration

### VPA for Database and Stateful Services

```yaml
# k8s/autoscaling/vpa-config.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: postgres-vpa
  namespace: uprez-production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: StatefulSet
    name: postgres-primary
  updatePolicy:
    updateMode: "Auto"  # Can be "Off", "Initial", or "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: postgres
      maxAllowed:
        cpu: 4
        memory: 8Gi
      minAllowed:
        cpu: 500m
        memory: 1Gi
      controlledResources: ["cpu", "memory"]

---
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: redis-vpa
  namespace: uprez-production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: StatefulSet
    name: redis-cache
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: redis
      maxAllowed:
        cpu: 2
        memory: 4Gi
      minAllowed:
        cpu: 100m
        memory: 256Mi
      controlledResources: ["cpu", "memory"]

---
# VPA for API service (mainly for memory optimization)
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: api-service-vpa
  namespace: uprez-production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-service
  updatePolicy:
    updateMode: "Off"  # Recommendation only, HPA handles scaling
  resourcePolicy:
    containerPolicies:
    - containerName: api
      maxAllowed:
        cpu: 2
        memory: 4Gi
      minAllowed:
        cpu: 200m
        memory: 512Mi
      controlledResources: ["memory"]  # Only optimize memory
```

## 3. Cluster Autoscaler Configuration

### Node Pool Auto-scaling

```yaml
# k8s/autoscaling/cluster-autoscaler.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
  labels:
    app: cluster-autoscaler
spec:
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
      annotations:
        prometheus.io/scrape: 'true'
        prometheus.io/port: '8085'
    spec:
      serviceAccountName: cluster-autoscaler
      containers:
      - image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.21.0
        name: cluster-autoscaler
        resources:
          limits:
            cpu: 100m
            memory: 300Mi
          requests:
            cpu: 100m
            memory: 300Mi
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=gce
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=mig:name=gke-cluster-default-pool
        - --scale-down-enabled=true
        - --scale-down-delay-after-add=10m
        - --scale-down-unneeded-time=10m
        - --scale-down-utilization-threshold=0.5
        - --max-node-provision-time=15m
        - --nodes=1:10:gke-uprez-cluster-default-pool
        - --nodes=0:20:gke-uprez-cluster-compute-pool
        - --nodes=0:5:gke-uprez-cluster-gpu-pool
        env:
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: /etc/gcp/service-account.json
        volumeMounts:
        - name: gcp-service-account
          mountPath: /etc/gcp
          readOnly: true
      volumes:
      - name: gcp-service-account
        secret:
          secretName: gcp-service-account

---
# Service Account for Cluster Autoscaler
apiVersion: v1
kind: ServiceAccount
metadata:
  name: cluster-autoscaler
  namespace: kube-system
  labels:
    app: cluster-autoscaler

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: cluster-autoscaler
  labels:
    app: cluster-autoscaler
rules:
- apiGroups: [""]
  resources: ["events", "endpoints"]
  verbs: ["create", "patch"]
- apiGroups: [""]
  resources: ["pods/eviction"]
  verbs: ["create"]
- apiGroups: [""]
  resources: ["pods/status"]
  verbs: ["update"]
- apiGroups: [""]
  resources: ["endpoints"]
  resourceNames: ["cluster-autoscaler"]
  verbs: ["get", "update"]
- apiGroups: [""]
  resources: ["nodes"]
  verbs: ["watch", "list", "get", "update"]
- apiGroups: [""]
  resources: ["pods", "services", "replicationcontrollers", "persistentvolumeclaims", "persistentvolumes"]
  verbs: ["watch", "list", "get"]
- apiGroups: ["extensions"]
  resources: ["replicasets", "daemonsets"]
  verbs: ["watch", "list", "get"]
- apiGroups: ["policy"]
  resources: ["poddisruptionbudgets"]
  verbs: ["watch", "list"]
- apiGroups: ["apps"]
  resources: ["statefulsets", "replicasets", "daemonsets"]
  verbs: ["watch", "list", "get"]
- apiGroups: ["storage.k8s.io"]
  resources: ["storageclasses", "csinodes"]
  verbs: ["watch", "list", "get"]
- apiGroups: ["batch", "extensions"]
  resources: ["jobs"]
  verbs: ["get", "list", "watch", "patch"]
- apiGroups: ["coordination.k8s.io"]
  resources: ["leases"]
  verbs: ["create"]
- apiGroups: ["coordination.k8s.io"]
  resourceNames: ["cluster-autoscaler"]
  resources: ["leases"]
  verbs: ["get", "update"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: cluster-autoscaler
  labels:
    app: cluster-autoscaler
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cluster-autoscaler
subjects:
- kind: ServiceAccount
  name: cluster-autoscaler
  namespace: kube-system
```

## 4. Advanced Scaling Policies

### Predictive Scaling Configuration

```python
# scripts/autoscaling/predictive_scaling.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import requests
import json

class PredictiveScaler:
    """
    Predictive auto-scaling based on historical patterns and external factors
    """
    
    def __init__(self, prometheus_url: str = "http://localhost:9090",
                 kubernetes_config: str = None):
        self.prometheus_url = prometheus_url
        self.models = {}
        self.scaling_history = []
        
        # Initialize Kubernetes client
        from kubernetes import client, config
        if kubernetes_config:
            config.load_kube_config(kubernetes_config)
        else:
            config.load_incluster_config()
        
        self.k8s_client = client.AppsV1Api()
        self.k8s_autoscaling = client.AutoscalingV2Api()
    
    def collect_metrics(self, lookback_hours: int = 24) -> pd.DataFrame:
        """
        Collect historical metrics for prediction
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_hours)
        
        metrics_queries = {
            'cpu_usage': 'avg(rate(container_cpu_usage_seconds_total[5m])) by (pod)',
            'memory_usage': 'avg(container_memory_usage_bytes) by (pod)',
            'request_rate': 'sum(rate(http_requests_total[5m]))',
            'response_time': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))',
            'active_users': 'active_users_total',
            'queue_length': 'sum(model_inference_queue_length)'
        }
        
        all_data = []
        
        for metric_name, query in metrics_queries.items():
            try:
                params = {
                    'query': query,
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'step': '5m'
                }
                
                response = requests.get(f"{self.prometheus_url}/api/v1/query_range", params=params)
                data = response.json()
                
                if data.get('data', {}).get('result'):
                    for series in data['data']['result']:
                        for timestamp, value in series['values']:
                            all_data.append({
                                'timestamp': pd.to_datetime(timestamp, unit='s'),
                                'metric': metric_name,
                                'value': float(value),
                                'pod': series['metric'].get('pod', 'aggregate')
                            })
                            
            except Exception as e:
                logging.error(f"Error collecting metric {metric_name}: {e}")
        
        df = pd.DataFrame(all_data)
        if not df.empty:
            df = df.pivot_table(index=['timestamp', 'pod'], columns='metric', values='value').reset_index()
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features for seasonality detection
        """
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def add_external_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add external factors that might influence scaling needs
        """
        # Market hours (business days 9-5 Australian time)
        df['is_market_hours'] = (
            (df['timestamp'].dt.hour >= 9) & 
            (df['timestamp'].dt.hour <= 17) & 
            (df['timestamp'].dt.dayofweek < 5)
        ).astype(int)
        
        # IPO calendar events (placeholder - would integrate with real IPO calendar)
        # For now, assume higher activity on certain days
        df['is_high_activity_period'] = (
            (df['timestamp'].dt.dayofweek == 0) |  # Mondays
            (df['timestamp'].dt.day <= 5)  # Beginning of month
        ).astype(int)
        
        return df
    
    def train_scaling_model(self, service_name: str, target_metric: str = 'cpu_usage') -> None:
        """
        Train predictive model for a specific service
        """
        # Collect historical data
        df = self.collect_metrics(lookback_hours=168)  # 1 week
        
        if df.empty:
            logging.warning(f"No data available for training {service_name} model")
            return
        
        # Filter for specific service
        service_df = df[df['pod'].str.contains(service_name, case=False, na=False)]
        
        if service_df.empty:
            logging.warning(f"No data available for service {service_name}")
            return
        
        # Add features
        service_df = self.add_time_features(service_df)
        service_df = self.add_external_features(service_df)
        
        # Prepare features and target
        feature_columns = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'is_market_hours', 'is_high_activity_period',
            'request_rate', 'active_users', 'queue_length'
        ]
        
        # Remove rows with missing values
        service_df = service_df.dropna(subset=feature_columns + [target_metric])
        
        if len(service_df) < 100:
            logging.warning(f"Insufficient data for training {service_name} model")
            return
        
        X = service_df[feature_columns]
        y = service_df[target_metric]
        
        # Train ensemble model
        models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        trained_models = {}
        for name, model in models.items():
            try:
                model.fit(X, y)
                trained_models[name] = model
                logging.info(f"Trained {name} model for {service_name}")
            except Exception as e:
                logging.error(f"Error training {name} model for {service_name}: {e}")
        
        self.models[service_name] = {
            'models': trained_models,
            'feature_columns': feature_columns,
            'target_metric': target_metric,
            'last_updated': datetime.now()
        }
    
    def predict_scaling_needs(self, service_name: str, hours_ahead: int = 1) -> Dict[str, Any]:
        """
        Predict scaling needs for a service
        """
        if service_name not in self.models:
            logging.error(f"No trained model available for {service_name}")
            return {}
        
        model_info = self.models[service_name]
        feature_columns = model_info['feature_columns']
        
        # Create future time points
        future_times = [
            datetime.now() + timedelta(hours=h) for h in range(1, hours_ahead + 1)
        ]
        
        predictions = []
        
        for future_time in future_times:
            # Create feature vector for future time
            features = {}
            
            # Time features
            features['hour_sin'] = np.sin(2 * np.pi * future_time.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * future_time.hour / 24)
            features['day_sin'] = np.sin(2 * np.pi * future_time.weekday() / 7)
            features['day_cos'] = np.cos(2 * np.pi * future_time.weekday() / 7)
            
            # External features
            features['is_market_hours'] = int(
                9 <= future_time.hour <= 17 and future_time.weekday() < 5
            )
            features['is_high_activity_period'] = int(
                future_time.weekday() == 0 or future_time.day <= 5
            )
            
            # Current metrics (assuming they'll be similar in near future)
            current_metrics = self.get_current_metrics()
            features['request_rate'] = current_metrics.get('request_rate', 0)
            features['active_users'] = current_metrics.get('active_users', 0)
            features['queue_length'] = current_metrics.get('queue_length', 0)
            
            # Make predictions with ensemble
            feature_vector = np.array([[features[col] for col in feature_columns]])
            
            model_predictions = {}
            for model_name, model in model_info['models'].items():
                try:
                    pred = model.predict(feature_vector)[0]
                    model_predictions[model_name] = pred
                except Exception as e:
                    logging.error(f"Error making prediction with {model_name}: {e}")
            
            # Ensemble prediction (average)
            if model_predictions:
                ensemble_pred = np.mean(list(model_predictions.values()))
                predictions.append({
                    'timestamp': future_time,
                    'predicted_value': ensemble_pred,
                    'individual_predictions': model_predictions
                })
        
        return {
            'service': service_name,
            'metric': model_info['target_metric'],
            'predictions': predictions
        }
    
    def get_current_metrics(self) -> Dict[str, float]:
        """
        Get current system metrics
        """
        metrics = {}
        
        queries = {
            'request_rate': 'sum(rate(http_requests_total[5m]))',
            'active_users': 'active_users_total',
            'queue_length': 'sum(model_inference_queue_length)'
        }
        
        for metric_name, query in queries.items():
            try:
                params = {'query': query}
                response = requests.get(f"{self.prometheus_url}/api/v1/query", params=params)
                data = response.json()
                
                if data.get('data', {}).get('result'):
                    metrics[metric_name] = float(data['data']['result'][0]['value'][1])
                    
            except Exception as e:
                logging.error(f"Error getting current metric {metric_name}: {e}")
                metrics[metric_name] = 0
        
        return metrics
    
    def apply_predictive_scaling(self, service_name: str, namespace: str = "uprez-production") -> bool:
        """
        Apply predictive scaling recommendations
        """
        try:
            predictions = self.predict_scaling_needs(service_name, hours_ahead=2)
            
            if not predictions or not predictions['predictions']:
                logging.warning(f"No predictions available for {service_name}")
                return False
            
            # Get current HPA
            hpa_name = f"{service_name}-hpa"
            try:
                hpa = self.k8s_autoscaling.read_namespaced_horizontal_pod_autoscaler(
                    name=hpa_name, namespace=namespace
                )
            except Exception as e:
                logging.error(f"Error reading HPA {hpa_name}: {e}")
                return False
            
            # Analyze predictions
            future_values = [p['predicted_value'] for p in predictions['predictions']]
            max_predicted_value = max(future_values)
            current_replicas = hpa.status.current_replicas or 1
            
            # Calculate scaling recommendation
            if predictions['metric'] == 'cpu_usage':
                # If predicted CPU usage > 70%, pre-scale
                target_cpu = 0.7  # 70% CPU target
                if max_predicted_value > target_cpu:
                    scale_factor = max_predicted_value / target_cpu
                    recommended_replicas = min(
                        int(current_replicas * scale_factor),
                        hpa.spec.max_replicas
                    )
                else:
                    recommended_replicas = current_replicas
            else:
                # Generic scaling logic
                if max_predicted_value > current_replicas * 1.5:
                    recommended_replicas = min(
                        int(max_predicted_value / 1.5),
                        hpa.spec.max_replicas
                    )
                else:
                    recommended_replicas = current_replicas
            
            # Apply scaling if significant change needed
            if abs(recommended_replicas - current_replicas) >= 1:
                # Update deployment directly for predictive scaling
                try:
                    deployment = self.k8s_client.read_namespaced_deployment(
                        name=service_name, namespace=namespace
                    )
                    deployment.spec.replicas = recommended_replicas
                    
                    self.k8s_client.patch_namespaced_deployment(
                        name=service_name, namespace=namespace, body=deployment
                    )
                    
                    logging.info(f"Predictively scaled {service_name} from {current_replicas} to {recommended_replicas} replicas")
                    
                    # Record scaling event
                    self.scaling_history.append({
                        'timestamp': datetime.now(),
                        'service': service_name,
                        'from_replicas': current_replicas,
                        'to_replicas': recommended_replicas,
                        'reason': 'predictive_scaling',
                        'predicted_values': future_values
                    })
                    
                    return True
                    
                except Exception as e:
                    logging.error(f"Error applying predictive scaling to {service_name}: {e}")
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error in predictive scaling for {service_name}: {e}")
            return False

class AutoScalingOrchestrator:
    """
    Orchestrate different types of auto-scaling
    """
    
    def __init__(self):
        self.predictive_scaler = PredictiveScaler()
        self.services = ['api-service', 'ml-service', 'frontend-service']
        
    def run_scaling_cycle(self) -> None:
        """
        Run complete auto-scaling cycle
        """
        logging.info("Starting auto-scaling cycle")
        
        for service in self.services:
            try:
                # Train/update predictive models
                self.predictive_scaler.train_scaling_model(service)
                
                # Apply predictive scaling
                self.predictive_scaler.apply_predictive_scaling(service)
                
            except Exception as e:
                logging.error(f"Error in scaling cycle for {service}: {e}")
        
        logging.info("Auto-scaling cycle completed")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create orchestrator
    orchestrator = AutoScalingOrchestrator()
    
    # Run scaling cycle (would be scheduled as a cron job)
    orchestrator.run_scaling_cycle()
```

## 5. Load Balancing Configuration

### NGINX Ingress with Advanced Load Balancing

```yaml
# k8s/load-balancing/advanced-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: uprez-advanced-ingress
  namespace: uprez-production
  annotations:
    # Load balancing algorithm
    nginx.ingress.kubernetes.io/upstream-hash-by: "$request_uri"
    nginx.ingress.kubernetes.io/load-balance: "round_robin"
    
    # Session affinity for stateful operations
    nginx.ingress.kubernetes.io/session-cookie-name: "uprez-session"
    nginx.ingress.kubernetes.io/session-cookie-expires: "3600"
    nginx.ingress.kubernetes.io/session-cookie-max-age: "3600"
    nginx.ingress.kubernetes.io/affinity: "cookie"
    
    # Rate limiting
    nginx.ingress.kubernetes.io/rate-limit-connections: "20"
    nginx.ingress.kubernetes.io/rate-limit-requests-per-second: "10"
    nginx.ingress.kubernetes.io/rate-limit-window-size: "1m"
    
    # Connection and timeout settings
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "30"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "120"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "120"
    nginx.ingress.kubernetes.io/proxy-next-upstream: "error timeout http_500 http_502 http_503 http_504"
    nginx.ingress.kubernetes.io/proxy-next-upstream-tries: "3"
    nginx.ingress.kubernetes.io/proxy-next-upstream-timeout: "30"
    
    # SSL and security
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/ssl-protocols: "TLSv1.2 TLSv1.3"
    nginx.ingress.kubernetes.io/ssl-ciphers: "ECDHE-ECDSA-AES128-GCM-SHA256,ECDHE-RSA-AES128-GCM-SHA256"
    
    # Custom configuration for different endpoints
    nginx.ingress.kubernetes.io/server-snippet: |
      location ~* ^/api/v1/ml/ {
          proxy_buffering off;
          proxy_request_buffering off;
          client_max_body_size 10m;
      }
      
      location ~* ^/api/v1/data/ {
          proxy_cache static_cache;
          proxy_cache_valid 200 5m;
          proxy_cache_key "$scheme$request_method$host$request_uri";
      }

spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - uprez.com
    - api.uprez.com
    secretName: uprez-tls
  rules:
  - host: api.uprez.com
    http:
      paths:
      # ML endpoints with special handling
      - path: /api/v1/ml
        pathType: Prefix
        backend:
          service:
            name: ml-service
            port:
              number: 8000
      
      # API endpoints
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 8000
      
      # Health checks bypass rate limiting
      - path: /health
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 8000

---
# Service mesh configuration with Istio (optional)
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: api-service-destination-rule
  namespace: uprez-production
spec:
  host: api-service
  trafficPolicy:
    loadBalancer:
      simple: LEAST_CONN  # Connection-based load balancing
    connectionPool:
      tcp:
        maxConnections: 100
        connectTimeout: 30s
      http:
        http1MaxPendingRequests: 50
        http2MaxRequests: 100
        maxRequestsPerConnection: 10
        maxRetries: 3
        consecutiveGatewayErrors: 5
        interval: 30s
        baseEjectionTime: 30s
        maxEjectionPercent: 50
    outlierDetection:
      consecutiveGatewayErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
  portLevelSettings:
  - port:
      number: 8000
    connectionPool:
      tcp:
        maxConnections: 50
```

This auto-scaling configuration documentation provides:

1. **Horizontal Pod Autoscaler (HPA)**: Advanced HPA configurations with custom metrics, behavior policies, and service-specific scaling strategies
2. **Vertical Pod Autoscaler (VPA)**: Resource optimization for stateful services and memory-intensive workloads
3. **Cluster Autoscaler**: Node-level scaling with multiple node pools and cost optimization
4. **Predictive Scaling**: ML-based predictive scaling using historical patterns and external factors
5. **Load Balancing**: Advanced load balancing with session affinity, rate limiting, and service mesh integration

The configuration includes production-ready settings for handling variable workloads typical in financial applications, with specific optimizations for ML inference workloads and database services.