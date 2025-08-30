# Kubernetes Deployment Strategy for IPO Valuation Platform

## Overview

This document outlines the comprehensive Kubernetes deployment strategy for the Uprez IPO valuation platform, including namespace organization, resource management, auto-scaling, monitoring, and production-ready configurations.

## Architecture Overview

```
Kubernetes Cluster Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                        Ingress Controller                       │
│                    (NGINX/Istio Gateway)                       │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      Production Namespace                       │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Frontend      │   API Gateway   │     ML Services             │
│   (3 replicas)  │   (2 replicas)  │     (2 replicas)           │
├─────────────────┼─────────────────┼─────────────────────────────┤
│   Backend API   │   Auth Service  │   Data Pipeline             │
│   (3 replicas)  │   (2 replicas)  │   (1 replica)               │
├─────────────────┼─────────────────┼─────────────────────────────┤
│   PostgreSQL    │   Redis Cache   │   MinIO Storage             │
│   (StatefulSet) │   (StatefulSet) │   (StatefulSet)             │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## 1. Namespace Configuration

### Base Namespace Setup

```yaml
# k8s/namespaces/namespaces.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: uprez-production
  labels:
    environment: production
    app: uprez-ipo-valuation
---
apiVersion: v1
kind: Namespace
metadata:
  name: uprez-staging
  labels:
    environment: staging
    app: uprez-ipo-valuation
---
apiVersion: v1
kind: Namespace
metadata:
  name: uprez-development
  labels:
    environment: development
    app: uprez-ipo-valuation
---
# Monitoring namespace
apiVersion: v1
kind: Namespace
metadata:
  name: uprez-monitoring
  labels:
    environment: all
    app: uprez-monitoring
```

### Resource Quotas

```yaml
# k8s/namespaces/resource-quotas.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: production-quota
  namespace: uprez-production
spec:
  hard:
    # Compute resources
    requests.cpu: "20"
    requests.memory: 40Gi
    limits.cpu: "40" 
    limits.memory: 80Gi
    
    # Storage resources
    requests.storage: 200Gi
    persistentvolumeclaims: "10"
    
    # Object counts
    pods: "50"
    replicationcontrollers: "20"
    secrets: "20"
    services: "20"
    
---
apiVersion: v1
kind: LimitRange
metadata:
  name: production-limits
  namespace: uprez-production
spec:
  limits:
  - default:
      cpu: "1"
      memory: "2Gi"
    defaultRequest:
      cpu: "100m"
      memory: "256Mi"
    type: Container
  - default:
      storage: "10Gi"
    type: PersistentVolumeClaim
```

## 2. ConfigMaps and Secrets

### Application Configuration

```yaml
# k8s/configs/api-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: api-config
  namespace: uprez-production
data:
  # Application settings
  LOG_LEVEL: "INFO"
  MAX_WORKERS: "4"
  TIMEOUT: "120"
  
  # Database settings
  DB_POOL_SIZE: "20"
  DB_MAX_OVERFLOW: "10"
  DB_POOL_TIMEOUT: "30"
  
  # Cache settings
  CACHE_TTL: "3600"
  CACHE_MAX_CONNECTIONS: "100"
  
  # ML Model settings
  MODEL_BATCH_SIZE: "32"
  MODEL_TIMEOUT: "30"
  
  # Feature flags
  ENABLE_CACHING: "true"
  ENABLE_METRICS: "true"
  ENABLE_TRACING: "true"

---
# ML Service Configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-service-config
  namespace: uprez-production
data:
  # Model serving
  MODEL_SERVER_WORKERS: "2"
  MODEL_BATCH_SIZE: "64"
  MODEL_MAX_BATCH_DELAY: "100"
  
  # Performance settings
  TORCH_THREADS: "4"
  OMP_NUM_THREADS: "4"
  
  # Monitoring
  METRICS_PORT: "9090"
  HEALTH_CHECK_INTERVAL: "30"

---
# Frontend Configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: frontend-config
  namespace: uprez-production
data:
  # API endpoints
  NEXT_PUBLIC_API_URL: "https://api.uprez.com"
  NEXT_PUBLIC_ML_SERVICE_URL: "https://ml.uprez.com"
  
  # Feature flags
  NEXT_PUBLIC_ENABLE_ANALYTICS: "true"
  NEXT_PUBLIC_ENABLE_DEBUG: "false"
  
  # Performance
  NEXT_PUBLIC_CDN_URL: "https://cdn.uprez.com"
```

### Secret Management

```yaml
# k8s/secrets/database-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: database-secrets
  namespace: uprez-production
type: Opaque
stringData:
  DATABASE_URL: "postgresql://username:password@postgres-service:5432/uprez_production"
  DB_USERNAME: "uprez_user"
  DB_PASSWORD: "secure_password_here"

---
# JWT and Encryption Secrets
apiVersion: v1
kind: Secret
metadata:
  name: auth-secrets
  namespace: uprez-production
type: Opaque
stringData:
  JWT_SECRET: "jwt_secret_key_here"
  JWT_ALGORITHM: "HS256"
  ENCRYPTION_KEY: "encryption_key_here"

---
# External API Secrets
apiVersion: v1
kind: Secret
metadata:
  name: external-api-secrets
  namespace: uprez-production
type: Opaque
stringData:
  ASX_API_KEY: "asx_api_key_here"
  MARKET_DATA_API_KEY: "market_data_api_key_here"
  NOTIFICATION_API_KEY: "notification_api_key_here"

---
# Google Cloud Service Account
apiVersion: v1
kind: Secret
metadata:
  name: gcp-service-account
  namespace: uprez-production
type: Opaque
data:
  service-account.json: <base64-encoded-service-account-json>
```

## 3. Database Deployments (StatefulSets)

### PostgreSQL Primary Database

```yaml
# k8s/database/postgresql.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres-primary
  namespace: uprez-production
  labels:
    app: postgres-primary
    component: database
spec:
  serviceName: postgres-primary
  replicas: 1
  selector:
    matchLabels:
      app: postgres-primary
  template:
    metadata:
      labels:
        app: postgres-primary
        component: database
    spec:
      securityContext:
        fsGroup: 999
        runAsUser: 999
        runAsNonRoot: true
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: uprez_production
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: DB_USERNAME
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: DB_PASSWORD
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata
        ports:
        - containerPort: 5432
          name: postgres
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        - name: postgres-config
          mountPath: /etc/postgresql/postgresql.conf
          subPath: postgresql.conf
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          exec:
            command:
            - /bin/bash
            - -c
            - pg_isready -U $POSTGRES_USER -d $POSTGRES_DB
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - /bin/bash
            - -c
            - pg_isready -U $POSTGRES_USER -d $POSTGRES_DB
          initialDelaySeconds: 15
          periodSeconds: 5
      volumes:
      - name: postgres-config
        configMap:
          name: postgres-config
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: ssd-persistent
      resources:
        requests:
          storage: 100Gi

---
# PostgreSQL Service
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: uprez-production
  labels:
    app: postgres-primary
spec:
  ports:
  - port: 5432
    targetPort: 5432
    name: postgres
  selector:
    app: postgres-primary
  type: ClusterIP
```

### Redis Cache

```yaml
# k8s/cache/redis.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cache
  namespace: uprez-production
  labels:
    app: redis-cache
    component: cache
spec:
  serviceName: redis-cache
  replicas: 1
  selector:
    matchLabels:
      app: redis-cache
  template:
    metadata:
      labels:
        app: redis-cache
        component: cache
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command:
        - redis-server
        - "/redis-master/redis.conf"
        env:
        - name: MASTER
          value: "true"
        ports:
        - containerPort: 6379
          name: redis
        volumeMounts:
        - name: redis-config
          mountPath: /redis-master
        - name: redis-storage
          mountPath: /data
        resources:
          requests:
            memory: "1Gi"
            cpu: "200m"
          limits:
            memory: "2Gi"
            cpu: "500m"
        livenessProbe:
          tcpSocket:
            port: 6379
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 15
          periodSeconds: 5
      volumes:
      - name: redis-config
        configMap:
          name: redis-config
          defaultMode: 0755
  volumeClaimTemplates:
  - metadata:
      name: redis-storage
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: ssd-persistent
      resources:
        requests:
          storage: 20Gi

---
# Redis Service
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: uprez-production
  labels:
    app: redis-cache
spec:
  ports:
  - port: 6379
    targetPort: 6379
    name: redis
  selector:
    app: redis-cache
  type: ClusterIP
```

## 4. Application Deployments

### Backend API Service

```yaml
# k8s/api/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-service
  namespace: uprez-production
  labels:
    app: api-service
    component: backend
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: api-service
  template:
    metadata:
      labels:
        app: api-service
        component: backend
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      serviceAccountName: api-service-account
      initContainers:
      - name: database-migration
        image: gcr.io/uprez-project/api:latest
        command: ['alembic', 'upgrade', 'head']
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: DATABASE_URL
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
      containers:
      - name: api
        image: gcr.io/uprez-project/api:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        env:
        # Database configuration
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: DATABASE_URL
        # Cache configuration
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        # Auth configuration
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: auth-secrets
              key: JWT_SECRET
        envFrom:
        - configMapRef:
            name: api-config
        volumeMounts:
        - name: gcp-service-account
          mountPath: /etc/gcp
          readOnly: true
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
      volumes:
      - name: gcp-service-account
        secret:
          secretName: gcp-service-account

---
# API Service
apiVersion: v1
kind: Service
metadata:
  name: api-service
  namespace: uprez-production
  labels:
    app: api-service
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9090"
spec:
  ports:
  - port: 8000
    targetPort: 8000
    name: http
  - port: 9090
    targetPort: 9090
    name: metrics
  selector:
    app: api-service
  type: ClusterIP

---
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-service-hpa
  namespace: uprez-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### ML Service Deployment

```yaml
# k8s/ml/ml-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
  namespace: uprez-production
  labels:
    app: ml-service
    component: ml-inference
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: ml-service
  template:
    metadata:
      labels:
        app: ml-service
        component: ml-inference
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
      containers:
      - name: ml-service
        image: gcr.io/uprez-project/ml-service:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: MODEL_STORE_PATH
          value: "/models"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        envFrom:
        - configMapRef:
            name: ml-service-config
        volumeMounts:
        - name: model-storage
          mountPath: /models
          readOnly: true
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60  # ML models take time to load
          periodSeconds: 30
          timeoutSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 45
          periodSeconds: 10
          timeoutSeconds: 5
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc

---
# ML Service
apiVersion: v1
kind: Service
metadata:
  name: ml-service
  namespace: uprez-production
  labels:
    app: ml-service
spec:
  ports:
  - port: 8000
    targetPort: 8000
    name: http
  - port: 9090
    targetPort: 9090
    name: metrics
  selector:
    app: ml-service
  type: ClusterIP

---
# ML Service HPA with custom metrics
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-service-hpa
  namespace: uprez-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-service
  minReplicas: 2
  maxReplicas: 8
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
  # Custom metric for inference queue length
  - type: Pods
    pods:
      metric:
        name: inference_queue_length
      target:
        type: AverageValue
        averageValue: "30"
```

### Frontend Deployment

```yaml
# k8s/frontend/frontend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend-service
  namespace: uprez-production
  labels:
    app: frontend-service
    component: frontend
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 0
  selector:
    matchLabels:
      app: frontend-service
  template:
    metadata:
      labels:
        app: frontend-service
        component: frontend
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
      containers:
      - name: frontend
        image: gcr.io/uprez-project/frontend:latest
        ports:
        - containerPort: 3000
          name: http
        envFrom:
        - configMapRef:
            name: frontend-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "200m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/health
            port: 3000
          initialDelaySeconds: 15
          periodSeconds: 5

---
# Frontend Service
apiVersion: v1
kind: Service
metadata:
  name: frontend-service
  namespace: uprez-production
  labels:
    app: frontend-service
spec:
  ports:
  - port: 3000
    targetPort: 3000
    name: http
  selector:
    app: frontend-service
  type: ClusterIP
```

## 5. Ingress and Load Balancing

### NGINX Ingress Controller

```yaml
# k8s/ingress/nginx-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: uprez-ingress
  namespace: uprez-production
  annotations:
    # NGINX specific annotations
    nginx.ingress.kubernetes.io/use-regex: "true"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    
    # Rate limiting
    nginx.ingress.kubernetes.io/rate-limit-connections: "10"
    nginx.ingress.kubernetes.io/rate-limit-requests-per-second: "5"
    
    # CORS configuration
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://uprez.com,https://www.uprez.com"
    nginx.ingress.kubernetes.io/cors-allow-methods: "GET, POST, PUT, DELETE, OPTIONS"
    nginx.ingress.kubernetes.io/cors-allow-headers: "Authorization, Content-Type"
    
    # Security headers
    nginx.ingress.kubernetes.io/configuration-snippet: |
      add_header X-Frame-Options DENY;
      add_header X-Content-Type-Options nosniff;
      add_header X-XSS-Protection "1; mode=block";
      add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
      
    # Load balancing
    nginx.ingress.kubernetes.io/upstream-hash-by: "$remote_addr"
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "30"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "120"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "120"
    
    # Certificate management
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - uprez.com
    - www.uprez.com
    - api.uprez.com
    - ml.uprez.com
    secretName: uprez-tls
  rules:
  # Main application
  - host: uprez.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 3000
  - host: www.uprez.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 3000
  
  # API endpoints
  - host: api.uprez.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 8000
  
  # ML service endpoints
  - host: ml.uprez.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ml-service
            port:
              number: 8000

---
# Certificate Issuer for Let's Encrypt
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: devops@uprez.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
```

### Network Policies

```yaml
# k8s/network/network-policies.yaml
# Default deny all traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: uprez-production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress

---
# Allow frontend to API communication
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: frontend-to-api
  namespace: uprez-production
spec:
  podSelector:
    matchLabels:
      app: api-service
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend-service
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000

---
# Allow API to database communication
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-to-database
  namespace: uprez-production
spec:
  podSelector:
    matchLabels:
      app: postgres-primary
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: api-service
    - podSelector:
        matchLabels:
          app: ml-service
    ports:
    - protocol: TCP
      port: 5432

---
# Allow services to cache
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: services-to-cache
  namespace: uprez-production
spec:
  podSelector:
    matchLabels:
      app: redis-cache
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          component: backend
    - podSelector:
        matchLabels:
          component: ml-inference
    ports:
    - protocol: TCP
      port: 6379

---
# Allow outbound traffic for external APIs
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-external-api
  namespace: uprez-production
spec:
  podSelector:
    matchLabels:
      app: api-service
  policyTypes:
  - Egress
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
```

## 6. Service Accounts and RBAC

### Service Accounts

```yaml
# k8s/rbac/service-accounts.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: api-service-account
  namespace: uprez-production
  annotations:
    iam.gke.io/gcp-service-account: uprez-api@uprez-project.iam.gserviceaccount.com

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ml-service-account
  namespace: uprez-production
  annotations:
    iam.gke.io/gcp-service-account: uprez-ml@uprez-project.iam.gserviceaccount.com

---
# RBAC Configuration
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: uprez-production
  name: api-service-role
rules:
# Allow reading ConfigMaps
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["get", "list"]
# Allow reading Secrets
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["get", "list"]
# Allow creating/updating service status
- apiGroups: [""]
  resources: ["services"]
  verbs: ["get", "patch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: api-service-binding
  namespace: uprez-production
subjects:
- kind: ServiceAccount
  name: api-service-account
  namespace: uprez-production
roleRef:
  kind: Role
  name: api-service-role
  apiGroup: rbac.authorization.k8s.io
```

## 7. Persistent Storage

### Storage Classes and PVCs

```yaml
# k8s/storage/storage-classes.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ssd-persistent
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-ssd
  zones: us-central1-a,us-central1-b,us-central1-c
  replication-type: regional-pd
allowVolumeExpansion: true
reclaimPolicy: Retain

---
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: hdd-persistent
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-standard
  zones: us-central1-a,us-central1-b,us-central1-c
allowVolumeExpansion: true
reclaimPolicy: Delete

---
# ML Model Storage
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage-pvc
  namespace: uprez-production
spec:
  accessModes:
  - ReadWriteOnce
  storageClassName: ssd-persistent
  resources:
    requests:
      storage: 50Gi

---
# Backup Storage
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: backup-storage-pvc
  namespace: uprez-production
spec:
  accessModes:
  - ReadWriteMany
  storageClassName: hdd-persistent
  resources:
    requests:
      storage: 200Gi
```

## 8. Monitoring and Observability

### Prometheus ServiceMonitor

```yaml
# k8s/monitoring/service-monitors.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: api-service-monitor
  namespace: uprez-production
  labels:
    app: api-service
spec:
  selector:
    matchLabels:
      app: api-service
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
    targetPort: 9090

---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ml-service-monitor
  namespace: uprez-production
  labels:
    app: ml-service
spec:
  selector:
    matchLabels:
      app: ml-service
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
    targetPort: 9090

---
# PrometheusRule for alerting
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: uprez-alerts
  namespace: uprez-production
spec:
  groups:
  - name: uprez.rules
    rules:
    - alert: HighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: High error rate detected
        description: "Error rate is {{ $value }} errors per second"
        
    - alert: HighMemoryUsage
      expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: High memory usage
        description: "Memory usage is {{ $value | humanizePercentage }}"
        
    - alert: PodCrashLooping
      expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: Pod is crash looping
        description: "Pod {{ $labels.pod }} is crash looping"
```

This Kubernetes deployment documentation provides:

1. **Complete namespace organization** with resource quotas and limits
2. **Comprehensive configuration management** with ConfigMaps and Secrets
3. **Production-ready StatefulSets** for databases and caches
4. **Auto-scaling deployments** for application services
5. **Advanced ingress configuration** with security and performance optimizations
6. **Network security policies** for micro-segmentation
7. **RBAC and service accounts** for secure access control
8. **Persistent storage management** with different storage classes
9. **Monitoring and alerting** configuration with Prometheus

The configurations include production-ready settings, security best practices, and scalability considerations specifically for an IPO valuation platform.