# Containerization Strategy for IPO Valuation Platform

## Overview

This document outlines the comprehensive containerization strategy for the Uprez IPO valuation platform, including Docker configurations, multi-stage builds, security considerations, and orchestration patterns.

## Architecture Overview

```
Production Deployment Stack:
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Frontend      │  │   API Gateway   │  │   ML Services   │
│   (React/Next)  │  │   (Kong/Nginx)  │  │   (FastAPI)     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                     │                     │
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Backend API   │  │   Auth Service  │  │   Data Pipeline │
│   (FastAPI)     │  │   (OAuth2/JWT)  │  │   (Airflow)     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                     │                     │
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   PostgreSQL    │  │   Redis Cache   │  │   MinIO Storage │
│   (Primary DB)  │  │   (Sessions)    │  │   (Models/Data) │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## 1. Base Images and Multi-Stage Builds

### Python ML Service Dockerfile

```dockerfile
# docker/ml-service/Dockerfile
# Multi-stage build for ML inference service

# =====================
# Stage 1: Dependencies
# =====================
FROM python:3.11-slim as dependencies

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# =====================
# Stage 2: Development
# =====================
FROM dependencies as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY --chown=appuser:appuser . .

# Switch to app user
USER appuser

EXPOSE 8000

# Development command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =====================
# Stage 3: Production
# =====================
FROM dependencies as production

# Copy only necessary files
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser config/ ./config/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to app user
USER appuser

EXPOSE 8000

# Production command
CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", "--timeout", "120"]

# =====================
# Stage 4: Model Training
# =====================
FROM dependencies as training

# Install additional training dependencies
RUN pip install --no-cache-dir \
    optuna \
    mlflow \
    jupyter \
    matplotlib \
    seaborn

# Copy training scripts
COPY --chown=appuser:appuser training/ ./training/
COPY --chown=appuser:appuser data/ ./data/
COPY --chown=appuser:appuser notebooks/ ./notebooks/

# Switch to app user
USER appuser

# Training command
CMD ["python", "training/train_model.py"]
```

### Frontend Application Dockerfile

```dockerfile
# docker/frontend/Dockerfile
# Multi-stage build for React/Next.js frontend

# =====================
# Stage 1: Dependencies
# =====================
FROM node:18-alpine as dependencies

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY yarn.lock ./

# Install dependencies
RUN yarn install --frozen-lockfile

# =====================
# Stage 2: Build
# =====================
FROM dependencies as builder

WORKDIR /app

# Copy source code
COPY . .

# Build application
RUN yarn build

# =====================
# Stage 3: Production
# =====================
FROM node:18-alpine as production

# Create app user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

WORKDIR /app

# Copy built application
COPY --from=builder --chown=nextjs:nodejs /app/.next ./.next
COPY --from=builder --chown=nextjs:nodejs /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/package.json ./package.json
COPY --from=builder --chown=nextjs:nodejs /app/node_modules ./node_modules

# Switch to app user
USER nextjs

EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

# Production command
CMD ["yarn", "start"]
```

### Database Migration Dockerfile

```dockerfile
# docker/migrations/Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r migrator && useradd --no-log-init -r -g migrator migrator

WORKDIR /app

# Copy requirements
COPY requirements-migration.txt .
RUN pip install --no-cache-dir -r requirements-migration.txt

# Copy migration scripts
COPY --chown=migrator:migrator migrations/ ./migrations/
COPY --chown=migrator:migrator alembic.ini .
COPY --chown=migrator:migrator alembic/ ./alembic/

# Switch to app user
USER migrator

# Migration command
CMD ["alembic", "upgrade", "head"]
```

## 2. Docker Compose Configurations

### Development Environment

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  # Database
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: uprez_ipo_dev
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: devpassword
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - uprez-network
    
  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - uprez-network
      
  # MinIO for object storage
  minio:
    image: minio/minio:latest
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data
    networks:
      - uprez-network
      
  # Backend API
  api:
    build:
      context: .
      dockerfile: docker/api/Dockerfile
      target: development
    environment:
      - DATABASE_URL=postgresql://postgres:devpassword@postgres:5432/uprez_ipo_dev
      - REDIS_URL=redis://redis:6379
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=minioadmin
      - MINIO_SECRET_KEY=minioadmin
    ports:
      - "8000:8000"
    volumes:
      - .:/app
      - /app/node_modules
    depends_on:
      - postgres
      - redis
      - minio
    networks:
      - uprez-network
      
  # ML Service
  ml-service:
    build:
      context: .
      dockerfile: docker/ml-service/Dockerfile
      target: development
    environment:
      - MODEL_STORE_URL=http://minio:9000
      - REDIS_URL=redis://redis:6379
    ports:
      - "8001:8000"
    volumes:
      - ./models:/app/models
      - ./training:/app/training
    depends_on:
      - minio
      - redis
    networks:
      - uprez-network
      
  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: ../docker/frontend/Dockerfile
      target: dependencies
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
      - NEXT_PUBLIC_ML_SERVICE_URL=http://localhost:8001
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    command: yarn dev
    networks:
      - uprez-network
      
  # Data Pipeline (Airflow)
  airflow-webserver:
    build:
      context: .
      dockerfile: docker/airflow/Dockerfile
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql://postgres:devpassword@postgres:5432/airflow_dev
      - AIRFLOW__CORE__FERNET_KEY=your-fernet-key-here
      - AIRFLOW__WEBSERVER__SECRET_KEY=your-secret-key-here
    ports:
      - "8080:8080"
    volumes:
      - ./dags:/opt/airflow/dags
      - ./plugins:/opt/airflow/plugins
    depends_on:
      - postgres
    networks:
      - uprez-network
    command: >
      bash -c "airflow db init &&
               airflow users create --role Admin --username admin --email admin@example.com --firstname admin --lastname admin --password admin &&
               airflow webserver"

volumes:
  postgres_data:
  redis_data:
  minio_data:

networks:
  uprez-network:
    driver: bridge
```

### Production Environment

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # Reverse Proxy / Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./docker/nginx/ssl:/etc/nginx/ssl
    depends_on:
      - api
      - frontend
    networks:
      - uprez-network
    restart: unless-stopped
    
  # Database (Production should use managed service)
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - uprez-network
    restart: unless-stopped
    
  # Redis Cache
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    networks:
      - uprez-network
    restart: unless-stopped
    
  # Backend API (Multiple replicas)
  api:
    build:
      context: .
      dockerfile: docker/api/Dockerfile
      target: production
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - JWT_SECRET=${JWT_SECRET}
      - ENCRYPTION_KEY=${ENCRYPTION_KEY}
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
    depends_on:
      - postgres
      - redis
    networks:
      - uprez-network
    restart: unless-stopped
    
  # ML Service (Multiple replicas)
  ml-service:
    build:
      context: .
      dockerfile: docker/ml-service/Dockerfile
      target: production
    environment:
      - MODEL_STORE_URL=${MODEL_STORE_URL}
      - REDIS_URL=${REDIS_URL}
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
    volumes:
      - model_storage:/app/models:ro
    networks:
      - uprez-network
    restart: unless-stopped
    
  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: ../docker/frontend/Dockerfile
      target: production
    environment:
      - NEXT_PUBLIC_API_URL=${PUBLIC_API_URL}
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
    networks:
      - uprez-network
    restart: unless-stopped
    
  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./docker/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - uprez-network
    restart: unless-stopped
    
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - uprez-network
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  model_storage:
  grafana_data:

networks:
  uprez-network:
    driver: overlay
    attachable: true
```

## 3. Container Security Configuration

### Security Best Practices Implementation

```dockerfile
# docker/security/secure-base.Dockerfile
# Secure base image with security hardening

FROM python:3.11-slim

# Update and install security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

# Create non-root user with specific UID/GID
RUN groupadd -r appuser -g 1000 && \
    useradd --no-log-init -r -u 1000 -g appuser appuser

# Set up secure directory structure
RUN mkdir -p /app && \
    chown -R appuser:appuser /app && \
    chmod 755 /app

# Remove unnecessary packages and files
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Security labels
LABEL maintainer="security@uprez.com" \
      security.scan="enabled" \
      security.policy="restricted"

WORKDIR /app
USER appuser

# Security constraints
# - No root access
# - Read-only filesystem where possible
# - Minimal attack surface
# - Regular security updates
```

### Container Security Scanner

```python
# scripts/security/container_security_scan.py
import docker
import subprocess
import json
import logging
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class SecurityVulnerability:
    severity: str
    description: str
    package: str
    version: str
    fixed_version: str = None

class ContainerSecurityScanner:
    """
    Container security scanning and vulnerability assessment
    """
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.scanners = ['trivy', 'grype', 'clair']
        
    def scan_image(self, image_name: str, scanner: str = 'trivy') -> Dict[str, Any]:
        """
        Scan container image for vulnerabilities
        """
        if scanner == 'trivy':
            return self._scan_with_trivy(image_name)
        elif scanner == 'grype':
            return self._scan_with_grype(image_name)
        else:
            raise ValueError(f"Unsupported scanner: {scanner}")
    
    def _scan_with_trivy(self, image_name: str) -> Dict[str, Any]:
        """
        Scan image using Trivy
        """
        try:
            cmd = [
                'trivy', 'image',
                '--format', 'json',
                '--severity', 'HIGH,CRITICAL',
                '--no-progress',
                image_name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            scan_results = json.loads(result.stdout)
            
            vulnerabilities = []
            
            if 'Results' in scan_results:
                for result in scan_results['Results']:
                    if 'Vulnerabilities' in result:
                        for vuln in result['Vulnerabilities']:
                            vulnerabilities.append(SecurityVulnerability(
                                severity=vuln.get('Severity', 'UNKNOWN'),
                                description=vuln.get('Description', ''),
                                package=vuln.get('PkgName', ''),
                                version=vuln.get('InstalledVersion', ''),
                                fixed_version=vuln.get('FixedVersion', None)
                            ))
            
            return {
                'scanner': 'trivy',
                'image': image_name,
                'vulnerabilities': vulnerabilities,
                'total_count': len(vulnerabilities),
                'critical_count': len([v for v in vulnerabilities if v.severity == 'CRITICAL']),
                'high_count': len([v for v in vulnerabilities if v.severity == 'HIGH'])
            }
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Trivy scan failed: {e}")
            return {'error': str(e)}
    
    def _scan_with_grype(self, image_name: str) -> Dict[str, Any]:
        """
        Scan image using Grype
        """
        try:
            cmd = [
                'grype',
                '--output', 'json',
                '--fail-on', 'high',
                image_name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            scan_results = json.loads(result.stdout)
            
            vulnerabilities = []
            
            if 'matches' in scan_results:
                for match in scan_results['matches']:
                    vuln_info = match.get('vulnerability', {})
                    artifact_info = match.get('artifact', {})
                    
                    vulnerabilities.append(SecurityVulnerability(
                        severity=vuln_info.get('severity', 'UNKNOWN'),
                        description=vuln_info.get('description', ''),
                        package=artifact_info.get('name', ''),
                        version=artifact_info.get('version', ''),
                        fixed_version=vuln_info.get('fix', {}).get('versions', [None])[0]
                    ))
            
            return {
                'scanner': 'grype',
                'image': image_name,
                'vulnerabilities': vulnerabilities,
                'total_count': len(vulnerabilities),
                'critical_count': len([v for v in vulnerabilities if v.severity == 'Critical']),
                'high_count': len([v for v in vulnerabilities if v.severity == 'High'])
            }
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Grype scan failed: {e}")
            return {'error': str(e)}
    
    def generate_security_report(self, scan_results: Dict[str, Any]) -> str:
        """
        Generate human-readable security report
        """
        report = []
        report.append(f"Security Scan Report for {scan_results['image']}")
        report.append("=" * 50)
        report.append(f"Scanner: {scan_results['scanner']}")
        report.append(f"Total Vulnerabilities: {scan_results['total_count']}")
        report.append(f"Critical: {scan_results['critical_count']}")
        report.append(f"High: {scan_results['high_count']}")
        report.append("")
        
        if scan_results['vulnerabilities']:
            report.append("Vulnerabilities:")
            report.append("-" * 20)
            
            for vuln in scan_results['vulnerabilities']:
                report.append(f"• {vuln.severity}: {vuln.package} ({vuln.version})")
                report.append(f"  Description: {vuln.description}")
                if vuln.fixed_version:
                    report.append(f"  Fixed in: {vuln.fixed_version}")
                report.append("")
        
        return "\n".join(report)
    
    def check_image_compliance(self, image_name: str) -> Dict[str, Any]:
        """
        Check if image meets security compliance requirements
        """
        compliance_checks = {
            'non_root_user': False,
            'no_critical_vulns': False,
            'minimal_attack_surface': False,
            'regular_base_image': False,
            'signed_image': False
        }
        
        try:
            # Get image inspection
            image = self.docker_client.images.get(image_name)
            config = image.attrs.get('Config', {})
            
            # Check if running as non-root
            user = config.get('User', '')
            if user and user != 'root' and user != '0':
                compliance_checks['non_root_user'] = True
            
            # Scan for vulnerabilities
            scan_results = self.scan_image(image_name)
            if scan_results.get('critical_count', 1) == 0:
                compliance_checks['no_critical_vulns'] = True
            
            # Check base image age (simplified check)
            created = image.attrs.get('Created', '')
            # Add logic to check if base image is recent
            
            # Calculate compliance score
            score = sum(compliance_checks.values()) / len(compliance_checks) * 100
            
            return {
                'image': image_name,
                'compliance_score': score,
                'checks': compliance_checks,
                'passed': score >= 80  # Require 80% compliance
            }
            
        except Exception as e:
            logging.error(f"Compliance check failed: {e}")
            return {'error': str(e)}

# Example usage and automation script
def automated_security_pipeline(images: List[str]):
    """
    Automated security scanning pipeline
    """
    scanner = ContainerSecurityScanner()
    results = {}
    
    for image in images:
        logging.info(f"Scanning {image}...")
        
        # Vulnerability scan
        scan_result = scanner.scan_image(image)
        
        # Compliance check
        compliance_result = scanner.check_image_compliance(image)
        
        # Generate report
        if 'error' not in scan_result:
            report = scanner.generate_security_report(scan_result)
            
            results[image] = {
                'vulnerabilities': scan_result,
                'compliance': compliance_result,
                'report': report,
                'passed': (scan_result['critical_count'] == 0 and 
                          compliance_result.get('passed', False))
            }
        else:
            results[image] = {'error': scan_result['error']}
            
        logging.info(f"Scan completed for {image}")
    
    return results

if __name__ == "__main__":
    # Scan production images
    production_images = [
        'uprez/api:latest',
        'uprez/ml-service:latest',
        'uprez/frontend:latest'
    ]
    
    results = automated_security_pipeline(production_images)
    
    # Output results
    for image, result in results.items():
        if 'error' not in result:
            print(f"\n{result['report']}")
            print(f"Security Status: {'PASSED' if result['passed'] else 'FAILED'}")
        else:
            print(f"\nError scanning {image}: {result['error']}")
```

## 4. Container Registry and Image Management

### Google Container Registry Integration

```python
# scripts/deployment/container_registry.py
import docker
import subprocess
import json
import logging
from google.cloud import container_v1
from google.oauth2 import service_account
from typing import Dict, List, Any
from datetime import datetime, timedelta

class ContainerRegistryManager:
    """
    Manage container images in Google Container Registry
    """
    
    def __init__(self, project_id: str, registry_url: str = "gcr.io"):
        self.project_id = project_id
        self.registry_url = registry_url
        self.docker_client = docker.from_env()
        
    def build_and_push_image(self, dockerfile_path: str, image_name: str,
                           tag: str = "latest", build_args: Dict[str, str] = None) -> str:
        """
        Build and push image to GCR
        """
        full_image_name = f"{self.registry_url}/{self.project_id}/{image_name}:{tag}"
        
        try:
            # Build image
            logging.info(f"Building image: {full_image_name}")
            
            image, build_logs = self.docker_client.images.build(
                path=dockerfile_path,
                tag=full_image_name,
                buildargs=build_args or {},
                rm=True,
                forcerm=True
            )
            
            # Log build output
            for log in build_logs:
                if 'stream' in log:
                    logging.info(log['stream'].strip())
            
            # Push image
            logging.info(f"Pushing image: {full_image_name}")
            
            push_logs = self.docker_client.images.push(
                full_image_name,
                stream=True,
                decode=True
            )
            
            # Log push output
            for log in push_logs:
                if 'status' in log:
                    logging.info(f"{log['status']}: {log.get('progress', '')}")
            
            logging.info(f"Successfully pushed: {full_image_name}")
            return full_image_name
            
        except Exception as e:
            logging.error(f"Failed to build/push image: {e}")
            raise
    
    def list_images(self, repository: str) -> List[Dict[str, Any]]:
        """
        List images in GCR repository
        """
        try:
            cmd = [
                'gcloud', 'container', 'images', 'list-tags',
                f"{self.registry_url}/{self.project_id}/{repository}",
                '--format=json'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            images = json.loads(result.stdout)
            
            return images
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to list images: {e}")
            return []
    
    def cleanup_old_images(self, repository: str, keep_count: int = 10) -> List[str]:
        """
        Clean up old images, keeping only the most recent ones
        """
        images = self.list_images(repository)
        
        if len(images) <= keep_count:
            logging.info(f"Only {len(images)} images found, no cleanup needed")
            return []
        
        # Sort by timestamp (newest first)
        sorted_images = sorted(
            images, 
            key=lambda x: x.get('timestamp', {}).get('datetime', ''), 
            reverse=True
        )
        
        # Keep the most recent images, delete the rest
        images_to_delete = sorted_images[keep_count:]
        deleted_images = []
        
        for image in images_to_delete:
            digest = image.get('digest')
            if digest:
                image_ref = f"{self.registry_url}/{self.project_id}/{repository}@{digest}"
                
                try:
                    cmd = ['gcloud', 'container', 'images', 'delete', image_ref, '--quiet']
                    subprocess.run(cmd, check=True)
                    deleted_images.append(image_ref)
                    logging.info(f"Deleted: {image_ref}")
                    
                except subprocess.CalledProcessError as e:
                    logging.error(f"Failed to delete {image_ref}: {e}")
        
        return deleted_images
    
    def scan_image_vulnerabilities(self, image_name: str) -> Dict[str, Any]:
        """
        Scan image for vulnerabilities using Google Container Analysis
        """
        from google.cloud import containeranalysis_v1
        
        client = containeranalysis_v1.ContainerAnalysisClient()
        grafeas_client = client.get_grafeas_client()
        
        project_name = f"projects/{self.project_id}"
        image_url = f"{self.registry_url}/{self.project_id}/{image_name}"
        
        # Wait for scan to complete
        import time
        time.sleep(30)  # Give time for scan to process
        
        # Get vulnerabilities
        occurrence_filter = f'resourceUrl="https://{image_url}" AND kind="VULNERABILITY"'
        
        occurrences = grafeas_client.list_occurrences(
            parent=project_name,
            filter=occurrence_filter
        )
        
        vulnerabilities = []
        severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        
        for occurrence in occurrences:
            vuln = occurrence.vulnerability
            severity = vuln.severity.name if vuln.severity else 'UNKNOWN'
            
            vulnerabilities.append({
                'severity': severity,
                'cve': occurrence.note_name.split('/')[-1],
                'description': vuln.short_description,
                'package': vuln.package_issue[0].affected_package if vuln.package_issue else '',
                'fixed_version': vuln.package_issue[0].fixed_version if vuln.package_issue else ''
            })
            
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        return {
            'image': image_name,
            'vulnerabilities': vulnerabilities,
            'total_count': len(vulnerabilities),
            'severity_counts': severity_counts
        }

class ImageBuildPipeline:
    """
    Automated image build and deployment pipeline
    """
    
    def __init__(self, registry_manager: ContainerRegistryManager):
        self.registry = registry_manager
        
    def build_all_services(self, version_tag: str) -> Dict[str, str]:
        """
        Build all service images with version tag
        """
        services = [
            {
                'name': 'api',
                'dockerfile': 'docker/api/Dockerfile',
                'context': '.',
                'target': 'production'
            },
            {
                'name': 'ml-service',
                'dockerfile': 'docker/ml-service/Dockerfile',
                'context': '.',
                'target': 'production'
            },
            {
                'name': 'frontend',
                'dockerfile': 'docker/frontend/Dockerfile',
                'context': './frontend',
                'target': 'production'
            },
            {
                'name': 'data-pipeline',
                'dockerfile': 'docker/airflow/Dockerfile',
                'context': '.',
                'target': 'production'
            }
        ]
        
        built_images = {}
        
        for service in services:
            try:
                build_args = {}
                if 'target' in service:
                    build_args['--target'] = service['target']
                
                # Build with both version tag and latest
                version_image = self.registry.build_and_push_image(
                    dockerfile_path=service['context'],
                    image_name=service['name'],
                    tag=version_tag,
                    build_args=build_args
                )
                
                latest_image = self.registry.build_and_push_image(
                    dockerfile_path=service['context'],
                    image_name=service['name'],
                    tag='latest',
                    build_args=build_args
                )
                
                built_images[service['name']] = {
                    'version': version_image,
                    'latest': latest_image
                }
                
            except Exception as e:
                logging.error(f"Failed to build {service['name']}: {e}")
                raise
        
        return built_images
    
    def deploy_to_staging(self, images: Dict[str, str]) -> bool:
        """
        Deploy images to staging environment
        """
        try:
            # Update staging deployment with new images
            # This would integrate with Kubernetes or Cloud Run
            
            # Example: Update Kubernetes deployment
            for service_name, image_info in images.items():
                self._update_kubernetes_deployment(
                    service_name, 
                    image_info['version'],
                    namespace='staging'
                )
            
            logging.info("Staging deployment completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Staging deployment failed: {e}")
            return False
    
    def _update_kubernetes_deployment(self, service_name: str, 
                                    image: str, namespace: str) -> None:
        """
        Update Kubernetes deployment with new image
        """
        cmd = [
            'kubectl', 'set', 'image',
            f'deployment/{service_name}',
            f'{service_name}={image}',
            f'--namespace={namespace}'
        ]
        
        subprocess.run(cmd, check=True)
        
        # Wait for rollout to complete
        cmd = [
            'kubectl', 'rollout', 'status',
            f'deployment/{service_name}',
            f'--namespace={namespace}',
            '--timeout=300s'
        ]
        
        subprocess.run(cmd, check=True)

# Build script for CI/CD
def main():
    """
    Main build script for CI/CD pipeline
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Build and deploy container images')
    parser.add_argument('--project-id', required=True, help='GCP Project ID')
    parser.add_argument('--version', required=True, help='Version tag')
    parser.add_argument('--deploy-staging', action='store_true', help='Deploy to staging')
    
    args = parser.parse_args()
    
    # Initialize registry manager
    registry = ContainerRegistryManager(args.project_id)
    
    # Initialize build pipeline
    pipeline = ImageBuildPipeline(registry)
    
    try:
        # Build all images
        built_images = pipeline.build_all_services(args.version)
        
        # Deploy to staging if requested
        if args.deploy_staging:
            success = pipeline.deploy_to_staging(built_images)
            if not success:
                exit(1)
        
        # Cleanup old images
        for service_name in built_images.keys():
            registry.cleanup_old_images(service_name, keep_count=10)
        
        logging.info("Build pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Build pipeline failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
```

This containerization strategy documentation provides:

1. **Multi-stage Docker builds** for optimized production images
2. **Development and production** Docker Compose configurations
3. **Security hardening** and vulnerability scanning
4. **Container registry management** with Google Cloud integration
5. **Automated build pipelines** for CI/CD integration

The documentation includes executable scripts and configuration files that development and DevOps teams can use to maintain secure, scalable containerized deployments.