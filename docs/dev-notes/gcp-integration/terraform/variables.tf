# Variables for IPO Valuation Platform GCP Infrastructure

variable "project_id" {
  description = "The GCP project ID"
  type        = string
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{4,28}[a-z0-9]$", var.project_id))
    error_message = "Project ID must be between 6 and 30 characters, start with a letter, and contain only lowercase letters, numbers, and hyphens."
  }
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "production"
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

variable "region" {
  description = "The GCP region for resources"
  type        = string
  default     = "australia-southeast1"
  validation {
    condition     = can(regex("^[a-z]+-[a-z]+[0-9]$", var.region))
    error_message = "Region must be a valid GCP region format."
  }
}

variable "zone" {
  description = "The GCP zone for resources"
  type        = string
  default     = "australia-southeast1-a"
  validation {
    condition     = can(regex("^[a-z]+-[a-z]+[0-9]-[a-z]$", var.zone))
    error_message = "Zone must be a valid GCP zone format."
  }
}

# Notification and alerting configuration
variable "alert_email" {
  description = "Email address for critical alerts"
  type        = string
  validation {
    condition     = can(regex("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$", var.alert_email))
    error_message = "Alert email must be a valid email address."
  }
}

variable "slack_channel" {
  description = "Slack channel name for alerts"
  type        = string
  default     = "#ipo-platform-alerts"
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for notifications"
  type        = string
  sensitive   = true
  default     = ""
}

# Machine learning configuration
variable "ml_training_machine_type" {
  description = "Machine type for ML training workloads"
  type        = string
  default     = "n1-standard-8"
}

variable "ml_training_accelerator_type" {
  description = "Accelerator type for ML training"
  type        = string
  default     = "NVIDIA_TESLA_T4"
}

variable "ml_training_accelerator_count" {
  description = "Number of accelerators for ML training"
  type        = number
  default     = 1
}

variable "model_serving_machine_type" {
  description = "Machine type for model serving"
  type        = string
  default     = "n1-standard-4"
}

variable "model_serving_min_replicas" {
  description = "Minimum number of serving replicas"
  type        = number
  default     = 1
}

variable "model_serving_max_replicas" {
  description = "Maximum number of serving replicas"
  type        = number
  default     = 10
}

# Feature Store configuration
variable "featurestore_online_serving_nodes" {
  description = "Number of nodes for Feature Store online serving"
  type        = number
  default     = 2
}

variable "featurestore_min_nodes" {
  description = "Minimum nodes for Feature Store scaling"
  type        = number
  default     = 1
}

variable "featurestore_max_nodes" {
  description = "Maximum nodes for Feature Store scaling"
  type        = number
  default     = 10
}

# BigQuery configuration
variable "bigquery_default_table_expiration_days" {
  description = "Default table expiration in days for BigQuery datasets"
  type        = number
  default     = 365
}

variable "bigquery_slot_reservation" {
  description = "BigQuery slot reservation for consistent performance"
  type        = number
  default     = 100
}

# Storage configuration
variable "storage_retention_policy_days" {
  description = "Retention policy for Cloud Storage buckets in days"
  type        = number
  default     = 2555  # 7 years for financial data
}

variable "enable_storage_versioning" {
  description = "Enable versioning for Cloud Storage buckets"
  type        = bool
  default     = true
}

# Document AI configuration
variable "document_ai_processors" {
  description = "Map of Document AI processors to create"
  type = map(object({
    type         = string
    display_name = string
  }))
  default = {
    prospectus = {
      type         = "FORM_PARSER_PROCESSOR"
      display_name = "IPO Prospectus Parser"
    }
    annual_report = {
      type         = "FORM_PARSER_PROCESSOR"
      display_name = "Annual Report Parser"
    }
    financial_statement = {
      type         = "FORM_PARSER_PROCESSOR"
      display_name = "Financial Statement Parser"
    }
  }
}

# Natural Language AI configuration
variable "natural_language_sentiment_enabled" {
  description = "Enable sentiment analysis in Natural Language AI"
  type        = bool
  default     = true
}

variable "natural_language_entity_sentiment_enabled" {
  description = "Enable entity sentiment analysis"
  type        = bool
  default     = true
}

# Monitoring configuration
variable "monitoring_retention_days" {
  description = "Monitoring data retention period in days"
  type        = number
  default     = 90
}

variable "enable_detailed_monitoring" {
  description = "Enable detailed monitoring with custom metrics"
  type        = bool
  default     = true
}

variable "alerting_policy_enabled" {
  description = "Enable alerting policies"
  type        = bool
  default     = true
}

# Security configuration
variable "enable_audit_logs" {
  description = "Enable audit logging for all services"
  type        = bool
  default     = true
}

variable "enable_vpc_flow_logs" {
  description = "Enable VPC flow logs"
  type        = bool
  default     = true
}

variable "require_ssl" {
  description = "Require SSL for all external communications"
  type        = bool
  default     = true
}

variable "enable_private_google_access" {
  description = "Enable private Google access for VPC"
  type        = bool
  default     = true
}

# Networking configuration
variable "vpc_cidr_range" {
  description = "CIDR range for VPC subnet"
  type        = string
  default     = "10.0.0.0/16"
}

variable "authorized_networks" {
  description = "List of authorized networks for access"
  type = list(object({
    name  = string
    value = string
  }))
  default = []
}

# Kubernetes and container configuration
variable "enable_workload_identity" {
  description = "Enable Workload Identity for Kubernetes integration"
  type        = bool
  default     = false
}

variable "kubernetes_namespace" {
  description = "Kubernetes namespace for Workload Identity"
  type        = string
  default     = "ipo-valuation"
}

# CI/CD configuration
variable "enable_cicd" {
  description = "Enable CI/CD with Cloud Build"
  type        = bool
  default     = false
}

variable "github_owner" {
  description = "GitHub repository owner for CI/CD"
  type        = string
  default     = ""
}

variable "github_repo" {
  description = "GitHub repository name for CI/CD"
  type        = string
  default     = ""
}

# Cost optimization configuration
variable "enable_preemptible_instances" {
  description = "Enable preemptible instances for cost optimization"
  type        = bool
  default     = false
}

variable "budget_amount" {
  description = "Monthly budget amount for cost alerting"
  type        = number
  default     = 5000
}

variable "budget_threshold_percentage" {
  description = "Budget threshold percentage for alerts"
  type        = number
  default     = 80
}

# Backup and disaster recovery
variable "enable_cross_region_backup" {
  description = "Enable cross-region backup for critical data"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 90
}

variable "disaster_recovery_region" {
  description = "Secondary region for disaster recovery"
  type        = string
  default     = "australia-southeast2"
}

# Feature flags
variable "enable_experimental_features" {
  description = "Enable experimental ML features"
  type        = bool
  default     = false
}

variable "enable_auto_scaling" {
  description = "Enable auto-scaling for ML workloads"
  type        = bool
  default     = true
}

variable "enable_spot_instances" {
  description = "Enable spot/preemptible instances for training workloads"
  type        = bool
  default     = false
}

# Data residency and compliance
variable "data_residency_region" {
  description = "Region for data residency compliance"
  type        = string
  default     = "australia-southeast1"
}

variable "enable_cmek" {
  description = "Enable Customer-Managed Encryption Keys (CMEK)"
  type        = bool
  default     = false
}

variable "cmek_key_ring" {
  description = "Cloud KMS key ring for CMEK"
  type        = string
  default     = "ipo-valuation-keyring"
}

# Resource quotas and limits
variable "vertex_ai_quota_models" {
  description = "Quota limit for Vertex AI models"
  type        = number
  default     = 100
}

variable "vertex_ai_quota_endpoints" {
  description = "Quota limit for Vertex AI endpoints"
  type        = number
  default     = 50
}

variable "bigquery_slot_commitment" {
  description = "BigQuery slot commitment for cost optimization"
  type        = string
  default     = "FLEX"
  validation {
    condition     = contains(["FLEX", "ANNUAL", "MONTHLY"], var.bigquery_slot_commitment)
    error_message = "BigQuery slot commitment must be one of: FLEX, ANNUAL, MONTHLY."
  }
}

# External integration configuration
variable "external_apis" {
  description = "Configuration for external API integrations"
  type = map(object({
    enabled     = bool
    api_key_secret = string
    rate_limit  = number
  }))
  default = {
    asx_data_api = {
      enabled        = true
      api_key_secret = "asx-data-api-key"
      rate_limit     = 1000
    }
    reuters_api = {
      enabled        = false
      api_key_secret = "reuters-api-key"
      rate_limit     = 500
    }
  }
}

# Tagging and labeling
variable "additional_labels" {
  description = "Additional labels to apply to all resources"
  type        = map(string)
  default     = {}
}

variable "cost_center" {
  description = "Cost center for billing allocation"
  type        = string
  default     = "ml-platform"
}

variable "owner_team" {
  description = "Team responsible for the infrastructure"
  type        = string
  default     = "data-science"
}

# Performance and scaling
variable "enable_gpu_sharing" {
  description = "Enable GPU sharing for training workloads"
  type        = bool
  default     = false
}

variable "max_training_jobs_concurrent" {
  description = "Maximum number of concurrent training jobs"
  type        = number
  default     = 5
}

variable "prediction_cache_ttl" {
  description = "Prediction cache TTL in seconds"
  type        = number
  default     = 300
}

# Development and testing
variable "enable_sandbox_mode" {
  description = "Enable sandbox mode for development"
  type        = bool
  default     = false
}

variable "enable_debug_logging" {
  description = "Enable debug-level logging"
  type        = bool
  default     = false
}