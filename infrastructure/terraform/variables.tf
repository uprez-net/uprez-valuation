# IPO Valuation Platform - Terraform Variables
# Comprehensive variable definitions with validation

# Project Configuration
variable "project_id" {
  description = "The GCP project ID"
  type        = string
  validation {
    condition     = length(var.project_id) > 4 && can(regex("^[a-z][a-z0-9-]*[a-z0-9]$", var.project_id))
    error_message = "Project ID must be valid GCP project identifier."
  }
}

variable "project_name" {
  description = "Name of the project for resource naming"
  type        = string
  default     = "uprez-valuation"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for zonal resources"
  type        = string
  default     = "us-central1-a"
}

variable "terraform_state_bucket" {
  description = "GCS bucket for Terraform state"
  type        = string
}

# Networking Configuration
variable "vpc_cidr_range" {
  description = "CIDR range for the VPC"
  type        = string
  default     = "10.0.0.0/16"
  validation {
    condition     = can(cidrhost(var.vpc_cidr_range, 0))
    error_message = "VPC CIDR range must be a valid CIDR block."
  }
}

variable "private_subnet_cidr_range" {
  description = "CIDR range for private subnet"
  type        = string
  default     = "10.0.1.0/24"
}

variable "public_subnet_cidr_range" {
  description = "CIDR range for public subnet"
  type        = string
  default     = "10.0.2.0/24"
}

variable "pod_cidr_range" {
  description = "CIDR range for Kubernetes pods"
  type        = string
  default     = "10.1.0.0/16"
}

variable "service_cidr_range" {
  description = "CIDR range for Kubernetes services"
  type        = string
  default     = "10.2.0.0/16"
}

variable "enable_nat_gateway" {
  description = "Enable Cloud NAT gateway"
  type        = bool
  default     = true
}

variable "enable_cloud_armor" {
  description = "Enable Cloud Armor DDoS protection"
  type        = bool
  default     = true
}

# GKE Configuration
variable "gke_cluster_version" {
  description = "GKE cluster version"
  type        = string
  default     = "1.27.8-gke.1067004"
}

variable "gke_release_channel" {
  description = "GKE release channel"
  type        = string
  default     = "STABLE"
  validation {
    condition     = contains(["RAPID", "REGULAR", "STABLE"], var.gke_release_channel)
    error_message = "GKE release channel must be RAPID, REGULAR, or STABLE."
  }
}

variable "enable_gke_autopilot" {
  description = "Enable GKE Autopilot mode"
  type        = bool
  default     = false
}

variable "enable_workload_identity" {
  description = "Enable Workload Identity"
  type        = bool
  default     = true
}

variable "enable_network_policy" {
  description = "Enable Kubernetes Network Policy"
  type        = bool
  default     = true
}

variable "enable_pod_security_policy" {
  description = "Enable Pod Security Policy"
  type        = bool
  default     = true
}

variable "enable_shielded_nodes" {
  description = "Enable Shielded GKE Nodes"
  type        = bool
  default     = true
}

variable "enable_image_streaming" {
  description = "Enable Image Streaming for faster pod startup"
  type        = bool
  default     = true
}

variable "node_pools" {
  description = "GKE node pools configuration"
  type = map(object({
    machine_type     = string
    disk_size_gb     = number
    disk_type        = string
    image_type       = string
    auto_repair      = bool
    auto_upgrade     = bool
    preemptible      = bool
    initial_node_count = number
    min_node_count   = number
    max_node_count   = number
    node_labels      = map(string)
    node_taints = list(object({
      key    = string
      value  = string
      effect = string
    }))
  }))
  default = {
    general = {
      machine_type       = "e2-standard-4"
      disk_size_gb       = 100
      disk_type          = "pd-standard"
      image_type         = "COS_CONTAINERD"
      auto_repair        = true
      auto_upgrade       = true
      preemptible        = false
      initial_node_count = 1
      min_node_count     = 1
      max_node_count     = 10
      node_labels = {
        role = "general"
      }
      node_taints = []
    }
    compute = {
      machine_type       = "c2-standard-8"
      disk_size_gb       = 200
      disk_type          = "pd-ssd"
      image_type         = "COS_CONTAINERD"
      auto_repair        = true
      auto_upgrade       = true
      preemptible        = true
      initial_node_count = 0
      min_node_count     = 0
      max_node_count     = 20
      node_labels = {
        role = "compute-intensive"
      }
      node_taints = [
        {
          key    = "compute-intensive"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
}

# Database Configuration
variable "database_version" {
  description = "PostgreSQL version"
  type        = string
  default     = "POSTGRES_15"
}

variable "database_tier" {
  description = "Cloud SQL tier"
  type        = string
  default     = "db-standard-2"
}

variable "database_disk_size" {
  description = "Database disk size in GB"
  type        = number
  default     = 100
  validation {
    condition     = var.database_disk_size >= 10
    error_message = "Database disk size must be at least 10 GB."
  }
}

variable "database_disk_type" {
  description = "Database disk type"
  type        = string
  default     = "PD_SSD"
  validation {
    condition     = contains(["PD_SSD", "PD_HDD"], var.database_disk_type)
    error_message = "Database disk type must be PD_SSD or PD_HDD."
  }
}

variable "backup_retention_days" {
  description = "Database backup retention in days"
  type        = number
  default     = 30
  validation {
    condition     = var.backup_retention_days >= 7 && var.backup_retention_days <= 365
    error_message = "Backup retention must be between 7 and 365 days."
  }
}

variable "enable_point_in_time_recovery" {
  description = "Enable point-in-time recovery"
  type        = bool
  default     = true
}

variable "enable_ha_replica" {
  description = "Enable high availability replica"
  type        = bool
  default     = true
}

# Storage Configuration
variable "storage_buckets" {
  description = "Storage bucket configurations"
  type = map(object({
    location      = string
    storage_class = string
    lifecycle_rules = list(object({
      action = object({
        type = string
      })
      condition = object({
        age        = number
        with_state = string
      })
    }))
  }))
  default = {
    data = {
      location      = "US"
      storage_class = "STANDARD"
      lifecycle_rules = [
        {
          action = {
            type = "SetStorageClass"
          }
          condition = {
            age        = 30
            with_state = "LIVE"
          }
        }
      ]
    }
    backups = {
      location      = "US"
      storage_class = "COLDLINE"
      lifecycle_rules = [
        {
          action = {
            type = "Delete"
          }
          condition = {
            age        = 365
            with_state = "LIVE"
          }
        }
      ]
    }
  }
}

variable "enable_bucket_versioning" {
  description = "Enable bucket versioning"
  type        = bool
  default     = true
}

variable "bucket_retention_policy" {
  description = "Bucket retention policy in seconds"
  type        = number
  default     = 86400 # 1 day
}

# Security Configuration
variable "enable_binary_authorization" {
  description = "Enable Binary Authorization"
  type        = bool
  default     = true
}

# Monitoring Configuration
variable "enable_uptime_checks" {
  description = "Enable uptime monitoring"
  type        = bool
  default     = true
}

variable "notification_channels" {
  description = "Notification channels for alerts"
  type = list(object({
    type         = string
    display_name = string
    description  = string
    labels       = map(string)
  }))
  default = [
    {
      type         = "email"
      display_name = "DevOps Team Email"
      description  = "Primary notification channel for DevOps team"
      labels = {
        email_address = "devops@uprez.com"
      }
    }
  ]
}

variable "custom_dashboards" {
  description = "Custom dashboard configurations"
  type = map(object({
    display_name = string
    grid_layout = object({
      columns = number
      widgets = list(object({
        title = string
        type  = string
      }))
    })
  }))
  default = {}
}

variable "log_based_metrics" {
  description = "Log-based metrics configuration"
  type = map(object({
    description = string
    filter      = string
    metric_kind = string
    value_type  = string
  }))
  default = {
    error_rate = {
      description = "Application error rate"
      filter      = "resource.type=\"k8s_container\" severity>=ERROR"
      metric_kind = "GAUGE"
      value_type  = "DOUBLE"
    }
  }
}

# Feature Flags
variable "enable_cost_optimization" {
  description = "Enable cost optimization features"
  type        = bool
  default     = true
}

variable "enable_disaster_recovery" {
  description = "Enable disaster recovery setup"
  type        = bool
  default     = true
}

variable "enable_multi_region" {
  description = "Enable multi-region deployment"
  type        = bool
  default     = false
}

# Tags and Labels
variable "additional_labels" {
  description = "Additional labels to apply to all resources"
  type        = map(string)
  default     = {}
}

# Compliance
variable "enable_audit_logs" {
  description = "Enable audit logging"
  type        = bool
  default     = true
}

variable "enable_data_encryption" {
  description = "Enable data encryption at rest"
  type        = bool
  default     = true
}