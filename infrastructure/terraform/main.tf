# IPO Valuation Platform - Main Terraform Configuration
# Complete GCP infrastructure setup with best practices

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.4"
    }
  }

  backend "gcs" {
    bucket = var.terraform_state_bucket
    prefix = "terraform/state"
  }
}

# Configure the Google Cloud Provider
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Data sources
data "google_client_config" "default" {}

data "google_container_cluster" "primary" {
  name     = module.gke.cluster_name
  location = var.region
  depends_on = [module.gke]
}

# Configure Kubernetes provider
provider "kubernetes" {
  host  = "https://${data.google_container_cluster.primary.endpoint}"
  token = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(
    data.google_container_cluster.primary.master_auth[0].cluster_ca_certificate,
  )
}

provider "helm" {
  kubernetes {
    host  = "https://${data.google_container_cluster.primary.endpoint}"
    token = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(
      data.google_container_cluster.primary.master_auth[0].cluster_ca_certificate,
    )
  }
}

# Local values for consistent naming
locals {
  common_labels = {
    project     = "uprez-valuation"
    environment = var.environment
    managed_by  = "terraform"
    created_by  = "devops-team"
  }

  resource_prefix = "${var.project_name}-${var.environment}"
}

# Random suffix for unique resource names
resource "random_id" "suffix" {
  byte_length = 4
}

# Core Infrastructure Modules
module "networking" {
  source = "./modules/networking"

  project_id       = var.project_id
  region          = var.region
  environment     = var.environment
  resource_prefix = local.resource_prefix
  common_labels   = local.common_labels

  vpc_cidr_range              = var.vpc_cidr_range
  private_subnet_cidr_range   = var.private_subnet_cidr_range
  public_subnet_cidr_range    = var.public_subnet_cidr_range
  pod_cidr_range             = var.pod_cidr_range
  service_cidr_range         = var.service_cidr_range
  enable_nat_gateway         = var.enable_nat_gateway
  enable_cloud_armor         = var.enable_cloud_armor
}

module "security" {
  source = "./modules/security"

  project_id       = var.project_id
  region          = var.region
  environment     = var.environment
  resource_prefix = local.resource_prefix
  common_labels   = local.common_labels

  vpc_network_name    = module.networking.vpc_network_name
  private_subnet_name = module.networking.private_subnet_name
  enable_binary_auth  = var.enable_binary_authorization
  enable_pod_security = var.enable_pod_security_policy
}

module "database" {
  source = "./modules/database"

  project_id       = var.project_id
  region          = var.region
  environment     = var.environment
  resource_prefix = local.resource_prefix
  common_labels   = local.common_labels

  vpc_network_id          = module.networking.vpc_network_id
  private_subnet_name     = module.networking.private_subnet_name
  database_version        = var.database_version
  database_tier          = var.database_tier
  database_disk_size     = var.database_disk_size
  database_disk_type     = var.database_disk_type
  backup_retention_days  = var.backup_retention_days
  enable_point_in_time_recovery = var.enable_point_in_time_recovery
  enable_ha_replica      = var.enable_ha_replica
}

module "gke" {
  source = "./modules/compute"

  project_id       = var.project_id
  region          = var.region
  environment     = var.environment
  resource_prefix = local.resource_prefix
  common_labels   = local.common_labels

  vpc_network_name        = module.networking.vpc_network_name
  private_subnet_name     = module.networking.private_subnet_name
  pod_cidr_range         = var.pod_cidr_range
  service_cidr_range     = var.service_cidr_range

  # GKE Cluster Configuration
  cluster_version        = var.gke_cluster_version
  release_channel       = var.gke_release_channel
  enable_autopilot      = var.enable_gke_autopilot
  enable_workload_identity = var.enable_workload_identity
  enable_network_policy  = var.enable_network_policy
  enable_pod_security_policy = var.enable_pod_security_policy
  
  # Node Pool Configuration
  node_pools = var.node_pools
  
  # Security
  enable_shielded_nodes  = var.enable_shielded_nodes
  enable_image_streaming = var.enable_image_streaming
}

module "storage" {
  source = "./modules/storage"

  project_id       = var.project_id
  region          = var.region
  environment     = var.environment
  resource_prefix = local.resource_prefix
  common_labels   = local.common_labels

  storage_buckets = var.storage_buckets
  enable_versioning = var.enable_bucket_versioning
  retention_policy  = var.bucket_retention_policy
}

module "monitoring" {
  source = "./modules/monitoring"

  project_id       = var.project_id
  region          = var.region
  environment     = var.environment
  resource_prefix = local.resource_prefix
  common_labels   = local.common_labels

  # Monitoring Configuration
  enable_uptime_checks    = var.enable_uptime_checks
  notification_channels   = var.notification_channels
  custom_dashboards      = var.custom_dashboards
  
  # Log-based Metrics
  log_based_metrics      = var.log_based_metrics
  
  # Dependencies
  cluster_name = module.gke.cluster_name
  database_name = module.database.instance_name
}