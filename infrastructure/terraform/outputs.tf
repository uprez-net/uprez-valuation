# IPO Valuation Platform - Terraform Outputs
# Comprehensive output definitions for infrastructure components

# Project Information
output "project_id" {
  description = "The GCP project ID"
  value       = var.project_id
}

output "region" {
  description = "The GCP region"
  value       = var.region
}

output "environment" {
  description = "The deployment environment"
  value       = var.environment
}

# Networking Outputs
output "vpc_network_name" {
  description = "The name of the VPC network"
  value       = module.networking.vpc_network_name
}

output "vpc_network_id" {
  description = "The ID of the VPC network"
  value       = module.networking.vpc_network_id
}

output "vpc_network_self_link" {
  description = "The self-link of the VPC network"
  value       = module.networking.vpc_network_self_link
}

output "private_subnet_name" {
  description = "The name of the private subnet"
  value       = module.networking.private_subnet_name
}

output "private_subnet_cidr" {
  description = "The CIDR range of the private subnet"
  value       = module.networking.private_subnet_cidr
}

output "public_subnet_name" {
  description = "The name of the public subnet"
  value       = module.networking.public_subnet_name
}

output "public_subnet_cidr" {
  description = "The CIDR range of the public subnet"
  value       = module.networking.public_subnet_cidr
}

output "nat_gateway_ip" {
  description = "The external IP address of the NAT gateway"
  value       = module.networking.nat_gateway_ip
}

output "cloud_router_name" {
  description = "The name of the Cloud Router"
  value       = module.networking.cloud_router_name
}

# GKE Outputs
output "cluster_name" {
  description = "The name of the GKE cluster"
  value       = module.gke.cluster_name
}

output "cluster_endpoint" {
  description = "The endpoint of the GKE cluster"
  value       = module.gke.cluster_endpoint
  sensitive   = true
}

output "cluster_location" {
  description = "The location of the GKE cluster"
  value       = module.gke.cluster_location
}

output "cluster_ca_certificate" {
  description = "The cluster CA certificate"
  value       = module.gke.cluster_ca_certificate
  sensitive   = true
}

output "cluster_master_version" {
  description = "The Kubernetes master version"
  value       = module.gke.cluster_master_version
}

output "node_pools" {
  description = "Information about the node pools"
  value       = module.gke.node_pools
}

output "workload_identity_service_account" {
  description = "The Workload Identity service account"
  value       = module.gke.workload_identity_service_account
}

# Database Outputs
output "database_instance_name" {
  description = "The name of the database instance"
  value       = module.database.instance_name
}

output "database_connection_name" {
  description = "The connection name of the database instance"
  value       = module.database.connection_name
}

output "database_private_ip_address" {
  description = "The private IP address of the database instance"
  value       = module.database.private_ip_address
  sensitive   = true
}

output "database_public_ip_address" {
  description = "The public IP address of the database instance"
  value       = module.database.public_ip_address
  sensitive   = true
}

output "database_backup_configuration" {
  description = "The backup configuration of the database"
  value       = module.database.backup_configuration
}

output "database_replica_names" {
  description = "The names of database replicas"
  value       = module.database.replica_names
}

# Storage Outputs
output "storage_bucket_names" {
  description = "The names of the storage buckets"
  value       = module.storage.bucket_names
}

output "storage_bucket_urls" {
  description = "The URLs of the storage buckets"
  value       = module.storage.bucket_urls
}

output "storage_bucket_self_links" {
  description = "The self-links of the storage buckets"
  value       = module.storage.bucket_self_links
}

# Security Outputs
output "service_accounts" {
  description = "Service accounts created for the infrastructure"
  value       = module.security.service_accounts
}

output "iam_policies" {
  description = "IAM policies applied"
  value       = module.security.iam_policies
}

output "firewall_rules" {
  description = "Firewall rules created"
  value       = module.security.firewall_rules
}

output "secret_manager_secrets" {
  description = "Secret Manager secrets created"
  value       = module.security.secret_manager_secrets
  sensitive   = true
}

# Monitoring Outputs
output "monitoring_workspace" {
  description = "The monitoring workspace information"
  value       = module.monitoring.workspace
}

output "alerting_policies" {
  description = "Alerting policies created"
  value       = module.monitoring.alerting_policies
}

output "uptime_checks" {
  description = "Uptime checks configured"
  value       = module.monitoring.uptime_checks
}

output "notification_channels" {
  description = "Notification channels configured"
  value       = module.monitoring.notification_channels
}

output "custom_dashboards" {
  description = "Custom dashboards created"
  value       = module.monitoring.custom_dashboards
}

output "log_based_metrics" {
  description = "Log-based metrics created"
  value       = module.monitoring.log_based_metrics
}

# Kubernetes Configuration
output "kubectl_config_command" {
  description = "Command to configure kubectl"
  value       = "gcloud container clusters get-credentials ${module.gke.cluster_name} --region ${var.region} --project ${var.project_id}"
}

output "kubernetes_namespace" {
  description = "Default Kubernetes namespace for the application"
  value       = "${var.project_name}-${var.environment}"
}

# Application URLs
output "application_urls" {
  description = "Application URLs (will be populated after deployment)"
  value = {
    frontend = "https://${var.project_name}-${var.environment}-frontend.${var.region}.run.app"
    api      = "https://${var.project_name}-${var.environment}-api.${var.region}.run.app"
    admin    = "https://${var.project_name}-${var.environment}-admin.${var.region}.run.app"
  }
}

# Load Balancer Information
output "load_balancer_ip" {
  description = "External IP address of the load balancer"
  value       = module.networking.load_balancer_ip
}

output "ssl_certificate_name" {
  description = "Name of the SSL certificate"
  value       = module.networking.ssl_certificate_name
}

# Cost Management
output "resource_labels" {
  description = "Labels applied to resources for cost tracking"
  value       = local.common_labels
}

output "estimated_monthly_cost" {
  description = "Estimated monthly cost (USD) - requires cost calculation"
  value       = "Please run terraform plan with cost estimation enabled"
}

# Backup and Recovery
output "backup_schedule" {
  description = "Backup schedule configuration"
  value = {
    database = "Daily at 2 AM UTC"
    storage  = "Continuous with lifecycle rules"
    cluster  = "Weekly cluster snapshots"
  }
}

output "disaster_recovery_regions" {
  description = "Configured disaster recovery regions"
  value = var.enable_multi_region ? [
    var.region,
    var.region == "us-central1" ? "us-east1" : "us-central1"
  ] : [var.region]
}

# Compliance and Security
output "compliance_status" {
  description = "Compliance features enabled"
  value = {
    audit_logs        = var.enable_audit_logs
    data_encryption   = var.enable_data_encryption
    binary_auth       = var.enable_binary_authorization
    workload_identity = var.enable_workload_identity
    network_policy    = var.enable_network_policy
    pod_security      = var.enable_pod_security_policy
  }
}

output "security_features" {
  description = "Security features enabled"
  value = {
    shielded_nodes    = var.enable_shielded_nodes
    cloud_armor       = var.enable_cloud_armor
    private_cluster   = true
    secret_management = true
  }
}

# Terraform State
output "terraform_state_bucket" {
  description = "GCS bucket storing Terraform state"
  value       = var.terraform_state_bucket
}

# Version Information
output "infrastructure_version" {
  description = "Infrastructure version and build information"
  value = {
    terraform_version = "~> 1.0"
    google_provider   = "~> 5.0"
    kubernetes_version = var.gke_cluster_version
    deployment_time   = timestamp()
  }
}

# Documentation Links
output "documentation_links" {
  description = "Links to relevant documentation"
  value = {
    runbooks    = "infrastructure/docs/runbooks/"
    monitoring  = "infrastructure/docs/monitoring/"
    security    = "infrastructure/docs/security/"
    disaster_recovery = "infrastructure/docs/disaster-recovery/"
  }
}