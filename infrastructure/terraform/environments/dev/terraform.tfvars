# Development Environment Configuration
# IPO Valuation Platform - Dev Environment

# Project Configuration
project_id = "uprez-valuation-dev"
project_name = "uprez-valuation"
environment = "dev"
region = "us-central1"
zone = "us-central1-a"

# Terraform State
terraform_state_bucket = "uprez-valuation-dev-terraform-state"

# Networking
vpc_cidr_range = "10.0.0.0/16"
private_subnet_cidr_range = "10.0.1.0/24"
public_subnet_cidr_range = "10.0.2.0/24"
pod_cidr_range = "10.1.0.0/16"
service_cidr_range = "10.2.0.0/16"
enable_nat_gateway = true
enable_cloud_armor = false  # Disabled for dev to save costs

# GKE Configuration
gke_cluster_version = "1.27.8-gke.1067004"
gke_release_channel = "REGULAR"
enable_gke_autopilot = false
enable_workload_identity = true
enable_network_policy = true
enable_pod_security_policy = true
enable_shielded_nodes = true
enable_image_streaming = true

# Node Pools (smaller for dev)
node_pools = {
  general = {
    machine_type       = "e2-medium"
    disk_size_gb       = 50
    disk_type          = "pd-standard"
    image_type         = "COS_CONTAINERD"
    auto_repair        = true
    auto_upgrade       = true
    preemptible        = true  # Use preemptible for cost savings
    initial_node_count = 1
    min_node_count     = 1
    max_node_count     = 5
    node_labels = {
      role = "general"
      environment = "dev"
    }
    node_taints = []
  }
}

# Database Configuration (smaller for dev)
database_version = "POSTGRES_15"
database_tier = "db-f1-micro"  # Smallest tier for dev
database_disk_size = 20
database_disk_type = "PD_SSD"
backup_retention_days = 7
enable_point_in_time_recovery = false  # Disabled for cost savings
enable_ha_replica = false  # Disabled for dev

# Storage Configuration
storage_buckets = {
  data = {
    location      = "US"
    storage_class = "STANDARD"
    lifecycle_rules = [
      {
        action = {
          type = "Delete"
        }
        condition = {
          age        = 30
          with_state = "LIVE"
        }
      }
    ]
  }
}

enable_bucket_versioning = false  # Disabled for dev
bucket_retention_policy = 86400  # 1 day

# Security Configuration
enable_binary_authorization = false  # Disabled for dev

# Monitoring Configuration
enable_uptime_checks = true
notification_channels = [
  {
    type         = "email"
    display_name = "Dev Team Email"
    description  = "Notification channel for dev environment"
    labels = {
      email_address = "dev-team@uprez.com"
    }
  }
]

# Feature Flags
enable_cost_optimization = true
enable_disaster_recovery = false  # Disabled for dev
enable_multi_region = false
enable_audit_logs = false  # Disabled for cost savings
enable_data_encryption = false  # Disabled for dev

# Additional Labels
additional_labels = {
  cost_center = "engineering"
  owner       = "dev-team"
  purpose     = "development"
}