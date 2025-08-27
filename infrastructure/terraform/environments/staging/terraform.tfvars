# Staging Environment Configuration
# IPO Valuation Platform - Staging Environment

# Project Configuration
project_id = "uprez-valuation-staging"
project_name = "uprez-valuation"
environment = "staging"
region = "us-central1"
zone = "us-central1-a"

# Terraform State
terraform_state_bucket = "uprez-valuation-staging-terraform-state"

# Networking
vpc_cidr_range = "10.0.0.0/16"
private_subnet_cidr_range = "10.0.1.0/24"
public_subnet_cidr_range = "10.0.2.0/24"
pod_cidr_range = "10.1.0.0/16"
service_cidr_range = "10.2.0.0/16"
enable_nat_gateway = true
enable_cloud_armor = true

# GKE Configuration
gke_cluster_version = "1.27.8-gke.1067004"
gke_release_channel = "REGULAR"
enable_gke_autopilot = false
enable_workload_identity = true
enable_network_policy = true
enable_pod_security_policy = true
enable_shielded_nodes = true
enable_image_streaming = true

# Node Pools (production-like sizing)
node_pools = {
  general = {
    machine_type       = "e2-standard-2"
    disk_size_gb       = 100
    disk_type          = "pd-standard"
    image_type         = "COS_CONTAINERD"
    auto_repair        = true
    auto_upgrade       = true
    preemptible        = true  # Use preemptible for cost savings
    initial_node_count = 2
    min_node_count     = 2
    max_node_count     = 8
    node_labels = {
      role = "general"
      environment = "staging"
    }
    node_taints = []
  }
  compute = {
    machine_type       = "c2-standard-4"
    disk_size_gb       = 100
    disk_type          = "pd-ssd"
    image_type         = "COS_CONTAINERD"
    auto_repair        = true
    auto_upgrade       = true
    preemptible        = true
    initial_node_count = 0
    min_node_count     = 0
    max_node_count     = 10
    node_labels = {
      role = "compute-intensive"
      environment = "staging"
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

# Database Configuration
database_version = "POSTGRES_15"
database_tier = "db-standard-1"
database_disk_size = 50
database_disk_type = "PD_SSD"
backup_retention_days = 14
enable_point_in_time_recovery = true
enable_ha_replica = false  # Single replica for staging

# Storage Configuration
storage_buckets = {
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
      },
      {
        action = {
          type = "Delete"
        }
        condition = {
          age        = 90
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
          age        = 180
          with_state = "LIVE"
        }
      }
    ]
  }
}

enable_bucket_versioning = true
bucket_retention_policy = 86400  # 1 day

# Security Configuration
enable_binary_authorization = true

# Monitoring Configuration
enable_uptime_checks = true
notification_channels = [
  {
    type         = "email"
    display_name = "Staging Team Email"
    description  = "Notification channel for staging environment"
    labels = {
      email_address = "staging-alerts@uprez.com"
    }
  },
  {
    type         = "slack"
    display_name = "Staging Slack Channel"
    description  = "Slack notifications for staging"
    labels = {
      channel_name = "#staging-alerts"
    }
  }
]

custom_dashboards = {
  application_overview = {
    display_name = "Application Overview - Staging"
    grid_layout = {
      columns = 2
      widgets = [
        {
          title = "Request Rate"
          type  = "line_chart"
        },
        {
          title = "Error Rate"
          type  = "line_chart"
        },
        {
          title = "Response Time"
          type  = "line_chart"
        },
        {
          title = "Active Users"
          type  = "scorecard"
        }
      ]
    }
  }
}

log_based_metrics = {
  error_rate = {
    description = "Application error rate"
    filter      = "resource.type=\"k8s_container\" resource.labels.namespace_name=\"uprez-valuation-staging\" severity>=ERROR"
    metric_kind = "GAUGE"
    value_type  = "DOUBLE"
  }
  request_count = {
    description = "HTTP request count"
    filter      = "resource.type=\"k8s_container\" resource.labels.namespace_name=\"uprez-valuation-staging\" httpRequest.requestMethod!=\"\""
    metric_kind = "GAUGE"
    value_type  = "INT64"
  }
}

# Feature Flags
enable_cost_optimization = true
enable_disaster_recovery = true
enable_multi_region = false
enable_audit_logs = true
enable_data_encryption = true

# Additional Labels
additional_labels = {
  cost_center = "engineering"
  owner       = "platform-team"
  purpose     = "staging"
  criticality = "medium"
}