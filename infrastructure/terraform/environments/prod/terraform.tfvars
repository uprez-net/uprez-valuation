# Production Environment Configuration
# IPO Valuation Platform - Production Environment

# Project Configuration
project_id = "uprez-valuation-prod"
project_name = "uprez-valuation"
environment = "prod"
region = "us-central1"
zone = "us-central1-a"

# Terraform State
terraform_state_bucket = "uprez-valuation-prod-terraform-state"

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
gke_release_channel = "STABLE"  # Use stable channel for production
enable_gke_autopilot = false
enable_workload_identity = true
enable_network_policy = true
enable_pod_security_policy = true
enable_shielded_nodes = true
enable_image_streaming = true

# Node Pools (production sizing)
node_pools = {
  general = {
    machine_type       = "e2-standard-4"
    disk_size_gb       = 100
    disk_type          = "pd-ssd"  # Use SSD for production
    image_type         = "COS_CONTAINERD"
    auto_repair        = true
    auto_upgrade       = true
    preemptible        = false  # No preemptible instances in production
    initial_node_count = 3
    min_node_count     = 3
    max_node_count     = 20
    node_labels = {
      role = "general"
      environment = "prod"
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
    preemptible        = false
    initial_node_count = 2
    min_node_count     = 2
    max_node_count     = 30
    node_labels = {
      role = "compute-intensive"
      environment = "prod"
    }
    node_taints = [
      {
        key    = "compute-intensive"
        value  = "true"
        effect = "NO_SCHEDULE"
      }
    ]
  }
  memory = {
    machine_type       = "n2-highmem-4"
    disk_size_gb       = 100
    disk_type          = "pd-ssd"
    image_type         = "COS_CONTAINERD"
    auto_repair        = true
    auto_upgrade       = true
    preemptible        = false
    initial_node_count = 1
    min_node_count     = 1
    max_node_count     = 10
    node_labels = {
      role = "memory-intensive"
      environment = "prod"
    }
    node_taints = [
      {
        key    = "memory-intensive"
        value  = "true"
        effect = "NO_SCHEDULE"
      }
    ]
  }
}

# Database Configuration (production-grade)
database_version = "POSTGRES_15"
database_tier = "db-standard-4"
database_disk_size = 500
database_disk_type = "PD_SSD"
backup_retention_days = 30
enable_point_in_time_recovery = true
enable_ha_replica = true  # High availability for production

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
          type = "SetStorageClass"
        }
        condition = {
          age        = 90
          with_state = "NEARLINE"
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
          age        = 2555  # 7 years for compliance
          with_state = "LIVE"
        }
      }
    ]
  }
  archive = {
    location      = "US"
    storage_class = "ARCHIVE"
    lifecycle_rules = []
  }
}

enable_bucket_versioning = true
bucket_retention_policy = 2592000  # 30 days

# Security Configuration
enable_binary_authorization = true

# Monitoring Configuration
enable_uptime_checks = true
notification_channels = [
  {
    type         = "email"
    display_name = "Production Alerts - Primary"
    description  = "Primary email for production alerts"
    labels = {
      email_address = "production-alerts@uprez.com"
    }
  },
  {
    type         = "email"
    display_name = "Production Alerts - CTO"
    description  = "CTO notification for critical issues"
    labels = {
      email_address = "cto@uprez.com"
    }
  },
  {
    type         = "slack"
    display_name = "Production Slack"
    description  = "Slack notifications for production"
    labels = {
      channel_name = "#production-alerts"
    }
  },
  {
    type         = "pagerduty"
    display_name = "Production PagerDuty"
    description  = "PagerDuty integration for production incidents"
    labels = {
      service_key = "PAGERDUTY_SERVICE_KEY_PLACEHOLDER"
    }
  }
]

custom_dashboards = {
  application_overview = {
    display_name = "Application Overview - Production"
    grid_layout = {
      columns = 3
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
          title = "Response Time (P95)"
          type  = "line_chart"
        },
        {
          title = "Active Users"
          type  = "scorecard"
        },
        {
          title = "Database Connections"
          type  = "line_chart"
        },
        {
          title = "Cache Hit Rate"
          type  = "line_chart"
        }
      ]
    }
  }
  infrastructure_health = {
    display_name = "Infrastructure Health - Production"
    grid_layout = {
      columns = 2
      widgets = [
        {
          title = "CPU Utilization"
          type  = "line_chart"
        },
        {
          title = "Memory Utilization"
          type  = "line_chart"
        },
        {
          title = "Disk Usage"
          type  = "line_chart"
        },
        {
          title = "Network Throughput"
          type  = "line_chart"
        }
      ]
    }
  }
  business_metrics = {
    display_name = "Business Metrics - Production"
    grid_layout = {
      columns = 2
      widgets = [
        {
          title = "Valuations Processed"
          type  = "scorecard"
        },
        {
          title = "Revenue Impact"
          type  = "line_chart"
        },
        {
          title = "User Engagement"
          type  = "line_chart"
        },
        {
          title = "Data Processing Time"
          type  = "histogram"
        }
      ]
    }
  }
}

log_based_metrics = {
  error_rate = {
    description = "Application error rate"
    filter      = "resource.type=\"k8s_container\" resource.labels.namespace_name=\"uprez-valuation-prod\" severity>=ERROR"
    metric_kind = "GAUGE"
    value_type  = "DOUBLE"
  }
  request_count = {
    description = "HTTP request count"
    filter      = "resource.type=\"k8s_container\" resource.labels.namespace_name=\"uprez-valuation-prod\" httpRequest.requestMethod!=\"\""
    metric_kind = "GAUGE"
    value_type  = "INT64"
  }
  security_events = {
    description = "Security-related events"
    filter      = "resource.type=\"k8s_container\" resource.labels.namespace_name=\"uprez-valuation-prod\" (jsonPayload.level=\"SECURITY\" OR textPayload:\"unauthorized\" OR textPayload:\"forbidden\")"
    metric_kind = "GAUGE"
    value_type  = "INT64"
  }
  business_events = {
    description = "Business metric events"
    filter      = "resource.type=\"k8s_container\" resource.labels.namespace_name=\"uprez-valuation-prod\" jsonPayload.event_type=\"business_metric\""
    metric_kind = "GAUGE"
    value_type  = "DOUBLE"
  }
}

# Feature Flags (all enabled for production)
enable_cost_optimization = true
enable_disaster_recovery = true
enable_multi_region = true  # Multi-region for production
enable_audit_logs = true
enable_data_encryption = true

# Additional Labels
additional_labels = {
  cost_center = "production"
  owner       = "platform-team"
  purpose     = "production"
  criticality = "high"
  compliance  = "required"
  backup      = "required"
}