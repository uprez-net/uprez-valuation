# IPO Valuation SaaS - Comprehensive Security Infrastructure
# Main Terraform configuration for Google Cloud Platform

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
  }
  
  backend "gcs" {
    bucket = "ipo-valuation-terraform-state"
    prefix = "security/terraform.tfstate"
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
data "google_organization" "org" {
  domain = var.organization_domain
}

data "google_project" "project" {
  project_id = var.project_id
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "compute.googleapis.com",
    "container.googleapis.com",
    "cloudsql.googleapis.com",
    "storage.googleapis.com",
    "bigquery.googleapis.com",
    "cloudkms.googleapis.com",
    "secretmanager.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "securitycenter.googleapis.com",
    "dlp.googleapis.com",
    "binaryauthorization.googleapis.com",
    "iap.googleapis.com",
    "accesscontextmanager.googleapis.com",
    "cloudasset.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
    "cloudbuild.googleapis.com",
    "pubsub.googleapis.com",
    "cloudfunctions.googleapis.com",
    "networkservices.googleapis.com",
    "certificatemanager.googleapis.com"
  ])

  service            = each.value
  disable_on_destroy = false
  
  timeouts {
    create = "30m"
    update = "40m"
  }
}

# ===== KMS ENCRYPTION INFRASTRUCTURE =====

# KMS Key Rings
resource "google_kms_key_ring" "key_rings" {
  for_each = var.kms_key_rings
  
  name     = each.key
  location = var.region
  
  depends_on = [google_project_service.required_apis]
}

# KMS Crypto Keys
resource "google_kms_crypto_key" "crypto_keys" {
  for_each = var.kms_crypto_keys
  
  name     = each.value.name
  key_ring = google_kms_key_ring.key_rings[each.value.key_ring].id
  purpose  = each.value.purpose
  
  rotation_period = each.value.rotation_period
  
  version_template {
    algorithm        = each.value.algorithm
    protection_level = each.value.protection_level
  }
  
  labels = merge(
    var.common_labels,
    {
      data_type = each.value.data_type
      environment = var.environment
    }
  )
  
  lifecycle {
    prevent_destroy = true
  }
}

# KMS Key IAM Bindings
resource "google_kms_crypto_key_iam_binding" "key_bindings" {
  for_each = var.kms_key_bindings
  
  crypto_key_id = google_kms_crypto_key.crypto_keys[each.value.key_name].id
  role          = each.value.role
  members       = each.value.members
  
  dynamic "condition" {
    for_each = each.value.condition != null ? [each.value.condition] : []
    content {
      title       = condition.value.title
      description = condition.value.description
      expression  = condition.value.expression
    }
  }
}

# ===== NETWORK SECURITY INFRASTRUCTURE =====

# Production VPC
resource "google_compute_network" "prod_vpc" {
  name                    = "ipo-valuation-prod-vpc"
  auto_create_subnetworks = false
  description            = "Production VPC for IPO Valuation SaaS"
  routing_mode           = "REGIONAL"
  mtu                    = 1460
  
  depends_on = [google_project_service.required_apis]
}

# VPC Subnets
resource "google_compute_subnetwork" "subnets" {
  for_each = var.vpc_subnets
  
  name                     = each.value.name
  ip_cidr_range           = each.value.ip_cidr_range
  region                  = var.region
  network                 = google_compute_network.prod_vpc.id
  private_ip_google_access = each.value.private_ip_google_access
  description             = each.value.description
  
  log_config {
    aggregation_interval = "INTERVAL_5_SEC"
    flow_sampling       = each.value.flow_sampling
    metadata           = "INCLUDE_ALL_METADATA"
    metadata_fields    = [
      "src_instance",
      "dest_instance",
      "project_id",
      "vpc_name",
      "subnet_name"
    ]
  }
  
  dynamic "secondary_ip_range" {
    for_each = each.value.secondary_ip_ranges != null ? each.value.secondary_ip_ranges : []
    content {
      range_name    = secondary_ip_range.value.range_name
      ip_cidr_range = secondary_ip_range.value.ip_cidr_range
    }
  }
}

# Cloud Router for NAT
resource "google_compute_router" "prod_router" {
  name    = "ipo-prod-router"
  region  = var.region
  network = google_compute_network.prod_vpc.id
  
  bgp {
    asn = var.bgp_asn
  }
}

# Static IPs for NAT
resource "google_compute_address" "nat_ips" {
  count  = 2
  name   = "ipo-prod-nat-ip-${count.index + 1}"
  region = var.region
}

# Cloud NAT
resource "google_compute_router_nat" "prod_nat" {
  name                               = "ipo-prod-nat"
  router                            = google_compute_router.prod_router.name
  region                            = var.region
  nat_ip_allocate_option            = "MANUAL_ONLY"
  nat_ips                           = google_compute_address.nat_ips[*].self_link
  source_subnetwork_ip_ranges_to_nat = "LIST_OF_SUBNETWORKS"
  
  subnetwork {
    name                    = google_compute_subnetwork.subnets["app-subnet"].id
    source_ip_ranges_to_nat = ["PRIMARY_IP_RANGE"]
  }
  
  log_config {
    enable = true
    filter = "ALL"
  }
}

# Firewall Rules
resource "google_compute_firewall" "firewall_rules" {
  for_each = var.firewall_rules
  
  name        = each.key
  network     = google_compute_network.prod_vpc.name
  description = each.value.description
  direction   = each.value.direction
  priority    = each.value.priority
  
  dynamic "allow" {
    for_each = each.value.allow != null ? each.value.allow : []
    content {
      protocol = allow.value.protocol
      ports    = allow.value.ports
    }
  }
  
  dynamic "deny" {
    for_each = each.value.deny != null ? each.value.deny : []
    content {
      protocol = deny.value.protocol
      ports    = deny.value.ports
    }
  }
  
  source_ranges      = each.value.source_ranges
  destination_ranges = each.value.destination_ranges
  source_tags        = each.value.source_tags
  target_tags        = each.value.target_tags
  
  log_config {
    metadata = "INCLUDE_ALL_METADATA"
  }
}

# Cloud Armor Security Policy
resource "google_compute_security_policy" "main_policy" {
  name        = "ipo-valuation-security-policy"
  description = "Main security policy for IPO valuation platform"
  
  # Rate limiting rule
  rule {
    action   = "rate_based_ban"
    priority = "1000"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    rate_limit_options {
      conform_action = "allow"
      exceed_action  = "deny(429)"
      enforce_on_key = "IP"
      rate_limit_threshold {
        count        = var.rate_limit_requests_per_minute
        interval_sec = 60
      }
      ban_threshold {
        count        = var.rate_limit_ban_threshold
        interval_sec = 300
      }
      ban_duration_sec = 1800
    }
    description = "Rate limiting - requests per minute"
  }
  
  # OWASP protection rule
  rule {
    action   = "deny(403)"
    priority = "2000"
    match {
      expr {
        expression = <<-EOT
          evaluatePreconfiguredExpr('sqli-stable') ||
          evaluatePreconfiguredExpr('xss-stable') ||
          evaluatePreconfiguredExpr('lfi-stable') ||
          evaluatePreconfiguredExpr('rfi-stable') ||
          evaluatePreconfiguredExpr('rce-stable') ||
          evaluatePreconfiguredExpr('methodenforcement-stable') ||
          evaluatePreconfiguredExpr('scannerdetection-stable') ||
          evaluatePreconfiguredExpr('protocolattack-stable')
        EOT
      }
    }
    description = "OWASP Top 10 protection"
  }
  
  # Geographic restriction
  rule {
    action   = "deny(403)"
    priority = "3000"
    match {
      expr {
        expression = var.geographic_restriction_enabled ? "origin.region_code != 'AU' && origin.region_code != 'NZ'" : "false"
      }
    }
    description = "Geographic restriction to Australia and New Zealand"
  }
  
  # Default allow rule
  rule {
    action   = "allow"
    priority = "2147483647"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    description = "Default allow rule"
  }
  
  adaptive_protection_config {
    layer_7_ddos_defense_config {
      enable          = true
      rule_visibility = "STANDARD"
    }
  }
}

# ===== IDENTITY AND ACCESS MANAGEMENT =====

# Custom IAM Roles
resource "google_project_iam_custom_role" "custom_roles" {
  for_each = var.custom_iam_roles
  
  role_id     = each.key
  title       = each.value.title
  description = each.value.description
  permissions = each.value.permissions
  stage       = each.value.stage
}

# IAM Policy Bindings
resource "google_project_iam_binding" "project_bindings" {
  for_each = var.iam_bindings
  
  project = var.project_id
  role    = each.value.role
  members = each.value.members
  
  dynamic "condition" {
    for_each = each.value.condition != null ? [each.value.condition] : []
    content {
      title       = condition.value.title
      description = condition.value.description
      expression  = condition.value.expression
    }
  }
}

# Service Accounts
resource "google_service_account" "service_accounts" {
  for_each = var.service_accounts
  
  account_id   = each.key
  display_name = each.value.display_name
  description  = each.value.description
}

# Service Account Keys (only for external integrations)
resource "google_service_account_key" "external_keys" {
  for_each = {
    for k, v in var.service_accounts : k => v
    if v.create_key == true
  }
  
  service_account_id = google_service_account.service_accounts[each.key].name
}

# ===== ACCESS CONTEXT MANAGER (VPC SERVICE CONTROLS) =====

# Access Policy
resource "google_access_context_manager_access_policy" "policy" {
  parent = "organizations/${data.google_organization.org.org_id}"
  title  = "IPO Valuation Access Policy"
  
  lifecycle {
    prevent_destroy = true
  }
}

# Access Levels
resource "google_access_context_manager_access_level" "access_levels" {
  for_each = var.access_levels
  
  parent = google_access_context_manager_access_policy.policy.name
  name   = "accessPolicies/${google_access_context_manager_access_policy.policy.name}/accessLevels/${each.key}"
  title  = each.value.title
  
  basic {
    combining_function = each.value.combining_function
    
    dynamic "conditions" {
      for_each = each.value.conditions
      content {
        ip_subnetworks = conditions.value.ip_subnetworks
        members        = conditions.value.members
        regions        = conditions.value.regions
        
        dynamic "device_policy" {
          for_each = conditions.value.device_policy != null ? [conditions.value.device_policy] : []
          content {
            require_screen_lock              = device_policy.value.require_screen_lock
            require_admin_approval          = device_policy.value.require_admin_approval
            require_corp_owned              = device_policy.value.require_corp_owned
            allowed_device_management_levels = device_policy.value.allowed_device_management_levels
          }
        }
      }
    }
  }
}

# Service Perimeter
resource "google_access_context_manager_service_perimeter" "perimeter" {
  parent = google_access_context_manager_access_policy.policy.name
  name   = "accessPolicies/${google_access_context_manager_access_policy.policy.name}/servicePerimeters/ipo_production"
  title  = "IPO Production Perimeter"
  
  status {
    resources = [
      "projects/${data.google_project.project.number}"
    ]
    
    access_levels = values(google_access_context_manager_access_level.access_levels)[*].name
    
    restricted_services = [
      "storage.googleapis.com",
      "cloudsql.googleapis.com",
      "secretmanager.googleapis.com",
      "cloudkms.googleapis.com",
      "bigquery.googleapis.com"
    ]
    
    vpc_accessible_services {
      enable_restriction = true
      allowed_services = [
        "storage.googleapis.com",
        "cloudsql.googleapis.com"
      ]
    }
    
    ingress_policies {
      ingress_from {
        sources {
          access_level = google_access_context_manager_access_level.access_levels["high_security"].name
        }
        identity_type = "ANY_IDENTITY"
      }
      
      ingress_to {
        resources = ["*"]
        operations {
          service_name = "storage.googleapis.com"
          method_selectors {
            method = "google.storage.objects.get"
          }
          method_selectors {
            method = "google.storage.objects.list"
          }
        }
      }
    }
  }
  
  lifecycle {
    prevent_destroy = true
  }
}

# ===== CLOUD STORAGE WITH ENCRYPTION =====

# Encrypted Storage Buckets
resource "google_storage_bucket" "encrypted_buckets" {
  for_each = var.storage_buckets
  
  name                        = each.value.name
  location                    = each.value.location
  storage_class              = each.value.storage_class
  uniform_bucket_level_access = true
  
  encryption {
    default_kms_key_name = google_kms_crypto_key.crypto_keys[each.value.kms_key].id
  }
  
  versioning {
    enabled = each.value.versioning_enabled
  }
  
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
  
  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }
  
  lifecycle_rule {
    condition {
      age = 2555 # 7 years for financial records
    }
    action {
      type = "Delete"
    }
  }
  
  logging {
    log_bucket        = google_storage_bucket.access_logs.name
    log_object_prefix = "${each.key}/"
  }
  
  labels = merge(
    var.common_labels,
    {
      data_classification = each.value.data_classification
    }
  )
}

# Access logs bucket
resource "google_storage_bucket" "access_logs" {
  name                        = "${var.project_id}-access-logs"
  location                    = var.region
  storage_class              = "COLDLINE"
  uniform_bucket_level_access = true
  
  lifecycle_rule {
    condition {
      age = 2555 # 7 years retention
    }
    action {
      type = "Delete"
    }
  }
}

# ===== CLOUD SQL WITH ENCRYPTION =====

# Cloud SQL instances
resource "google_sql_database_instance" "instances" {
  for_each = var.cloudsql_instances
  
  name             = each.value.name
  database_version = each.value.database_version
  region           = var.region
  
  settings {
    tier                        = each.value.tier
    availability_type          = each.value.availability_type
    disk_type                  = "PD_SSD"
    disk_size                  = each.value.disk_size
    disk_autoresize           = true
    disk_autoresize_limit     = each.value.disk_autoresize_limit
    
    database_flags {
      name  = "log_statement"
      value = "all"
    }
    
    database_flags {
      name  = "log_min_duration_statement"
      value = "1000" # Log queries taking more than 1 second
    }
    
    backup_configuration {
      enabled                        = true
      start_time                    = "02:00"
      location                      = var.region
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
      backup_retention_settings {
        retained_backups = 30
        retention_unit   = "COUNT"
      }
    }
    
    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.prod_vpc.id
      require_ssl     = true
      
      dynamic "authorized_networks" {
        for_each = each.value.authorized_networks != null ? each.value.authorized_networks : []
        content {
          name  = authorized_networks.value.name
          value = authorized_networks.value.cidr
        }
      }
    }
    
    maintenance_window {
      day          = 7 # Sunday
      hour         = 3 # 3 AM AEST
      update_track = "stable"
    }
    
    user_labels = merge(
      var.common_labels,
      {
        database_type = each.value.database_type
      }
    )
  }
  
  encryption_key_name = google_kms_crypto_key.crypto_keys[each.value.kms_key].id
  
  deletion_protection = each.value.deletion_protection
  
  depends_on = [
    google_service_networking_connection.private_vpc_connection
  ]
}

# Private VPC connection for Cloud SQL
resource "google_compute_global_address" "private_ip_address" {
  name          = "private-ip-address"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.prod_vpc.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.prod_vpc.id
  service                = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
}

# ===== MONITORING AND LOGGING =====

# BigQuery datasets for security logs
resource "google_bigquery_dataset" "security_datasets" {
  for_each = var.bigquery_datasets
  
  dataset_id    = each.key
  friendly_name = each.value.friendly_name
  description   = each.value.description
  location      = each.value.location
  
  default_table_expiration_ms = each.value.default_table_expiration_ms
  
  default_encryption_configuration {
    kms_key_name = google_kms_crypto_key.crypto_keys[each.value.kms_key].id
  }
  
  labels = var.common_labels
}

# Log sinks
resource "google_logging_project_sink" "log_sinks" {
  for_each = var.log_sinks
  
  name        = each.key
  destination = each.value.destination
  filter      = each.value.filter
  
  unique_writer_identity = true
  
  bigquery_options {
    use_partitioned_tables = each.value.use_partitioned_tables
  }
}

# Monitoring alert policies
resource "google_monitoring_alert_policy" "alert_policies" {
  for_each = var.alert_policies
  
  display_name = each.value.display_name
  combiner     = each.value.combiner
  enabled      = each.value.enabled
  
  dynamic "conditions" {
    for_each = each.value.conditions
    content {
      display_name = conditions.value.display_name
      
      condition_threshold {
        filter         = conditions.value.filter
        duration       = conditions.value.duration
        comparison     = conditions.value.comparison
        threshold_value = conditions.value.threshold_value
        
        aggregations {
          alignment_period   = conditions.value.alignment_period
          per_series_aligner = conditions.value.per_series_aligner
        }
      }
    }
  }
  
  notification_channels = each.value.notification_channels
  
  alert_strategy {
    auto_close = each.value.auto_close
  }
  
  documentation {
    content   = each.value.documentation
    mime_type = "text/markdown"
  }
}

# ===== OUTPUTS =====

output "project_id" {
  description = "The project ID"
  value       = var.project_id
}

output "vpc_network_name" {
  description = "The name of the VPC network"
  value       = google_compute_network.prod_vpc.name
}

output "vpc_network_id" {
  description = "The ID of the VPC network"
  value       = google_compute_network.prod_vpc.id
}

output "subnet_ids" {
  description = "The IDs of the subnets"
  value = {
    for k, v in google_compute_subnetwork.subnets : k => v.id
  }
}

output "kms_key_ids" {
  description = "The IDs of KMS keys"
  value = {
    for k, v in google_kms_crypto_key.crypto_keys : k => v.id
  }
}

output "service_account_emails" {
  description = "The email addresses of service accounts"
  value = {
    for k, v in google_service_account.service_accounts : k => v.email
  }
}

output "storage_bucket_names" {
  description = "The names of storage buckets"
  value = {
    for k, v in google_storage_bucket.encrypted_buckets : k => v.name
  }
}

output "cloudsql_instance_connection_names" {
  description = "The connection names of Cloud SQL instances"
  value = {
    for k, v in google_sql_database_instance.instances : k => v.connection_name
  }
}

output "access_policy_name" {
  description = "The name of the access context manager policy"
  value       = google_access_context_manager_access_policy.policy.name
}

output "service_perimeter_name" {
  description = "The name of the service perimeter"
  value       = google_access_context_manager_service_perimeter.perimeter.name
}

output "cloud_armor_policy_id" {
  description = "The ID of the Cloud Armor security policy"
  value       = google_compute_security_policy.main_policy.id
}