# Variables for IPO Valuation SaaS Security Infrastructure

variable "project_id" {
  description = "The GCP project ID"
  type        = string
  validation {
    condition     = length(var.project_id) > 0
    error_message = "Project ID cannot be empty."
  }
}

variable "organization_domain" {
  description = "The organization domain for access policies"
  type        = string
  default     = "uprez.com"
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "australia-southeast1"
  validation {
    condition = contains([
      "australia-southeast1",
      "australia-southeast2"
    ], var.region)
    error_message = "Region must be in Australia for data residency compliance."
  }
}

variable "zone" {
  description = "The GCP zone"
  type        = string
  default     = "australia-southeast1-a"
}

variable "environment" {
  description = "Environment name (prod, staging, dev)"
  type        = string
  default     = "prod"
  validation {
    condition     = contains(["prod", "staging", "dev"], var.environment)
    error_message = "Environment must be prod, staging, or dev."
  }
}

variable "bgp_asn" {
  description = "BGP ASN for Cloud Router"
  type        = number
  default     = 64512
}

variable "rate_limit_requests_per_minute" {
  description = "Rate limit for requests per minute in Cloud Armor"
  type        = number
  default     = 100
}

variable "rate_limit_ban_threshold" {
  description = "Ban threshold for rate limiting"
  type        = number
  default     = 500
}

variable "geographic_restriction_enabled" {
  description = "Enable geographic restriction to Australia and New Zealand"
  type        = bool
  default     = true
}

variable "common_labels" {
  description = "Common labels to apply to all resources"
  type        = map(string)
  default = {
    project     = "ipo-valuation"
    environment = "production"
    managed_by  = "terraform"
    compliance  = "sox-apra-iso27001"
  }
}

# KMS Configuration
variable "kms_key_rings" {
  description = "KMS key rings configuration"
  type        = map(object({
    location = string
  }))
  default = {
    "ipo-documents" = {
      location = "australia-southeast1"
    }
    "financial-data" = {
      location = "australia-southeast1"
    }
    "user-data" = {
      location = "australia-southeast1"
    }
    "database" = {
      location = "australia-southeast1"
    }
    "compute" = {
      location = "australia-southeast1"
    }
    "audit-logs" = {
      location = "australia-southeast1"
    }
  }
}

variable "kms_crypto_keys" {
  description = "KMS crypto keys configuration"
  type = map(object({
    name             = string
    key_ring         = string
    purpose          = string
    rotation_period  = string
    algorithm        = string
    protection_level = string
    data_type        = string
  }))
  default = {
    "documents-key" = {
      name             = "documents-key"
      key_ring         = "ipo-documents"
      purpose          = "ENCRYPT_DECRYPT"
      rotation_period  = "2592000s" # 30 days
      algorithm        = "GOOGLE_SYMMETRIC_ENCRYPTION"
      protection_level = "HSM"
      data_type        = "documents"
    }
    "financial-key" = {
      name             = "financial-key"
      key_ring         = "financial-data"
      purpose          = "ENCRYPT_DECRYPT"
      rotation_period  = "1296000s" # 15 days
      algorithm        = "GOOGLE_SYMMETRIC_ENCRYPTION"
      protection_level = "HSM"
      data_type        = "financial"
    }
    "user-key" = {
      name             = "user-key"
      key_ring         = "user-data"
      purpose          = "ENCRYPT_DECRYPT"
      rotation_period  = "2592000s"
      algorithm        = "GOOGLE_SYMMETRIC_ENCRYPTION"
      protection_level = "HSM"
      data_type        = "personal"
    }
    "database-key" = {
      name             = "sql-key"
      key_ring         = "database"
      purpose          = "ENCRYPT_DECRYPT"
      rotation_period  = "2592000s"
      algorithm        = "GOOGLE_SYMMETRIC_ENCRYPTION"
      protection_level = "HSM"
      data_type        = "database"
    }
    "compute-key" = {
      name             = "boot-disk-key"
      key_ring         = "compute"
      purpose          = "ENCRYPT_DECRYPT"
      rotation_period  = "2592000s"
      algorithm        = "GOOGLE_SYMMETRIC_ENCRYPTION"
      protection_level = "HSM"
      data_type        = "compute"
    }
    "audit-key" = {
      name             = "audit-key"
      key_ring         = "audit-logs"
      purpose          = "ENCRYPT_DECRYPT"
      rotation_period  = "5184000s" # 60 days
      algorithm        = "GOOGLE_SYMMETRIC_ENCRYPTION"
      protection_level = "HSM"
      data_type        = "audit"
    }
  }
}

variable "kms_key_bindings" {
  description = "KMS key IAM bindings"
  type = map(object({
    key_name = string
    role     = string
    members  = list(string)
    condition = optional(object({
      title       = string
      description = string
      expression  = string
    }))
  }))
  default = {
    "financial-key-app-access" = {
      key_name = "financial-key"
      role     = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
      members  = ["serviceAccount:ipo-app@PROJECT_ID.iam.gserviceaccount.com"]
      condition = {
        title       = "Business hours only"
        description = "Only allow access during business hours"
        expression  = "request.time.getHours() >= 9 && request.time.getHours() <= 17"
      }
    }
  }
}

# VPC Configuration
variable "vpc_subnets" {
  description = "VPC subnet configuration"
  type = map(object({
    name                     = string
    ip_cidr_range           = string
    private_ip_google_access = bool
    description             = string
    flow_sampling           = number
    secondary_ip_ranges = optional(list(object({
      range_name    = string
      ip_cidr_range = string
    })))
  }))
  default = {
    "app-subnet" = {
      name                     = "ipo-app-subnet"
      ip_cidr_range           = "10.1.0.0/24"
      private_ip_google_access = true
      description             = "Application tier subnet"
      flow_sampling           = 0.5
      secondary_ip_ranges = [
        {
          range_name    = "pod-range"
          ip_cidr_range = "10.11.0.0/16"
        },
        {
          range_name    = "service-range"
          ip_cidr_range = "10.12.0.0/16"
        }
      ]
    }
    "database-subnet" = {
      name                     = "ipo-db-subnet"
      ip_cidr_range           = "10.1.1.0/24"
      private_ip_google_access = true
      description             = "Database tier subnet (private)"
      flow_sampling           = 1.0
    }
    "management-subnet" = {
      name                     = "ipo-mgmt-subnet"
      ip_cidr_range           = "10.1.2.0/24"
      private_ip_google_access = true
      description             = "Management and monitoring subnet"
      flow_sampling           = 1.0
    }
  }
}

# Firewall Rules Configuration
variable "firewall_rules" {
  description = "Firewall rules configuration"
  type = map(object({
    description        = string
    direction          = string
    priority           = number
    source_ranges      = optional(list(string))
    destination_ranges = optional(list(string))
    source_tags        = optional(list(string))
    target_tags        = optional(list(string))
    allow = optional(list(object({
      protocol = string
      ports    = optional(list(string))
    })))
    deny = optional(list(object({
      protocol = string
      ports    = optional(list(string))
    })))
  }))
  default = {
    "allow-health-checks" = {
      description   = "Allow Google Cloud health checks"
      direction     = "INGRESS"
      priority      = 1000
      source_ranges = ["130.211.0.0/22", "35.191.0.0/16"]
      target_tags   = ["http-server", "https-server"]
      allow = [{
        protocol = "tcp"
        ports    = ["80", "443", "8080"]
      }]
    }
    "allow-internal-communication" = {
      description   = "Allow communication between internal subnets"
      direction     = "INGRESS"
      priority      = 1000
      source_ranges = ["10.1.0.0/16"]
      target_tags   = ["internal-communication"]
      allow = [{
        protocol = "all"
      }]
    }
    "deny-all-external-ingress" = {
      description   = "Deny all external ingress by default"
      direction     = "INGRESS"
      priority      = 65534
      source_ranges = ["0.0.0.0/0"]
      deny = [{
        protocol = "all"
      }]
    }
  }
}

# IAM Configuration
variable "custom_iam_roles" {
  description = "Custom IAM roles"
  type = map(object({
    title       = string
    description = string
    permissions = list(string)
    stage       = string
  }))
  default = {
    "ipoValuationAnalyst" = {
      title       = "IPO Valuation Analyst"
      description = "Custom role for IPO valuation analysts"
      permissions = [
        "cloudsql.instances.connect",
        "storage.objects.get",
        "storage.objects.list",
        "bigquery.jobs.create",
        "bigquery.tables.get",
        "monitoring.timeSeries.list"
      ]
      stage = "GA"
    }
    "ipoDataManager" = {
      title       = "IPO Data Manager"
      description = "Custom role for IPO data managers"
      permissions = [
        "cloudsql.instances.connect",
        "storage.objects.create",
        "storage.objects.delete",
        "storage.objects.get",
        "storage.objects.list",
        "storage.objects.update",
        "bigquery.datasets.get",
        "bigquery.tables.create",
        "bigquery.tables.update"
      ]
      stage = "GA"
    }
    "ipoSecurityAuditor" = {
      title       = "IPO Security Auditor"
      description = "Custom role for security auditors"
      permissions = [
        "logging.entries.list",
        "logging.logEntries.list",
        "monitoring.timeSeries.list",
        "securitycenter.findings.list",
        "securitycenter.sources.list",
        "cloudkms.cryptoKeys.list",
        "iam.roles.list",
        "iam.serviceAccounts.list"
      ]
      stage = "GA"
    }
  }
}

variable "iam_bindings" {
  description = "IAM bindings for the project"
  type = map(object({
    role    = string
    members = list(string)
    condition = optional(object({
      title       = string
      description = string
      expression  = string
    }))
  }))
  default = {
    "valuation-analysts" = {
      role    = "projects/PROJECT_ID/roles/ipoValuationAnalyst"
      members = ["group:ipo-valuation-analysts@uprez.com"]
      condition = {
        title       = "Business hours only"
        description = "Access only during business hours"
        expression  = "request.time.getHours() >= 9 && request.time.getHours() <= 17"
      }
    }
    "data-managers" = {
      role    = "projects/PROJECT_ID/roles/ipoDataManager"
      members = ["group:ipo-data-managers@uprez.com"]
    }
    "security-auditors" = {
      role    = "projects/PROJECT_ID/roles/ipoSecurityAuditor"
      members = ["group:ipo-security-team@uprez.com"]
    }
  }
}

variable "service_accounts" {
  description = "Service accounts configuration"
  type = map(object({
    display_name = string
    description  = string
    create_key   = optional(bool, false)
  }))
  default = {
    "ipo-application" = {
      display_name = "IPO Valuation Application"
      description  = "Service account for IPO valuation application"
      create_key   = false
    }
    "ipo-data-pipeline" = {
      display_name = "IPO Data Pipeline"
      description  = "Service account for data pipeline operations"
      create_key   = false
    }
    "ipo-monitoring" = {
      display_name = "IPO Monitoring"
      description  = "Service account for monitoring and alerting"
      create_key   = false
    }
  }
}

# Access Context Manager Configuration
variable "access_levels" {
  description = "Access levels for VPC Service Controls"
  type = map(object({
    title              = string
    combining_function = string
    conditions = list(object({
      ip_subnetworks = optional(list(string))
      members        = optional(list(string))
      regions        = optional(list(string))
      device_policy = optional(object({
        require_screen_lock              = bool
        require_admin_approval          = bool
        require_corp_owned              = bool
        allowed_device_management_levels = list(string)
      }))
    }))
  }))
  default = {
    "basic_access" = {
      title              = "Basic Access Level"
      combining_function = "AND"
      conditions = [{
        ip_subnetworks = ["203.0.113.0/24", "10.0.0.0/8"]
        members        = ["group:ipo-valuation-analysts@uprez.com"]
      }]
    }
    "high_security" = {
      title              = "High Security Access Level"
      combining_function = "AND"
      conditions = [{
        members = ["group:ipo-executives@uprez.com", "group:ipo-security-team@uprez.com"]
        device_policy = {
          require_screen_lock              = true
          require_admin_approval          = true
          require_corp_owned              = true
          allowed_device_management_levels = ["COMPLETE"]
        }
      }]
    }
  }
}

# Cloud Storage Configuration
variable "storage_buckets" {
  description = "Cloud Storage buckets configuration"
  type = map(object({
    name                  = string
    location             = string
    storage_class        = string
    versioning_enabled   = bool
    kms_key              = string
    data_classification  = string
  }))
  default = {
    "ipo-documents" = {
      name                = "ipo-documents-prod-encrypted"
      location           = "australia-southeast1"
      storage_class      = "STANDARD"
      versioning_enabled = true
      kms_key           = "documents-key"
      data_classification = "restricted"
    }
    "financial-data" = {
      name                = "financial-data-prod-encrypted"
      location           = "australia-southeast1"
      storage_class      = "STANDARD"
      versioning_enabled = true
      kms_key           = "financial-key"
      data_classification = "restricted"
    }
    "user-uploads" = {
      name                = "user-uploads-prod-encrypted"
      location           = "australia-southeast1"
      storage_class      = "STANDARD"
      versioning_enabled = true
      kms_key           = "user-key"
      data_classification = "confidential"
    }
  }
}

# Cloud SQL Configuration
variable "cloudsql_instances" {
  description = "Cloud SQL instances configuration"
  type = map(object({
    name                   = string
    database_version      = string
    tier                  = string
    availability_type     = string
    disk_size             = number
    disk_autoresize_limit = number
    kms_key               = string
    database_type         = string
    deletion_protection   = bool
    authorized_networks = optional(list(object({
      name = string
      cidr = string
    })))
  }))
  default = {
    "primary-database" = {
      name                   = "ipo-valuation-db-prod"
      database_version      = "POSTGRES_15"
      tier                  = "db-custom-4-16384" # 4 vCPUs, 16GB RAM
      availability_type     = "REGIONAL" # High availability
      disk_size             = 100
      disk_autoresize_limit = 500
      kms_key               = "database-key"
      database_type         = "primary"
      deletion_protection   = true
    }
    "read-replica" = {
      name                   = "ipo-valuation-db-read-replica"
      database_version      = "POSTGRES_15"
      tier                  = "db-custom-2-8192" # 2 vCPUs, 8GB RAM
      availability_type     = "ZONAL"
      disk_size             = 100
      disk_autoresize_limit = 300
      kms_key               = "database-key"
      database_type         = "replica"
      deletion_protection   = true
    }
  }
}

# BigQuery Configuration
variable "bigquery_datasets" {
  description = "BigQuery datasets configuration"
  type = map(object({
    friendly_name              = string
    description               = string
    location                  = string
    default_table_expiration_ms = number
    kms_key                   = string
  }))
  default = {
    "security_logs" = {
      friendly_name              = "Security Logs"
      description               = "Dataset for security-related logs"
      location                  = "australia-southeast1"
      default_table_expiration_ms = 220752000000 # 7 years in milliseconds
      kms_key                   = "audit-key"
    }
    "compliance_data" = {
      friendly_name              = "Compliance Data"
      description               = "Dataset for compliance reporting and metrics"
      location                  = "australia-southeast1"
      default_table_expiration_ms = 220752000000
      kms_key                   = "audit-key"
    }
    "financial_analytics" = {
      friendly_name              = "Financial Analytics"
      description               = "Dataset for financial data analytics"
      location                  = "australia-southeast1"
      default_table_expiration_ms = 157680000000 # 5 years
      kms_key                   = "financial-key"
    }
  }
}

# Logging Configuration
variable "log_sinks" {
  description = "Log sinks configuration"
  type = map(object({
    destination             = string
    filter                 = string
    use_partitioned_tables = bool
  }))
  default = {
    "security-events-bigquery" = {
      destination             = "bigquery.googleapis.com/projects/PROJECT_ID/datasets/security_logs"
      filter                 = "protoPayload.authenticationInfo.principalEmail!=\"\" OR severity>=ERROR OR protoPayload.serviceName=\"iap.googleapis.com\" OR protoPayload.serviceName=\"cloudkms.googleapis.com\""
      use_partitioned_tables = true
    }
    "audit-logs-storage" = {
      destination             = "storage.googleapis.com/audit-logs-bucket-encrypted"
      filter                 = "logName:\"cloudaudit.googleapis.com\""
      use_partitioned_tables = false
    }
  }
}

# Monitoring Configuration
variable "alert_policies" {
  description = "Monitoring alert policies"
  type = map(object({
    display_name = string
    combiner     = string
    enabled      = bool
    conditions = list(object({
      display_name       = string
      filter            = string
      duration          = string
      comparison        = string
      threshold_value   = number
      alignment_period  = string
      per_series_aligner = string
    }))
    notification_channels = list(string)
    auto_close           = string
    documentation        = string
  }))
  default = {
    "critical-security-finding" = {
      display_name = "Critical Security Finding"
      combiner     = "OR"
      enabled      = true
      conditions = [{
        display_name       = "Critical SCC Finding"
        filter            = "resource.type=\"global\" AND metric.type=\"securitycenter.googleapis.com/finding/count\" AND metric.labels.category=\"FINANCIAL_DATA_EXPOSURE\""
        duration          = "60s"
        comparison        = "COMPARISON_GREATER_THAN"
        threshold_value   = 0
        alignment_period  = "60s"
        per_series_aligner = "ALIGN_RATE"
      }]
      notification_channels = []
      auto_close           = "43200s"
      documentation        = "Critical security finding detected in Security Command Center. Immediate investigation required."
    }
    "failed-authentication-spike" = {
      display_name = "Failed Authentication Spike"
      combiner     = "OR"
      enabled      = true
      conditions = [{
        display_name       = "Authentication failure rate"
        filter            = "metric.type=\"logging.googleapis.com/user/failed_authentication_rate\""
        duration          = "300s"
        comparison        = "COMPARISON_GREATER_THAN"
        threshold_value   = 10
        alignment_period  = "300s"
        per_series_aligner = "ALIGN_RATE"
      }]
      notification_channels = []
      auto_close           = "1800s"
      documentation        = "High rate of failed authentication attempts detected. Possible brute force attack."
    }
  }
}