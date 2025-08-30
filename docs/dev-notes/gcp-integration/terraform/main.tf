# Terraform Configuration for IPO Valuation Platform GCP Infrastructure
# Main configuration file for deploying all GCP AI/ML services

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
    bucket = "uprez-valuation-terraform-state"
    prefix = "infrastructure/state"
  }
}

# Configure providers
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

# Local values for resource naming and tagging
locals {
  common_labels = {
    environment = var.environment
    application = "ipo-valuation"
    team       = "ml-platform"
    managed_by = "terraform"
  }
  
  resource_prefix = "uprez-${var.environment}"
  
  # Service accounts mapping
  service_accounts = {
    ml_training           = "${local.resource_prefix}-ml-training"
    model_serving        = "${local.resource_prefix}-model-serving"
    data_processing      = "${local.resource_prefix}-data-processing"
    document_processing  = "${local.resource_prefix}-document-ai"
    natural_language     = "${local.resource_prefix}-natural-language"
    monitoring          = "${local.resource_prefix}-monitoring"
    pipeline_orchestration = "${local.resource_prefix}-pipeline-orchestration"
    feature_store       = "${local.resource_prefix}-feature-store"
  }
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "ml.googleapis.com",
    "documentai.googleapis.com",
    "language.googleapis.com",
    "bigquery.googleapis.com",
    "bigqueryml.googleapis.com",
    "storage.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudfunctions.googleapis.com",
    "pubsub.googleapis.com",
    "secretmanager.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "cloudresourcemanager.googleapis.com"
  ])
  
  project = var.project_id
  service = each.value
  
  disable_on_destroy = false
}

# Cloud Storage buckets
resource "google_storage_bucket" "ml_staging" {
  name     = "${local.resource_prefix}-ml-staging"
  location = var.region
  
  uniform_bucket_level_access = true
  force_destroy              = var.environment != "production"
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
  
  labels = local.common_labels
}

resource "google_storage_bucket" "ml_models" {
  name     = "${local.resource_prefix}-ml-models"
  location = var.region
  
  uniform_bucket_level_access = true
  force_destroy              = var.environment != "production"
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "Delete"
    }
  }
  
  labels = local.common_labels
}

resource "google_storage_bucket" "document_storage" {
  name     = "${local.resource_prefix}-document-storage"
  location = var.region
  
  uniform_bucket_level_access = true
  force_destroy              = var.environment != "production"
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 2555  # 7 years for financial document retention
    }
    action {
      type = "Nearline"
    }
  }
  
  labels = local.common_labels
}

resource "google_storage_bucket" "feature_store_backup" {
  name     = "${local.resource_prefix}-feature-store-backup"
  location = var.region
  
  uniform_bucket_level_access = true
  force_destroy              = var.environment != "production"
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Nearline"
    }
  }
  
  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "Coldline"
    }
  }
  
  labels = local.common_labels
}

# BigQuery dataset for ML features and training data
resource "google_bigquery_dataset" "ml_features" {
  dataset_id  = "ml_features"
  description = "Dataset for ML feature engineering and training data"
  location    = var.region
  
  default_table_expiration_ms = 365 * 24 * 60 * 60 * 1000  # 1 year
  
  labels = local.common_labels
  
  depends_on = [google_project_service.required_apis]
}

resource "google_bigquery_dataset" "ipo_data" {
  dataset_id  = "ipo_data"
  description = "Dataset for IPO valuation raw data"
  location    = var.region
  
  # No expiration for core business data
  
  labels = local.common_labels
  
  depends_on = [google_project_service.required_apis]
}

resource "google_bigquery_dataset" "market_data" {
  dataset_id  = "market_data"
  description = "Dataset for market and financial data"
  location    = var.region
  
  default_table_expiration_ms = 2 * 365 * 24 * 60 * 60 * 1000  # 2 years
  
  labels = local.common_labels
  
  depends_on = [google_project_service.required_apis]
}

# Service accounts for different ML workloads
resource "google_service_account" "ml_training" {
  account_id   = local.service_accounts.ml_training
  display_name = "ML Training Service Account"
  description  = "Service account for ML model training operations"
}

resource "google_service_account" "model_serving" {
  account_id   = local.service_accounts.model_serving
  display_name = "Model Serving Service Account"
  description  = "Service account for model serving and inference"
}

resource "google_service_account" "data_processing" {
  account_id   = local.service_accounts.data_processing
  display_name = "Data Processing Service Account"
  description  = "Service account for data ingestion and processing"
}

resource "google_service_account" "document_processing" {
  account_id   = local.service_accounts.document_processing
  display_name = "Document AI Service Account"
  description  = "Service account for Document AI processing"
}

resource "google_service_account" "natural_language" {
  account_id   = local.service_accounts.natural_language
  display_name = "Natural Language Service Account"
  description  = "Service account for Natural Language AI processing"
}

resource "google_service_account" "monitoring" {
  account_id   = local.service_accounts.monitoring
  display_name = "Monitoring Service Account"
  description  = "Service account for monitoring and alerting"
}

resource "google_service_account" "pipeline_orchestration" {
  account_id   = local.service_accounts.pipeline_orchestration
  display_name = "Pipeline Orchestration Service Account"
  description  = "Service account for pipeline orchestration"
}

resource "google_service_account" "feature_store" {
  account_id   = local.service_accounts.feature_store
  display_name = "Feature Store Service Account"
  description  = "Service account for Feature Store operations"
}

# IAM bindings for ML training service account
resource "google_project_iam_member" "ml_training_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/storage.objectViewer",
    "roles/storage.objectCreator",
    "roles/ml.developer"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.ml_training.email}"
}

# IAM bindings for model serving service account
resource "google_project_iam_member" "model_serving_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/ml.modelUser",
    "roles/monitoring.metricWriter",
    "roles/logging.logWriter"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.model_serving.email}"
}

# IAM bindings for data processing service account
resource "google_project_iam_member" "data_processing_roles" {
  for_each = toset([
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/storage.objectAdmin",
    "roles/pubsub.subscriber",
    "roles/pubsub.publisher"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.data_processing.email}"
}

# IAM bindings for Document AI service account
resource "google_project_iam_member" "document_processing_roles" {
  for_each = toset([
    "roles/documentai.apiUser",
    "roles/storage.objectViewer",
    "roles/storage.objectCreator"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.document_processing.email}"
}

# IAM bindings for Natural Language service account
resource "google_project_iam_member" "natural_language_roles" {
  for_each = toset([
    "roles/language.serviceAgent",
    "roles/storage.objectViewer"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.natural_language.email}"
}

# IAM bindings for monitoring service account
resource "google_project_iam_member" "monitoring_roles" {
  for_each = toset([
    "roles/monitoring.editor",
    "roles/logging.viewer",
    "roles/errorreporting.writer",
    "roles/cloudtrace.agent"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.monitoring.email}"
}

# IAM bindings for pipeline orchestration service account
resource "google_project_iam_member" "pipeline_orchestration_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/bigquery.jobUser",
    "roles/storage.objectAdmin",
    "roles/iam.serviceAccountUser"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.pipeline_orchestration.email}"
}

# IAM bindings for Feature Store service account
resource "google_project_iam_member" "feature_store_roles" {
  for_each = toset([
    "roles/aiplatform.featurestoreUser",
    "roles/bigquery.dataViewer",
    "roles/bigquery.jobUser"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.feature_store.email}"
}

# Vertex AI Feature Store
resource "google_vertex_ai_featurestore" "ipo_valuation" {
  name     = "${local.resource_prefix}-featurestore"
  region   = var.region
  
  labels = local.common_labels
  
  online_serving_config {
    fixed_node_count = var.environment == "production" ? 3 : 1
    scaling {
      min_node_count           = var.environment == "production" ? 1 : 1
      max_node_count          = var.environment == "production" ? 10 : 3
      cpu_utilization_target  = 70
    }
  }
  
  depends_on = [google_project_service.required_apis]
}

# Entity types for Feature Store
resource "google_vertex_ai_featurestore_entitytype" "company" {
  name         = "company"
  featurestore = google_vertex_ai_featurestore.ipo_valuation.id
  
  labels = local.common_labels
  
  monitoring_config {
    snapshot_analysis {
      disabled = false
    }
    categorical_threshold_config {
      value = 0.3
    }
    numerical_threshold_config {
      value = 0.3
    }
  }
}

resource "google_vertex_ai_featurestore_entitytype" "market" {
  name         = "market"
  featurestore = google_vertex_ai_featurestore.ipo_valuation.id
  
  labels = local.common_labels
}

resource "google_vertex_ai_featurestore_entitytype" "ipo" {
  name         = "ipo"
  featurestore = google_vertex_ai_featurestore.ipo_valuation.id
  
  labels = local.common_labels
}

resource "google_vertex_ai_featurestore_entitytype" "financial" {
  name         = "financial"
  featurestore = google_vertex_ai_featurestore.ipo_valuation.id
  
  labels = local.common_labels
}

# Pub/Sub topics for event-driven processing
resource "google_pubsub_topic" "document_processing" {
  name = "${local.resource_prefix}-document-processing"
  
  labels = local.common_labels
  
  depends_on = [google_project_service.required_apis]
}

resource "google_pubsub_topic" "model_predictions" {
  name = "${local.resource_prefix}-model-predictions"
  
  labels = local.common_labels
  
  depends_on = [google_project_service.required_apis]
}

resource "google_pubsub_topic" "feature_updates" {
  name = "${local.resource_prefix}-feature-updates"
  
  labels = local.common_labels
  
  depends_on = [google_project_service.required_apis]
}

# Pub/Sub subscriptions
resource "google_pubsub_subscription" "document_processing_sub" {
  name  = "${local.resource_prefix}-document-processing-sub"
  topic = google_pubsub_topic.document_processing.name
  
  ack_deadline_seconds = 300
  
  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }
  
  expiration_policy {
    ttl = "2678400s"  # 31 days
  }
  
  labels = local.common_labels
}

resource "google_pubsub_subscription" "model_predictions_sub" {
  name  = "${local.resource_prefix}-model-predictions-sub"
  topic = google_pubsub_topic.model_predictions.name
  
  ack_deadline_seconds = 60
  
  labels = local.common_labels
}

# Secret Manager secrets for API keys and credentials
resource "google_secret_manager_secret" "ml_api_keys" {
  secret_id = "${local.resource_prefix}-ml-api-keys"
  
  labels = local.common_labels
  
  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }
  
  depends_on = [google_project_service.required_apis]
}

resource "google_secret_manager_secret" "external_data_credentials" {
  secret_id = "${local.resource_prefix}-external-data-credentials"
  
  labels = local.common_labels
  
  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }
  
  depends_on = [google_project_service.required_apis]
}

# Cloud Monitoring notification channels
resource "google_monitoring_notification_channel" "email_alerts" {
  display_name = "IPO Platform Email Alerts"
  type         = "email"
  
  labels = {
    email_address = var.alert_email
  }
  
  depends_on = [google_project_service.required_apis]
}

resource "google_monitoring_notification_channel" "slack_alerts" {
  display_name = "IPO Platform Slack Alerts"
  type         = "slack"
  
  labels = {
    channel_name = var.slack_channel
    url         = var.slack_webhook_url
  }
  
  depends_on = [google_project_service.required_apis]
}

# Cloud Logging log sinks for structured logging
resource "google_logging_project_sink" "ml_operations_sink" {
  name        = "${local.resource_prefix}-ml-operations-sink"
  destination = "storage.googleapis.com/${google_storage_bucket.ml_staging.name}"
  
  filter = jsonencode({
    "AND" = [
      {
        "resource.type" = "vertex_ai_model"
      },
      {
        "OR" = [
          { "severity" = "ERROR" },
          { "severity" = "WARNING" }
        ]
      }
    ]
  })
  
  unique_writer_identity = true
}

# IAM binding for log sink
resource "google_storage_bucket_iam_member" "log_sink_writer" {
  bucket = google_storage_bucket.ml_staging.name
  role   = "roles/storage.objectCreator"
  member = google_logging_project_sink.ml_operations_sink.writer_identity
}

# Workload Identity setup for Kubernetes integration (if using GKE)
resource "google_service_account_iam_member" "workload_identity_ml_training" {
  count = var.enable_workload_identity ? 1 : 0
  
  service_account_id = google_service_account.ml_training.name
  role              = "roles/iam.workloadIdentityUser"
  member            = "serviceAccount:${var.project_id}.svc.id.goog[${var.kubernetes_namespace}/ml-training-ksa]"
}

resource "google_service_account_iam_member" "workload_identity_model_serving" {
  count = var.enable_workload_identity ? 1 : 0
  
  service_account_id = google_service_account.model_serving.name
  role              = "roles/iam.workloadIdentityUser"
  member            = "serviceAccount:${var.project_id}.svc.id.goog[${var.kubernetes_namespace}/model-serving-ksa]"
}

# Cloud Build trigger for CI/CD (if using Cloud Build)
resource "google_cloudbuild_trigger" "ml_pipeline_trigger" {
  count = var.enable_cicd ? 1 : 0
  
  name     = "${local.resource_prefix}-ml-pipeline"
  location = var.region
  
  github {
    owner = var.github_owner
    name  = var.github_repo
    push {
      branch = "^main$"
    }
  }
  
  build {
    step {
      name = "gcr.io/cloud-builders/gcloud"
      args = [
        "ai",
        "custom-jobs",
        "create",
        "--region=${var.region}",
        "--display-name=ml-training-${substr(timestamp(), 0, 10)}",
        "--config=ml-training-config.yaml"
      ]
    }
  }
  
  depends_on = [google_project_service.required_apis]
}

# Outputs for other modules and external references
output "project_id" {
  description = "The GCP project ID"
  value       = var.project_id
}

output "region" {
  description = "The GCP region"
  value       = var.region
}

output "ml_staging_bucket" {
  description = "ML staging bucket name"
  value       = google_storage_bucket.ml_staging.name
}

output "ml_models_bucket" {
  description = "ML models bucket name"
  value       = google_storage_bucket.ml_models.name
}

output "document_storage_bucket" {
  description = "Document storage bucket name"
  value       = google_storage_bucket.document_storage.name
}

output "featurestore_id" {
  description = "Feature Store ID"
  value       = google_vertex_ai_featurestore.ipo_valuation.id
}

output "service_accounts" {
  description = "Service account emails"
  value = {
    ml_training           = google_service_account.ml_training.email
    model_serving        = google_service_account.model_serving.email
    data_processing      = google_service_account.data_processing.email
    document_processing  = google_service_account.document_processing.email
    natural_language     = google_service_account.natural_language.email
    monitoring          = google_service_account.monitoring.email
    pipeline_orchestration = google_service_account.pipeline_orchestration.email
    feature_store       = google_service_account.feature_store.email
  }
}

output "bigquery_datasets" {
  description = "BigQuery dataset IDs"
  value = {
    ml_features = google_bigquery_dataset.ml_features.dataset_id
    ipo_data   = google_bigquery_dataset.ipo_data.dataset_id
    market_data = google_bigquery_dataset.market_data.dataset_id
  }
}

output "pubsub_topics" {
  description = "Pub/Sub topic names"
  value = {
    document_processing = google_pubsub_topic.document_processing.name
    model_predictions  = google_pubsub_topic.model_predictions.name
    feature_updates    = google_pubsub_topic.feature_updates.name
  }
}

output "notification_channels" {
  description = "Monitoring notification channel IDs"
  value = {
    email = google_monitoring_notification_channel.email_alerts.id
    slack = google_monitoring_notification_channel.slack_alerts.id
  }
}