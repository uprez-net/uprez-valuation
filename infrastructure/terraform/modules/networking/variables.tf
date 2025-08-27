# Networking Module Variables

variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "zone" {
  description = "GCP zone"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "resource_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "common_labels" {
  description = "Common labels to apply to all resources"
  type        = map(string)
}

variable "private_subnet_cidr_range" {
  description = "CIDR range for private subnet"
  type        = string
}

variable "public_subnet_cidr_range" {
  description = "CIDR range for public subnet"
  type        = string
}

variable "pod_cidr_range" {
  description = "CIDR range for Kubernetes pods"
  type        = string
}

variable "service_cidr_range" {
  description = "CIDR range for Kubernetes services"
  type        = string
}

variable "enable_nat_gateway" {
  description = "Enable Cloud NAT gateway"
  type        = bool
  default     = true
}

variable "enable_cloud_armor" {
  description = "Enable Cloud Armor security policy"
  type        = bool
  default     = true
}

variable "authorized_ssh_ranges" {
  description = "IP ranges authorized for SSH access"
  type        = list(string)
  default     = ["10.0.0.0/8"]
}

variable "ssl_domains" {
  description = "Domains for SSL certificate"
  type        = list(string)
  default     = []
}