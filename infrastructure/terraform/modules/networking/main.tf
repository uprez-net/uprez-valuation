# Networking Module - VPC, Subnets, NAT, Firewall Rules
# Comprehensive networking setup for IPO Valuation Platform

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# VPC Network
resource "google_compute_network" "vpc_network" {
  name                    = "${var.resource_prefix}-vpc"
  auto_create_subnetworks = false
  description             = "VPC network for ${var.environment} environment"
  routing_mode           = "REGIONAL"
  
  project = var.project_id
}

# Private Subnet for GKE and internal services
resource "google_compute_subnetwork" "private_subnet" {
  name          = "${var.resource_prefix}-private-subnet"
  ip_cidr_range = var.private_subnet_cidr_range
  region        = var.region
  network       = google_compute_network.vpc_network.id
  description   = "Private subnet for GKE cluster and internal services"
  
  # Secondary IP ranges for GKE
  secondary_ip_range {
    range_name    = "pod-range"
    ip_cidr_range = var.pod_cidr_range
  }
  
  secondary_ip_range {
    range_name    = "service-range"
    ip_cidr_range = var.service_cidr_range
  }
  
  # Enable private Google access
  private_ip_google_access = true
  
  # Log configuration
  log_config {
    aggregation_interval = "INTERVAL_10_MIN"
    flow_sampling       = 0.5
    metadata           = "INCLUDE_ALL_METADATA"
  }

  project = var.project_id
}

# Public Subnet for load balancers and bastion hosts
resource "google_compute_subnetwork" "public_subnet" {
  name          = "${var.resource_prefix}-public-subnet"
  ip_cidr_range = var.public_subnet_cidr_range
  region        = var.region
  network       = google_compute_network.vpc_network.id
  description   = "Public subnet for load balancers and external access"

  log_config {
    aggregation_interval = "INTERVAL_10_MIN"
    flow_sampling       = 0.5
    metadata           = "INCLUDE_ALL_METADATA"
  }

  project = var.project_id
}

# Cloud Router for NAT gateway
resource "google_compute_router" "cloud_router" {
  count   = var.enable_nat_gateway ? 1 : 0
  name    = "${var.resource_prefix}-router"
  region  = var.region
  network = google_compute_network.vpc_network.id
  description = "Cloud router for NAT gateway"

  project = var.project_id
}

# External IP for NAT gateway
resource "google_compute_address" "nat_external_ip" {
  count  = var.enable_nat_gateway ? 1 : 0
  name   = "${var.resource_prefix}-nat-ip"
  region = var.region
  description = "External IP for NAT gateway"

  project = var.project_id
}

# Cloud NAT for outbound internet access from private subnet
resource "google_compute_router_nat" "cloud_nat" {
  count  = var.enable_nat_gateway ? 1 : 0
  name   = "${var.resource_prefix}-nat"
  router = google_compute_router.cloud_router[0].name
  region = var.region
  
  nat_ip_allocate_option             = "MANUAL_ONLY"
  nat_ips                           = [google_compute_address.nat_external_ip[0].self_link]
  source_subnetwork_ip_ranges_to_nat = "LIST_OF_SUBNETWORKS"
  
  subnetwork {
    name                    = google_compute_subnetwork.private_subnet.id
    source_ip_ranges_to_nat = ["ALL_IP_RANGES"]
  }
  
  # Logging configuration
  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }

  project = var.project_id
}

# External IP for Load Balancer
resource "google_compute_global_address" "lb_external_ip" {
  name        = "${var.resource_prefix}-lb-ip"
  description = "External IP for application load balancer"
  ip_version  = "IPV4"

  project = var.project_id
}

# Firewall Rules
# Allow internal communication within VPC
resource "google_compute_firewall" "allow_internal" {
  name    = "${var.resource_prefix}-allow-internal"
  network = google_compute_network.vpc_network.name
  description = "Allow internal communication within VPC"

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = [
    var.private_subnet_cidr_range,
    var.public_subnet_cidr_range,
    var.pod_cidr_range,
    var.service_cidr_range
  ]

  project = var.project_id
}

# Allow SSH access from specific IP ranges
resource "google_compute_firewall" "allow_ssh" {
  name    = "${var.resource_prefix}-allow-ssh"
  network = google_compute_network.vpc_network.name
  description = "Allow SSH access from authorized sources"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = var.authorized_ssh_ranges
  target_tags   = ["ssh-access"]

  project = var.project_id
}

# Allow HTTP/HTTPS traffic to load balancers
resource "google_compute_firewall" "allow_http_https" {
  name    = "${var.resource_prefix}-allow-http-https"
  network = google_compute_network.vpc_network.name
  description = "Allow HTTP/HTTPS traffic to load balancers"

  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["http-server", "https-server"]

  project = var.project_id
}

# Allow health checks from Google Load Balancers
resource "google_compute_firewall" "allow_health_check" {
  name    = "${var.resource_prefix}-allow-health-check"
  network = google_compute_network.vpc_network.name
  description = "Allow health checks from Google Load Balancers"

  allow {
    protocol = "tcp"
  }

  source_ranges = [
    "130.211.0.0/22",
    "35.191.0.0/16"
  ]

  target_tags = ["allow-health-check"]

  project = var.project_id
}

# Deny all other inbound traffic (implicit deny-all rule)
resource "google_compute_firewall" "deny_all" {
  name     = "${var.resource_prefix}-deny-all"
  network  = google_compute_network.vpc_network.name
  priority = 65534
  description = "Deny all other inbound traffic"

  deny {
    protocol = "all"
  }

  source_ranges = ["0.0.0.0/0"]

  project = var.project_id
}

# Cloud Armor Security Policy (if enabled)
resource "google_compute_security_policy" "security_policy" {
  count = var.enable_cloud_armor ? 1 : 0
  name  = "${var.resource_prefix}-security-policy"
  description = "Security policy for DDoS protection and WAF"

  # Default rule
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
        count        = 100
        interval_sec = 60
      }
      ban_duration_sec = 600
    }
    description = "Rate limit: 100 requests per minute per IP"
  }

  # Block common attack patterns
  rule {
    action   = "deny(403)"
    priority = "2000"
    match {
      expr {
        expression = "origin.region_code == 'CN' || origin.region_code == 'RU'"
      }
    }
    description = "Block traffic from high-risk regions"
  }

  project = var.project_id
}

# SSL Certificate for HTTPS
resource "google_compute_managed_ssl_certificate" "ssl_certificate" {
  name = "${var.resource_prefix}-ssl-cert"
  description = "Managed SSL certificate for application domains"

  managed {
    domains = var.ssl_domains
  }

  lifecycle {
    create_before_destroy = true
  }

  project = var.project_id
}

# VPC Peering for database access (if needed)
resource "google_compute_global_address" "private_ip_address" {
  name          = "${var.resource_prefix}-private-ip-address"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc_network.id
  description   = "IP range for VPC peering with managed services"

  project = var.project_id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.vpc_network.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]

  project = var.project_id
}

# Network Endpoint Group for load balancer backend
resource "google_compute_network_endpoint_group" "neg" {
  name         = "${var.resource_prefix}-neg"
  network      = google_compute_network.vpc_network.id
  subnetwork   = google_compute_subnetwork.private_subnet.id
  default_port = 80
  zone         = var.zone
  description  = "Network endpoint group for application load balancer"

  project = var.project_id
}