# Networking Module Outputs

output "vpc_network_name" {
  description = "The name of the VPC network"
  value       = google_compute_network.vpc_network.name
}

output "vpc_network_id" {
  description = "The ID of the VPC network"
  value       = google_compute_network.vpc_network.id
}

output "vpc_network_self_link" {
  description = "The self-link of the VPC network"
  value       = google_compute_network.vpc_network.self_link
}

output "private_subnet_name" {
  description = "The name of the private subnet"
  value       = google_compute_subnetwork.private_subnet.name
}

output "private_subnet_cidr" {
  description = "The CIDR range of the private subnet"
  value       = google_compute_subnetwork.private_subnet.ip_cidr_range
}

output "private_subnet_self_link" {
  description = "The self-link of the private subnet"
  value       = google_compute_subnetwork.private_subnet.self_link
}

output "public_subnet_name" {
  description = "The name of the public subnet"
  value       = google_compute_subnetwork.public_subnet.name
}

output "public_subnet_cidr" {
  description = "The CIDR range of the public subnet"
  value       = google_compute_subnetwork.public_subnet.ip_cidr_range
}

output "public_subnet_self_link" {
  description = "The self-link of the public subnet"
  value       = google_compute_subnetwork.public_subnet.self_link
}

output "nat_gateway_ip" {
  description = "The external IP address of the NAT gateway"
  value       = var.enable_nat_gateway ? google_compute_address.nat_external_ip[0].address : null
}

output "cloud_router_name" {
  description = "The name of the Cloud Router"
  value       = var.enable_nat_gateway ? google_compute_router.cloud_router[0].name : null
}

output "load_balancer_ip" {
  description = "The external IP address of the load balancer"
  value       = google_compute_global_address.lb_external_ip.address
}

output "ssl_certificate_name" {
  description = "The name of the SSL certificate"
  value       = google_compute_managed_ssl_certificate.ssl_certificate.name
}

output "security_policy_name" {
  description = "The name of the Cloud Armor security policy"
  value       = var.enable_cloud_armor ? google_compute_security_policy.security_policy[0].name : null
}

output "network_endpoint_group_name" {
  description = "The name of the network endpoint group"
  value       = google_compute_network_endpoint_group.neg.name
}

output "private_vpc_connection_network" {
  description = "The network used for private VPC connection"
  value       = google_service_networking_connection.private_vpc_connection.network
}