resource "google_project_service" "services" {
  for_each = toset(var.required_services)

  service            = each.value
  disable_on_destroy = false
}

resource "google_compute_network" "vpc_network" {
  name                    = "${var.environment}-vpc-network"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "vpc_subnetwork" {
  name          = "${var.environment}-vpc-subnetwork"
  network       = google_compute_network.vpc_network.id
  region        = var.region
  ip_cidr_range = "10.0.1.0/24"
}

resource "google_compute_global_address" "vpc_connector_ip" {
  name          = "${var.environment}-vpc-connector-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc_network.id
}

resource "google_service_networking_connection" "vpc_connection" {
  network                 = google_compute_network.vpc_network.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.vpc_connector_ip.name]
}

resource "google_certificate_manager_certificate" "ssl_cert" {
  name        = "my-ssl-certificate"
  description = "Managed SSL certificate for my load balancer"

  managed {
    domains = ["sst.datakind.org"] # Replace with your domains
  }
}