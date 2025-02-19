resource "google_project_service" "services" {
  for_each = toset(var.required_services)

  service            = each.value
  disable_on_destroy = false
}

resource "google_compute_network" "vpc_network" {
  name                    = "${var.environment}-vpc-network"
  auto_create_subnetworks = false
}

data "google_compute_network" "vpc_network" {
  name    = var.vpc_host_network
  project = var.vpc_host_project
}

resource "google_compute_subnetwork" "subnetwork" {
  name          = "${var.environment}-vpc-subnetwork"
  network       = data.google_compute_network.vpc_network.id
  region        = var.region
  ip_cidr_range = var.subnet_ip_cidr_range
  project       = var.vpc_host_project
}

resource "google_compute_global_address" "connector_ip" {
  name          = "${var.environment}-vpc-connector-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = data.google_compute_network.vpc_network.id
  project       = var.vpc_host_project
}

resource "google_service_networking_connection" "connection" {
  network                 = data.google_compute_network.vpc_network.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.connector_ip.name]
}

data "google_project" "service_project" {
  project_id = var.project
}

resource "google_project_iam_member" "host_project_network_user" {
  project = var.vpc_host_project
  role    = "roles/compute.networkViewer"
  member  = "serviceAccount:service-${data.google_project.service_project.number}@serverless-robot-prod.iam.gserviceaccount.com"
}

resource "google_compute_subnetwork_iam_member" "subnet_network_user" {
  project    = var.vpc_host_project
  region     = var.region
  subnetwork = google_compute_subnetwork.subnetwork.name
  role       = "roles/compute.networkUser"
  member     = "serviceAccount:service-${data.google_project.service_project.number}@serverless-robot-prod.iam.gserviceaccount.com"
}

