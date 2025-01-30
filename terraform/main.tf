terraform {
  backend "gcs" {
    bucket = "sst-terraform-state"
  }
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "6.8.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "3.6.3"
    }
  }
}

# Configure the Google Cloud provider
provider "google" {
  project = var.project
  region  = var.region
  zone    = var.zone
}

module "network" {
  source = "./modules/network"

  environment = var.environment
  region      = var.region
}

module "iam" {
  source = "./modules/iam"

  environment = var.environment
}

module "database" {
  source = "./modules/database"

  environment      = var.environment
  region           = var.region
  zone             = var.zone
  database_name    = var.database_name
  database_version = var.database_version

  cloud_run_service_account_email = module.iam.cloud_run_service_account_email
  network_id                      = module.network.network_id
}

module "migrate" {
  source = "./modules/migrate"

  environment   = var.environment
  region        = var.region
  image         = var.frontend_image
  database_name = var.database_name

  database_password_secret_id       = module.database.password_secret_id
  database_instance_connection_name = module.database.instance_connection_name
  database_instance_private_ip      = module.database.instance_private_ip
  network_id                        = module.network.network_id
  subnetwork_id                     = module.network.subnetwork_id
  cloud_run_service_account_email   = module.iam.cloud_run_service_account_email
}

module "webapp" {
  source = "./modules/service"

  name          = "webapp"
  project       = var.project
  environment   = var.environment
  region        = var.region
  image         = var.webapp_image
  database_name = var.database_name

  database_password_secret_id       = module.database.password_secret_id
  database_instance_connection_name = module.database.instance_connection_name
  database_instance_private_ip      = module.database.instance_private_ip
  network_id                        = module.network.network_id
  subnetwork_id                     = module.network.subnetwork_id
  cloud_run_service_account_email   = module.iam.cloud_run_service_account_email
}

module "frontend" {
  source = "./modules/service"

  name          = "frontend"
  project       = var.project
  environment   = var.environment
  region        = var.region
  image         = var.frontend_image
  database_name = var.database_name

  database_password_secret_id       = module.database.password_secret_id
  database_instance_connection_name = module.database.instance_connection_name
  database_instance_private_ip      = module.database.instance_private_ip
  network_id                        = module.network.network_id
  subnetwork_id                     = module.network.subnetwork_id
  cloud_run_service_account_email   = module.iam.cloud_run_service_account_email
}

module "load_balancer" {
  source = "./modules/load_balancer"

  project     = var.project
  environment = var.environment
  region      = var.region
  domain      = var.domain

  # webapp_service_name   = module.webapp.service_name
  # frontend_service_name = module.frontend.service_name
}
