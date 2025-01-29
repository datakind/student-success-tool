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

locals {
  webapp_image   = "us-central1-docker.pkg.dev/dev-sst-439514/cloud-run-source-deploy/student-success-tool/central-dev-app-deploy@sha256:ccc3b670a56fd9c9678805e2b38a7636dcf3c0300a5eff0245b1e51b2f589492"
  frontend_image = "gcr.io/dev-sst-439514/github.com/datakind/sst-app-ui@sha256:381e12f87acdbd6cab7371ee41696f958dba59f704df4377a817f8c48a5af9e0"
}

module "migrate" {
  source = "./modules/migrate"

  environment   = var.environment
  region        = var.region
  image         = local.frontend_image
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
  image         = local.webapp_image
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
  image         = local.frontend_image
  database_name = var.database_name

  database_password_secret_id       = module.database.password_secret_id
  database_instance_connection_name = module.database.instance_connection_name
  database_instance_private_ip      = module.database.instance_private_ip
  network_id                        = module.network.network_id
  subnetwork_id                     = module.network.subnetwork_id
  cloud_run_service_account_email   = module.iam.cloud_run_service_account_email
}

module "lb-http" {
  source  = "terraform-google-modules/lb-http/google//modules/serverless_negs"
  version = "~> 12.0"

  project = var.project
  name    = "tf-cr-lb-1"

  ssl                             = true
  managed_ssl_certificate_domains = [var.domain]
  https_redirect                  = true

  backends = {
    default = {
      description = "Cloud Run backend"
      groups      = []
      serverless_neg_backends = [{
        region : var.region,
        type : "cloud-run",
        service : {
          name : "${var.environment}-webapp",
        }
      }]
      enable_cdn = false

      iap_config = {
        enable = false
      }
      log_config = {
        enable = false
      }
    }
  }
}
