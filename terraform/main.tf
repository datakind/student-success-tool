terraform {
  backend "gcs" {
    bucket = "sst-terraform-state"
    prefix = "dev"
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

module "deployment" {
  source = "./modules/deployment"

  project          = var.project
  region           = var.region
  environment      = var.environment
  zone             = var.zone
  database_version = var.database_version
  database_name    = var.database_name
  domain           = var.domain
  webapp_image     = var.webapp_image
  frontend_image   = var.frontend_image
}

module "cloudbuild" {
  source = "./modules/cloudbuild"

  project        = var.project
  domain         = var.domain
  environment    = var.environment
  region         = var.region
  webapp_image   = var.webapp_image
  frontend_image = var.frontend_image

  cloudbuild_service_account_id = module.deployment.iam.cloudbuild_service_account_id
}
