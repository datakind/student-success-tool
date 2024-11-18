terraform {
  backend "gcs" {
    bucket = "sst-terraform-state"
  }
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "6.8.0"
    }
  }
}

provider "google" {
  project = var.project
  region  = var.region
  zone    = var.zone
}

locals {
  institutions = [
    {
      name = "Example College"
      id   = "example-college"
    },
    {
      name = "Example University"
      id   = "example-university"
    }
  ]
}

resource "google_storage_bucket" "upload_buckets" {
  for_each = { for inst in local.institutions : inst.id => inst }

  name     = each.value.id
  location = "US"

  cors {
    origin = ["*"]
    method = ["POST", "PUT"]
    response_header = [
      "Content-Type",
      "Access-Control-Allow-Origin",
      "X-Goog-Content-Length-Range"
    ]
    max_age_seconds = 3600
  }
}

resource "google_service_account" "webapp_service_acccount" {
  account_id   = "webapp"
  display_name = "Webapp Service Account"
  description  = "Service account for the webapp"
}

resource "google_project_iam_member" "webapp_service_acccount" {
  project = var.project
  role    = "roles/storage.objectUser"
  member  = "serviceAccount:${google_service_account.webapp_service_acccount.email}"
}

resource "google_project_iam_member" "token_creator" {
  project = var.project
  role    = "roles/iam.serviceAccountTokenCreator"
  member  = "serviceAccount:${google_service_account.webapp_service_acccount.email}"
}

resource "google_storage_bucket" "default" {
  name          = "sst-terraform-state"
  force_destroy = true
  location      = "US"
  storage_class = "STANDARD"
  versioning {
    enabled = true
  }
}
