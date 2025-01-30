variable "project" {
  description = "The GCP project ID"
  type        = string
}

variable "domain" {
  description = "The domain for the managed SSL certificate"
  type        = string
}

variable "region" {
  description = "The region where the Cloud Run service is deployed"
  type        = string
}

variable "environment" {
  description = "The environment name (e.g., dev, prod)"
  type        = string
}

module "lb-http" {
  source  = "terraform-google-modules/lb-http/google//modules/serverless_negs"
  version = "~> 12.0"

  project = var.project
  name    = "tf-cr-lb-1"

  address = "35.227.226.31"
  ssl                             = true
  managed_ssl_certificate_domains = [var.domain]
  https_redirect                  = true

  backends = {
    frontend = {
      description = "Cloud Run frontend"
      groups      = []
      serverless_neg_backends = [{
        region : var.region,
        type : "cloud-run",
        service : {
          name : "${var.environment}-frontend",
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
    webapp = {
      description = "Cloud Run webapp"
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
  create_url_map = false
  url_map = google_compute_url_map.url_map.self_link
}

resource "google_compute_url_map" "url_map" {
  name            = "tf-cr-url-map-1" # Choose a unique name
  default_service = module.lb-http.backend_services["frontend"].self_link

  host_rule {
    hosts = ["*"]
    path_matcher = "allpaths"
  }

  path_matcher {
    name               = "allpaths"
    default_service    = module.lb-http.backend_services["frontend"].self_link

    path_rule {
      paths   = ["/api", "/api/*"]
      service = module.lb-http.backend_services["webapp"].self_link
    }

    path_rule {
      paths = [
        "/build",
        "/build/*"
      ]
      service = google_compute_backend_bucket.build.self_link
    }
  }
}

resource "google_compute_backend_bucket" "build" {
  name        = "tf-cr-static-build-1" # Choose a unique name
  bucket_name = "dev-frontend-dev-sst-439514-static"
  enable_cdn  = true  
}