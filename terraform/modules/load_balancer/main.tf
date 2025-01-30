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
      }, {
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
  }
}

# resource "google_compute_url_map" "url_map" {
#   name            = "tf-cr-url-map-1" # Choose a unique name
#   default_service = module.lb-http.backend_services[""]

#   path_matcher {
#     name               = "api-matcher"
#     default_service    = google_cloud_run_v2_service.frontend.status[0].url
#     path_rule {
#       paths   = ["/api/*"]
#       service = google_cloud_run_v2_service.webapp.status[0].url
#     }
#   }
# }
