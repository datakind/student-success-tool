variable "project" {
  type = string
}

variable "environment" {
  type = string
}

variable "domain" {
  type = string
}

# variable "network_id" {
#   type = string
# }

# variable "subnetwork_id" {
#   type = string
# }

variable "region" {
  type = string
}

variable "cloud_run_service_name" {
  type = string
}

resource "google_certificate_manager_certificate" "ssl_cert" {
  name        = "${var.environment}-lb-ssl-certificate"
  description = "Managed SSL certificate"

  managed {
    domains = [var.domain, "www.${var.domain}"]
  }
}


resource "google_compute_region_network_endpoint_group" "cloud_run_neg" {
  name                  = "${var.environment}-lb-cloud-run-neg"
  network_endpoint_type = "SERVERLESS"
  region                = var.region
  # network               = var.network_id
  # subnetwork            = var.subnetwork_id

  cloud_run {
    service = var.cloud_run_service_name
  }
}


module "lb-http" {
  source  = "terraform-google-modules/lb-http/google//modules/serverless_negs"
  version = "~> 12.0"

  project = var.project # Why is this required?
  name    = "tf-cr-lb-1"

  ssl                             = true
  managed_ssl_certificate_domains = [var.domain, "www.${var.domain}"]
  https_redirect                  = true
  # labels         = { "example-label" = "cloud-run-example" }

  backends = {
    default = {
      description             = "Cloud Run backend"
      groups                  = []
      serverless_neg_backends = [{ region : "us-central1", type : "cloud-run", service : { name : var.cloud_run_service_name } }]
      enable_cdn              = false

      iap_config = {
        enable = false
      }
      log_config = {
        enable = false
      }
    }
  }
}
