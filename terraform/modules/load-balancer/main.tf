variable "project" {
  type = string
}

variable "environment" {
  type = string
}

variable "domain" {
  type = string
}

variable "region" {
  type = string
}

variable "cloud_run_service_name" {
  type = string
}

module "lb-http" {
  source  = "terraform-google-modules/lb-http/google//modules/serverless_negs"
  version = "~> 12.0"

  project = var.project # Why is this required?
  name    = "tf-cr-lb-1"

  ssl                             = true
  managed_ssl_certificate_domains = [var.domain]
  https_redirect                  = true
  # labels         = { "example-label" = "cloud-run-example" }

  backends = {
    default = {
      description = "Cloud Run backend"
      groups      = []
      serverless_neg_backends = [{
        region : var.region,
        type : "cloud-run",
        service : {
          name : var.cloud_run_service_name
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
