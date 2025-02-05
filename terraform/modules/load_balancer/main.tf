resource "google_compute_global_address" "lb_ip" {
  name         = "tf-cr-lb-1-address"
  address_type = "EXTERNAL"
}

module "lb-http" {
  source  = "terraform-google-modules/lb-http/google//modules/serverless_negs"
  version = "~> 12.0"

  project = var.project
  name    = "tf-cr-lb-1"

  address                         = google_compute_global_address.lb_ip.address
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
        enable = true
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
        enable = true
      }
      log_config = {
        enable = false
      }
    }
  }
  create_url_map = false
  url_map        = google_compute_url_map.url_map.self_link
}

resource "google_compute_url_map" "url_map" {
  name            = "tf-cr-url-map-1"
  default_service = module.lb-http.backend_services["frontend"].self_link

  host_rule {
    hosts        = ["*"]
    path_matcher = "allpaths"
  }

  path_matcher {
    name            = "allpaths"
    default_service = module.lb-http.backend_services["frontend"].self_link

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
  name        = "tf-cr-static-build-1"
  bucket_name = "dev-frontend-dev-sst-439514-static"
  enable_cdn  = true
}

data "google_iam_policy" "admin" {
  binding {
    role = "roles/iap.httpsResourceAccessor"
    members = [
      "domain:datakind.org"
    ]
  }
}

# TODO: disable this for the prod environment
resource "google_iap_web_backend_service_iam_policy" "web_backend_service_iam_policy" {
  for_each            = module.lb-http.backend_services
  web_backend_service = each.value.name
  policy_data         = data.google_iam_policy.admin.policy_data
}
