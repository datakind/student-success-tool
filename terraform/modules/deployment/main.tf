module "network" {
  source = "../network"

  environment = var.environment
  region      = var.region
}

module "iam" {
  source = "../iam"

  project     = var.project
  environment = var.environment
}

module "database" {
  source = "../database"

  environment      = var.environment
  region           = var.region
  zone             = var.zone
  database_name    = var.database_name
  database_version = var.database_version

  cloudrun_service_account_email   = module.iam.cloudrun_service_account_email
  cloudbuild_service_account_email = module.iam.cloudbuild_service_account_email
  network_id                       = module.network.network_id
}

module "migrate" {
  source = "../migrate"

  environment   = var.environment
  region        = var.region
  image         = var.frontend_image
  database_name = var.database_name

  database_password_secret_id       = module.database.password_secret_id
  database_instance_connection_name = module.database.instance_connection_name
  database_instance_private_ip      = module.database.instance_private_ip
  network_id                        = module.network.network_id
  subnetwork_id                     = module.network.subnetwork_id
  cloudrun_service_account_email    = module.iam.cloudrun_service_account_email
}

locals {
  services = [
    {
      name  = "webapp"
      image = var.webapp_image
    },
    {
      name  = "frontend"
      image = var.frontend_image
    },
    {
      name  = "worker"
      image = var.webapp_image
    }
  ]
}

module "services" {
  for_each = { for s in local.services : s.name => s }

  source = "../service"

  name          = each.value.name
  project       = var.project
  environment   = var.environment
  region        = var.region
  image         = each.value.image
  database_name = var.database_name

  database_password_secret_id       = module.database.password_secret_id
  database_instance_connection_name = module.database.instance_connection_name
  database_instance_private_ip      = module.database.instance_private_ip
  network_id                        = module.network.network_id
  subnetwork_id                     = module.network.subnetwork_id
  cloudrun_service_account_email    = module.iam.cloudrun_service_account_email
}

module "load_balancer" {
  source = "../load_balancer"

  project     = var.project
  environment = var.environment
  region      = var.region
  domain      = var.domain
}

resource "google_compute_global_address" "worker_lb_ip" {
  name         = "${var.environment}-tf-cr-lb-worker-address"
  address_type = "EXTERNAL"
}

module "worker_lb" {
  source  = "terraform-google-modules/lb-http/google//modules/serverless_negs"
  version = "~> 12.0"

  project = var.project
  name    = "${var.environment}-tf-cr-lb-worker"

  address                         = google_compute_global_address.worker_lb_ip.address
  create_address                  = false
  ssl                             = true
  managed_ssl_certificate_domains = [var.admin_domain]
  https_redirect                  = true

  backends = {
    "${var.environment}-worker" = {
      description = "Cloud Run worker"
      groups      = []
      serverless_neg_backends = [{
        region : var.region,
        type : "cloud-run",
        service : {
          name : "${var.environment}-worker",
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
}
