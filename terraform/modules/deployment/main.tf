resource "google_project_service" "services" {
  for_each = toset(var.required_services)

  service            = each.value
  disable_on_destroy = false
}

module "network" {
  source = "../network"

  project              = var.project
  environment          = var.environment
  region               = var.region
  subnet_ip_cidr_range = var.subnet_ip_cidr_range
  vpc_host_network     = var.vpc_host_network
  vpc_host_project     = var.vpc_host_project
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

  depends_on = [module.network, module.iam]
}

locals {
  jobs = [
    {
      name    = "migrate"
      image   = var.frontend_image
      command = ["launcher"]
      args    = ["php artisan migrate --force"]
    }
  ]
}

module "jobs" {
  source = "../job"

  for_each = { for j in local.jobs : j.name => j }

  name          = each.value.name
  command       = each.value.command
  args          = each.value.args
  image         = each.value.image
  environment   = var.environment
  region        = var.region
  database_name = var.database_name

  database_password_secret_id       = module.database.password_secret_id
  database_instance_connection_name = module.database.instance_connection_name
  database_instance_private_ip      = module.database.instance_private_ip
  network_id                        = module.network.network_id
  subnetwork_id                     = module.network.subnetwork_id
  cloudrun_service_account_email    = module.iam.cloudrun_service_account_email

  depends_on = [module.database, module.network, module.iam]
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
  source = "../service"

  for_each = { for s in local.services : s.name => s }

  name          = each.value.name
  image         = each.value.image
  project       = var.project
  environment   = var.environment
  region        = var.region
  database_name = var.database_name

  database_password_secret_id            = module.database.password_secret_id
  database_instance_connection_name      = module.database.instance_connection_name
  database_instance_private_ip           = module.database.instance_private_ip
  network_id                             = module.network.network_id
  subnetwork_id                          = module.network.subnetwork_id
  cloudrun_service_account_email         = module.iam.cloudrun_service_account_email
  cloudbuild_service_account_email       = module.iam.cloudbuild_service_account_email

  depends_on = [module.database, module.network, module.iam]
}

resource "google_compute_global_address" "lb_ip" {
  name         = "${var.environment}-tf-cr-lb-1-address"
  address_type = "EXTERNAL"
}

module "lb-http" {
  source  = "terraform-google-modules/lb-http/google//modules/serverless_negs"
  version = "~> 12.0"

  project = var.project
  name    = "${var.environment}-tf-cr-lb-1"

  address                         = google_compute_global_address.lb_ip.address
  create_address                  = false
  ssl                             = true
  managed_ssl_certificate_domains = [var.domain]
  https_redirect                  = true

  backends = {
    for s in local.services : "${var.environment}-${s.name}" => {
      description = "Cloud Run ${s.name}"
      groups      = []
      serverless_neg_backends = [{
        region : var.region,
        type : "cloud-run",
        service : {
          name : "${var.environment}-${s.name}",
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
  name            = "${var.environment}-tf-cr-url-map-1"
  default_service = module.lb-http.backend_services["${var.environment}-frontend"].self_link

  host_rule {
    hosts        = ["*"]
    path_matcher = "allpaths"
  }

  path_matcher {
    name            = "allpaths"
    default_service = module.lb-http.backend_services["${var.environment}-frontend"].self_link

    path_rule {
      paths   = ["/api", "/api/*"]
      service = module.lb-http.backend_services["${var.environment}-webapp"].self_link
    }

    path_rule {
      paths   = ["/worker", "/worker/*"]
      service = module.lb-http.backend_services["${var.environment}-webapp"].self_link
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

resource "google_storage_bucket" "static_assets" {
  name                        = "${var.project}-${var.environment}-static"
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true
}

resource "google_storage_bucket_iam_binding" "public_rule" {
  bucket = google_storage_bucket.static_assets.name
  role   = "roles/storage.objectViewer"
  members = [
    "allUsers",
  ]
}

resource "google_compute_backend_bucket" "build" {
  name        = "${var.environment}-tf-cr-static-build-1"
  bucket_name = google_storage_bucket.static_assets.name
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

module "cloudbuild" {
  source = "../cloudbuild"

  project              = var.project
  domain               = var.domain
  environment          = var.environment
  region               = var.region
  subnet_ip_cidr_range = var.subnet_ip_cidr_range
  vpc_host_network     = var.vpc_host_network
  vpc_host_project     = var.vpc_host_project

  cloudbuild_service_account_id = module.iam.cloudbuild_service_account_id
  terraform_service_account_id  = module.iam.terraform_service_account_id
  static_assets_bucket_name     = google_storage_bucket.static_assets.name

  depends_on = [module.services]
}
