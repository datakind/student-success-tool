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

module "webapp" {
  source = "../service"

  name          = "webapp"
  project       = var.project
  environment   = var.environment
  region        = var.region
  image         = var.webapp_image
  database_name = var.database_name

  database_password_secret_id       = module.database.password_secret_id
  database_instance_connection_name = module.database.instance_connection_name
  database_instance_private_ip      = module.database.instance_private_ip
  network_id                        = module.network.network_id
  subnetwork_id                     = module.network.subnetwork_id
  cloudrun_service_account_email    = module.iam.cloudrun_service_account_email
}

module "frontend" {
  source = "../service"

  name          = "frontend"
  project       = var.project
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

module "load_balancer" {
  source = "../load_balancer"

  project     = var.project
  environment = var.environment
  region      = var.region
  domain      = var.domain
}
