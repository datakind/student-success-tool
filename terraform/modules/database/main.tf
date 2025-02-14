resource "google_project_service" "services" {
  for_each = toset(var.required_services)

  service            = each.value
  disable_on_destroy = false
}

resource "random_password" "db_password" {
  length  = 16
  special = true
}

resource "google_secret_manager_secret" "db_password_secret" {
  secret_id = "${var.environment}-db-password"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "db_password_secret_version" {
  secret      = google_secret_manager_secret.db_password_secret.id
  secret_data = random_password.db_password.result
}

data "google_secret_manager_secret_version" "db_password_secret_version" {
  secret  = google_secret_manager_secret.db_password_secret.id
  version = "latest"

  depends_on = [google_secret_manager_secret_version.db_password_secret_version]
}

resource "google_sql_database_instance" "db_instance" {
  deletion_protection = false
  name                = "${var.environment}-db-instance"
  database_version    = var.database_version
  region              = var.region
  settings {
    tier = "db-f1-micro"
    location_preference {
      zone = var.zone
    }
    ip_configuration {
      ipv4_enabled    = false
      private_network = var.network_id
      ssl_mode        = "ENCRYPTED_ONLY"
    }
  }
  timeouts {
    create = "1h"
  }
}

resource "google_sql_database" "db" {
  name     = var.database_name
  instance = google_sql_database_instance.db_instance.name
}

resource "google_sql_user" "db_user" {
  name     = "root"
  instance = google_sql_database_instance.db_instance.name
  password = data.google_secret_manager_secret_version.db_password_secret_version.secret_data
}

resource "google_sql_ssl_cert" "db_ssl_cert" {
  instance    = google_sql_database_instance.db_instance.name
  common_name = "${var.environment}-db-ssl-cert"
}

resource "google_secret_manager_secret" "db_client_cert" {
  secret_id = "${var.environment}-db-client-cert"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "db_client_cert_version" {
  deletion_policy = "DELETE"
  enabled         = true
  secret          = google_secret_manager_secret.db_client_cert.id
  secret_data     = google_sql_ssl_cert.db_ssl_cert.cert
}

resource "google_secret_manager_secret" "db_client_key" {
  secret_id = "${var.environment}-db-client-key"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "db_client_key_version" {
  deletion_policy = "DELETE"
  enabled         = true
  secret          = google_secret_manager_secret.db_client_key.id
  secret_data     = google_sql_ssl_cert.db_ssl_cert.private_key
}

resource "google_secret_manager_secret" "db_server_ca" {
  secret_id = "${var.environment}-db-server-ca"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "db_server_ca_version" {
  deletion_policy = "DELETE"
  enabled         = true
  secret          = google_secret_manager_secret.db_server_ca.id
  secret_data     = google_sql_ssl_cert.db_ssl_cert.server_ca_cert
}

resource "google_secret_manager_secret_iam_member" "cloudrun_sa_db_password_access" {
  secret_id = google_secret_manager_secret.db_password_secret.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.cloudrun_service_account_email}"
}

resource "google_secret_manager_secret_iam_member" "cloudrun_sa_db_client_cert_access" {
  secret_id = google_secret_manager_secret.db_client_cert.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.cloudrun_service_account_email}"
}

resource "google_secret_manager_secret_iam_member" "cloudrun_sa_db_client_key_access" {
  secret_id = google_secret_manager_secret.db_client_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.cloudrun_service_account_email}"
}

resource "google_secret_manager_secret_iam_member" "cloudrun_sa_db_server_ca_access" {
  secret_id = google_secret_manager_secret.db_server_ca.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.cloudrun_service_account_email}"
}

resource "google_secret_manager_secret_iam_member" "cloudbuild_sa_db_password_access" {
  secret_id = google_secret_manager_secret.db_password_secret.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.cloudbuild_service_account_email}"
}

resource "google_secret_manager_secret_iam_member" "cloudbuild_sa_db_client_cert_access" {
  secret_id = google_secret_manager_secret.db_client_cert.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.cloudbuild_service_account_email}"
}

resource "google_secret_manager_secret_iam_member" "cloudbuild_sa_db_client_key_access" {
  secret_id = google_secret_manager_secret.db_client_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.cloudbuild_service_account_email}"
}

resource "google_secret_manager_secret_iam_member" "cloudbuild_sa_db_server_ca_access" {
  secret_id = google_secret_manager_secret.db_server_ca.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.cloudbuild_service_account_email}"
}
