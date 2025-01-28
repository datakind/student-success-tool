resource "google_project_service" "services" {
  for_each = toset(var.required_services)

  service            = each.value
  disable_on_destroy = false
}

resource "google_secret_manager_secret_iam_member" "cloudrun_sa_db_client_cert_access" {
  secret_id = "projects/${var.project}/secrets/test-api-env-file"
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.cloud_run_service_account_email}"
}

data "google_secret_manager_secret_version" "database_password_secret_version" {
  secret  = var.database_password_secret_id
  version = "latest"
}

resource "google_cloud_run_v2_service" "webapp" {
  deletion_protection = false
  location            = var.region
  name                = "${var.environment}-webapp"
  launch_stage        = "GA"
  template {
    service_account = var.cloud_run_service_account_email
    containers {
      image = var.image
      env {
        name  = "ENV_FILE_PATH"
        value = "/vol_mt/env_vars/.env"
      }
      env {
        name  = "DB_USER"
        value = "root"
      }
      env {
        name  = "DB_PASS"
        value = data.google_secret_manager_secret_version.database_password_secret_version.secret_data
      }
      env {
        name  = "DB_NAME"
        value = var.database_name
      }
      env {
        name  = "INSTANCE_HOST"
        value = var.database_instance_private_ip
      }
      env {
        name  = "DB_PORT"
        value = "3306"
      }
      volume_mounts {
        mount_path = "/vol_mt/env_vars"
        name       = "env_file"
      }
      volume_mounts {
        mount_path = "/vol_mt/certs/cert"
        name       = "db-client-cert-vol"
      }
      volume_mounts {
        mount_path = "/vol_mt/certs/key"
        name       = "db-client-key-vol"
      }
      volume_mounts {
        mount_path = "/vol_mt/certs/server"
        name       = "db-server-ca-vol"
      }
    }
    vpc_access {
      network_interfaces {
        network    = var.network_id
        subnetwork = var.subnetwork_id
      }
      egress = "ALL_TRAFFIC"
    }
    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [var.database_instance_connection_name]
      }
    }
    volumes {
      name = "env_file"
      secret {
        items {
          path    = ".env"
          version = "latest"
        }

        secret = "${var.environment}-api-env-file"
      }
    }
    volumes {
      name = "db-client-cert-vol"
      secret {
        items {
          path    = "client-cert.pem"
          version = "latest"
        }

        secret = "${var.environment}-db-client-cert"
      }
    }
    volumes {
      name = "db-client-key-vol"
      secret {
        items {
          path    = "client-key.pem"
          version = "latest"
        }
        secret = "${var.environment}-db-client-key"
      }
    }
    volumes {
      name = "db-server-ca-vol"
      secret {
        items {
          path    = "server-ca.pem"
          version = "latest"
        }
        secret = "${var.environment}-db-server-ca"
      }
    }
  }
}
