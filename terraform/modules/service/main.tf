resource "google_project_service" "services" {
  for_each = toset(var.required_services)

  service            = each.value
  disable_on_destroy = false
}

resource "google_secret_manager_secret" "env_file" {
  secret_id = "${var.environment}-${var.name}-env-file"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "env_file_secret_version" {
  secret      = google_secret_manager_secret.env_file.id
  secret_data = "APP_NAME=SST"
}

resource "google_secret_manager_secret_iam_member" "cloudrun_sa_env_file_access" {
  secret_id = google_secret_manager_secret.env_file.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.cloudrun_service_account_email}"
}

resource "google_cloud_run_v2_service" "webapp" {
  deletion_protection = false
  location            = var.region
  name                = "${var.environment}-${var.name}"
  launch_stage        = "GA"
  ingress             = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"
  template {
    service_account = var.cloudrun_service_account_email
    containers {
      image = var.image
      env {
        name  = "ENV_FILE_PATH"
        value = "/vol_mt/env_vars/.env"
      }
      env {
        name  = "APP_SETTINGS_DIR"
        value = "/vol_mt/env_vars"
      }
      env {
        name  = "DB_USER"
        value = "root"
      }
      env {
        name = "DB_PASS"
        value_source {
          secret_key_ref {
            secret  = var.database_password_secret_id
            version = "latest"
          }
        }
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

        secret = "${var.environment}-${var.name}-env-file"
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

resource "google_cloud_run_v2_service_iam_member" "allow_all" {
  location = google_cloud_run_v2_service.webapp.location
  name     = google_cloud_run_v2_service.webapp.name
  role     = "roles/run.invoker"
  member   = "allUsers" # Makes the service public
}
