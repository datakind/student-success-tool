resource "google_cloud_run_v2_job" "migrate" {
  location            = var.region
  name                = "${var.environment}-${var.name}"

  template {
    task_count = 1

    template {
      containers {
        command = var.command
        args    = var.args

        env {
          name  = "DB_USERNAME"
          value = "root"
        }

        env {
          name  = "DB_DATABASE"
          value = var.database_name
        }

        env {
          name  = "DB_HOST"
          value = var.database_instance_private_ip
        }

        env {
          name  = "DB_CONNECTION"
          value = "mysql"
        }

        env {
          name  = "DB_PORT"
          value = "3306"
        }

        env {
          name  = "SSL_CA_PATH"
          value = "\"/var/www/html/certs/server/server-ca.pem\""
        }

        env {
          name  = "SSL_KEY_PATH"
          value = "\"/var/www/html/certs/key/client-key.pem\""
        }

        env {
          name  = "SSL_CERT_PATH"
          value = "\"/var/www/html/certs/cert/client-cert.pem\""
        }

        env {
          name = "DB_PASSWORD"
          value_source {
            secret_key_ref {
              secret  = var.database_password_secret_id
              version = "latest"
            }
          }
        }

        image = var.image

        resources {
          limits = {
            cpu    = "1000m"
            memory = "512Mi"
          }
        }

        volume_mounts {
          mount_path = "/var/www/html/certs/cert"
          name       = "${var.environment}-client-cert-red-rok-fuj"
        }

        volume_mounts {
          mount_path = "/var/www/html/certs/key"
          name       = "${var.environment}-client-key-tax-gap-beq"
        }

        volume_mounts {
          mount_path = "/var/www/html/certs/server"
          name       = "${var.environment}-server-ca-vah-faw-wup"
        }

        volume_mounts {
          mount_path = "/cloudsql"
          name       = "cloudsql"
        }
      }

      execution_environment = "EXECUTION_ENVIRONMENT_GEN2"
      max_retries           = 3
      service_account       = var.cloudrun_service_account_email
      timeout               = "600s"

      volumes {
        name = "${var.environment}-client-cert-red-rok-fuj"

        secret {
          items {
            path    = "client-cert.pem"
            version = "latest"
          }

          secret = "${var.environment}-db-client-cert"
        }
      }

      volumes {
        name = "${var.environment}-client-key-tax-gap-beq"

        secret {
          items {
            path    = "client-key.pem"
            version = "latest"
          }

          secret = "${var.environment}-db-client-key"
        }
      }

      volumes {
        name = "${var.environment}-server-ca-vah-faw-wup"

        secret {
          items {
            path    = "server-ca.pem"
            version = "latest"
          }

          secret = "${var.environment}-db-server-ca"
        }
      }

      volumes {
        name = "cloudsql"
        cloud_sql_instance {
          instances = [var.database_instance_connection_name]
        }
      }

      vpc_access {
        network_interfaces {
          network    = var.network_id
          subnetwork = var.subnetwork_id
        }
        egress = "ALL_TRAFFIC"
      }
    }
  }
  # This is a workaround to avoid unnecessary updates to the container image
  lifecycle {
    ignore_changes = [
      template[0].template[0].containers[0].image,
    ]
  }
}
