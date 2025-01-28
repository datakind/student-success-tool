output "instance_connection_name" {
  value = google_sql_database_instance.db_instance.connection_name
}

output "password_secret_id" {
  value     = google_secret_manager_secret.db_password_secret.id
  sensitive = true
}

output "instance_private_ip" {
  value = google_sql_database_instance.db_instance.private_ip_address
}
