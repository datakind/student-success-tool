variable "environment" {
  type = string
}

variable "region" {
  type = string
}

variable "image" {
  type = string
}

variable "database_instance_connection_name" {
  type = string
}

variable "database_instance_private_ip" {
  type = string
}

variable "database_name" {
  type = string
}

variable "database_password_secret_id" {
  type      = string
  sensitive = true
}

variable "network_id" {
  type = string
}

variable "subnetwork_id" {
  type = string
}

variable "cloudrun_service_account_email" {
  type = string
}
