variable "required_services" {
  type = list(string)
  default = [
    "sqladmin.googleapis.com",
    "secretmanager.googleapis.com"
  ]
}

variable "environment" {
  type = string
}

variable "region" {
  type = string
}

variable "zone" {
  type = string
}

variable "database_version" {
  type = string
}

variable "database_name" {
  type = string
}

variable "network_id" {
  type = string
}

variable "cloud_run_service_account_email" {
  type = string
}
