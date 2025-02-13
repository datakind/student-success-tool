variable "required_services" {
  type = list(string)
  default = [
    "cloudresourcemanager.googleapis.com",
  ]
}

variable "project" {
  type = string
}

variable "environment" {
  type = string
}

variable "region" {
  type = string
}

variable "cloudbuild_service_account_id" {
  type = string
}

variable "terraform_service_account_id" {
  type = string
}

variable "webapp_image" {
  type = string
}

variable "frontend_image" {
  type = string
}

variable "domain" {
  type = string
}

variable "static_assets_bucket_name" {
  type = string
}
