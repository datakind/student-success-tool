variable "project" {
  description = "The project ID"
}

variable "environment" {
  description = "The environment"
}

variable "region" {
  description = "The region"
}

variable "cloudbuild_service_account_id" {
  description = "The Cloud Build service account ID"
  type        = string
}

variable "webapp_image" {
  description = "The webapp Docker image"
  type        = string
}

variable "frontend_image" {
  description = "The frontend Docker image"
  type        = string
}

variable "domain" {
  type = string
}
