variable "project" {
  description = "The project ID"
  type        = string
}

variable "region" {
  description = "The region to deploy resources"
  default     = "us-east4"
}

variable "environment" {
  description = "The environment to deploy resources"
  default     = "prod"
}

variable "zone" {
  description = "The zone to deploy resources"
  default     = "us-east4-c"
}

variable "database_version" {
  description = "The database version"
  default     = "MYSQL_8_0"
}

variable "database_name" {
  description = "The database name"
  default     = "all_tables"
}

variable "domain" {
  description = "The domain name"
  type        = string
}

variable "webapp_image" {
  description = "Image for the webapp service"
  type        = string
}

variable "worker_image" {
  description = "Image for the worker service"
  type        = string
}

variable "frontend_image" {
  description = "Image for the frontend service"
  type        = string
}
