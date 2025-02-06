variable "project" {
  description = "The project ID"
  type        = string
}

variable "region" {
  description = "The region to deploy resources"
  type        = string
}

variable "environment" {
  description = "The environment to deploy resources"
  type        = string
}

variable "zone" {
  description = "The zone to deploy resources"
  type        = string
}

variable "database_version" {
  description = "The database version"
  type        = string
}

variable "database_name" {
  description = "The database name"
  type        = string
}

variable "domain" {
  description = "The domain name"
  type        = string
}

variable "admin_domain" {
  description = "The domain name for the admin interface"
  type        = string
}

variable "webapp_image" {
  description = "Image for the webapp service"
  type        = string
}

variable "frontend_image" {
  description = "Image for the frontend service"
  type        = string
}
