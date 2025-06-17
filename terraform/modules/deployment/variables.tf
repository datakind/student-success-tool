variable "required_services" {
  type = list(string)
  default = [
    "iap.googleapis.com",
    "servicenetworking.googleapis.com",
  ]
}

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

variable "webapp_image" {
  description = "Image for the webapp service"
  type        = string
}

variable "frontend_image" {
  description = "Image for the frontend service"
  type        = string
}

variable "worker_image" {
  description = "Image for the worker service"
  type        = string
}


variable "subnet_ip_cidr_range" {
  description = "The CIDR range for the subnet"
  type        = string
}

variable "vpc_host_project" {
  description = "The project ID of the VPC host"
  type        = string
}

variable "vpc_host_network" {
  description = "The name of the VPC host network"
  type        = string
}

variable "managed_ssl_certificate_domains" {
  description = "List of domains for the Google-managed SSL certificate"
  type        = list(string)
  default     = []
}
