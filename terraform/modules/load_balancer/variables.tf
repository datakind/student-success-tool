variable "project" {
  description = "The GCP project ID"
  type        = string
}

variable "domain" {
  description = "The domain for the managed SSL certificate"
  type        = string
}

variable "region" {
  description = "The region where the Cloud Run service is deployed"
  type        = string
}

variable "environment" {
  description = "The environment name (e.g., dev, prod)"
  type        = string
}
