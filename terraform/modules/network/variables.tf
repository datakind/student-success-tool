variable "required_services" {
  type = list(string)
  default = [
    "compute.googleapis.com",
  ]
}

variable "region" {
  type = string
}

variable "environment" {
  type = string
}

variable "subnet_ip_cidr_range" {
  type = string
}

variable "vpc_host_project" {
  type = string
}

variable "vpc_host_network" {
  type = string
}