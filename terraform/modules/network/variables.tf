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
