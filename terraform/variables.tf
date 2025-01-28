variable "project" {
  description = "The project ID"
  type = "string"
}

variable "region" {
  description = "The region to deploy resources"
  default     = "us-central1"
}

variable "environment" {
  description = "The environment to deploy resources"
  default     = "test"
}

variable "zone" {
  description = "The zone to deploy resources"
  default     = "us-central1-c"
}

variable "database_version" {
  description = "The database version"
  default     = "MYSQL_8_0"
}

variable "database_name" {
  description = "The database name"
  default     = "all_tables"
}
