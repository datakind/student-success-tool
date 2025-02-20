variable "required_services" {
  type = list(string)
  default = [
    "iam.googleapis.com",
  ]
}

variable "project" {
  type = string
}

variable "environment" {
  type = string
}

variable "cloudbuild_sa_roles" {
  type        = list(string)
  description = "List of IAM roles to grant to the Cloud Build service account."
  default = [
    "roles/artifactregistry.writer",
    "roles/cloudbuild.serviceAgent",
    "roles/compute.instanceAdmin.v1",
    "roles/compute.networkUser",
    "roles/iam.serviceAccountUser",
    "roles/storage.objectCreator",
    "roles/run.developer",
  ]
}

variable "cloudrun_sa_roles" {
  type        = list(string)
  description = "List of IAM roles to grant to the Cloud Run service account."
  default = [
    "roles/run.viewer",
    "roles/iam.serviceAccountTokenCreator",
    "roles/storage.admin",
    "roles/secretmanager.secretAccessor",
  ]
}

variable "databricks_sa_roles" {
  type        = list(string)
  description = "List of IAM roles to grant to the Databricks service account."
  default = [
    
  ]
}
