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
    "roles/secretmanager.secretAccessor",
    "roles/storage.objectCreator",
  ]
}

variable "cloudrun_sa_roles" {
  type        = list(string)
  description = "List of IAM roles to grant to the Cloud Run service account."
  default = [
    "roles/run.invoker",
  ]
}

variable "terraform_sa_roles" {
  type        = list(string)
  description = "List of IAM roles to grant to the Terraform service account."
  default = [
    "roles/compute.networkAdmin",
    "roles/compute.storageAdmin",
    "roles/compute.securityAdmin",
    "roles/iam.securityAdmin",
    "roles/iam.serviceAccountUser",
    "roles/logging.logWriter",
    "roles/run.developer",
    "roles/run.invoker",
    "roles/secretmanager.secretAccessor",
    "roles/storage.admin",
  ]
}
