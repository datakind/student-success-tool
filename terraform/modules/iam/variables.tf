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
    "roles/cloudbuild.serviceAgent",
    "roles/iam.serviceAccountUser",
    "roles/storage.objectCreator",
    "roles/compute.networkUser",
    "roles/compute.instanceAdmin.v1",
    "roles/secretmanager.secretAccessor",
    "roles/artifactregistry.writer",
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
    "roles/iam.serviceAccountUser",
    "roles/logging.logWriter",
    "roles/storage.admin",
    "roles/artifactregistry.admin",
    "roles/run.sourceDeveloper",
    "roles/run.developer",
    "roles/run.invoker",
    "roles/compute.networkAdmin",
    "roles/compute.storageAdmin",
    "roles/secretmanager.secretAccessor",
    "roles/iam.serviceAccountAdmin",
  ]
}
