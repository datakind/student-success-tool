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

resource "google_project_service" "services" {
  for_each = toset(var.required_services)

  service            = each.value
  disable_on_destroy = false
}

resource "google_service_account" "cloudrun_sa" {
  account_id   = "${var.environment}-cloudrun-sa"
  display_name = "${var.environment} Cloud Run Service Account"
}

# Add the cloud run invoker role to the service account
resource "google_project_iam_member" "cloudrun_sa_invoker" {
  project = var.project
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.cloudrun_sa.email}"
}

output "cloud_run_service_account_email" {
  value = google_service_account.cloudrun_sa.email
}
