variable "required_services" {
  type = list(string)
  default = [
    "iam.googleapis.com",
  ]
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

output "cloud_run_service_account_email" {
  value = google_service_account.cloudrun_sa.email
}
