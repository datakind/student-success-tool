resource "google_project_service" "services" {
  for_each = toset(var.required_services)

  service            = each.value
  disable_on_destroy = false
}

resource "google_service_account" "cloudrun_sa" {
  account_id   = "${var.environment}-cloudrun-sa"
  display_name = "Cloud Run Service Account"
}

resource "google_project_iam_member" "cloudrun_sa_invoker" {
  for_each = toset(var.cloudrun_sa_roles)
  project  = var.project
  role     = each.key
  member   = "serviceAccount:${google_service_account.cloudrun_sa.email}"
}

resource "google_service_account" "cloudbuild_sa" {
  account_id   = "${var.environment}-cloudbuild-sa"
  display_name = "Cloud Build Service Account"
}

resource "google_project_iam_member" "cloudbuild_sa_member" {
  for_each = toset(var.cloudbuild_sa_roles)
  project  = var.project
  role     = each.key
  member   = "serviceAccount:${google_service_account.cloudbuild_sa.email}"
}
