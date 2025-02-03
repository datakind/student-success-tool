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

# TODO: Narrow down the permissions for the Cloud Build service account
resource "google_service_account" "cloudbuild_sa" {
  account_id = "cloudbuild-sa"
}

resource "google_project_iam_member" "act_as" {
  project = var.project
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${google_service_account.cloudbuild_sa.email}"
}

resource "google_project_iam_member" "logs_writer" {
  project = var.project
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.cloudbuild_sa.email}"
}

resource "google_project_iam_member" "storage_object_delete" {
  project = var.project
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.cloudbuild_sa.email}"
}

resource "google_project_iam_member" "upload_artifacts" {
  project = var.project
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.cloudbuild_sa.email}"
}

resource "google_project_iam_member" "cloud_run_deployer" {
  project = var.project
  role    = "roles/run.sourceDeveloper"
  member  = "serviceAccount:${google_service_account.cloudbuild_sa.email}"
}

resource "google_project_iam_member" "cloudbuild_run_developer" {
  project = var.project
  role    = "roles/run.developer"
  member  = "serviceAccount:${google_service_account.cloudbuild_sa.email}"
}

resource "google_project_iam_member" "cloudbuild_run_invoker" {
  project = var.project
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.cloudbuild_sa.email}"
}

resource "google_project_iam_member" "cloudbuild_sa_viewer" {
  project = var.project
  role    = "roles/viewer"
  member  = "serviceAccount:cloudbuild-sa@dev-sst-439514.iam.gserviceaccount.com"
}

resource "google_project_iam_member" "cloudbuild_sa_compute_network_admin" {
  project = var.project
  role    = "roles/compute.networkAdmin"
  member  = "serviceAccount:cloudbuild-sa@dev-sst-439514.iam.gserviceaccount.com"
}

resource "google_project_iam_member" "cloudbuild_sa_compute_storage_admin" {
  project = var.project
  role    = "roles/compute.storageAdmin"
  member  = "serviceAccount:cloudbuild-sa@dev-sst-439514.iam.gserviceaccount.com"
}

resource "google_project_iam_member" "cloudbuild_sa_secret_accessor" {
  project = var.project
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.cloudbuild_sa.email}"
}

resource "google_project_iam_member" "cloudbuild_sa_security_admin" {
  project = var.project
  role    = "roles/iam.securityAdmin"
  member  = "serviceAccount:${google_service_account.cloudbuild_sa.email}"
}

resource "google_project_iam_member" "cloudbuild_sa_artifact_admin" {
  project = var.project
  role    = "roles/artifactregistry.admin"
  member  = "serviceAccount:${google_service_account.cloudbuild_sa.email}"
}

resource "google_project_iam_member" "cloudbuild_sa_storage_admin" {
  project = var.project
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.cloudbuild_sa.email}"
}

