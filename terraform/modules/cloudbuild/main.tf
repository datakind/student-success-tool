resource "google_artifact_registry_repository" "student_success_tool" {
  location      = "us-central1"
  repository_id = "student-success-tool"
  format        = "DOCKER"
}

resource "google_artifact_registry_repository" "sst_app_ui" {
  location      = "us-central1"
  repository_id = "sst-app-ui"
  format        = "DOCKER"
}

resource "google_cloudbuild_trigger" "webapp" {
  name            = "${var.environment}-student-success-tool-webapp"
  service_account = var.cloudbuild_service_account_id
  substitutions = {
    "_ENVIRONMENT" = var.environment,
  }
  github {
    owner = "datakind"
    name  = "student-success-tool"
    push {
      branch = "fellows-experimental"
    }
  }
  build {
    step {
      name = "gcr.io/cloud-builders/docker"
      args = [
        "build",
        "-t",
        "${var.region}-docker.pkg.dev/${var.project}/student-success-tool/webapp:$COMMIT_SHA",
        "-t",
        "${var.region}-docker.pkg.dev/${var.project}/student-success-tool/webapp:latest",
        "."
      ]
    }
    step {
      name = "gcr.io/cloud-builders/docker"
      args = ["push", "${var.region}-docker.pkg.dev/${var.project}/student-success-tool/webapp:$COMMIT_SHA"]
    }
    step {
      name = "gcr.io/cloud-builders/docker"
      args = ["push", "${var.region}-docker.pkg.dev/${var.project}/student-success-tool/webapp:latest"]
    }
    step {
      name = "gcr.io/cloud-builders/gcloud"
      args = [
        "run",
        "deploy",
        "$_ENVIRONMENT-webapp",
        "--image",
        "${var.region}-docker.pkg.dev/${var.project}/student-success-tool/webapp:$COMMIT_SHA",
        "--region",
        "${var.region}",
      ]
    }
    step {
      name = "gcr.io/cloud-builders/gcloud"
      args = [
        "run",
        "deploy",
        "$_ENVIRONMENT-webapp",
        "--image",
        "${var.region}-docker.pkg.dev/${var.project}/student-success-tool/webapp:$COMMIT_SHA",
        "--region",
        "${var.region}",
        "--command",
        "fastapi",
        "--args",
        "run",
        "src/worker",
        "--port",
        "8080",
        "--host",
        "0.0.0.0"
      ]
    }
    options {
      logging               = "CLOUD_LOGGING_ONLY"
      dynamic_substitutions = true
    }
  }
}

resource "google_cloudbuild_trigger" "frontend" {
  name            = "${var.environment}-sst-app-ui-frontend"
  service_account = var.cloudbuild_service_account_id
  substitutions = {
    "_ENVIRONMENT" = var.environment,
  }
  github {
    owner = "datakind"
    name  = "sst-app-ui"
    push {
      branch = "develop"
    }
  }
  build {
    step {
      id         = "INSTALL npm"
      name       = "node"
      entrypoint = "npm"
      args       = ["install"]
    }
    step {
      id         = "CREATE vite assets"
      name       = "node"
      entrypoint = "npm"
      args       = ["run", "build"]
    }
    step {
      id         = "COPY to gcs bucket"
      name       = "gcr.io/cloud-builders/gsutil"
      entrypoint = "bash"
      args       = ["-c", "gsutil -m cp -r public/* gs://${var.static_assets_bucket_name}"]
    }
    step {
      id         = "BUILD and PUSH with cloudpacks"
      name       = "gcr.io/k8s-skaffold/pack"
      entrypoint = "pack"
      args = [
        "build",
        "--builder=gcr.io/buildpacks/builder",
        "--publish",
        "${var.region}-docker.pkg.dev/${var.project}/sst-app-ui/frontend:$COMMIT_SHA",
      ]
    }
    step {
      id   = "PULL and TAG latest"
      name = "gcr.io/cloud-builders/docker"
      args = ["pull", "${var.region}-docker.pkg.dev/${var.project}/sst-app-ui/frontend:$COMMIT_SHA"]
    }
    step {
      name = "gcr.io/cloud-builders/docker"
      args = [
        "tag",
        "${var.region}-docker.pkg.dev/${var.project}/sst-app-ui/frontend:$COMMIT_SHA",
        "${var.region}-docker.pkg.dev/${var.project}/sst-app-ui/frontend:latest"
      ]
    }
    step {
      name = "gcr.io/cloud-builders/docker"
      args = ["push", "${var.region}-docker.pkg.dev/${var.project}/sst-app-ui/frontend:$COMMIT_SHA"]
    }
    step {
      name = "gcr.io/cloud-builders/docker"
      args = ["push", "${var.region}-docker.pkg.dev/${var.project}/sst-app-ui/frontend:latest"]
    }
    step {
      id         = "DEPLOY and RUN migration job"
      name       = "gcr.io/google.com/cloudsdktool/cloud-sdk:slim"
      entrypoint = "gcloud"
      args = [
        "run",
        "jobs",
        "deploy",
        "$_ENVIRONMENT-migrate",
        "--image=${var.region}-docker.pkg.dev/${var.project}/sst-app-ui/frontend:$COMMIT_SHA",
        "--region=${var.region}",
        "--execute-now"
      ]
    }
    step {
      id         = "DEPLOY to cloud run"
      name       = "gcr.io/cloud-builders/gcloud"
      entrypoint = "gcloud"
      args = [
        "run",
        "deploy",
        "$_ENVIRONMENT-frontend",
        "--image",
        "${var.region}-docker.pkg.dev/${var.project}/sst-app-ui/frontend:$COMMIT_SHA",
        "--region",
        "${var.region}",
      ]
    }
    options {
      logging               = "CLOUD_LOGGING_ONLY"
      dynamic_substitutions = true
    }
  }
}
