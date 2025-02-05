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
  name            = "student-success-tool-webapp"
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
    options {
      logging               = "CLOUD_LOGGING_ONLY"
      dynamic_substitutions = true
    }
  }
}

resource "google_cloudbuild_trigger" "frontend" {
  name            = "sst-app-ui-frontend"
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
      # TODO the storage bucket should be a variable
      args = ["-c", "gsutil -m cp -r public/* gs://dev-frontend-dev-sst-439514-static"]
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

resource "google_cloudbuild_trigger" "terraform" {
  name            = "student-success-tool-terraform"
  service_account = var.cloudbuild_service_account_id
  substitutions = {
    "_PROJECT"        = var.project
    "_REGION"         = var.region
    "_ENVIRONMENT"    = var.environment
    "_DOMAIN"         = var.domain
    "_WEBAPP_IMAGE"   = "${var.region}-docker.pkg.dev/${var.project}/student-success-tool/webapp:latest"
    "_FRONTEND_IMAGE" = "${var.region}-docker.pkg.dev/${var.project}/sst-app-ui/frontend:latest"
  }
  source_to_build {
    ref = "refs/heads/fellows-experimental"
    repo_type = "GITHUB"
    uri = "https://github.com/datakind/student-success-tool"
  }
  build {
    step {
      name = "hashicorp/terraform:1.10.1"
      dir  = "terraform"
      args = ["init"]
    }
    step {
      name = "hashicorp/terraform:1.10.1"
      dir  = "terraform"
      args = [
        "apply",
        "-auto-approve",
        "-var",
        "project=$_PROJECT",
        "-var",
        "region=$_REGION",
        "-var",
        "environment=$_ENVIRONMENT",
        "-var",
        "webapp_image=$_WEBAPP_IMAGE",
        "-var",
        "frontend_image=$_FRONTEND_IMAGE",
        "-var",
        "domain=$_DOMAIN",
      ]
    }
    options {
      logging               = "CLOUD_LOGGING_ONLY"
      dynamic_substitutions = true
    }
  }
}
