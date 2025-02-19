resource "google_project_service" "services" {
  for_each = toset(var.required_services)

  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "student_success_tool" {
  location      = var.region
  repository_id = "student-success-tool"
  format        = "DOCKER"
}

resource "google_artifact_registry_repository" "sst_app_ui" {
  location      = var.region
  repository_id = "sst-app-ui"
  format        = "DOCKER"
}

resource "google_cloudbuild_trigger" "python_apps" {
  for_each        = toset(["webapp", "worker"])
  name            = "${var.environment}-${each.key}"
  service_account = var.cloudbuild_service_account_id
  dynamic "github" {
    for_each = var.environment == "dev" ? [1] : []
    content {
      owner = "datakind"
      name  = "student-success-tool"
      push {
        branch = "fellows-experimental"
      }
    }
  }
  dynamic "source_to_build" {
    for_each = var.environment != "dev" ? [1] : []
    content {
      ref       = "refs/heads/fellows-experimental"
      repo_type = "GITHUB"
      uri       = "https://github.com/datakind/student-success-tool"
    }
  }
  build {
    step {
      name = "gcr.io/cloud-builders/docker"
      args = [
        "build",
        "-f",
        "src/${each.key}/Dockerfile",
        "-t",
        "${var.region}-docker.pkg.dev/${var.project}/student-success-tool/${each.key}:$COMMIT_SHA",
        "-t",
        "${var.region}-docker.pkg.dev/${var.project}/student-success-tool/${each.key}:latest",
        "."
      ]
    }
    step {
      name = "gcr.io/cloud-builders/docker"
      args = ["push", "${var.region}-docker.pkg.dev/${var.project}/student-success-tool/${each.key}:$COMMIT_SHA"]
    }
    step {
      name = "gcr.io/cloud-builders/docker"
      args = ["push", "${var.region}-docker.pkg.dev/${var.project}/student-success-tool/${each.key}:latest"]
    }
    step {
      name = "gcr.io/cloud-builders/gcloud"
      args = [
        "run",
        "deploy",
        "${var.environment}-${each.key}",
        "--image",
        "${var.region}-docker.pkg.dev/${var.project}/student-success-tool/${each.key}:$COMMIT_SHA",
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
  name            = "${var.environment}-frontend"
  service_account = var.cloudbuild_service_account_id
  dynamic "github" {
    for_each = var.environment == "dev" ? [1] : []
    content {
      owner = "datakind"
      name  = "sst-app-ui"
      push {
        branch = "develop"
      }
    }
  }
  dynamic "source_to_build" {
    for_each = var.environment != "dev" ? [1] : []
    content {
      ref       = "refs/heads/fellows-experimental"
      repo_type = "GITHUB"
      uri       = "https://github.com/datakind/sst-app-ui"
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
        "${var.environment}-migrate",
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
        "${var.environment}-frontend",
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
  name            = "${var.environment}-terraform"
  service_account = var.terraform_service_account_id
  source_to_build {
    ref       = "refs/heads/fellows-experimental"
    repo_type = "GITHUB"
    uri       = "https://github.com/datakind/student-success-tool"
  }
  build {
    step {
      name = "hashicorp/terraform:1.10.1"
      dir  = "terraform/environments/${var.environment}"
      args = ["init"]
    }
    step {
      name = "hashicorp/terraform:1.10.1"
      dir  = "terraform/environments/${var.environment}"
      args = [
        "apply",
        "-auto-approve",
        "-var",
        "project=${var.project}",
        "-var",
        "domain=${var.domain}",
        "-var",
        "subnet_ip_cidr_range=${var.subnet_ip_cidr_range}",
        "-var",
        "vpc_host_project=${var.vpc_host_project}",
        "-var",
        "vpc_host_network=${var.vpc_host_network}",
      ]
    }
    options {
      logging               = "CLOUD_LOGGING_ONLY"
      dynamic_substitutions = true
    }
  }
}
