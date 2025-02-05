# Terraform Configuration for Student Success Tool

This directory contains the Terraform configuration for the Student Success Tool. The configuration is organized into different environments, such as `dev`, `staging`, and `prod`.

## Environments

Each environment has its own directory under `environments/`. For example, the `dev` environment configuration is located in `environments/dev/`.

## Initial Application of the Configuration

To apply the Terraform configuration, navigate to the desired environment directory and run `terraform apply`. For example, to apply the configuration for the `dev` environment:

```sh
cd environments/dev/
terraform apply
```

You can provide variable values using a `terraform.tfvars` file or by supplying them directly on the command line. For example, to use a `terraform.tfvars` file:

```sh
terraform apply -var-file="terraform.tfvars"
```

Or to supply variables on the command line:

```sh
terraform apply -var="project=my-project" -var="region=us-central1"
```

## Applying Updates

After an environment has been applied for the first time, future updates may be applied via a Cloud Build trigger that can apply Terraform configurations. This allows for automated and continuous deployment of infrastructure changes.

## Configuration Details

### Backend Configuration

The Terraform state is stored in a Google Cloud Storage (GCS) bucket. The backend configuration specifies the bucket name and the prefix for the state files.

```hcl
terraform {
  backend "gcs" {
    bucket = "sst-terraform-state"
    prefix = "dev"
  }
}
```

### Providers

The configuration uses the Google Cloud provider and the Random provider. The required versions are specified in the `required_providers` block.

```hcl
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "6.8.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "3.6.3"
    }
  }
}
```

### Google Cloud Provider

The Google Cloud provider is configured with the project ID, region, and zone.

```hcl
provider "google" {
  project = var.project
  region  = var.region
  zone    = var.zone
}
```

### Modules

The configuration uses two modules: `deployment` and `cloudbuild`.

#### Deployment Module

The `deployment` module is responsible for deploying the application. It requires several variables, such as project ID, region, environment, database version, database name, domain, and Docker images for the web application and frontend.

```hcl
module "deployment" {
  source = "../../modules/deployment"

  project          = var.project
  region           = var.region
  environment      = var.environment
  zone             = var.zone
  database_version = var.database_version
  database_name    = var.database_name
  domain           = var.domain
  webapp_image     = var.webapp_image
  frontend_image   = var.frontend_image
}
```

#### Cloud Build Module

The `cloudbuild` module is responsible for configuring Cloud Build. It requires variables such as project ID, domain, environment, region, and Docker images. It also uses the service account ID from the `deployment` module.

```hcl
module "cloudbuild" {
  source = "../../modules/cloudbuild"

  project        = var.project
  domain         = var.domain
  environment    = var.environment
  region         = var.region
  webapp_image   = var.webapp_image
  frontend_image = var.frontend_image

  cloudbuild_service_account_id = module.deployment.iam.cloudbuild_service_account_id
}
```

This module is only configured for the `dev` environment and sets up continuous deployment with cloudbuild triggers on code push. The triggers for `dev` can also deploy the other environments by changing the variable `_ENVIRONMENT` for a manual trigger run.