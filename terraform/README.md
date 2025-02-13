# Terraform Configuration for Student Success Tool

This directory contains the Terraform configuration for the Student Success Tool. The configuration is organized into different environments, such as `dev`, `staging`, and `prod`.

## Environments

Each environment has its own directory under `environments/`. For example, the `dev` environment configuration is located in `environments/dev/`.

Each environment directory contains the following files:

- `main.tf`: The main Terraform configuration file for the environment.
- `variables.tf`: Defines the variables used in the environment.
- `terraform.tfvars`: Contains the values for the variables specific to the environment. These files are usually added to `.gitignore` to prevent sensitive information from being tracked in version control.

Example directory structure:

```sh
environments/
  ├── dev/
  │   ├── main.tf
  │   ├── variables.tf
  │   └── terraform.tfvars
  ├── staging/
  │   ├── main.tf
  │   ├── variables.tf
  │   └── terraform.tfvars
  └── prod/
      ├── main.tf
      ├── variables.tf
      └── terraform.tfvars
```

Example tfvars file:
```sh
domain         = "sst.datakind.org"
frontend_image = "us-docker.pkg.dev/cloudrun/container/placeholder"
webapp_image   = "us-docker.pkg.dev/cloudrun/container/placeholder"
project        = "my-project-id"
```

## Application of the Configuration

### Prerequesites

- [Install the Terraform CLI Tool](https://developer.hashicorp.com/terraform/install)
- [Configure Google Auth Platform](https://console.cloud.google.com/auth/overview)
- [Connect GitHub Repositories](https://console.cloud.google.com/cloud-build/repositories/2nd-gen)

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

## DNS Configs

To ensure that your environment is properly set up, you need to register the IP address of the load balancer to the domain used by the environment. This step is crucial for directing traffic to your application.

Here's how you can do it:

[Obtain the IP Address](https://console.cloud.google.com/net-services/loadbalancing/details/http/dev-tf-cr-url-map-1): After setting up your load balancer, you will receive a Frontend IP address. This IP address is what you will use to point your domain to your load balancer.

Update DNS Records: Go to your domain registrar's DNS settings and create an A record. The A record should point your domain (e.g., www.example.com) to the IP address of your load balancer.

Verify the Setup: After updating the DNS records, it may take some time for the changes to propagate. Once propagated, accessing your domain should route traffic through the load balancer to your application.

Here's an example of how you might update your DNS settings:

```
Type: A
Name: @ (or your subdomain, e.g., www)
Value: [Your Load Balancer IP Address]
TTL: 3600 (or your preferred TTL)
```

### Initial Environment Setup

For the initial environment setup, use placeholder images like the "Hello World" example for Cloud Run, as the images for the web application and frontend UI will not exist yet. These placeholder images will take the place of the `webapp_image` and `frontend_image` variables.

Example placeholder images:

```sh
terraform apply -var="webapp_image=gcr.io/cloudrun/hello" -var="frontend_image=gcr.io/cloudrun/hello"
```

## Configuration Details

### Backend Configuration

The Terraform state is stored in a Google Cloud Storage (GCS) bucket. The backend configuration specifies the bucket name and, optionally, the prefix for the state files. The bucket name must be globally unique.

```hcl
terraform {
  backend "gcs" {
    bucket = "<bucket name>"
    prefix = "<optional object prefix>"
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

The Terraform configuration is organized into several modules, each responsible for a specific part of the infrastructure. Below is a brief description of each module:

- **network**: Configures the VPC network, subnetwork, and VPC peering.
- **iam**: Manages IAM roles and service accounts for Cloud Run and Cloud Build.
- **database**: Sets up the Cloud SQL database instance, users, and secrets for database credentials.
- **service**: Deploys Cloud Run services and configures environment variables and secrets.
- **deployment**: Coordinates the deployment of the entire infrastructure by invoking other modules.
- **cloudbuild**: Configures Cloud Build triggers for building and deploying the web application and frontend UI.
- **job**: Manages Cloud Run jobs for tasks such as database migrations and other background processes.

Each module is located in its respective directory under `modules/`.

```sh
modules/
  ├── cloudbuild/
  ├── database/
  ├── deployment/
  ├── iam/
  ├── job/
  ├── network/
  └── service/
```
