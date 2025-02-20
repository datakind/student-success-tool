output "cloudrun_service_account_email" {
  value = google_service_account.cloudrun_sa.email
}

output "cloudbuild_service_account_id" {
  value = google_service_account.cloudbuild_sa.id
}

output "cloudbuild_service_account_email" {
  value = google_service_account.cloudbuild_sa.email
}

output "terraform_service_account_id" {
  value = google_service_account.terraform_sa.id
}

#output "terraform_service_account_private_key" {
#  value     = google_service_account_key.cloudrun_sa_key.private_key
#  sensitive = true
#}

#output "decoded_private_key" {
#  value     = base64decode(google_service_account_key.cloudrun_sa_key.private_key)
#  sensitive = true
#}