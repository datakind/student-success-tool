# SST Model Inference Pipeline - README

This pipeline is a proof of concept (POC) Model Inference Pipeline for PDP Institutions, it facilitates batch inference using a pre-trained enrollment model to generate predictions for student success.

## Overview

This pipeline automates the process of:

1. **Data generation:** This is an optional step to generate synthetic test data. 
2. **Data Ingestion:** Retrieving raw CSV data from the SST App's designated Google Cloud Storage (GCS) bucket.
3. **Data Processing:** Transforming and filtering the ingested data according to institution-specific configurations.
4. **Model Inference:** Applying a pre-trained machine learning model to generate predictions. 

This pipeline is intended for batch inference jobs and is deployed as a Databricks Workflow.

## Current Assumptions

Before running the pipeline, ensure the following prerequisites are met:

* **Institution Onboarding:** The target institution must be onboarded with the following Databricks components:
    * **Databricks Schemas:** Three schemas: `institution_ID_bronze`, `institution_ID_silver`, and `institution_ID_gold`.
    * **Databricks Volume:** A Databricks Volume named `pdp_pipeline_internal` within the bronze schema (e.g., `/Volumes/{workspace}/{institution_ID}_bronze/pdp_pipeline_internal`).
    * **External GCS Bucket:** A GCS bucket for storing raw CSV files, with the URI structure: `gs://{workspace}_{institution_ID}_sst_application/validated/`.
    * **Registered Model:** A trained model registered within the institution's bronze schema (e.g., `{workspace}.{institution_ID}_bronze.latest_enrollment_model`).
    * **Institution Configuration File:** A `.toml` configuration file stored in the Databricks Volume: `/Volumes/{workspace}/{institution_ID}_bronze/pdp_pipeline_internal/configuration_files/{institution_ID}.toml`.
* **Data Schema Compliance:** The student and cohort CSV files must adhere to the PDP schema.
    * Sample files are available for reference at: https://github.com/datakind/student-success-tool/tree/pedro-develop/synthetic-data/pdp

## Pipeline Tasks

The pipeline executes the following tasks:

1.  **Optional Synthetic Data Generation (For Development/Testing):** If needed, synthetic data can be generated for testing purposes.
2.  **Data Ingestion from GCS:** CSV files are ingested from the institution's GCS bucket.
3.  **Data Processing:** The ingested data is processed based on the parameters defined in the institution's `.toml` configuration file. This includes data cleaning, transformation, and feature engineering.
4.  **Model Inference:** The institution's pre-trained registered model is loaded and used to generate predictions on the processed data.
5.  **Result Storage:** The inference results are stored in the appropriate Databricks schema and tables.

## Running Inference with an Existing Test Institution ("standard_pdp_institution")

This section provides instructions for running the inference pipeline with a pre-configured test institution.

### Prerequisites

* [x] A test institution named "standard_pdp_institution" must be onboarded.
* [x] Sample student and cohort CSV files synthetically generated have already been saved on the bucket URI: gs://dev_sst_02_standard_pdp_institution_sst_application/validated/ 
    - File names: 
        - standard_pdp_institution_sample_STUDENT_SEMESTER_AR_DEIDENTIFIED.csv
        - standard_pdp_institution_sample_COURSE_LEVEL_AR_DEID.csv
* [x] The standard_pdp_institution.toml file is stored in the Databricks Volume 
* [x] The Workflow called ["Github_Sourced_PDP_Inference_Pipeline"](./Github_Sourced_PDP_Inference_Pipeline.yaml) is already deployed on Databricks and configured to run on a job cluster (ephemeral cluster for each run).

### Steps to run with default values

1.  **Trigger the Databricks Workflow from user interface:**
    * Navigate to the Databricks Workflows section.
    * Locate the Workflow called "Github_Sourced_PDP_Inference_Pipeline".
    * Trigger a new run of the workflow "Run now" with default settings.
2.  **Monitor the Workflow Run:**
    * Observe the progress of each task within the workflow.
    * Check for any errors or failures.
3.  **Verify Inference Results:**
    * Once the workflow completes successfully, query the results table in the institution silver schema (e.g., `standard_pdp_institution_silver.{job_id}_predicted_dataset`).
    * Verify the data output matches the expected schema and values.

### Important Notes

* **Configuration Files:** The `.toml` configuration file is crucial for customizing the data processing steps. Ensure it is correctly configured for the target institution.
* **Model Versioning:** The pipeline utilizes the "latest" version of the registered model. If you need to use a specific model version, update the pipeline code accordingly.