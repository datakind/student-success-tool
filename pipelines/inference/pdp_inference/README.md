# SST Model Inference Pipeline - README

This pipeline is a proof of concept (POC) Model Inference Pipeline for PDP Institutions. It facilitates batch inference using a pre-trained enrollment model to generate predictions for student success. It is intended for batch inference jobs submitted from SST web application, but this guide includes an example to execute the job from Databricks console. 

## Overview

The pipeline is deployed as a Databricks Workflow automates the process of:

1. **Data Ingestion:** Retrieving raw CSV data from the SST App's designated Google Cloud Storage (GCS) bucket.
2. **Data Preprocessing:** Transforming and filtering the ingested data according to institution-specific configurations.
3. **Data validation:** **This task is still to be implemented** 
4. **Model Inference:** Applying a pre-trained machine learning model to generate predictions. 

## Current Assumptions

Before running the pipeline, ensure the following prerequisites are met:

* **Institution Onboarding:** The target institution must be onboarded with the following Databricks components:
    * **Databricks Schemas:** Three schemas: `institution_name_bronze`, `institution_name_silver`, and `institution_name_gold`.
    * **Databricks Volume:** A Databricks Volume named `pdp_pipeline_internal` within the bronze schema (e.g., `/Volumes/{workspace}/{institution_name}_bronze/bronze_volume`).
    * **Databricks Volume:** A Databricks Volume named `gold_volume` within the gold schema (e.g., `/Volumes/{workspace}/{institution_name}_bronze/gold_volume`).
    * **External GCS Bucket:** A GCS bucket for storing raw CSV files, with the URI structure: `gs://{workspace}_{institution_ID}/validated/`.
    * **Registered Model:** A trained model registered within the institution's bronze schema (e.g., `{workspace}.{institution_name}_bronze.latest_enrollment_model`).
    * **Institution Configuration File:** A `.toml` configuration file stored in the Gold volume: `/Volumes/{workspace}/{institution_name}_gold/gold_volume/configuration_files/{institution_name}.toml`.
* **Data Schema Compliance:** The student and cohort CSV files must adhere to the PDP schema.
    * Sample files are available for reference at: https://github.com/datakind/student-success-tool/tree/develop/synthetic-data/pdp
* **Databricks Workflow is deployed:** The Workflow called ["github_sourced_pdp_inference_pipeline"](./workflow_asset_bundle/resources/github_sourced_pdp_inference_pipeline.yml) is already deployed using Databricks Asset Bundles, the workflow is configured to run on a Databricks Job cluster (ephemeral cluster for each run). See [Workflow Readme file](./workflow_asset_bundle/README.md) with details to deploy the workflow.

## Pipeline Tasks

The pipeline executes the following tasks:

1.  **Data Ingestion from GCS:** CSV files are ingested from the institution's GCS bucket.
2.  **Data Preprocessing:** The ingested data is processed based on the parameters defined in the institution's `.toml` configuration file. This includes data cleaning, transformation, and feature engineering.
3.  **Data Validation:** This task will use Tensorflow Data Validation (TFDV) library to validate the data before prediction.
4.  **Model Inference:** The institution's pre-trained registered model is loaded and used to generate predictions on the processed data.
5.  **Result Storage:** The inference results are stored in the institution's silver schema for review.

## Running Inference with an Existing Test Institution

This section provides instructions for running the inference pipeline with a pre-configured test institution.

### Prerequisites

* [x] A test institution must be onboarded.
* [x] Sample student and cohort CSV files synthetically generated have already been saved on the institution's external bucket 
    - File names: 
        - synthetic_STUDENT_SEMESTER_AR_DEIDENTIFIED.csv
        - synthetic_COURSE_LEVEL_AR_DEID.csv
* [x] The institution's .toml file is stored in the gold Volume 
* [x] Workflow ["github_sourced_pdp_inference_pipeline"](./workflow_asset_bundle/resources/github_sourced_pdp_inference_pipeline.yml) is already deployed 

### Steps to run with default values

1.  **Trigger the Databricks Workflow from user interface:**
    * Navigate to the Databricks Workflows section.
    * Locate the Workflow called "Github_Sourced_PDP_Inference_Pipeline".
    * Trigger a new run of the workflow "Run now" with default settings.
2.  **Monitor the Workflow Run:**
    * Observe the progress of each task within the workflow.
    * Check for any errors or failures.
3.  **Verify Inference Results:**
    * Once the workflow completes successfully, query the results table in the institution silver schema (e.g., `institution_name_silver.{job_id}_predicted_dataset`).
    * Verify the data output matches the expected schema and values.

### Important Notes

* **Configuration Files:** The `.toml` configuration file is crucial for customizing the data processing steps, it is model-specific.  Ensure it is correctly configured for the target institution and the specific model.
* **Model Versioning:** The pipeline utilizes the "latest" version of the registered model. If you need to use a specific model version, update the pipeline parameter accordingly.