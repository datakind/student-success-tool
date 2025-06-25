# SST Model Inference Pipeline - README

This pipeline is a proof of concept (POC) Model Inference Pipeline for PDP Institutions. It facilitates batch inference using a pre-trained enrollment model to generate predictions for student success. It is intended for batch inference jobs submitted from the SST web application, but this guide includes an example to execute the job from the Databricks console.

## Overview

The pipeline is deployed as a Databricks Workflow that automates the process of:

1.  **Data Ingestion:** Retrieving raw CSV data from the SST App's designated Google Cloud Storage (GCS) bucket.
2.  **Data Preprocessing:** Transforming and filtering the ingested data according to institution-specific configurations.
3.  **Data validation:** This task uses TensorFlow Data Validation (TFDV) to validate data against a reference schema and detect anomalies.
4.  **Model Inference:** Applying a pre-trained machine learning model to generate predictions and calculate SHAP values.
5.  **Inference Output Publish:** Copying the model inference output data to the SST App's designated Google Cloud Storage (GCS) bucket.

## Current Assumptions

Before running the pipeline, ensure the following prerequisites are met:

* **Institution Onboarding:** The target institution must be onboarded with the following Databricks components:
    * **Databricks Schemas:** Three schemas: `institution_name_bronze`, `institution_name_silver`, and `institution_name_gold`.
    * **Databricks Volumes:** A Databricks Volume for each of the institution's corresponding schemas: `gold_volume`, `silver_volume`, and `bronze_volume`.
    * **External GCS Bucket:** A GCS bucket for storing raw CSV files, with the URI structure: `gs://{workspace}_{institution_ID}/validated/`.

* **Items from Model Training:**
    * **Trained Model:** The model used should be registered within the institution's gold schema (e.g., `{workspace}.{institution_name}_gold.latest_enrollment_model`).
    * **Institution Configuration File:** A `.toml` configuration file stored in the Gold volume: `/Volumes/{workspace}/{institution_name}_gold/gold_volume/configuration_files/{institution_name}_{model_name}_configuration_file.toml`.
    * **Data Validation Schema:** A `.pbtxt` file is expected for the [data validation task](https://github.com/datakind/student-success-tool/tree/develop/pipelines/tasks/data_validation). This file should be persisted during model training.

* **Data Schema Compliance:**
    * The student and cohort CSV files must adhere to the PDP schema.
    * Sample files are available for reference at: https://github.com/datakind/student-success-tool/tree/develop/synthetic-data/pdp

* **Databricks Workflow:**
    * The Workflow called ["github_sourced_pdp_inference_pipeline"](./workflow_asset_bundle/resources/github_sourced_pdp_inference_pipeline.yml) is already deployed using Databricks Asset Bundles. The workflow is configured to run on a Databricks Job cluster (an ephemeral cluster for each run). See the [Workflow README file](./workflow_asset_bundle/README.md) for details on deploying the workflow.

## Pipeline Tasks

The pipeline executes the following tasks:

1.  **Data Ingestion from GCS:** CSV files are ingested from the institution's GCS bucket.
2.  **Data Preprocessing:** The ingested data is processed based on the parameters defined in the institution's `.toml` configuration file. This includes data cleaning, transformation, and feature engineering.
3.  **Data Validation:** This task uses the TensorFlow Data Validation (TFDV) library to validate the data before prediction.
4.  **Model Inference:** The institution's pre-trained registered model is loaded and used to generate predictions on the processed data.
5.  **Inference Output Publish:** The inference results, including the SHAP chart and CSV file, are stored in the institution's bucket for review.

## Running Inference with an Existing Test Institution

This section provides instructions for running the inference pipeline with a pre-configured test institution.

### Prerequisites

* [x] A test institution must be onboarded.
* [x] Sample student and cohort CSV files, synthetically generated, have already been saved in the institution's external bucket.
    * File names:
        * synthetic_STUDENT_SEMESTER_AR_DEIDENTIFIED.csv
        * synthetic_COURSE_LEVEL_AR_DEID.csv
* [x] The institution's `.toml` file is stored in the Gold Volume with the correct parameters.
* [x] The Workflow ["github_sourced_pdp_inference_pipeline"](./workflow_asset_bundle/resources/github_sourced_pdp_inference_pipeline.yml) is already deployed.

### Steps to Run with Default Values

1.  **Trigger the Databricks Workflow from the User Interface:**
    * Navigate to the Databricks Workflows section.
    * Locate the Workflow called "github_sourced_pdp_inference_pipeline".
    * Trigger a new run of the workflow using "Run now" with default settings.
2.  **Monitor the Workflow Run:**
    * Observe the progress of each task within the workflow.
    * Check for any errors or failures.
3.  **Verify Inference Results:**
    * Once the workflow completes successfully, query the results table in the institution's silver schema (e.g., `institution_name_silver.{job_id}_predicted_dataset`).
    * Verify the data output matches the expected schema and values.

### Important Notes

* **Configuration Files:** The `.toml` configuration file is crucial for customizing the data processing steps. Ensure it is correctly configured for the target institution and the specific model.
* **Model Versioning:** The pipeline utilizes the "latest" version of the registered model. If you need to use a specific model version, update the pipeline parameter accordingly.