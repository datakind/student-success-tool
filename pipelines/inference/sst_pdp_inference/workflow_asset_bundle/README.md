# SST Pipeline Workflow Asset Bundle

The 'SST Pipeline Workflow Asset Bundle' is used to deploy the inference pipeline from github

## Getting started

1. Install the Databricks CLI from https://docs.databricks.com/dev-tools/cli/databricks-cli.html

2. Authenticate to your Databricks workspace, if you have not done so already:
    ```
    $ databricks auth login --host <Databricks host URL>
    ```

3. To deploy a development copy of this project:

   Set local env variables:
   ```
   export ingestion_pipeline_service_account=<ingestion pipline service account email>
   export notification_email=<email address>
   export group_to_manage_workflow=<managing group>
   export user_service_account=<user service account email>
   ```

   Then execute the command:

   ```
   $ databricks bundle deploy \
   --var="pipeline_sa_email=$ingestion_pipeline_service_account" \
   --var="notification_email=$notification_email" --var="can_manage_group=$group_to_manage_workflow" \
   --var="can_run_account_email=$user_service_account" \
   --target dev
   ```
    Note: "dev" is the default target, so the `--target dev` parameter is optional here.

    This deploys a single pipeline called `[dev yourname] github_sourced_pdp_inference_pipeline` to your workspace.
    You can find that job by opening your workpace and clicking on **Workflows**.



4. The job could be executed from the SST application or from CLI type:
   ```
   $ databricks bundle run \
   --var="pipeline_sa_email=$ingestion_pipeline_service_account" \
   --var="notification_email=$notification_email" --var="can_manage_group=$group_to_manage_workflow" \
   --var="can_run_account_email=$user_service_account" \
   --target dev
   ```


5. Similarly, to deploy a production copy, type:
   ```
   $ databricks bundle deploy \
   --var="pipeline_sa_email=$ingestion_pipeline_service_account" \
   --var="notification_email=$notification_email" --var="can_manage_group=$group_to_manage_workflow" \
   --var="can_run_account_email=$user_service_account" \
   --target prod
   ```

   Note: The production pipeline is not prefixed by the username. It is called `github_sourced_pdp_inference_pipeline`
