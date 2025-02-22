# SST Pipeline Workflow Asset Bundle

The 'SST Pipeline Workflow Asset Bundle' is used to deploy the demo inference pipeline based on [the workflow YAML definition](resources/github_sourced_pdp_inference_pipeline.yml)

## Getting started

1. Install the Databricks CLI from https://docs.databricks.com/dev-tools/cli/databricks-cli.html

2. Authenticate to your Databricks workspace, if you have not done so already:
    ```
    $ databricks auth login --host <Databricks host URL>
    ```
   This will create a local **Databricks profile name** that is needed for Step 3.
   

3. To deploy a development copy of this workflow:

   Set local env variables:
   ```
   export profile_name=<Profile name created on Step 2>
   export group_to_manage_workflow=<Databrics managing group>
   export service_account_user=<Service account user>
   export pipeline_sa_email=<Ingestion pipline service account email>
   export datakind_notification_email=<Datakind email to receive notifications>
   export end_user_notification_email=<End user email to receive notifications>
   ```

   Then execute the command:

   ```
   $ databricks bundle deploy \
   --profile=$profile_name \
   --var="service_account_user=$service_account_user" \
   --var="group_to_manage_workflow=$group_to_manage_workflow" \
   --var="pipeline_sa_email=$pipeline_sa_email" \
   --var="end_user_notification_email=$end_user_notification_email" \
   --var="datakind_notification_email=$datakind_notification_email" \
   --target dev
   ```
    Note: "dev" is the default target, so the `--target dev` parameter is optional here.

    This deploys a single pipeline called `[dev yourname] github_sourced_pdp_inference_pipeline` to your workspace.
    You can find that job by opening your workpace and clicking on **Workflows**.



4. Similarly, to deploy a production copy, execute:
   ```
   $ databricks bundle deploy \
   --profile=$profile_name \
   --var="service_account_user=$service_account_user" \
   --var="group_to_manage_workflow=$group_to_manage_workflow" \
   --var="pipeline_sa_email=$pipeline_sa_email" \
   --var="end_user_notification_email=$end_user_notification_email" \
   --var="datakind_notification_email=$datakind_notification_email" \
   --target prod
   ```

   Note: The production pipeline is not prefixed by the username. It is called `github_sourced_pdp_inference_pipeline`

5. The job could be executed via SST web application or to run it from CLI type:
   ```
   $ databricks bundle run \
   --profile=$profile_name \
   --var="service_account_user=$service_account_user" \
   --var="group_to_manage_workflow=$group_to_manage_workflow" \
   --var="pipeline_sa_email=$pipeline_sa_email" \
   --var="end_user_notification_email=$end_user_notification_email" \
   --var="datakind_notification_email=$datakind_notification_email" \
   --target dev
   ```
   Note: SST app will use only the prod version of the workflow
