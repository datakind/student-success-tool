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
   export DB_workspace=<Databricks Workspace name>
   export service_account_executer=<Service account to trigger the pipeline>
   export service_principal_run_as=<Service principal for the workflow to run as>
   export datakind_group_to_manage_workflow=<Datakind group to manahe the workflow>
   export pipeline_sa_email=<Service accoount to ingest the data>
   export datakind_notification_email=<Datakind email address to receive notifications>
   export end_user_notification_email=<Institution email address to receive notifications>
   export pip_install_branch=<Branch to install the SST libraries from>
   export custom_schemas_path=<Path to custom schemas>
   ```

   Then execute the command:

   ```
   $ databricks bundle deploy \
   --profile=$profile_name \
   --var="DB_workspace=$DB_workspace" \
   --var="service_account_executer=$service_account_executer" \
   --var="service_principal_run_as=$service_principal_run_as" \
   --var="datakind_group_to_manage_workflow=$datakind_group_to_manage_workflow" \
   --var="pipeline_sa_email=$pipeline_sa_email" \
   --var="end_user_notification_email=$end_user_notification_email" \
   --var="datakind_notification_email=$datakind_notification_email" \
   --var="pip_install_branch=$pip_install_branch" \
   --var="custom_schemas_path=$custom_schemas_path" \
   --target dev
   ```
    Note: "dev" is the default target, so the `--target dev` parameter is optional here.

    This deploys a single pipeline called `[dev yourname] github_sourced_pdp_inference_pipeline` to your workspace.
    You can find that job by opening your workspace and clicking **Workflows**.



4. Similarly, to deploy a production copy, execute:
   ```
   $ databricks bundle deploy \
   --profile=$profile_name \
   --var="DB_workspace=$DB_workspace" \
   --var="service_account_executer=$service_account_executer" \
   --var="service_principal_run_as=$service_principal_run_as" \
   --var="datakind_group_to_manage_workflow=$datakind_group_to_manage_workflow" \
   --var="pipeline_sa_email=$pipeline_sa_email" \
   --var="end_user_notification_email=$end_user_notification_email" \
   --var="datakind_notification_email=$datakind_notification_email" \
   --var="pip_install_branch=$pip_install_branch" \
   --var="custom_schemas_path=$custom_schemas_path" \
   --target prod
   ```

   Note: The production pipeline is not prefixed by the username. It is called `github_sourced_pdp_inference_pipeline`

5. The job could be executed via SST web application or to run it from CLI type:
   ```
   $ databricks bundle run \
   --profile=$profile_name \
   --var="DB_workspace=$DB_workspace" \
   --var="service_account_executer=$service_account_executer" \
   --var="service_principal_run_as=$service_principal_run_as" \
   --var="datakind_group_to_manage_workflow=$datakind_group_to_manage_workflow" \
   --var="pipeline_sa_email=$pipeline_sa_email" \
   --var="end_user_notification_email=$end_user_notification_email" \
   --var="datakind_notification_email=$datakind_notification_email" \
   --var="pip_install_branch=$pip_install_branch" \
   --var="custom_schemas_path=$custom_schemas_path" \
   --target dev
   ```
   Note: SST app will use only the prod version of the workflow
