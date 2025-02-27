"""Databricks SDk related helper functions.
"""

import os
import datetime
from pydantic import BaseModel
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Task, NotebookTask, Source
from databricks.sdk.service import catalog

from .config import databricks_vars, gcs_vars
from .utilities import databricksify_inst_name, SchemaType

# List of data medallion levels
medallion_levels = ["silver", "gold", "bronze"]
pdp_inference_job_name = "github_sourced_pdp_inference_pipeline"


class DatabricksInferenceRunRequest(BaseModel):
    """Databricks parameters for an inference run."""

    inst_name: str
    # Note that the following should be the filepath.
    filepath_to_type: dict[str, list[SchemaType]]
    model_name: str
    model_type: str = "sklearn"
    # The email where notifications will get sent.
    email: str
    gcp_external_bucket_name: str


class DatabricksInferenceRunResponse(BaseModel):
    """Databricks parameters for an inference run."""

    job_run_id: int


# Helper functions to get a file of a given file_type. For both, we will return the first file that matches the schema.
def get_filepath_of_filetype(
    file_dict: dict[str, list[SchemaType]], file_type: SchemaType
):
    for k, v in file_dict.items():
        if file_type in v:
            return k
    return ""


def check_types(dict_values, file_type: SchemaType):
    for elem in dict_values:
        if file_type in elem:
            return True
    return False


# Wrapping the usages in a class makes it easier to unit test via mocks.
class DatabricksControl(BaseModel):
    """Object to manage interfacing with GCS."""

    def setup_new_inst(self, inst_name: str) -> None:
        """Sets up Databricks resources for a new institution."""
        w = WorkspaceClient(
            host=databricks_vars["DATABRICKS_HOST_URL"],
            # This should still be cloud run, since it's cloud run triggering the databricks
            # this account needs to exist on Databricks as well and needs to hvae the creation and job management permissions
            google_service_account=gcs_vars["GCP_SERVICE_ACCOUNT_EMAIL"],
        )
        db_inst_name = databricksify_inst_name(inst_name)
        cat_name = databricks_vars["CATALOG_NAME"]
        for medallion in medallion_levels:
            w.schemas.create(name=f"{db_inst_name}_{medallion}", catalog_name=cat_name)
        # Create a managed volume in the bronze schema for internal pipeline data.
        # update to include a managed volume for toml files
        created_volume_bronze = w.volumes.create(
            catalog_name=cat_name,
            schema_name=f"{db_inst_name}_bronze",
            name=f"bronze_volume",
            volume_type=catalog.VolumeType.MANAGED,
        )
        created_volume_silver = w.volumes.create(
            catalog_name=cat_name,
            schema_name=f"{db_inst_name}_silver",
            name=f"silver_volume",
            volume_type=catalog.VolumeType.MANAGED,
        )
        created_volume_gold = w.volumes.create(
            catalog_name=cat_name,
            schema_name=f"{db_inst_name}_gold",
            name=f"gold_volume",
            volume_type=catalog.VolumeType.MANAGED,
        )

        # Create directory on the volume
        os.makedirs(
            f"/Volumes/{cat_name}/{db_inst_name}_gold/gold_volume/configuration_files/",
            exist_ok=True,
        )
        # Create directory on the volume
        os.makedirs(
            f"/Volumes/{cat_name}/{db_inst_name}_bronze/bronze_volume/raw_files/",
            exist_ok=True,
        )

    """Note that for each unique PIPELINE, we'll need a new function, this is by nature of how unique pipelines 
    may have unique parameters and would have a unique name (i.e. the name field specified in w.jobs.list()). But any run of a given pipeline (even across institutions) can use the same function. 
    E.g. there is one PDP inference pipeline, so one PDP inference function here."""

    def run_pdp_inference(
        self, req: DatabricksInferenceRunRequest
    ) -> DatabricksInferenceRunResponse:
        """Triggers PDP inference Databricks run."""
        print('aaaaaaaaaaaaaaaaaa0')
        if (
            not req.filepath_to_type
            or not check_types(req.filepath_to_type.values(), SchemaType.PDP_COURSE)
            or not check_types(req.filepath_to_type.values(), SchemaType.PDP_COHORT)
        ):
            raise ValueError(
                "run_pdp_inference() requires PDP_COURSE and PDP_COHORT type files to run."
            )
        print('aaaaaaaaaaaaaaaaaa1')
        print(databricks_vars["DATABRICKS_HOST_URL"])
        print(gcs_vars["GCP_SERVICE_ACCOUNT_EMAIL"])
        w = WorkspaceClient(
            host=databricks_vars["DATABRICKS_HOST_URL"],
            google_service_account=gcs_vars["GCP_SERVICE_ACCOUNT_EMAIL"],
        )
        db_inst_name = databricksify_inst_name(req.inst_name)
        print('aaaaaaaaaaaaaaaaaa2')
        list_jobs = w.jobs.list(name=pdp_inference_job_name)
        print('aaaaaaaaaaaaaaaaaa2.5')
        print(list_jobs)
        job = next(list_jobs)
        print('aaaaaaaaaaaaaaaaaa2.75')
        print(job)
        job_id = job.job_id
        print('aaaaaaaaaaaaaaaaaa3:'+str(job_id))
        print(databricks_vars["DATABRICKS_WORKSPACE"])
        run_job = w.jobs.run_now(
            job_id,
            job_parameters={
                "cohort_file_name": get_filepath_of_filetype(
                    req.filepath_to_type, SchemaType.PDP_COHORT
                ),
                "course_file_name": get_filepath_of_filetype(
                    req.filepath_to_type, SchemaType.PDP_COURSE
                ),
                "databricks_institution_name": db_inst_name,
                "DB_workspace": databricks_vars[
                    "DATABRICKS_WORKSPACE"
                ],  # is this value the same PER environ? dev/staging/prod
                "gcp_bucket_name": req.gcp_external_bucket_name,
                "model_name": req.model_name,
                "model_type": req.model_type,
                "notification_email": req.email,
            },
        )
        print('aaaaaaaaaaaaaaaaaa4')
        return DatabricksInferenceRunResponse(job_run_id=run_job.response.run_id)

    def delete_inst(self, inst_name: str) -> None:
        db_inst_name = databricksify_inst_name(inst_name)
        cat_name = databricks_vars["CATALOG_NAME"]
        w = WorkspaceClient(
            host=databricks_vars["DATABRICKS_HOST_URL"],
            # This should still be cloud run, since it's cloud run triggering the databricks
            # this account needs to exist on Databricks as well and needs to have permissions.
            google_service_account=gcs_vars["GCP_SERVICE_ACCOUNT_EMAIL"],
        )
        # Delete the managed volume.
        w.volumes.delete(name=f"{cat_name}.{db_inst_name}_bronze.bronze_volume")
        w.volumes.delete(name=f"{cat_name}.{db_inst_name}_silver.silver_volume")
        w.volumes.delete(name=f"{cat_name}.{db_inst_name}_gold.gold_volume")

        # Delete the MLflow model.
        # TODO how to handle deleting all models?
        """
        model_name = "latest_enrollment_model"
        new_institution_model_uri = f"{cat_name}.{db_inst_name}_gold.{model_name}"
        mlflow_client.delete_registered_model(name=new_institution_model_uri)
        """

        # Delete tables and schemas for each medallion level.
        for medallion in medallion_levels:
            all_tables = [
                table.name
                for table in w.tables.list(
                    catalog_name=cat_name,
                    schema_name=f"{db_inst_name}_{medallion}",
                )
            ]
            for table in all_tables:
                w.tables.delete(
                    full_name=f"{cat_name}.{db_inst_name}_{medallion}.{table}"
                )
            w.schemas.delete(full_name=f"{cat_name}.{db_inst_name}_{medallion}")
