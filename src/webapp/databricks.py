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

medallion_levels = ["silver", "gold", "bronze"]  # List of data medallion levels
# For every unique pipeline with unique param sets, you'll need a separate name.
pdp_inference_job_name = "pdp_inference_pipeline"  # can this be the job name for every inst or does it havve to be inst specific


class DatabricksInferenceRunRequest(BaseModel):
    """Databricks parameters for an inference run."""

    inst_name: str
    # Note that the following should be the filepath.
    filepath_to_type: dict[str, list[SchemaType]]
    model_name: str
    model_type: str = "sklearn"
    # The email where notifications will get sent.
    email: str


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


# Wrapping the usages in a class makes it easier to unit test via mocks.
class DatabricksControl(BaseModel):
    """Object to manage interfacing with GCS."""

    def setup_new_inst(self, inst_name: str) -> None:
        """Sets up Databricks resources for a new institution."""
        w = WorkspaceClient(
            host=databricks_vars["DATABRICKS_HOST_URL"],
            google_service_account=gcs_vars["GCP_SERVICE_ACCOUNT_EMAIL"],
        )
        db_inst_name = databricksify_inst_name(inst_name)
        cat_name = databricks_vars["CATALOG_NAME"]
        for medallion in medallion_levels:
            w.schemas.create(name=f"{db_inst_name}_{medallion}", catalog_name=cat_name)

        # Create a managed volume in the bronze schema for internal pipeline data.
        created_volume = w.volumes.create(
            catalog_name=cat_name,
            schema_name=f"{db_inst_name}_bronze",
            name=f"pipeline_internal",
            volume_type=catalog.VolumeType.MANAGED,
        )
        # Create directory on the volume
        os.makedirs(
            f"/Volumes/{cat_name}/{db_inst_name}_bronze/pipeline_internal/configuration_files/",
            exist_ok=True,
        )

    """Note that for each unique PIPELINE, we'll need a new function, this is by nature of how unique pipelines 
    may have unique parameters and would have a unique name (i.e. the name field specified in w.jobs.list()). But any run of a given pipeline (even across institutions) can use the same function. 
    E.g. there is one PDP inference pipeline, so one PDP inference function here."""

    def run_pdp_inference(
        self, req: DatabricksInferenceRunRequest
    ) -> DatabricksInferenceRunResponse:
        """Triggers PDP inference Databricks run."""
        """
        if (
            not req.file_to_type or SchemaType.PDP_COURSE not in req.file_to_type.values()
            or SchemaType.PDP_COHORT not in req.file_to_type.values()
        ):
            raise ValueError(
                "run_pdp_inference() requires PDP_COURSE and PDP_COHORT type files to run."
            )
        """
        print("xxxxxxxxxxxxxxxxxxxx1")
        w = WorkspaceClient(
            host=databricks_vars["DATABRICKS_HOST_URL"],
            google_service_account=gcs_vars["GCP_SERVICE_ACCOUNT_EMAIL"],
        )
        print("xxxxxxxxxxxxxxxxxxxx2")
        db_inst_name = databricksify_inst_name(req.inst_name)
        print("xxxxxxxxxxxxxxxxxxxx3")
        job_id = next(w.jobs.list(name=pdp_inference_job_name)).job_id
        # TODO xxxx delete hardcoded values
        print("xxxxxxxxxxxxxxxxxxxx4")
        run_job = w.jobs.run_now(
            job_id,
            job_parameters={
                "cohort_file_name": "standard_pdp_institution_sample_STUDENT_SEMESTER_AR_DEIDENTIFIED.csv",
                # get_filepath_of_filetype(
                #    req.file_to_type, SchemaType.PDP_COHORT
                # ),
                "course_file_name": "standard_pdp_institution_sample_COURSE_LEVEL_AR_DEID.csv",
                # get_filepath_of_filetype(
                #    req.file_to_type, SchemaType.PDP_COURSE
                # ),
                "institution_id": "standard_pdp_institution",  # db_inst_name,
                # "sst_job_id": f"{institution_id}_inference_job_id_{str(random.randint(1, 1000))}",
                "DB_workspace": databricks_vars[
                    "DATABRICKS_WORKSPACE"
                ],  # is this value the same PER environemtn? dev/staging/prod
                "model_name": "latest_enrollment_model",  # req.model_name,
                "model_type": req.model_type,
                # "notification_email": req.email,
            },
        )
        print("xxxxxxxxxxxxxxxxxxxx5")
        return DatabricksInferenceRunResponse(job_run_id=run_job.response.run_id)
