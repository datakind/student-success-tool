"""Databricks SDk related helper functions.
"""

import datetime
from pydantic import BaseModel
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Task, NotebookTask, Source
from databricks.sdk.service import catalog

from .config import databricks_vars, gcs_vars
from .utilities import databricksify_inst_name

medallion_levels = ["silver", "gold", "bronze"]  # List of data medallion levels
inference_job_name = "inference_pipeline"  # can this be the job name for every inst or does it havve to be inst specific


class DatabricksInferenceRunRequest(BaseModel):
    """Databricks parameters for an inference run."""

    inst_name: str
    file_to_type: dict[str, str]
    model_name: str


class DatabricksInferenceRunResponse(BaseModel):
    """Databricks parameters for an inference run."""

    job_run_id: int


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

    def run_inference(
        self, req: DatabricksInferenceRunRequest
    ) -> DatabricksInferenceRunResponse:
        """Triggers Databricks run."""
        w = WorkspaceClient(
            host=databricks_vars["DATABRICKS_HOST_URL"],
            google_service_account=gcs_vars["GCP_SERVICE_ACCOUNT_EMAIL"],
        )
        db_inst_name = databricksify_inst_name(req.inst_name)
        job_id = next(w.jobs.list(name=inference_job_name)).job_id
        run_job = w.jobs.run_now(
            job_id,
            job_parameters={
                # "synthetic_needed": "True",
                "institution_id": db_inst_name,
                # "sst_job_id": f"{institution_id}_inference_job_id_{str(random.randint(1, 1000))}",
                "DB_workspace": databricks_vars[
                    "DATABRICKS_WORKSPACE"
                ],  # is this value the same PER environemtn? dev/staging/prod
                "model_name": req.model_name,
                "model_type": None,  # ?
                # how do we pass the files
            },
        )
        return DatabricksInferenceRunResponse(job_run_id=run_job.response.run_id)
