"""Databricks SDk related helper functions.
"""

import datetime
from pydantic import BaseModel

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Task, NotebookTask, Source
from databricks.sdk.service import catalog

from .config import databricks_vars

from .utilities import databricksify_inst_name

# TODO uncomment once authn with Databricks is resolved
# The Databricks workspace client
# w = WorkspaceClient()

medallion_levels = ["silver", "gold", "bronze"]  # List of data medallion levels


# Wrapping the usages in a class makes it easier to unit test via mocks.
class DatabricksControl(BaseModel):
    """Object to manage interfacing with GCS."""

    def setup_catalog_new_inst(self, inst_name: str) -> None:
        """Sets up Databricks resources for a new institution."""
        w = WorkspaceClient()
        db_inst_name = databricksify_inst_name(inst_name)
        cat_name = databricks_vars["CATALOG_NAME"]
        for medallion in medallion_levels:
            w.schemas.create(name=f"{db_inst_name}_{medallion}", catalog_name=cat_name)

        # Create a managed volume in the bronze schema for internal pipeline data.
        created_volume = w.volumes.create(
            catalog_name=cat_name,
            schema_name=f"{db_inst_name}_bronze",
            name=f"pdp_pipeline_internal",
            volume_type=catalog.VolumeType.MANAGED,
        )
