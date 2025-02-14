# Databricks bundle export

This was created with the databricks CLI using the command `databricks bundle init`.

You can import jobs from an existing databricks instance with e.g. `databricks bundle generate job --existing-job-id 50461440212450`. Workflows are types of jobs and they can be found by clicking on the workflow.

Exporting a job will export both the pipeline yaml (under resources) and also the underlying notebooks (under src).

You can deploy the bundle to a given workspace with `databricks bundle deploy` (haven't tried this yet). It looks like this will deploy both the notebooks (into a workspace) and the resources.