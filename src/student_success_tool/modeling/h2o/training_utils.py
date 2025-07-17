import mlflow
import os
import pandas as pd
import datetime
from mlflow.tracking import MlflowClient


def set_or_create_experiment(workspace_path, institution_id, target_name, checkpoint_name, use_timestamp=True, client=None):
    if client is None:
        client = MlflowClient()

    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S") if use_timestamp else ""

    name_parts = [
        cfg.institution_id,
        cfg.preprocessing.target.name,
        cfg.preprocessing.checkpoint.name,
        prefix,
        timestamp
    ]
    experiment_name = "/".join([
        workspace_path.rstrip("/"),
        "h2o_automl",
        "_".join([part for part in name_parts if part]),
    ])

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)
        return experiment_id
    except Exception as e:
        raise RuntimeError(f"Failed to create or set MLflow experiment: {e}")