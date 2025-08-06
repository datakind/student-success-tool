import logging
import typing as t

import os
import datetime
import tempfile

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts
import pandas as pd

import h2o
from h2o.automl import H2OAutoML

from . import evaluation_h2o as evaluation
from . import imputation

LOGGER = logging.getLogger(__name__)


def download_model_artifact(run_id: str, artifact_subdir: str = "model") -> str:
    """
    Downloads a model directory artifact from MLflow and returns the local path.

    Args:
        run_id: MLflow run ID.
        artifact_subdir: Subdirectory in the run artifacts, usually 'model'.

    Returns:
        Path to the downloaded model directory.
    """
    local_dir = tempfile.mkdtemp()
    artifact_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_subdir, dst_path=local_dir
    )
    return artifact_path  # already includes artifact_subdir


def load_h2o_model(
    run_id: str, artifact_path: str = "model"
) -> h2o.model.model_base.ModelBase:
    """
    Initializes H2O, downloads the model artifact from MLflow, and loads it.
    Cleans up the temp directory after loading.
    """
    if not h2o.connection():
        h2o.init()

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_model_dir = download_artifacts(
            run_id=run_id, artifact_path=artifact_path, dst_path=tmp_dir
        )

        # Find the actual model file inside the directory
        files = os.listdir(local_model_dir)
        if not files:
            raise FileNotFoundError(f"No model file found in {local_model_dir}")

        model_path = os.path.join(local_model_dir, files[0])
        return h2o.load_model(model_path)


def log_h2o_experiment(
    aml: H2OAutoML,
    *,
    train: h2o.H2OFrame,
    valid: h2o.H2OFrame,
    test: h2o.H2OFrame,
    institution_id: str,
    target_col: str,
    target_name: str,
    checkpoint_name: str,
    workspace_path: str,
    experiment_id: str,
    imputer: t.Optional[imputation.SklearnImputerWrapper] = None,
    client: t.Optional["MLflowClient"] = None,
) -> tuple[str, pd.DataFrame]:
    """
    Logs evaluation metrics, plots, and model artifacts for all models in an H2O AutoML leaderboard to MLflow.

    Args:
        aml: Trained H2OAutoML object.
        train: H2OFrame containing the training split.
        valid: H2OFrame containing the validation split.
        test: H2OFrame containing the test split.
        institution_id: Institution identifier, used to namespace the MLflow experiment.
        target_col: Column name of target (used for plotting and label extraction).
        target_name: Name of the target of the model from the config.
        checkpoint_name: Name of the checkpoint of the model from the config.
        workspace_path: Path prefix for experiment naming within MLflow.
        experiment_id: ID of experiment set during training call
        client: Optional MLflowClient instance. If not provided, one will be created.

    Returns:
        experiment_id (str): The MLflow experiment ID used for logging.
        results_df (pd.DataFrame): DataFrame with metrics and MLflow run IDs for all successfully logged models.
    """
    LOGGER.info("Logging experiment to MLflow with classification plots...")

    leaderboard_df = aml.leaderboard.as_data_frame()

    log_h2o_experiment_summary(
        aml=aml,
        leaderboard_df=leaderboard_df,
        train=train,
        target_col=target_col,
    )

    # Capping # of models that we're logging to save some time
    MAX_MODELS_TO_LOG = 50
    top_model_ids = leaderboard_df["model_id"].tolist()[:MAX_MODELS_TO_LOG]

    if not top_model_ids:
        LOGGER.warning("No models found in leaderboard.")
        return experiment_id, pd.DataFrame()

    results = []
    num_models = len(top_model_ids)

    for idx, model_id in enumerate(top_model_ids):
        # Show status update
        model_num = idx + 1

        if model_num == 1 or model_num % 10 == 0 or model_num == num_models:
            LOGGER.info(
                f"Evaluating model {model_num}/{len(top_model_ids)}: {model_id}"
            )

        # Setting threshold to 0.5 due to binary classification
        metrics = evaluation.evaluate_and_log_model(
            aml=aml,
            model_id=model_id,
            train=train,
            valid=valid,
            test=test,
            threshold=0.5,
            imputer=imputer,
            client=client,
        )

        if metrics:
            results.append(metrics)

    results_df = pd.DataFrame(results)
    LOGGER.info(f"Logged {len(results_df)} model runs to MLflow.")

    return results_df


def log_h2o_experiment_summary(
    *,
    aml: H2OAutoML,
    leaderboard_df: pd.DataFrame,
    train: h2o.H2OFrame,
    target_col: str,
    run_name: str = "h2o_automl_experiment_summary",
    sample_size: int = 1000,
) -> None:
    """
    Logs summary information about the H2O AutoML experiment to a dedicated MLflow run in
    the experiment with the leaderboard as a CSV, list of input features, training dataset,
    target distribution, and the schema (column names and types)

    Args:
        aml: Trained H2OAutoML object.
        leaderboard_df (pd.DataFrame): Leaderboard as DataFrame.
        train (H2OFrame): Training H2OFrame.
        target_col (str): Name of the target column.
        run_name (str): Name of the MLflow run. Defaults to "h2o_automl_experiment_summary".
        sample_size (int): Number of rows to sample from train data. Defaults to 1000.
    """
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name=run_name):
        # Log basic experiment metadata
        mlflow.log_metric("num_models_trained", len(leaderboard_df))
        mlflow.log_param("best_model_id", aml.leader.model_id)

        # Create tmp directory for artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            # Log leaderboard
            leaderboard_path = os.path.join(tmpdir, "h2o_leaderboard.csv")
            leaderboard_df.to_csv(leaderboard_path, index=False)
            mlflow.log_artifact(leaderboard_path, artifact_path="leaderboard")

            # Log feature list
            features = [col for col in train.columns if col != target_col]
            features_path = os.path.join(tmpdir, "train_features.txt")
            with open(features_path, "w") as f:
                for feat in features:
                    f.write(f"{feat}\n")
            mlflow.log_artifact(features_path, artifact_path="inputs")

            # Log sampled training data
            train_df = train.as_data_frame(use_pandas=True)
            train_path = os.path.join(tmpdir, "train.csv")
            train_df.to_csv(train_path, index=False)
            mlflow.log_artifact(train_path, artifact_path="inputs")

            # Log target distribution
            target_dist_df = train[target_col].table().as_data_frame()
            target_dist_path = os.path.join(tmpdir, "target_distribution.csv")
            target_dist_df.to_csv(target_dist_path, index=False)
            mlflow.log_artifact(target_dist_path, artifact_path="inputs")

            # Log schema
            schema_df = pd.DataFrame(train.types.items(), columns=["column", "dtype"])
            schema_path = os.path.join(tmpdir, "train_schema.csv")
            schema_df.to_csv(schema_path, index=False)
            mlflow.log_artifact(schema_path, artifact_path="inputs")


def set_or_create_experiment(
    workspace_path: str,
    institution_id: str,
    target_name: str,
    checkpoint_name: str,
    client: t.Optional["MlflowClient"] = None,
) -> str:
    """
    Creates or retrieves a structured MLflow experiment and sets it as the active experiment.

    Args:
        workspace_path: Base MLflow workspace path.
        institution_id: Institution or tenant identifier used for experiment naming.
        target_name: Name of the target variable.
        checkpoint_name: Name of the modeling checkpoint.
        client: MLflow client. A new one is created if not provided.

    Returns:
        MLflow experiment ID (created or retrieved).
    """
    if client is None:
        client = MlflowClient()

    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    name_parts = [institution_id, target_name, checkpoint_name, "h20_automl", timestamp]
    experiment_name = "/".join(
        [
            workspace_path.rstrip("/"),
            "h2o_automl",
            "_".join([part for part in name_parts if part]),
        ]
    )

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
