import logging
import typing as t

import os
import datetime
import tempfile
import contextlib

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts
import pandas as pd

import h2o
from h2o.automl import H2OAutoML
from h2o.model.model_base import ModelBase

from . import evaluation
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
    target_col: str,
    experiment_id: str,
    imputer: t.Optional[imputation.SklearnImputerWrapper] = None,
) -> pd.DataFrame:
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
        results_df (pd.DataFrame): DataFrame with metrics and MLflow run IDs for all successfully logged models.
    """
    LOGGER.info("Logging experiment to MLflow with classification plots...")

    leaderboard_df = aml.leaderboard.as_data_frame()

    log_h2o_experiment_summary(
        aml=aml,
        leaderboard_df=leaderboard_df,
        train=train,
        valid=valid,
        test=test,
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
                f"Completed logging on {model_num}/{len(top_model_ids)} top models..."
            )

        # Setting threshold to 0.5 due to binary classification
        metrics = log_h2o_model(
            model_id=model_id,
            train=train,
            valid=valid,
            test=test,
            threshold=0.5,
            imputer=imputer,
            primary_metric=aml.sort_metric,
        )

        if metrics:
            results.append(metrics)

    results_df = pd.DataFrame(results)
    LOGGER.info(f"Finished logging on {len(results_df)} top model runs to MLflow.")

    return results_df


def log_h2o_experiment_summary(
    *,
    aml: H2OAutoML,
    leaderboard_df: pd.DataFrame,
    train: h2o.H2OFrame,
    valid: h2o.H2OFrame,
    test: h2o.H2OFrame,
    target_col: str,
    run_name: str = "H2O AutoML Experiment Summary and Storage",
) -> None:
    """
    Logs summary information about the H2O AutoML experiment to a dedicated MLflow run in
    the experiment with the leaderboard as a CSV, list of input features, training dataset
    (with splits e.g. "train", "test", "val"), target distribution, and the
    schema (column names and types).

    Args:
        aml: Trained H2OAutoML object.
        leaderboard_df (pd.DataFrame): Leaderboard as DataFrame.
        train (H2OFrame): Training H2OFrame.
        valid (H2OFrame): Validation H2OFrame.
        test (H2OFrame): Test H2OFrame.
        target_col (str): Name of the target column.
        run_name (str): Name of the MLflow run. Defaults to "h2o_automl_experiment_summary".
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
            valid_df = valid.as_data_frame(use_pandas=True)
            test_df = test.as_data_frame(use_pandas=True)
            full_df = pd.concat([train_df, valid_df, test_df], axis=0)
            df_parquet_path = os.path.join(tmpdir, "full_dataset.parquet")
            full_df.to_parquet(df_parquet_path, index=False)
            mlflow.log_artifact(df_parquet_path, artifact_path="inputs")

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


def log_h2o_model(
    *,
    model_id: str,
    train: h2o.H2OFrame,
    valid: h2o.H2OFrame,
    test: h2o.H2OFrame,
    threshold: float = 0.5,
    imputer: t.Optional[imputation.SklearnImputerWrapper] = None,
    primary_metric: str = "logloss",
) -> dict | None:
    """
    Evaluates a single H2O model and logs metrics, plots, and artifacts to MLflow.

    Args:
        model_id: The H2O model ID to evaluate.
        train: H2OFrame for training.
        valid: H2OFrame for validation.
        test: H2OFrame for testing.
        threshold: Classification threshold for binary metrics.
        imputer: Optional SklearnImputerWrapper used in preprocessing.
        artifact_path: MLflow artifact path for saving imputer files.

    Returns:
        dict of metrics with `mlflow_run_id`, or None on failure.
    """
    try:
        model = h2o.get_model(model_id)
        with (
            open(os.devnull, "w") as fnull,
            contextlib.redirect_stdout(fnull),
            contextlib.redirect_stderr(fnull),
        ):
            metrics = evaluation.get_metrics_near_threshold_all_splits(
                model, train, valid, test, threshold=threshold
            )

            if mlflow.active_run():
                mlflow.end_run()

            with mlflow.start_run():
                active_run = mlflow.active_run()
                if active_run is not None:  # type check
                    run_id = active_run.info.run_id

                log_model_metadata_to_mlflow(
                    model_id=model_id,
                    model=model,
                    metrics=metrics,
                    exclude_keys={"model_id"},
                )

                # Assign initial sort key for mlflow UI
                primary_metric_key = f"validate_{primary_metric}"
                mlflow.set_tag("mlflow.primaryMetric", primary_metric_key)

                # Log Classification Plots
                for split_name, frame in zip(
                    ["train", "val", "test"], [train, valid, test]
                ):
                    y_true = frame["target"].as_data_frame().values.flatten()
                    preds = model.predict(frame)
                    positive_class_label = preds.col_names[-1]
                    y_proba = (
                        preds[positive_class_label].as_data_frame().values.flatten()
                    )
                    y_pred = (y_proba >= threshold).astype(int)

                    evaluation.generate_all_classification_plots(
                        y_true, y_pred, y_proba, prefix=split_name
                    )

                # Log H2O Model
                local_model_dir = f"/tmp/h2o_models/{model_id}"
                os.makedirs(local_model_dir, exist_ok=True)
                h2o.save_model(model, path=local_model_dir, force=True)
                mlflow.log_artifacts(local_model_dir, artifact_path="model")

                # Log Imputer Artifacts
                if imputer is not None:
                    try:
                        imputer.log_pipeline(artifact_path="sklearn_imputer")
                    except Exception as e:
                        LOGGER.warning(f"Failed to log imputer artifacts: {e}")

            metrics["mlflow_run_id"] = run_id
            return metrics

    except Exception as e:
        LOGGER.exception(f"Failed to evaluate and log model {model_id}: {e}")
        return None


def log_model_metadata_to_mlflow(
    model_id: str,
    model: ModelBase,
    metrics: dict[str, t.Any],
    exclude_keys: t.Optional[set[str]] = None,
) -> None:
    """
    Logs model ID, hyperparameters, and metrics to MLflow.

    Args:
        model_id: ID string of the H2O model.
        model: H2O model object.
        metrics: Dictionary of metrics to log.
        exclude_keys: Optional set of metric keys to exclude from logging.
    """
    exclude_keys = exclude_keys or set()

    # Log model ID
    mlflow.log_param("model_id", model_id)

    # Log hyperparameters
    try:
        hyperparams = {
            k: str(v)
            for k, v in model._parms.items()
            if (
                v is not None
                and k != "model_id"
                and not isinstance(v, (h2o.H2OFrame, list, dict))
            )
        }
        if hyperparams:
            mlflow.log_params(hyperparams)
    except Exception as e:
        LOGGER.warning(f"Failed to log hyperparameters for model {model_id}: {e}")

    # Log metrics
    for k, v in metrics.items():
        if k in exclude_keys:
            continue
        try:
            if isinstance(v, (float, int)):
                mlflow.log_metric(k, float(v))
            elif isinstance(v, str):
                mlflow.log_metric(k, float(v))  # Best-effort conversion
            else:
                LOGGER.warning(
                    f"Skipping metric '{k}': unsupported type {type(v).__name__}"
                )
        except (ValueError, TypeError) as e:
            LOGGER.warning(f"Could not log metric '{k}' with value '{v}': {e}")


def set_or_create_experiment(
    workspace_path: str,
    institution_id: str,
    target_name: str,
    checkpoint_name: str,
    client: t.Optional[MlflowClient] = None,
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
