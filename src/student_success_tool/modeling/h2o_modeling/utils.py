import yaml
import logging
import typing as t

import os
import datetime
import tempfile
import contextlib
import random

import mlflow
from mlflow.models import Model, infer_signature
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts
import pandas as pd
import numpy as np
from pandas.api.types import is_categorical_dtype, is_object_dtype, is_string_dtype


import h2o
from h2o.automl import H2OAutoML
from h2o.model.model_base import ModelBase
from h2o.frame import H2OFrame
from h2o.two_dim_table import H2OTwoDimTable

from sklearn.metrics import confusion_matrix

from . import evaluation
from . import imputation

LOGGER = logging.getLogger(__name__)


def safe_h2o_init(base_port: int = 54321, mem_per_cluster: str = "4G") -> None:
    """
    Initialize a unique H2O cluster per Databricks task (or randomly if no task id).
    Ensures isolation across parallel runs and caps memory usage. This also works in
    a databricks workflow or interactively in a notebook.
    """

    task_id = os.environ.get("DATABRICKS_TASK_RUN_ID")
    if task_id:
        port = base_port + (int(task_id) % 10000)
    else:
        port = base_port + random.randint(0, 1000)

    LOGGER.info(f"Starting H2O cluster at port {port}...")
    h2o.init(port=port, nthreads=-1, max_mem_size=mem_per_cluster)


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
    Initializes H2O, downloads the UC-compatible H2O model artifact from MLflow, and loads it.
    """
    if not h2o.connection():
        h2o.init()

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_model_dir = download_artifacts(
            run_id=run_id, artifact_path=artifact_path, dst_path=tmp_dir
        )

        model_file = os.path.join(local_model_dir, "model.h2o")
        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"Expected model.h2o not found in {local_model_dir}"
            )

        return h2o.load_model(model_file)


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

    leaderboard_df = _to_pandas(aml.leaderboard)

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
            aml=aml,
            model_id=model_id,
            train=train,
            valid=valid,
            test=test,
            imputer=imputer,
            target_col=target_col,
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
            train_df = _to_pandas(train)
            valid_df = _to_pandas(valid)
            test_df = _to_pandas(test)
            full_df = pd.concat([train_df, valid_df, test_df], axis=0)
            df_parquet_path = os.path.join(tmpdir, "full_dataset.parquet")
            full_df.to_parquet(df_parquet_path, index=False)
            mlflow.log_artifact(df_parquet_path, artifact_path="inputs")

            # Log target distribution
            target_dist_df = _to_pandas(train[target_col].table())
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
    aml: H2OAutoML,
    model_id: str,
    train: h2o.H2OFrame,
    valid: h2o.H2OFrame,
    test: h2o.H2OFrame,
    threshold: float = 0.5,
    target_col: str = "target",
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

                # Assign initial sort key for mlflow UI
                primary_metric_key = f"validate_{primary_metric}"
                mlflow.set_tag("mlflow.primaryMetric", primary_metric_key)

                # Create & log model comparisons plot
                evaluation.create_and_log_h2o_model_comparison(aml=aml)

                # Log Classification Plots
                for split_name, frame in zip(
                    ["train", "val", "test"], [train, valid, test]
                ):
                    y_true = _to_pandas(frame[target_col]).values.flatten()
                    preds = model.predict(frame)
                    positive_class_label = preds.col_names[-1]
                    y_proba = _to_pandas(preds[positive_class_label]).values.flatten()
                    y_pred = (y_proba >= threshold).astype(int)

                    # Log Confusion matrix metrics for FE tables
                    label = "validate" if split_name == "val" else split_name
                    tn, fp, fn, tp = confusion_matrix(
                        y_true, y_pred, labels=[0, 1]
                    ).ravel()

                    metrics.update(
                        {
                            f"{label}_true_positives": float(tp),
                            f"{label}_true_negatives": float(tn),
                            f"{label}_false_positives": float(fp),
                            f"{label}_false_negatives": float(fn),
                        }
                    )

                    evaluation.generate_all_classification_plots(
                        y_true, y_pred, y_proba, prefix=split_name
                    )

                log_model_metadata_to_mlflow(
                    model_id=model_id,
                    model=model,
                    metrics=metrics,
                    exclude_keys={"model_id"},
                )

                X_sample = _to_pandas(train.drop(target_col, axis=1))
                y_pred_sample = model.predict(train).as_data_frame()
                signature = infer_signature(X_sample, y_pred_sample)

                log_h2o_model_metadata_for_uc(
                    h2o_model=model,
                    artifact_path="model",
                    signature=signature,
                )

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


def log_h2o_model_metadata_for_uc(
    h2o_model: ModelBase,
    artifact_path: str,
    signature: mlflow.models.signature.ModelSignature,
) -> None:
    """
    Custom H2O model logger (Unity Catalog-compatible & future-proof for MLflow 3.x).
    Mlflow 3.x will deprecate mlflow.h2o.log_model.
    Saves the H2O model + MLmodel metadata so Unity Catalog can register it.

    Args:
        h2o_model: Trained H2O model to log.
        artifact_path: Subdir in MLflow run artifacts (e.g. "model").
        signature: Optional MLflow signature object (mlflow.models.signature.ModelSignature).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Save raw H2O model
        model_path = h2o.save_model(h2o_model, path=tmpdir, force=True)

        # Normalize filename to "model.h2o"
        final_model_path = os.path.join(tmpdir, "model.h2o")
        if model_path != final_model_path:
            os.rename(model_path, final_model_path)

        # 2. Build MLmodel metadata
        mlmodel = Model(artifact_path=artifact_path, flavors={})
        mlmodel.add_flavor(
            "h2o",
            h2o_version=h2o.__version__,
            model_data="model.h2o",
        )
        if signature is not None:
            mlmodel.signature = signature

        mlmodel.save(os.path.join(tmpdir, "MLmodel"))

        # 3. Minimal environment specs
        reqs_path = os.path.join(tmpdir, "requirements.txt")
        with open(reqs_path, "w") as f:
            f.write(f"h2o=={h2o.__version__}\n")

        conda_env = {
            "name": "h2o_env",
            "channels": ["defaults", "conda-forge"],
            "dependencies": [
                f"h2o={h2o.__version__}",
                "pip",
                {"pip": [f"mlflow=={mlflow.__version__}"]},
            ],
        }
        with open(os.path.join(tmpdir, "conda.yaml"), "w") as f:
            yaml.safe_dump(conda_env, f)

        # 4. Log directory to MLflow artifacts
        mlflow.log_artifacts(tmpdir, artifact_path=artifact_path)


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

    name_parts = [institution_id, target_name, checkpoint_name, "h2o_automl", timestamp]
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


def correct_h2o_dtypes(
    h2o_df: h2o.H2OFrame,
    original_df: pd.DataFrame,
    force_enum_cols: t.Optional[t.List[str]] = None,
) -> h2o.H2OFrame:
    """
    Correct H2OFrame dtypes based on original pandas DataFrame, ensuring columns
    inferred as numeric in H2O are restored to categorical/enums if they were
    non-numeric in pandas.

    Args:
        h2o_df: H2OFrame created from original_df
        original_df: Original pandas DataFrame with dtype info
        force_enum_cols: Optional list of column names to forcibly convert to enum

    Returns:
        h2o_df (possibly modified)
    """
    force_enum_cols = set(force_enum_cols or [])
    converted_columns = []

    LOGGER.info("Starting H2O dtype correction.")

    for col in original_df.columns:
        if col not in h2o_df.columns:
            LOGGER.debug(f"Skipping '{col}': not found in H2OFrame.")
            continue

        orig_dtype = original_df[col].dtype
        h2o_type = h2o_df.types.get(col)
        is_non_numeric = (
            is_categorical_dtype(original_df[col])
            or is_object_dtype(original_df[col])
            or is_string_dtype(original_df[col])
        )
        h2o_is_numeric = h2o_type in ("int", "real")

        should_force = col in force_enum_cols and h2o_type not in ("enum",)
        needs_correction = (is_non_numeric and h2o_is_numeric) or should_force

        LOGGER.debug(
            f"Column '{col}': orig_dtype={orig_dtype}, h2o_dtype={h2o_type}, "
            f"non_numeric={is_non_numeric}, force={should_force}"
        )

        if needs_correction:
            try:
                h2o_df[col] = h2o_df[col].asfactor()
                converted_columns.append(col)
                LOGGER.info(
                    f"Converted '{col}' to enum "
                    f"(originally {orig_dtype}, inferred as {h2o_type})."
                )
            except Exception as e:
                LOGGER.warning(f"Failed to convert '{col}' to enum: {e}")

    LOGGER.info(
        f"H2O dtype correction complete. {len(converted_columns)} column(s) affected: {converted_columns}"
    )
    return h2o_df


def _to_h2o(
    pobj: t.Any, force_enum_cols: t.Optional[t.List[str]] = None
) -> h2o.H2OFrame:
    """Convert common Python objects to an H2OFrame.

    This function wraps multiple input types into an H2OFrame and applies
    `correct_h2o_dtypes` so that categorical columns from pandas are preserved
    as enums in H2O.

    Args:
        pobj (Any):
            The object to convert. Supported types:
              - `pandas.DataFrame`: Converted directly to H2OFrame.
              - `pandas.Series`: Converted to single-column H2OFrame.
              - `numpy.ndarray`: Converted to H2OFrame via a pandas.DataFrame wrapper.
              - `h2o.H2OFrame`: Returned as-is.
        force_enum_cols (Optional[List[str]]):
            Optional list of column names to force conversion to enum
            regardless of dtype.

    Returns:
        h2o.H2OFrame:
            The converted H2OFrame with corrected dtypes.

    Raises:
        TypeError: If the input type is unsupported or `None`.
    """
    if pobj is None:
        raise TypeError("_to_h2o: cannot convert None")

    # Already H2OFrame
    if H2OFrame is not None and isinstance(pobj, H2OFrame):
        return pobj

    # Pandas DataFrame
    if isinstance(pobj, pd.DataFrame):
        hf = h2o.H2OFrame(pobj)
        return correct_h2o_dtypes(hf, pobj, force_enum_cols=force_enum_cols)

    # Pandas Series
    if isinstance(pobj, pd.Series):
        df = pobj.to_frame()
        hf = h2o.H2OFrame(df)
        return correct_h2o_dtypes(hf, df, force_enum_cols=force_enum_cols)

    # Numpy array
    if isinstance(pobj, np.ndarray):
        if pobj.ndim == 1:
            df = pd.DataFrame({0: pobj})
        else:
            df = pd.DataFrame(pobj)
        hf = h2o.H2OFrame(df)
        return correct_h2o_dtypes(hf, df, force_enum_cols=force_enum_cols)

    raise TypeError(f"_to_h2o: unsupported object type {type(pobj)}")


def _to_pandas(hobj: t.Any) -> pd.DataFrame:
    """
    Convert common H2O objects to pandas.DataFrame.

    - H2OFrame.as_data_frame() supports `use_pandas` and `use_multi_thread` (for performance).
    - H2OTwoDimTable.as_data_frame() takes no arguments in H2O 3.46+.
    - For other objects, we'll use `as_data_frame()`.
    """
    # Case 1: Big data — use multithreaded pull for H2OFrame
    if H2OFrame is not None and isinstance(hobj, H2OFrame):
        try:
            return hobj.as_data_frame(use_pandas=True, use_multi_thread=True)
        except TypeError:
            # Very old H2O without use_multi_thread
            return hobj.as_data_frame(use_pandas=True)

    # Case 2: Metric tables such as H2OTwoDimTable doesn't support multi-thread
    if H2OTwoDimTable is not None and isinstance(hobj, H2OTwoDimTable):
        return hobj.as_data_frame()

    # Case 3: Fallback for any other hobj that supports as_dataframe
    if hasattr(hobj, "as_data_frame"):
        try:
            return hobj.as_data_frame()
        except TypeError:
            # Last-resort fallback for legacy signatures
            return hobj.as_data_frame(use_pandas=True)

    raise TypeError(f"_to_pandas: unsupported object type {type(hobj)}")
