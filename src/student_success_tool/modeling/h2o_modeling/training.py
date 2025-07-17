import logging
import mlflow
import pandas as pd

from . import evaluation

import h2o
from h2o.automl import H2OAutoML

LOGGER = logging.getLogger(__name__)


def run_h2o_automl_classification(
    df: pd.DataFrame,
    *,
    target_col: str,
    primary_metric: str,
    institution_id: str,
    job_run_id: str,
    student_id_col: str,
    **kwargs: object,
) -> H2OAutoML:
    """
    Runs H2O AutoML for classification tasks and logs the best model to MLflow.

    Args:
        df: Input Pandas DataFrame with features and target.
        target_col: Name of the target column.
        primary_metric: Used to sort models; supports "logloss", "AUC", "AUCPR", etc.
        institution_id: Institution ID for experiment naming.
        job_run_id: Job run ID to track lineage.
        student_id_col: Column name identifying students, excluded from training.
        **kwargs: Optional settings including timeout_minutes, max_models, etc.

    Returns:
        Trained H2OAutoML object.
    """

    # Validate input types
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not isinstance(target_col, str):
        raise TypeError("target_col must be a string.")
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in input DataFrame.")

    VALID_H2O_METRICS = {"auc", "logloss", "mean_per_class_error", "rmse", "mae", "aucpr"}
    primary_metric = primary_metric.lower()
    if primary_metric not in VALID_H2O_METRICS:
        raise ValueError(f"Invalid primary_metric '{primary_metric}'. Must be one of {VALID_H2O_METRICS}")

    # Set defaults and pop kwargs
    seed = kwargs.pop("seed", 42)
    timeout_minutes = kwargs.pop("timeout_minutes", 5)
    max_models = kwargs.pop("max_models", 100)
    exclude_cols = kwargs.pop("exclude_cols", [])
    split_col = kwargs.pop("split_col", "split")

    if student_id_col and student_id_col not in exclude_cols:
        exclude_cols.append(student_id_col)

    # Convert to H2OFrame and correct types
    # NOTE: H2O sometimes doesn't infer types correctly, so we need to manually check them here using our pandas DF.
    h2o_df = h2o.H2OFrame(df)
    h2o_df = correct_h2o_dtypes(h2o_df, df)

    if split_col not in h2o_df.columns:
        raise ValueError("Input data must contain a 'split' column with values ['train', 'validate', 'test'].")

    h2o_df[target_col] = h2o_df[target_col].asfactor()
    train = h2o_df[h2o_df[split_col] == "train"]
    valid = h2o_df[h2o_df[split_col] == "validate"]
    test = h2o_df[h2o_df[split_col] == "test"]

    features = [col for col in df.columns if col not in exclude_cols + [target_col]]

    LOGGER.info(f"Running H2O AutoML for target '{target_col}' with {len(features)} features...")

    aml = H2OAutoML(
        max_runtime_secs=timeout_minutes * 60,
        max_models=max_models,
        sort_metric=primary_metric,
        seed=seed,
        verbosity="info",
        include_algos=["XGBoost", "GBM", "GLM"],
    )
    aml.train(x=features, y=target_col, training_frame=train, validation_frame=valid, leaderboard_frame=test)

    LOGGER.info(f"Best model: {aml.leader.model_id}")

    return aml


def log_h2o_experiment(
    aml,
    train,
    valid,
    test,
    threshold=0.5,
    prefix="h2o_automl",
    use_timestamp=True,
    institution_id=None,
    target_name=None,
    checkpoint_name=None,
    workspace_path=None,
    client=None,
):
    if not isinstance(threshold, float) or not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be a float in [0.0, 1.0], got {threshold}")

    LOGGER.info("Logging experiment to MLflow with classification plots...")

    if client is None:
        client = MlflowClient()

    experiment_id = set_or_create_experiment(
        workspace_path,
        institution_id,
        target_name,
        checkpoint_name,
        prefix=prefix,
        use_timestamp=use_timestamp,
        client=client,
    )

    results = []
    leaderboard_df = aml.leaderboard.as_data_frame()
    top_model_ids = leaderboard_df["model_id"].tolist()

    if not top_model_ids:
        LOGGER.warning("No models found in leaderboard.")
        return experiment_id, pd.DataFrame()

    for model_id in top_model_ids:
        try:
            model = h2o.get_model(model_id)
            LOGGER.info(f"Evaluating model {model_id}...")

            metrics = evaluation.get_metrics_near_threshold_all_splits(model, train, valid, test, threshold=threshold)

            with mlflow.start_run(run_name=f"h2o_eval_{model_id}"):
                run_id = mlflow.active_run().info.run_id

                # Log metrics
                for k, v in metrics.items():
                    if k != "model_id":
                        mlflow.log_metric(k, v)

                # Generate & log classification plots
                for split_name, frame in zip(["train", "val", "test"], [train, valid, test]):
                    y_true = frame[target_name].as_data_frame().values.flatten()
                    y_proba = model.predict(frame)["p1"].as_data_frame().values.flatten()
                    y_pred = (y_proba >= threshold).astype(int)

                    generate_all_classification_plots(y_true, y_pred, y_proba, prefix=split_name)

                # Save model
                local_model_dir = f"/tmp/h2o_models/{model_id}"
                os.makedirs(local_model_dir, exist_ok=True)
                h2o.save_model(model, path=local_model_dir, force=True)
                mlflow.log_artifacts(local_model_dir, artifact_path="model")

            metrics["mlflow_run_id"] = run_id
            results.append(metrics)

        except Exception as e:
            LOGGER.exception(f"Failed to log model {model_id}: {e}")

    results_df = pd.DataFrame(results)
    LOGGER.info(f"Logged {len(results_df)} model runs to MLflow.")

    return experiment_id, results_df


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


def correct_h2o_dtypes(h2o_df, original_df):
    """
    Ensure that any columns that were categorical in original_df
    remain as enum in h2o_df, even if H2O inferred them as int or real.
    
    Args:
        h2o_df: H2OFrame created from original_df
        original_df: Original pandas DataFrame with dtype info
    
    Returns:
        h2o_df with corrected enum columns
    """
    for col in original_df.columns:
        if col not in h2o_df.columns:
            continue

        is_cat = pd.api.types.is_categorical_dtype(original_df[col]) or original_df[col].dtype == object

        h2o_type = h2o_df.types.get(col)
        if is_cat and h2o_type not in ("enum", "string"):
            # Convert to enum only if it was inferred as numeric
            h2o_df[col] = h2o_df[col].asfactor()
    
    return h2o_df
