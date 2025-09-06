import logging
import typing as t

from mlflow.tracking import MlflowClient
import pandas as pd

import h2o
from h2o.automl import H2OAutoML

from . import utils
from . import imputation

LOGGER = logging.getLogger(__name__)

VALID_H2O_METRICS = {
    "auc",
    "logloss",
    "mean_per_class_error",
    "rmse",
    "mae",
    "aucpr",
}

VALID_H2O_FRAMEWORKS = {
    "XGBoost",
    "GBM",
    "GLM",
    "DRF",
}


def run_h2o_automl_classification(
    df: pd.DataFrame,
    *,
    target_col: str,
    primary_metric: str,
    institution_id: str,
    student_id_col: str,
    client: t.Optional[MlflowClient] = None,
    **kwargs: object,
) -> tuple[str, H2OAutoML, h2o.H2OFrame, h2o.H2OFrame, h2o.H2OFrame]:
    if client is None:
        client = MlflowClient()

    # Set and validate inputs
    seed = kwargs.pop("seed", 42)
    timeout_minutes = int(float(str(kwargs.pop("timeout_minutes", 5))))
    split_col: str = str(kwargs.pop("split_col", "split"))
    sample_weight_col = str(kwargs.pop("sample_weight_col", "sample_weight"))
    target_name = kwargs.pop("target_name", None)
    checkpoint_name = kwargs.pop("checkpoint_name", None)
    workspace_path = kwargs.pop("workspace_path", None)
    metric = primary_metric.lower()

    exclude_cols = t.cast(list[str], kwargs.pop("exclude_cols", []) or [])
    exclude_cols = [c for c in exclude_cols if c is not None]

    exclude_frameworks = t.cast(list[str], kwargs.pop("exclude_frameworks", []) or [])
    exclude_frameworks = [c for c in exclude_frameworks if c is not None]

    if not all([target_name, checkpoint_name, workspace_path]):
        raise ValueError(
            "Missing logging parameters: target_name, checkpoint_name, workspace_path"
        )
    if target_col not in df or split_col not in df:
        raise ValueError("Missing target_col or split column in DataFrame.")
    if metric not in VALID_H2O_METRICS:
        raise ValueError(
            f"Invalid metric '{metric}', must be one of {VALID_H2O_METRICS}."
        )

    # Ensure columns that need to be excluded are from training & imputation
    if (
        student_id_col
        and student_id_col in df.columns
        and student_id_col not in exclude_cols
    ):
        exclude_cols.append(student_id_col)

    # Always exclude target & split; sample_weight only if it exists
    must_exclude: set[str] = {target_col, split_col}
    if sample_weight_col and sample_weight_col in df.columns:
        must_exclude.add(sample_weight_col)

    for c in must_exclude:
        if c not in exclude_cols:
            exclude_cols.append(c)

    # Only error on missing user-provided excludes; ignore optional system-added ones if absent
    missing_user_cols = [c for c in exclude_cols if c not in df.columns]
    if missing_user_cols:
        raise ValueError(f"exclude_cols contains missing columns: {missing_user_cols}")

    # Set frameworks for training
    frameworks = [fw for fw in VALID_H2O_FRAMEWORKS if fw not in exclude_frameworks]
    if not frameworks:
        raise ValueError(
            "All frameworks were excluded; must allow at least one of "
            f"{', '.join(VALID_H2O_FRAMEWORKS)}"
        )

    # Set training experiment
    experiment_id = utils.set_or_create_experiment(
        workspace_path=str(workspace_path),
        institution_id=institution_id,
        target_name=str(target_name),
        checkpoint_name=str(checkpoint_name),
        client=client,
    )

    # Fit and apply sklearn imputation
    LOGGER.info("Running sklearn-based imputation on feature columns only...")
    imputer = imputation.SklearnImputerWrapper()
    raw_model_features = [c for c in df.columns if c not in exclude_cols]
    imputer.fit(df.loc[df[split_col] == "train", raw_model_features])

    df_splits: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "validate", "test"):
        df_split = df[df[split_col] == split_name]
        df_split_processed = imputer.transform(df_split)
        LOGGER.info(
            "Processed '%s' split -> shape: %s (kept %d passthrough cols)",
            split_name,
            df_split_processed.shape,
            len(
                [
                    c
                    for c in df_split_processed.columns
                    if imputer.output_feature_names is not None
                    and c not in imputer.output_feature_names
                ]
            ),
        )
        df_splits[split_name] = df_split_processed

    # Convert to H2OFrames and fix dtypes
    h2o_splits: dict[str, h2o.H2OFrame] = {}
    for k, v in df_splits.items():
        missing_flags = [c for c in v.columns if c.endswith("_missing_flag")]
        hf = utils._to_h2o(v, force_enum_cols=missing_flags)
        hf[target_col] = hf[target_col].asfactor()
        h2o_splits[k] = hf

    train, valid, test = h2o_splits["train"], h2o_splits["validate"], h2o_splits["test"]

    # Run H2O AutoML
    processed_model_features = [c for c in train.columns if c not in exclude_cols]
    LOGGER.info(f"Running H2O AutoML with {len(processed_model_features)} features...")

    # Create stopping criteria based on dataset size
    n_rows = int(train.nrows)
    if n_rows < 3000:
        nfolds = 10
        stopping_tolerance = 3e-4
        stopping_rounds = 10
    else:
        nfolds = 5
        stopping_tolerance = 1e-4
        stopping_rounds = 5

    LOGGER.info(
        "H2O AutoML config -> training_rows=%d, nfolds=%d, "
        "metric=%s, stopping_tolerance=%.1e, stopping_rounds=%d, include_algos=%s",
        n_rows,
        nfolds,
        metric,
        stopping_tolerance,
        stopping_rounds,
        ",".join(frameworks),
    )

    aml = H2OAutoML(
        max_runtime_secs=timeout_minutes * 60,
        sort_metric=metric,
        stopping_metric=metric,
        seed=seed,
        verbosity="info",
        include_algos=frameworks,
        nfolds=nfolds,
        stopping_tolerance=stopping_tolerance,
        stopping_rounds=stopping_rounds,
    )

    # Only pass weights_column if it exists in the data
    train_kwargs = dict(
        x=processed_model_features,
        y=target_col,
        training_frame=train,
        validation_frame=valid,
        leaderboard_frame=valid,
    )
    if sample_weight_col in df.columns:
        train_kwargs["weights_column"] = sample_weight_col

    aml.train(**train_kwargs)

    LOGGER.info(f"Best model: {aml.leader.model_id}")

    utils.log_h2o_experiment(
        aml=aml,
        train=train,
        valid=valid,
        test=test,
        target_col=target_col,
        experiment_id=experiment_id,
        imputer=imputer,
    )

    return experiment_id, aml, train, valid, test
