import logging
import time
import typing as t

import pandas as pd

# TODO: how do we specify databricks.automl as a package dependency?
# i.e. what's the package name??

LOGGER = logging.getLogger(__name__)


def run_automl_classification(
    df: pd.DataFrame,
    *,
    target_col: str,
    primary_metric: str,
    institution_id: str,
    job_run_id: str,
    student_id_col: str,
    **kwargs: object,
) -> object:
    """
    Wrap :func:`databricks.automl.classify()` to allow testing and ensure that
    our parameters are used properly.

    Args:
        df: Dataset containing features and target used to train classifier model,
            as well as ``student_id_col`` and any other columns
            specified in the optional ``**kwargs``
        target_col: Column name for the target to be predicted
        primary_metric: Metric used to evaluate and rank model performance;
            currently supported classification metrics include "f1", "log_loss",
            "precision", "accuracy", "roc_auc"
        institution_id: Unique identifier for the dataset's institution,
            used to name the experiment in Databricks
        job_run_id: job run ID of Databricks workflow for labeling experiment
        student_id_col: column name containing student IDs to exclude from training.
        **kwargs: keyword arguments to be passed to :func:`databricks.automl.classify()`
            If time_col is provided, AutoML tries to split the dataset into training, validation,
            and test sets chronologically, using the earliest points as training data and the latest
            points as a test set. AutoML accepts timestamps and integeters. With Databricks Runtime
            10.2 ML and above, string columns are also supported using semantic detection. However,
            we have not found AutoML to accurately support string types relevant to our data,
            so our wrapper function requires that the column type is a timestamp or integer.

    Returns:
        AutoMLSummary: an AutoML object that describes and can be used to pull
            the metrics, parameters, and other details for each of the trials.

    References:
        - https://docs.databricks.com/en/machine-learning/automl/automl-api-reference.html#classify
    """
    if kwargs.get("time_col") is not None:
        time_col = kwargs["time_col"]
        if not (
            pd.api.types.is_datetime64_any_dtype(df[time_col].dtype)
            or pd.api.types.is_integer_dtype(df[time_col].dtype)
        ):
            raise ValueError(
                f"The time column specified ({time_col}) for splitting into training, "
                "testing, and validation datasets is not a datetime or integer, "
                f"rather it's of type {df[time_col].dtype}. Please revise!"
            )

    # set some sensible default arguments
    kwargs.setdefault("pos_label", True)
    # TODO: tune this! https://app.asana.com/0/0/1206779161097924/f
    kwargs.setdefault("timeout_minutes", 5)
    exclude_cols = kwargs.pop("exclude_cols", [])
    assert isinstance(exclude_cols, list)  # type guard
    exclude_cols = exclude_cols.copy()
    if student_id_col is not None and student_id_col not in exclude_cols:
        exclude_cols.append(student_id_col)

    experiment_name = get_experiment_name(
        institution_id=institution_id,
        job_run_id=job_run_id,
        primary_metric=primary_metric,
        timeout_minutes=kwargs["timeout_minutes"],  # type: ignore
        exclude_frameworks=kwargs.get("exclude_frameworks"),  # type: ignore
    )

    from databricks import automl  # type: ignore  # importing here for mocking in tests

    LOGGER.info("running experiment: %s ...", experiment_name)
    summary = automl.classify(
        dataset=df,
        target_col=target_col,
        primary_metric=primary_metric,
        experiment_name=experiment_name,
        exclude_cols=exclude_cols,
        **kwargs,
    )
    return summary


def get_experiment_name(
    *,
    institution_id: str,
    job_run_id: str,
    primary_metric: str,
    timeout_minutes: int,
    exclude_frameworks: t.Optional[list[str]] = None,
) -> str:
    """
    Get a descriptive experiment name based on more important input parameters.

    See Also:
        - :func:`run_automl_classification()`

    References:
        - https://docs.databricks.com/en/machine-learning/automl/automl-api-reference.html#classify
    """
    name_components = [
        institution_id,
        f"{job_run_id=}",
        f"{primary_metric=}",
        f"{timeout_minutes=}",
    ]
    if exclude_frameworks:
        name_components.append(f"exclude_frameworks={','.join(exclude_frameworks)}")
    name_components.append(time.strftime("%Y-%m-%dT%H:%M:%S"))

    name = "__".join(name_components)
    if len(name) > 500:
        LOGGER.warning("truncating long experiment name '%s' to first 500 chars", name)
        name = name[:500]
    return name
