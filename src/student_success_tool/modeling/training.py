import logging
import time

import pandas as pd

# TODO: how do we specify databricks.automl as a package dependency?
# i.e. what's the package name??

LOGGER = logging.getLogger(__name__)


def run_automl_classification(
    df: pd.DataFrame,
    *,
    target_col: str,
    optimization_metric: str,
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
        optimization_metric: Metric used to evaluate and rank model performance;
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
    if student_id_col is not None:
        exclude_cols.append(student_id_col)

    # generate a very descriptive experiment name
    experiment_name_components = [
        institution_id,
        target_col,
        str(job_run_id),
        optimization_metric,
    ]
    experiment_name_components.extend(f"{key}={val}" for key, val in kwargs.items())
    experiment_name_components.append(time.strftime("%Y-%m-%dT%H:%M:%S"))
    experiment_name = "_".join(experiment_name_components)

    from databricks import automl  # importing here for mocking in tests

    LOGGER.info("running experiment: %s ...", experiment_name)
    summary = automl.classify(
        dataset=df,
        target_col=target_col,
        primary_metric=optimization_metric,
        experiment_name=experiment_name,
        exclude_cols=exclude_cols,
        **kwargs,
    )
    return summary
