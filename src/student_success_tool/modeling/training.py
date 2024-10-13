import time

import pandas as pd


def run_automl_classification(
    institution_id: str,
    job_run_id: str,
    train_df: pd.DataFrame,
    outcome_col: str,
    optimization_metric: str,
    student_id_col: str,
    **kwargs: object,
) -> object:
    """
    Wrap around databricks.automl.classify to allow testing and ensure that
    our parameters are used properly.

    Args:
        institution_id: institution ID for labeling experiment
        job_run_id: job run ID of Databricks workflow for labeling experiment
        train_df: data containing features and outcome to model, as well as the student_id_col and any other columns
            specified in the optional **kwargs
        outcome_col: column name for the target to predict
        optimization_metric: Metric used to evaluate and rank model performance.
            Supported metrics for classification: “f1” (default), “log_loss”,
            “precision”, “accuracy”, “roc_auc”
        student_id_col: column name containing student IDs to exclude from training.
        **kwargs: keyword arguments to be passed to databricks.automl.classify(). For more information on the
            available optional arguments, see the API documentation here: https://docs.databricks.com/en/machine-learning/automl/automl-api-reference.html#classify.
            - If time_col is provided, AutoML tries to split the dataset into training, validation,
                and test sets chronologically, using the earliest points as training data and the latest
                points as a test set. AutoML accepts timestamps and integeters. With Databricks Runtime
                10.2 ML and above, string columns are also supported using semanting detection. However,
                we have not found AutoML to accurately support string types relevant to our data, so our wrapper function requires that the column type is a timestamp or integer.

    Returns:
        AutoMLSummary: an AutoML object that describes and can be used to pull
            the metrics, parameters, and other details for each of the trials.
    """
    if (time_col := kwargs.get("time_col")) is not None:
        assert pd.api.types.is_datetime64_any_dtype(
            train_df[time_col].dtype
        ) or pd.api.types.is_integer_dtype(train_df[time_col].dtype), (
            f"The time column specified ({time_col}) for splitting into training, "
            + "testing, and validation datasets is not a datetime or integer, but rather of type "
            + train_df[time_col].dtype
            + ". Please revise!"
        )

    experiment_name = "_".join(
        [
            institution_id,
            outcome_col,
            str(job_run_id),
            optimization_metric,
        ]
    )
    for key, val in kwargs.items():
        if key != "exclude_cols":
            experiment_name += "_" + key + str(val)
    experiment_name += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")

    # default arguments for SST
    if not kwargs.get("pos_label"):
        kwargs["pos_label"] = True
    if not kwargs.get("timeout_minutes"):
        kwargs["timeout_minutes"] = (
            5  # TODO: tune this! https://app.asana.com/0/0/1206779161097924/f
        )
    kwargs["exclude_cols"] = kwargs.get("exclude_cols", [])
    if student_id_col is not None:
        kwargs["exclude_cols"].append(student_id_col)

    # TODO: need to install this to poetry environment
    from databricks import automl  # importing here for mocking in tests

    print(f"Running experiment {experiment_name}")
    summary = automl.classify(
        experiment_name=experiment_name,
        dataset=train_df,
        target_col=outcome_col,
        primary_metric=optimization_metric,
        **kwargs,
    )

    return summary
