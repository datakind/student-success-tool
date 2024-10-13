import os
import shutil
import time
import uuid

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.outliers_influence import variance_inflation_factor

palette = sns.color_palette(
    "Paired"
)  # TODO: eventually we should use the custom_style.mplstyle colors, but currently the color palette does not have distinct enough colors for calibration by group where there are a lot of subgroups


def drop_incomplete_features(features, threshold):
    """Drop columns from dataframe that have at least some fraction of nulls.

    Args:
        features (pd.DataFrame): dataframe of columns to assess nulls across
        threshold (float): fraction of nulls deemed unacceptable. Any columns in
            the features dataframe with this fraction of nulls or more will be dropped.

    Returns:
        pd.DataFrame: features data without the incomplete columns
    """
    pct_null = features.isna().sum() / features.shape[0]
    incomplete_features = pct_null[pct_null >= threshold].index
    if len(incomplete_features) > 0:
        print(f"Dropping incomplete features: {incomplete_features}")
        return features.drop(columns=incomplete_features)
    else:
        print(f"No features with at least {threshold} proportion of nulls.")
        return features


def drop_low_variance_features(features, threshold):
    """Drop columns with low or no variance, according to a threshold.

    Args:
        features (pd.DataFrame): dataframe of columns to assess variance
        threshold (float): Features with a training-set variance lower than this threshold will be removed.

    Returns:
        pd.DataFrame: features data without the low variance columns
    """
    selector = VarianceThreshold(threshold=threshold)
    numeric_features = features.select_dtypes(include="number")

    column_contains_inf = np.isinf(numeric_features).any()
    if column_contains_inf.any():
        print("The following columns contain infinity. Remove and try again!")
        print(list(numeric_features.columns[column_contains_inf]))
        raise Exception
    selector.fit(numeric_features)
    no_variance_features = numeric_features.columns.values[~selector.get_support()]
    if (n := len(no_variance_features)) > 0:
        print(
            f"Dropping {n} low variance (<={threshold}) features: {no_variance_features}"
        )
        return features.drop(columns=no_variance_features)
    else:
        print(f"No features with less than {threshold} variance.")
        return features


def drop_collinear_features_iteratively(features, force_include_cols):
    """Use Variance Inflation Factor (VIF) to drop collinear features iteratively.
    The function takes the following steps:
    1. Selects only numeric features for VIF analysis
    2. Impute missing values - this is required by the variance_inflation_factor()
    function. Because we typically rely on default AutoML strategies for imputation,
    we expect to have null values in our data. TODO: we may want to revisit this
    decision if we want more consistent imputation between the VIF analysis and the
    modeling, perhaps deciding to do our own imputation prior to this VIF analysis.
    3. Find the highest VIF. If it is over 10, it is considered too high and the
    feature is contributing to multicollinearity - we drop this feature.
    4. Repeat step 3 with the remaining features until there are no VIF > 10.

    Note: the function used, variance_inflation_factor(), does not take an
    intercept into account. See: https://github.com/statsmodels/statsmodels/issues/2376
    If we want to use a centered VIF rather than an uncentered VIF, we would need to add
    a constant columns to the features dataframe and then re-calculate VIF.

    Troubleshooting: if this hangs, try disabling mlflow logging:
        import mlflow
        mlflow.autolog(disable=True)

    Args:
        features (pd.DataFrame): dataframe of features to assess for multicollinearity

    Returns:
        pd.DataFrame: features not considered collinear according to a VIF threshold of 10
    """

    np.seterr(divide="ignore", invalid="ignore")

    numeric_features = features.select_dtypes(include="number")
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    numeric_features_imputed = pd.DataFrame(imp.fit_transform(numeric_features))
    numeric_features_imputed.columns = numeric_features.columns.values

    # A VIF of 1 indicates the feature has no correlation with any of the other features. A VIF exceeding 10 is too high and contributing to multicollinearity
    n_features_dropped_so_far = 0
    while (
        max_vif := max(
            (
                uncentered_vif_dict := {
                    # calculate VIF of features that are not force-included, against all numeric variables
                    col: variance_inflation_factor(numeric_features_imputed, col_index)
                    for col_index, col in enumerate(
                        numeric_features_imputed.columns.values
                    )
                    if col not in force_include_cols
                }
            ).values()
        )
    ) > 10:
        highest_vif_cols = [
            col for col, vif in uncentered_vif_dict.items() if vif == max_vif
        ]
        n_drop_this_round = len(highest_vif_cols)
        n_features_dropped_so_far += n_drop_this_round
        print(f"Dropping {n_drop_this_round} column(s) with VIF {max_vif}")
        print(highest_vif_cols)
        features = features.drop(columns=highest_vif_cols)
        numeric_features_imputed = numeric_features_imputed.drop(
            columns=highest_vif_cols
        )
    print(f"Dropped {n_features_dropped_so_far} collinear features.")
    assert all(
        [col in features.columns for col in force_include_cols]
    ), "The dataset with selected features is missing one of the force include variables!"
    return features


def select_features(
    train_df,
    not_features_cols,
    force_include_cols,
    incomplete_threshold=0.5,
    low_variance_threshold=0.0,
):
    """Select features by dropping incomplete features, low variance features,
    and variables with high correlations to others.

    Args:
        train_df (pd.DataFrame): dataframe of features to select
        not_features_cols (list[str]): list of column names that are not features and should not
            be run through the feature selection algorithm. For example, demographics, IDs, outcome variables, etc.
        force_include_cols (list[str]): list of features to force include in the final dataset.
        incomplete_threshold (float, optional): threshold for level of incompleteness. Defaults to 0.5.
        low_variance_threshold (float, optional): threshold for level of low variance. Defaults to 0.0.

    Returns:
        pd.DataFrame: train_df with not_features_cols, force_include_cols, and any other columns selected
            by the algorithms
    """

    X = train_df.drop(columns=not_features_cols + force_include_cols)

    ###################################
    ##### Unsupervised selection ######
    ###################################

    # incomplete features - features with lots of nulls
    # Note: we should create dummy variables prior to running this process. If one of the dummies represents NaN, we won't drop the categoricals that have lots of nulls, but instead can capture any patterns that may relate to the nulls. If there are non-dummy categorical variables that get dropped here, consider pre-processing into dummy variables for this reason.
    X = drop_incomplete_features(
        X, threshold=incomplete_threshold
    )  # TODO: tune this threshold over time

    # features with low variance
    # Note: only removing features with 0 variance for now, as features with low
    # but non-zero variance may be powerful in predicting the outcome
    X = drop_low_variance_features(X, threshold=low_variance_threshold)

    # multi-collinearity: it may not interfere with the model's performance, but it does negatively impact the interpretation of the predictors
    features_force_include_df = train_df[
        list(set(list(X.columns.values) + force_include_cols))
    ]
    selected_features_df = drop_collinear_features_iteratively(
        features_force_include_df, force_include_cols
    )

    selected_cols = set(list(selected_features_df.columns.values)) - set(
        force_include_cols
    )
    print(f"Original N features: {train_df.shape[1]}")
    print(f"Selected {len(selected_cols)} features: {selected_cols}")

    keep_cols = list(selected_cols)
    additional_keep_cols = not_features_cols + force_include_cols
    keep_cols.extend(additional_keep_cols)
    print(
        f"Plus {len(additional_keep_cols)} demographic, outcome, and force-included columns: {additional_keep_cols}"
    )
    selected_features_df = train_df[keep_cols]
    return selected_features_df


def run_automl_classification(
    institution_id,
    job_run_id,
    train_df,
    outcome_col,
    optimization_metric,
    student_id_col,
    **kwargs,
):
    """Wrap around databricks.automl.classify to allow testing and ensure that
    our parameters are used properly.

    Args:
        institution_id (str): institution ID for labeling experiment
        job_run_id (str): job run ID of Databricks workflow for labeling experiment
        train_df (pd.DataFrame): data containing features and outcome to model, as well as the student_id_col and any other columns
            specified in the optional **kwargs
        outcome_col (str): column name for the target to predict
        optimization_metric (str): Metric used to evaluate and rank model performance.
            Supported metrics for classification: “f1” (default), “log_loss”,
            “precision”, “accuracy”, “roc_auc”
        student_id_col (str): column name containing student IDs to exclude from training.
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


def extract_training_data_from_model(
    automl_experiment_id, data_runname="Training Data Storage and Analysis"
):
    """Read training data from a model into a pandas DataFrame. This allows us to run more
    evaluations of the model, ensuring that we are using the same train/test/validation split

    Args:
        automl_experiment_id (str): Experiment ID of the AutoML experiment
        data_runname (str, optional): The runName tag designating where there training data is
            stored. Defaults to 'Training Data Storage and Analysis'.

    Returns:
        pd.DataFrame: the data used for training a model, with train/test/validation flags
    """
    run_df = mlflow.search_runs(experiment_ids=[automl_experiment_id])
    data_run_id = run_df[run_df["tags.mlflow.runName"] == data_runname]["run_id"].item()

    # Create temp directory to download input data from MLflow
    input_temp_dir = os.path.join(
        os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8]
    )
    os.makedirs(input_temp_dir)

    # Download the artifact and read it into a pandas DataFrame
    input_data_path = mlflow.artifacts.download_artifacts(
        run_id=data_run_id, artifact_path="data", dst_path=input_temp_dir
    )

    df_loaded = pd.read_parquet(os.path.join(input_data_path, "training_data"))
    # Delete the temp data
    shutil.rmtree(input_temp_dir)

    return df_loaded


def create_risk_score_histogram(risk_score, title_suffix):
    """Create histogram of risk scores

    Args:
        risk_score (array-like): risk scores
        title_suffix (str): suffix for plot title

    Returns:
        matplotlib.figure
    """

    fig1, ax1 = plt.subplots()
    sns.histplot(
        x=risk_score,
        ax=ax1,
        color=palette[1],
    )
    ax1.set_xlabel("Risk Score")
    ax1.set_title(f"Distribution of risk scores - {title_suffix}")
    return fig1


def check_array_of_arrays(input_array):
    """Check if an input array contains sub-arrays. Used for plotting different
    groups of predictions

    Args:
        input_array (array-like)

    Returns:
        bool: True if the input_array contains sub-arrays
    """
    try:
        assert isinstance(input_array, pd.Series)
        assert isinstance(input_array[0], list)
        return True
    except Exception:
        return False


def create_calibration_curve(
    y_true, risk_score, keys, title_suffix, pos_label, lowess_frac=None
):
    """Create calibration plot

    Args:
        y_true (array-like of shape (n_samples,) or (n_groups,)): overall or group-level true outcome class
        risk_score (array-like of shape (n_samples,) or (n_groups,)): overall or group level predicted risk scores
        keys (list[str] or str): overall or subgroup level labels for labeling lines
        title_suffix (str): suffix for plot title
        pos_label (int, float, bool or str, optional): label identifying the positive class. Defaults to True.

    Returns:
        matplotlib.figure: line plot of prediction bins X fraction of positive class
    """

    if not check_array_of_arrays(y_true):
        y_true = [y_true]
        risk_score = [risk_score]
        keys = [keys]

    fig2, ax2 = plt.subplots()

    for j in range(len(y_true)):
        prob_true, prob_pred = calibration_curve(
            y_true[j],
            risk_score[j],
            n_bins=10,
            strategy="uniform",
            pos_label=pos_label,
        )
        if lowess_frac:
            # When we create calibration curves with less data points (i.e. for subgroups),
            # it can look choppy and be difficult to interpret. We use locally weighted
            # scatterplot smoothing (LOWESS) to fit a smooth curve to the data points. LOWESS
            # is a non-parametric smoother that tries to find a curve of best fit without
            # assuming the data fits a particular shape.
            # Resource: https://www.statisticshowto.com/lowess-smoothing/
            prob_true = lowess(
                endog=prob_true,
                exog=prob_pred,
                frac=lowess_frac,
                it=0,
                is_sorted=False,
                return_sorted=False,
            )
        sns.lineplot(
            x=prob_pred, y=prob_true, color=palette[j + 1], ax=ax2, label=keys[j]
        )
    sns.lineplot(
        x=[0, 1],
        y=[0, 1],
        linestyle="dashed",
        color=palette[0],
        ax=ax2,
        label="Perfectly calibrated",
    )

    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Fraction of positives")
    ax2.set_title(f"Calibration Curve - {title_suffix}")
    ax2.legend(loc="lower right")

    return fig2


def get_sensitivity_of_top_q_pctl_thresh(y_true, risk_score, q, pos_label):
    """Report sensitivity (AKA recall score) using some percentile threshold.
    Calculation:
        number of true positivies / (number of true positives + number of false negatives)
        OR
        number of true positives / total number of actual true class

    Args:
        y_true (pd.Series of length n_samples): true outcome class
        risk_score (pd.Series of length n_samples): predicted risk scores
        q (float): probability for the quantiles to compute
        pos_label (int, float, bool or str): label identifying the positive class in y_true

    Returns:
        float
    """
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)
    if not isinstance(risk_score, pd.Series):
        risk_score = pd.Series(risk_score)

    prob_thresh = np.quantile(risk_score, q)
    high_risk = risk_score >= prob_thresh

    # convert actual outcome to booleans to match high_risk array
    y_true = y_true.apply(lambda x: True if x == pos_label else False)

    return recall_score(y_true, high_risk)


def plot_sla_curve(
    y_true,
    risk_score,
    keys,
    title_suffix,
    pos_label,
    alert_rates=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
    label_alert_rate=0.01,
):
    """Create Sensitivity at Low Alert Rates plot

    Args:
        y_true (array-like of shape (n_samples,) or (n_groups,)): overall or group-level true outcome class
        risk_score (array-like of shape (n_samples,) or (n_groups,)): overall or group level predicted risk scores
        keys (list[str] or str): overall or subgroup level labels for labeling lines
        title_suffix (str): suffix for plot title
        pos_label (int, float, bool or str): label identifying the positive class in y_true
        alert_rates (list, optional): alert rates to plot. Defaults to [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06].
        label_alert_rate (float, optional): alert rate of interest to report sensitivity at. Defaults to 0.01.

    Returns:
        matplotlib.figure: line plot of sensitivity at small alert rates
    """

    if not check_array_of_arrays(y_true):
        y_true = [y_true]
        risk_score = [risk_score]
        keys = [keys]

    fig, ax = plt.subplots()

    for j in range(len(risk_score)):
        ss = []
        for i in alert_rates:  # calculate sensitivity at different alert rates
            s = get_sensitivity_of_top_q_pctl_thresh(
                y_true[j], risk_score[j], 1 - i, pos_label
            )
            ss.append(s)
        s_lab = round(
            get_sensitivity_of_top_q_pctl_thresh(
                y_true[j], risk_score[j], 1 - label_alert_rate, pos_label
            ),
            2,
        )
        ax.plot(
            alert_rates,
            ss,
            color=palette[j + 1],
            label="{} (Sensitivity at {}% alert rate={})".format(
                keys[j], label_alert_rate * 100, s_lab
            ),
        )

    ax.set_ylabel("sensitivity (true positive rate)")
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Alert rate")
    ax.legend(loc="lower right")
    ax.set_title(f"Sensitivity vs. Low Alert Rate - {title_suffix}")

    return fig


def compare_trained_models(automl_experiment_id, automl_metric):
    """Retrieve, aggregate and sort performance data for models trained in a specified AutoML experiment.
    The validation dataset is used to tune hyperparameters. Metrics on the validation dataset are used to rank models, so we also use this metric to compare across models.

    Args:
        automl_experiment_id (str): Experiment ID of the AutoML experiment
        automl_metric (str): Chosen AutoML optimization metric
    Returns:
        pandas.DataFrame: DataFrame containing model types and highest scores for the given metric.
    """
    runs = mlflow.search_runs(automl_experiment_id)
    metric = str.lower(automl_metric)
    metric_score_column = (
        f"metrics.val_{metric}"
        if metric == "log_loss"
        else f"metrics.val_{metric}_score"
    )

    runs = runs[["tags.model_type", metric_score_column]].dropna()

    ascending = metric == "log_loss"
    df_sorted = (
        runs.groupby("tags.model_type", as_index=False)
        .agg({metric_score_column: ("min" if ascending else "max")})
        .sort_values(by=metric_score_column, ascending=ascending)
    )
    return df_sorted, metric_score_column


def compare_trained_models_plot(automl_experiment_id, automl_metric):
    """Create a plot to evaluate all the models trained by AutoML.

    Args:
        automl_experiment_id (str): Experiment ID of the AutoML experiment
        automl_metric (str): Chosen AutoML optimization metric
    Returns:
        matplotlib.figure.Figure: bar chart of model performance on test data by optimization metric.
    """

    df_sorted, metric_score_column = compare_trained_models(
        automl_experiment_id, automl_metric
    )

    df_sorted = df_sorted.sort_values(
        by=metric_score_column,
        ascending=True if not automl_metric == "log_loss" else False,
    )
    fig, ax = plt.subplots()

    ax.set_xlim(
        0,
        (
            df_sorted[metric_score_column].max() + 0.2
            if automl_metric == "log_loss"
            else 1
        ),
    )  # setting buffer after max log_loss to 0.2, for better viz

    colors = ["#A9A9A9"] * len(df_sorted)
    colors[-1] = "#5bc0de"  # Bright blue color for the best model

    bars = ax.barh(
        df_sorted["tags.model_type"], df_sorted[metric_score_column], color=colors
    )

    for bar in bars:
        ax.text(
            bar.get_width()
            - (
                max(df_sorted[metric_score_column]) * 0.005
            ),  # Position the text inside the bar
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.4f}",
            va="center",
            ha="right",
            fontweight="bold",
            color="white",
            fontsize=12,
        )

    sort_order = (
        "Lowest to Highest" if automl_metric == "log_loss" else "Highest to Lowest"
    )
    automl_metric = (
        automl_metric
        if automl_metric == "log_loss"
        else f"{automl_metric.capitalize()} Score"
    )

    ax.set_title(f"{automl_metric} by Model Type {sort_order}")
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=True)
    ax.tick_params(axis="x", colors="lightgrey", which="both")  # Color of ticks
    ax.xaxis.grid(True, color="lightgrey", linestyle="--", linewidth=0.5)

    ax.set_facecolor("none")
    ax.set_frame_on(False)
    fig.tight_layout()

    return fig


def create_evaluation_plots(data, risk_score_col, y_true_col, pos_label, split_type):
    """Create plots to evaluate a model overall - risk score histogram,
    calibration curve, and sensitivity at low alert rates

    Args:
        data (pd.DataFrame): containing predicted and actual outcome data
        risk_score_col (str): column name containing data of predicted risk scores
        y_true_col (str): column name containing data of actual outcome classes
        pos_label (int, float, bool or str): label identifying the positive class in y_true
        split_type (str): type of data being plotted for labeling plots - train, test, or validation

    Returns:
        list[matplotlib.figure]: risk score histogram, calibration curve, and
        sensitivity at low alert rates figures
    """
    title_suffix = f"{split_type} data - Overall"
    hist_fig = create_risk_score_histogram(data[risk_score_col], title_suffix)
    cal_fig = create_calibration_curve(
        data[y_true_col], data[risk_score_col], "Overall", title_suffix, pos_label
    )
    sla_fig = plot_sla_curve(
        data[y_true_col], data[risk_score_col], "Overall", title_suffix, pos_label
    )
    return hist_fig, cal_fig, sla_fig


def create_evaluation_plots_by_subgroup(
    data, risk_score_col, y_true_col, pos_label, group_col, split_type
):
    """Create plots to evaluate a model by group - calibration curve
    and sensitivity at low alert rates

    Args:
        data (pd.DataFrame): containing predicted and actual outcome data, as well as group label
        risk_score_col (str): column name containing data of predicted risk scores
        y_true_col (str): column name containing data of actual outcome classes
        pos_label (int, float, bool or str): label identifying the positive class in y_true
        group_col (str): column name containing data of subgroup labels
        split_type (str): type of data being plotted for labeling plots - train, test, or validation

    Returns:
        list[matplotlib.figure]: calibration curve sensitivity at low alert rates
        figures by group
    """
    title_suffix = f"{split_type} data - {group_col}"

    grouped_data = data.groupby(group_col).agg(list)

    ys = grouped_data[y_true_col]
    scores = grouped_data[risk_score_col]

    subgroups = grouped_data.index
    n_counts = [len(subgroup_scores) for subgroup_scores in scores]
    names = [
        subgroup_name + f" (N={n})" for subgroup_name, n in zip(subgroups, n_counts)
    ]

    if len(subgroups) > 4:
        lowess_frac = 0.7
    else:
        lowess_frac = 0.6

    sla_subgroup_plot = plot_sla_curve(ys, scores, names, title_suffix, pos_label)
    cal_subgroup_plot = create_calibration_curve(
        ys, scores, names, title_suffix, pos_label, lowess_frac=lowess_frac
    )

    return cal_subgroup_plot, sla_subgroup_plot
