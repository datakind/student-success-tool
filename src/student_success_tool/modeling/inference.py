import logging
import re
import typing as t
import mlflow

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.base
import shap
from shap import KernelExplainer

LOGGER = logging.getLogger(__name__)


def predict_probs(
    features: pd.DataFrame | np.ndarray,
    model: sklearn.base.BaseEstimator,
    *,
    feature_names: t.Optional[list[str]] = None,
    pos_label: t.Optional[bool | str] = None,
    dtypes: t.Optional[dict[str, object]] = None,
) -> np.ndarray:
    """
    Predict target probabilities for examples in ``features`` using ``model`` .

    Args:
        features
        model
        feature_names: Names of features corresponding to each column in ``features`` ,
            in cases where it must be passed as a numpy array. If not specified,
            feature names are inferred from the model's "column_selector"
            (a standard in models trained using Databricks AutoML).
        pos_label: Value in ``model`` classes that constitutes a "positive" prediction,
            often ``True`` in the case of binary classification.
        dtypes: Mapping of column name to dtype in ``featuress`` that needs to be overridden
            before passing it into ``model`` .

    Returns:
        Predicted probabilities, as a 1-dimensional array (when ``pos_label is True`` )
            or an N-dimensional array, where N corresponds to the number of pred classes.
    """
    if not sklearn.base.is_classifier(model):
        LOGGER.warning("predict_proba() expects a classifier, but received %s", model)

    if feature_names is None:
        feature_names = model.named_steps["column_selector"].get_params()["cols"]
    assert isinstance(feature_names, list)  # type guard

    assert features.shape[1] == len(feature_names)
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(data=features, columns=feature_names)

    if dtypes:
        features = features.astype(dtypes)

    pred_probs = model.predict_proba(features)
    assert isinstance(pred_probs, np.ndarray)  # type guard

    if pos_label is not None:
        return pred_probs[:, model.classes_.tolist().index(pos_label)]
    else:
        return pred_probs


def shap_summary_plot(
    df_shap_values: pd.DataFrame,
    df_test: pd.DataFrame,
    model_feature_names: list[str],
    model_classes: npt.NDArray,
    max_display: int = 20,
) -> None:
    """
    Generates and logs a SHAP summary plot using metadata config.

    Args:
        df_shap_values: DataFrame of SHAP values
        df_test: DataFrame of test features
        model_feature_names: List of feature names used in the model
        model_classes: Numpy array of model classes (e.g. True, False)
        max_display: Maximum number of features to display
    """

    shap.summary_plot(
        df_shap_values.loc[:, model_feature_names].to_numpy(),
        df_test.loc[:, model_feature_names],
        class_names=model_classes,
        max_display=20,
        show=False,
    )

    shap_fig = plt.gcf()

    mlflow.log_figure(shap_fig, "feature_importances_by_shap_plot.png")


def select_top_features_for_display(
    features: pd.DataFrame,
    unique_ids: pd.Series,
    predicted_probabilities: list[float],
    shap_values: npt.NDArray[np.float64],
    n_features: int = 3,
    needs_support_threshold_prob: t.Optional[float] = 0.5,
    features_table: t.Optional[dict[str, dict[str, str]]] = None,
) -> pd.DataFrame:
    """
    Select most important features from SHAP for each student
    and format for display

    Args:
        features: features used in modeling
        unique_ids: student IDs, of length ``features.shape[0]``
        predicted_probabilities: predicted probabilities for each student, in the same
            order as unique_ids, of shape len(unique_ids)
        shap_values: array of arrays of SHAP values, of shape len(unique_ids)
        n_features: number of important features to return
        needs_support_threshold_prob: Minimum probability in [0.0, 1.0] used to compute
            a boolean "needs support" field added to output records. Values in
            ``predicted_probabilities`` greater than or equal to this threshold result in
            a True value, otherwise it's False; if this threshold is set to null,
            then no "needs support" values are added to the output records.
            Note that this doesn't have to be the "optimal" decision threshold for
            the trained model that produced ``predicted_probabilities`` , it can
            be tailored to a school's preferences and use case.
        features_table: Optional mapping of column to human-friendly feature name/desc,
            loaded via :func:`utils.load_features_table()`

    Returns:
        explainability dataframe for display

    TODO: refactor this functionality so it's vectorized and aggregates by student
    """
    pred_probs = np.asarray(predicted_probabilities)

    top_features_info = []
    for i, (unique_id, predicted_proba) in enumerate(zip(unique_ids, pred_probs)):
        instance_shap_values = shap_values[i]
        top_indices = np.argsort(-np.abs(instance_shap_values))[:n_features]
        top_features = features.columns[top_indices]
        top_feature_values = features.iloc[i][top_features]
        top_shap_values = instance_shap_values[top_indices]

        student_output = {
            "Student ID": unique_id,
            "Support Score": predicted_proba,
        }
        if needs_support_threshold_prob is not None:
            student_output["Support Needed"] = (
                predicted_proba >= needs_support_threshold_prob
            )

        for feature_rank, (feature, feature_value, shap_value) in enumerate(
            zip(top_features, top_feature_values, top_shap_values), start=1
        ):
            feature_name = (
                _get_mapped_feature_name(feature, features_table)
                if features_table is not None
                else feature
            )
            feature_value = (
                str(round(feature_value, 2))
                if isinstance(feature_value, float)
                else str(feature_value)
            )
            student_output |= {
                f"Feature_{feature_rank}_Name": feature_name,
                f"Feature_{feature_rank}_Value": feature_value,
                f"Feature_{feature_rank}_Importance": round(shap_value, 2),
            }

        top_features_info.append(student_output)
    return pd.DataFrame(top_features_info)


def generate_ranked_feature_table(
    features: pd.DataFrame,
    shap_values: npt.NDArray[np.float64],
    features_table: t.Optional[dict[str, dict[str, str]]] = None,
) -> pd.DataFrame:
    """
    Creates a table of all selected features of the model ranked
    by average SHAP magnitude (aka feature importance). We utilize average
    SHAP magnitude & an absolute value because it removes directionality
    from the SHAP values and focuses specifically on importance. This table
    is used in the model cards to provide a comprehensive summary of the model's
    features.

    Args:
        features: feature data used in modeling where columns are the feature
            column names
        shap_values: array of arrays of SHAP values, of shape len(unique_ids)
        features_table: Optional mapping of column to human-friendly feature name/desc,
            loaded via :func:`utils.load_features_table()`

    Returns:
        A ranked pandas DataFrame by average shap magnitude
    """
    feature_metadata = []

    for idx, feature in enumerate(features.columns):
        feature_name = (
            _get_mapped_feature_name(feature, features_table)
            if features_table is not None
            else feature
        )
        dtype = features[feature].dtype
        data_type = (
            "Boolean"
            if pd.api.types.is_bool_dtype(dtype)
            else "Continuous"
            if pd.api.types.is_numeric_dtype(dtype)
            else "Categorical"
        )
        avg_shap_magnitude_raw = np.mean(np.abs(shap_values[:, idx]))
        feature_metadata.append(
            {
                "Feature Name": feature_name,
                "Data Type": data_type,
                "Average SHAP Magnitude (Raw)": avg_shap_magnitude_raw,
            }
        )

    df = (
        pd.DataFrame(feature_metadata)
        .sort_values(by="Average SHAP Magnitude (Raw)", ascending=False)
        .reset_index(drop=True)
    )

    # Format magnitudes after sorting to avoid type issues
    df["Average SHAP Magnitude"] = df["Average SHAP Magnitude (Raw)"].apply(
        lambda x: "<0.0000" if round(x, 4) == 0 else round(x, 4)
    )

    # Drop the raw magnitude column
    df = df.drop(columns=["Average SHAP Magnitude (Raw)"])

    # Log as an ML artifact
    df.to_csv("/tmp/ranked_selected_features.csv", index=False)
    mlflow.log_artifact(
        "/tmp/ranked_selected_features.csv", artifact_path="selected_features"
    )

    return df


def _get_mapped_feature_name(
    feature_col: str, features_table: dict[str, dict[str, str]], metadata: bool = False
) -> t.Any:
    feature_col = feature_col.lower()  # just in case
    if feature_col in features_table:
        entry = features_table[feature_col]
        feature_name = entry["name"]
        if metadata:
            short_desc = entry.get("short_desc")
            long_desc = entry.get("long_desc")
            return feature_name, short_desc, long_desc
        return feature_name
    else:
        for fkey, fval in features_table.items():
            if "(" in fkey and ")" in fkey:
                if match := re.fullmatch(fkey, feature_col):
                    feature_name = fval["name"].format(*match.groups())
                    if metadata:
                        short_desc = fval.get("short_desc")
                        long_desc = fval.get("long_desc")
                        return feature_name, short_desc, long_desc
                    return feature_name

        else:
            feature_name = feature_col
            if metadata:
                return feature_name, None, None
            return feature_name


def calculate_shap_values_spark_udf(
    dfs: t.Iterator[pd.DataFrame],
    *,
    student_id_col: str,
    model_features: list[str],
    explainer: KernelExplainer,
    mode: pd.Series,
) -> t.Iterator[pd.DataFrame]:
    """
    SHAP is computationally expensive, so this function enables parallelization,
    by calculating SHAP values over an iterator of DataFrames. Sparks' repartition
    performs a full shuffle (does not preserve row order), so it is critical to
    extract the student_id_col prior to creating shap values and then reattach
    for our final output.

    Args:
        dfs: An iterator over Pandas DataFrames.
        Each DataFrame is a batch of data points.
        student_id_col: The name of the column containing student_id
        model_features: A list of strings representing the names
        of the features for our model
        explainer: A KernelExplainer object used to compute
        shap values from our loaded model.
        mode: A Series containing values to impute missing values

    Returns:
        Iterator[pd.DataFrame]: An iterator over Pandas DataFrames. Each DataFrame
        contains the SHAP values for that partition of data.
    """
    for df in dfs:
        yield calculate_shap_values(
            df,
            explainer,
            feature_names=model_features,
            fillna_values=mode,
            student_id_col=student_id_col,
        )


def calculate_shap_values(
    df: pd.DataFrame,
    explainer: KernelExplainer,
    *,
    feature_names: list[str],
    fillna_values: pd.Series,
    student_id_col: str = "student_id",
) -> pd.DataFrame:
    """
    Compute SHAP values for the features in ``df`` using ``explainer`` and return result
    as a reassembled data frame with ``feature_names`` as columns and ``student_id_col``
    added as an extra (identifying) column.

    Args:
        df
        explainer
        feature_names
        fillna_values
        student_id_col

    Reference:
        https://shap.readthedocs.io/en/stable/generated/shap.KernelExplainer.html
    """
    # preserve student ids
    student_ids = df.loc[:, student_id_col].reset_index(drop=True)
    # impute missing values and run shap values using just features
    features_imp = df.loc[:, feature_names].fillna(fillna_values)
    explanation = explainer(features_imp)
    return (
        pd.DataFrame(data=explanation.values, columns=explanation.feature_names)
        # reattach student ids to their shap values
        .assign(**{student_id_col: student_ids})
    )


def top_shap_features(
    features: pd.DataFrame,
    unique_ids: pd.Series,
    shap_values: npt.NDArray[np.float64],
    top_n: int = 10,
    features_table: t.Optional[dict[str, dict[str, str]]] = None,
) -> pd.DataFrame:
    """
    Extracts the top N most important SHAP features across all samples.

    Args:
        features (pd.DataFrame): Input feature values.
        unique_ids (pd.Series): Unique identifiers for each sample.
        shap_values (np.ndarray): SHAP values for the input features.
        top_n (int): Number of top features to select (default is 10).
        features_table (dict, optional): Mapping of feature names to human-readable names.

    Returns:
        pd.DataFrame: Long-form DataFrame with columns:
            - student_id
            - feature_name
            - shap_value
            - feature_value
    """

    if features.empty or shap_values.size == 0 or unique_ids.empty:
        raise ValueError("Input data cannot be empty.")

    shap_long = (
        pd.DataFrame(shap_values, columns=features.columns)
        .assign(student_id=unique_ids.values)
        .melt(id_vars="student_id", var_name="feature_name", value_name="shap_value")
    )

    feature_long = features.assign(student_id=unique_ids.values).melt(
        id_vars="student_id", var_name="feature_name", value_name="feature_value"
    )

    summary_df = shap_long.merge(feature_long, on=["student_id", "feature_name"])

    top_n_features = (
        summary_df.groupby("feature_name")["shap_value"]
        .apply(lambda x: np.mean(np.abs(x)))
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    top_features = summary_df[summary_df["feature_name"].isin(top_n_features)].copy()

    if features_table is not None:
        top_features[
            ["feature_readable_name", "feature_short_desc", "feature_long_desc"]
        ] = top_features["feature_name"].apply(
            lambda feature: pd.Series(
                _get_mapped_feature_name(feature, features_table, metadata=True)
            )
        )

    top_features["feature_value"] = top_features["feature_value"].astype(str)

    return top_features


def top_feature_boxstats(
    features: pd.DataFrame,
    shap_values: npt.NDArray[np.float64],
    features_table: t.Optional[dict[str, dict[str, str]]] = None,
) -> pd.DataFrame:
    """
    Per-feature summary for the GLOBAL top-N features (by mean |SHAP|).
    Returns min, Q1, median, Q3, max suitable for box/whisker plotting,
    along with mean absolute SHAP for reference.
    """
    if features.empty or shap_values.size == 0:
        raise ValueError("Input data cannot be empty.")
    if shap_values.shape != (features.shape[0], features.shape[1]):
        raise ValueError(
            f"shap_values shape {shap_values.shape} must match features shape {features.shape}"
        )

    mean_abs = pd.Series(np.mean(np.abs(shap_values), axis=0), index=features.columns)
    top_feats = mean_abs.sort_values(ascending=False)

    # Restrict stats to numeric columns
    stats_source = features.select_dtypes(include=[np.number])

    rows = []
    for feat in top_feats.index:
        if feat not in stats_source.columns:
            # Non-numeric top feature — include row with NaN stats, but correct counts.
            rows.append(
                {
                    "feature_name": feat,
                    "feature_shap_value": float(top_feats[feat]),
                    "min": np.nan,
                    "Q1": np.nan,
                    "median": np.nan,
                    "Q3": np.nan,
                    "max": np.nan,
                    "count": int(features[feat].notna().sum()),
                    "n_missing": int(features[feat].isna().sum()),
                }
            )
            continue

        col = stats_source[feat]
        rows.append(
            {
                "feature_name": feat,
                "feature_shap_value": float(top_feats[feat]),
                "min": float(col.min()),
                "Q1": float(col.quantile(0.25, interpolation="linear")),
                "median": float(col.quantile(0.5, interpolation="linear")),
                "Q3": float(col.quantile(0.75, interpolation="linear")),
                "max": float(col.max()),
                "count": int(col.notna().sum()),
                "n_missing": int(col.isna().sum()),
            }
        )

    feature_boxstats = (
        pd.DataFrame(rows)
        .sort_values("feature_shap_value", ascending=False)
        .reset_index(drop=True)
    )
    if features_table is not None:
        feature_boxstats[
            ["feature_readable_name", "feature_short_desc", "feature_long_desc"]
        ] = feature_boxstats["feature_name"].apply(
            lambda feature: pd.Series(
                _get_mapped_feature_name(feature, features_table, metadata=True)
            )
        )
    return feature_boxstats


def support_score_distribution_table(
    df_serving: pd.DataFrame,
    unique_ids: t.Any,
    pred_probs: t.Any,
    shap_values: t.Any,
    inference_params: dict,
    features_table: t.Optional[dict[str, dict[str, str]]] = None,
) -> pd.DataFrame:
    """
    Selects top SHAP features for each student, and bins the support scores.

    Args:
        df_serving (pd.DataFrame): Input features used for prediction.
        unique_ids (pd.Series): Unique ids (student_id) for each student.
        pred_probs (list or np.ndarray): Predicted probabilities from the model.
        shap_values (np.ndarray or pd.DataFrame): SHAP values for the input features.
        inference_params (dict): Dictionary containing configuration for:
            - "num_top_features" (int): Number of top features to display.
            - "min_prob_pos_label" (float): Threshold to determine if support is needed.
        features_table (dict): Optional dictionary mapping feature names to understandable format.
        model_feature_names (list): List of feature names used by the model.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - bin_lower: Lower bound of the support score bin.
            - bin_upper: Upper bound of the support score bin.
            - support_score: Midpoint of the bin (used for plotting).
            - count_of_students: Number of students in the bin.
            - pct: Percentage of total students in the bin.

    """

    try:
        result = select_top_features_for_display(
            features=df_serving,
            unique_ids=unique_ids,
            predicted_probabilities=pred_probs,
            shap_values=shap_values.values,
            n_features=inference_params["num_top_features"],
            needs_support_threshold_prob=inference_params["min_prob_pos_label"],
            features_table=features_table,
        )

        # --- Bin support scores for histogram (e.g., 0.0 to 1.0 in 0.1 steps) ---
        bins = np.arange(0.1, 1.1, 0.1)
        result["score_bin"] = pd.cut(
            result["Support Score"], bins=bins, include_lowest=True, right=False
        )

        # Group and count
        bin_counts = (
            result.groupby("score_bin", observed=True)
            .size()
            .reset_index(name="count_of_students")
        )

        # Extract bin boundaries
        bin_counts["bin_lower"] = bin_counts["score_bin"].apply(
            lambda x: round(x.left, 2)
        )
        bin_counts["bin_upper"] = bin_counts["score_bin"].apply(
            lambda x: round(x.right, 2)
        )
        bin_counts["support_score"] = bin_counts["score_bin"].apply(
            lambda x: round((x.left + x.right) / 2, 2)
        )

        total_students = len(result)
        bin_counts["pct"] = (
            bin_counts["count_of_students"] / total_students * 100
        ).round(2)

        return bin_counts[
            ["bin_lower", "bin_upper", "support_score", "count_of_students", "pct"]
        ]

    except Exception:
        import traceback

        traceback.print_exc()
        raise  # <-- temporarily raise instead of returning None
