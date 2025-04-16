import logging
import re
import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.base
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


def _get_mapped_feature_name(
    feature_col: str, features_table: dict[str, dict[str, str]]
) -> str:
    feature_col = feature_col.lower()  # just in case
    if feature_col in features_table:
        feature_name = features_table[feature_col]["name"]
    else:
        for fkey, fval in features_table.items():
            if "(" in fkey and ")" in fkey:
                if match := re.match(fkey, feature_col):
                    feature_name = fval["name"].format(*match.groups())
                    break
        else:
            feature_name = feature_col
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
