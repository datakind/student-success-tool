import typing as t
from typing import Iterator

import numpy as np
import pandas as pd
from shap import KernelExplainer

def select_top_features_for_display(
    features: pd.DataFrame,
    unique_ids: pd.Series,
    predicted_probabilities: list[float],
    shap_values: pd.Series,
    n_features: int = 3,
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
        features_table: Optional mapping of column to human-friendly feature name/desc,
            loaded via :func:`utils.load_features_table()`

    Returns:
        explainability dataframe for display

    TODO: refactor this functionality so it's vectorized and aggregates by student
    """
    top_features_info = []

    for i, (unique_id, predicted_proba) in enumerate(
        zip(unique_ids, predicted_probabilities)
    ):
        instance_shap_values = shap_values[i]
        top_indices = np.argsort(-np.abs(instance_shap_values))[:n_features]
        top_features = features.columns[top_indices]
        top_feature_values = features.iloc[i][top_features]
        top_shap_values = instance_shap_values[top_indices]

        for rank, (feature, feature_value, shap_value) in enumerate(
            zip(top_features, top_feature_values, top_shap_values), start=1
        ):
            feature_name = (
                # HACK: lowercase feature column name in features table lookup
                # TODO: we should *ensure* feature column names are lowercased
                # before using them in a model; current behavior should be considered a bug
                features_table.get(feature.lower(), {}).get("name", feature)
                if features_table is not None
                else feature
            )
            top_features_info.append(
                {
                    "Student ID": unique_id,
                    "Support Score": predicted_proba,
                    "Top Indicators": feature_name,
                    "Indicator Value": feature_value,
                    "SHAP Value": shap_value,
                    "Rank": rank,
                }
            )
    return pd.DataFrame(top_features_info)


def calculate_shap_values(iterator: Iterator[pd.DataFrame], *, student_id_col: str,
                model_features: list[str], explainer: KernelExplainer,
                mode: pd.Series) -> Iterator[pd.DataFrame]:
    """
    SHAP is computationally expensive, so this function enables parallelization,
    by calculating SHAP values over an iterator of DataFrames. Sparks' repartition
    performs a full shuffle (does not preserve row order), so it is critical to 
    extract the student_id_col prior to creating shap values and then reattach 
    for our final output.

    Args:
        iterator (Iterator[pd.DataFrame]): An iterator over Pandas DataFrames.
        Each DataFrame is a batch of data points.
        student_id_col (str): The name of the column containing student_id
        model_features (list[str]): A list of strings representing the names 
        of the features for our model
        explainer (shap.KernelExplainer): A KernelExplainer object used to compute
        shap values from our loaded model.
        mode (pd.Series): A Series containing values to impute missing values

    Returns:
        Iterator[pd.DataFrame]: An iterator over Pandas DataFrames. Each DataFrame
        contains the SHAP values for that partition of data.
    """

    for pdf in iterator:
        # Preserve student_id column
        student_ids_batch = pdf.loc[:, student_id_col]

        # Impute missing values and run shap values using just pdf features
        pdf_features = pdf[model_features].fillna(mode)
        shap_values = explainer.shap_values(pdf_features)

        # Create a DataFrame from the SHAP values
        shap_df = pd.DataFrame(shap_values, columns=model_features)

        # Reattach the student_id column to our SHAP values DataFrame
        shap_df[student_id_col] = student_ids_batch

        yield shap_df
