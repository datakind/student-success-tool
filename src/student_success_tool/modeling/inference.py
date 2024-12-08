import numpy as np
import pandas as pd


def select_top_features_for_display(
    features: pd.DataFrame,
    unique_ids: pd.Series,
    predicted_probabilities: list[float],
    shap_values: pd.Series,
    n_features: int = 3,
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

    Returns:
        explainability dataframe for display
    """
    top_features_info = []

    for i, (unique_id, predicted_proba) in enumerate(
        zip(unique_ids, predicted_probabilities)
    ):
        instance_shap_values = shap_values[i]
        top_indices = np.argsort(-np.abs(instance_shap_values))[:n_features]
        top_features = features.columns[top_indices]
        top_shap_values = instance_shap_values[top_indices]

        for rank, (feature, shap_value) in enumerate(
            zip(top_features, top_shap_values), start=1
        ):
            top_features_info.append(
                {
                    "Student ID": unique_id,
                    "Support Score": predicted_proba,
                    "Top Indicators": feature,
                    "SHAP Value": shap_value,
                    "Rank": rank,
                }  # column names defined here https://app.asana.com/0/1206275396780585/1206834683873668/f
            )
    return pd.DataFrame(top_features_info)
