import logging
import pandas as pd

from ..utils import utils

LOGGER = logging.getLogger(__name__)


def register_data_sections(card, registry):
    """
    Registers sections related to data split and feature tables.
    It does not contain sections relevant to performance or bias. These tables
    are gathered from a combination of config.toml and mlflow artifacts.
    """

    @registry.register("data_split_table")
    def data_split():
        """
        Produces a markdown table of the data split.
        """
        total_students = len(card.modeling_data)
        splits = card.cfg.preprocessing.splits
        if splits:
            labels = {"train": "Training", "validate": "Validation", "test": "Test"}

            rows = [
                f"| {labels[k]:<10} | {round(total_students * splits[k]):<8} | {int(splits[k] * 100)}%       |"
                for k in ["train", "validate", "test"]
            ]

            return "\n".join(
                [
                    "| Split      | Students | Percentage |",
                    "|------------|----------|------------|",
                    *rows,
                ]
            )
        else:
            LOGGER.warning(
                "Unable to produce data split table. No splits found in config."
            )
            return f"{card.format.bold('Could not parse data split')}"

    @registry.register("selected_features_ranked_by_shap")
    def selected_features_ranked_by_shap():
        """
        Produces a markdown table of the selected features ranked by average SHAP
        magnitude.
        """
        feature_artifact_path = "selected_features/ranked_selected_features.csv"

        try:
            local_path = utils.download_artifact(
                run_id=card.run_id,
                local_folder=card.assets_folder,
                artifact_path=feature_artifact_path,
            )
            df = pd.read_csv(local_path)

            # Build markdown table
            headers = "| " + " | ".join(df.columns) + " |"
            separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
            rows = [
                "| " + " | ".join(str(val).replace("\n", "<br>") for val in row) + " |"
                for row in df.values
            ]

            title = f"{card.format.header_level(4)}Selected Features\n"
            subtitle = f"{card.format.header_level(5)}Full List of Selected Features Ranked by Importance\n\n"

            table_markdown = "\n".join([headers, separator] + rows)

            return f"{title}\n{subtitle}{table_markdown}"

        except Exception as e:
            LOGGER.warning(f"Could not load feature importance table: {str(e)}")
            return f"{card.format.bold('Selected Features Ranked by Importance')}\n\nCould not load data."
