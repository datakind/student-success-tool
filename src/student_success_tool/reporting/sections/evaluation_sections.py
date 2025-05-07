import logging
import typing as t
import pandas as pd

from ..utils import utils

LOGGER = logging.getLogger(__name__)


def register_evaluation_sections(card, registry):
    """
    Register evaluation metrics for each group. These metrics include both performance and bias.
    We assume all necessary formatting in terms of rows and columns of the table is done in the
    mlflow artifact.
    """
    evaluation_sections = [
        f"{card.format.header_level(4)}Evaluation Metrics by Student Group\n"
    ]
    group_eval_artifacts = utils.list_paths_in_directory(
        run_id=card.run_id, directory="group_metrics"
    )

    def make_group_metric_table(path: str, title: str) -> t.Callable[[], str]:
        """
        This method is used for dynamic section registration based on the number
        of student groups. Later, the registry will render all of these functions to create
        tables for all of our student groups.

        Args:
            path: Artifact path to the csv file containing the evaluation metrics.
            title: Title of the group.

        Returns:
            A function that returns a markdown table of the evaluation metrics for a group.
        """

        def group_metric_table():
            try:
                local_path = utils.download_artifact(
                    run_id=card.run_id,
                    local_folder=card.assets_folder,
                    artifact_path=path,
                )
                df = pd.read_csv(local_path)

                # Build markdown table
                headers = "| " + " | ".join(df.columns) + " |"
                separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
                rows = [
                    "| " + " | ".join(str(val) for val in row) + " |"
                    for row in df.values
                ]

                return f"{card.format.bold(f'{title} Metrics')}\n\n" + "\n".join(
                    [headers, separator] + rows
                )

            except Exception as e:
                LOGGER.warning(
                    f"Could not load evaluation metrics for {title}: {str(e)}"
                )
                return f"{card.format.bold(f'{title} Metrics')}\n\nCould not load data."

        return group_metric_table

    for csv_path in group_eval_artifacts:
        if csv_path.startswith("group_metrics/test_") and csv_path.endswith(
            "_metrics.csv"
        ):
            group_name = csv_path.replace("group_metrics/test_", "").replace(
                "_metrics.csv", ""
            )

            group_title = card.format.friendly_case(group_name)

            group_table_func = make_group_metric_table(csv_path, group_title)
            registry.register(f"group_metric_table_{group_name}")(group_table_func)
            evaluation_sections.append(group_table_func())

    @registry.register("evaluation_by_group_section")
    def evaluation_section():
        """
        Returns the evaluation metrics section for the model card.
        """
        if not evaluation_sections:
            return f"{card.format.bold('No group evaluation metrics available')}."
        return "\n\n".join(evaluation_sections)
