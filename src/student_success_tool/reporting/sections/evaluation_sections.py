import logging
import typing as t
import pandas as pd
import collections

from ..utils import utils

LOGGER = logging.getLogger(__name__)


def register_evaluation_sections(card, registry):
    """
    Register evaluation metrics for each group. These metrics include both performance and bias.
    We assume all necessary formatting in terms of rows and columns of the table is done in the
    mlflow artifact.
    """
    performance_section = [f"{card.format.header_level(4)}Model Performance\n"]
    split_artifacts = utils.list_paths_in_directory(
        run_id=card.run_id, directory="metrics"
    )

    evaluation_sections = [
        f"{card.format.header_level(4)}Evaluation Metrics by Student Group\n"
    ]
    group_eval_artifacts = utils.list_paths_in_directory(
        run_id=card.run_id, directory="group_metrics"
    )

    def make_metric_table(path: str, title: str) -> t.Callable[[], str]:
        """
        This method is used for dynamic section registration.
        Later, the registry will render all of these functions to create
        tables.

        Args:
            path: Artifact path to the csv file containing the evaluation metrics.
            title: Title of the group.

        Returns:
            A function that returns a markdown table of the evaluation metrics for a group.
        """

        def metric_table():
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

                return f"{card.format.header_level(5)}{title}\n\n" + "\n".join(
                    [headers, separator] + rows
                )

            except Exception as e:
                LOGGER.warning(
                    f"Could not load evaluation metrics for {title}: {str(e)}"
                )
                return f"{card.format.bold(f'{title} Metrics')}\n\nCould not load data."

        return metric_table

    # Evaluation by Group
    group_parts = collections.defaultdict(dict)

    # Group bias and performance parts for each group
    for csv_path in group_eval_artifacts:
        if csv_path.startswith("group_metrics/bias_test_") and csv_path.endswith(
            ".csv"
        ):
            group_name = csv_path.replace("group_metrics/bias_test_", "").replace(
                "_metrics.csv", ""
            )
            group_parts[group_name]["bias"] = csv_path

        if csv_path.startswith("group_metrics/perf_test_") and csv_path.endswith(
            ".csv"
        ):
            group_name = csv_path.replace("group_metrics/perf_test_", "").replace(
                "_metrics.csv", ""
            )
            group_parts[group_name]["perf"] = csv_path

    # Render both tables under the same group title without labeling them separately
    for group_name, parts in group_parts.items():
        if "bias" not in parts or "perf" not in parts:
            continue

        section_text = []

        # Bias Evaluation Table
        bias_title = f"{card.format.friendly_case(group_name)} Bias Metrics"
        bias_table_func = make_metric_table(parts["bias"], bias_title)
        registry.register(f"metric_table_{group_name}_bias")(bias_table_func)
        section_text.append(bias_table_func())

        # Performance Evaluation Table
        perf_title = f"{card.format.friendly_case(group_name)} Performance Metrics"
        perf_table_func = make_metric_table(parts["perf"], perf_title)
        registry.register(f"metric_table_{group_name}_perf")(perf_table_func)
        section_text.append(perf_table_func())

        evaluation_sections.append("\n\n".join(section_text))

    # Performance Across Splits
    for csv_path in split_artifacts:
        if csv_path.startswith("metrics/") and csv_path.endswith("_splits.csv"):
            title = "Performance across Splits"
            table_func = make_metric_table(csv_path, title)
            registry.register("performance_by_splits_metric_table")(table_func)
            performance_section.append(table_func())

    @registry.register("evaluation_by_group_section")
    def evaluation_section():
        """
        Returns the evaluation metrics section for the model card.
        """
        if not evaluation_sections:
            return f"{card.format.bold('No group evaluation metrics available')}."
        return "\n\n".join(evaluation_sections)

    @registry.register("performance_by_splits_section")
    def performance_by_splits_section():
        """
        Returns the performance metrics section by splits for the model card.
        """
        if not performance_section:
            return f"{card.format.bold('No performance metrics available')}."
        return "\n\n".join(performance_section)
