import logging
import typing as t
import pandas as pd
import collections
from ...utils import utils

LOGGER = logging.getLogger(__name__)


def register_evaluation_sections(card, registry):
    """
    Register evaluation metrics for each group. These metrics include both performance and bias.
    We assume all necessary formatting in terms of rows and columns of the table is done in the
    mlflow artifact.
    """
    performance_section = [f"{card.format.header_level(4)}Model Performance\n"]
    split_artifacts = utils.list_paths_in_directory(run_id=card.run_id, directory="metrics")

    evaluation_sections = [f"{card.format.header_level(4)}Evaluation Metrics by Student Group\n"]
    group_eval_artifacts = utils.list_paths_in_directory(run_id=card.run_id, directory="group_metrics")

    # Try to load group aliases from config
    try:
        alias_dict = card.cfg.student_group_aliases
        assert isinstance(alias_dict, dict)
    except Exception:
        alias_dict = {}

    def resolve_group_label(group_name: str) -> str:
        """Convert internal group name to user-friendly alias if available."""
        return alias_dict.get(group_name, card.format.friendly_case(group_name))

    def make_metric_table(path: str, title: str) -> t.Callable[[], str]:
        def metric_table():
            try:
                local_path = utils.download_artifact(
                    run_id=card.run_id,
                    local_folder=card.assets_folder,
                    artifact_path=path,
                )
                df = pd.read_csv(local_path)

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
                LOGGER.warning(f"Could not load evaluation metrics for {title}: {str(e)}")
                return f"{card.format.bold(f'{title} Metrics')}\n\nCould not load data."

        return metric_table

    # Group bias and performance paths
    group_parts = collections.defaultdict(dict)

    for csv_path in group_eval_artifacts:
        if csv_path.startswith("group_metrics/bias_test_") and csv_path.endswith(".csv"):
            group = csv_path.removeprefix("group_metrics/bias_test_").removesuffix("_metrics.csv")
            group_parts[group]["bias"] = csv_path
        elif csv_path.startswith("group_metrics/perf_test_") and csv_path.endswith(".csv"):
            group = csv_path.removeprefix("group_metrics/perf_test_").removesuffix("_metrics.csv")
            group_parts[group]["perf"] = csv_path

    # Register group-level evaluation tables
    for group, parts in group_parts.items():
        if "bias" not in parts or "perf" not in parts:
            continue

        label = resolve_group_label(group)
        section_text = []

        bias_title = f"{label} Bias Metrics"
        perf_title = f"{label} Performance Metrics"

        bias_table_func = make_metric_table(parts["bias"], bias_title)
        perf_table_func = make_metric_table(parts["perf"], perf_title)

        registry.register(f"metric_table_{group}_bias")(bias_table_func)
        registry.register(f"metric_table_{group}_perf")(perf_table_func)

        section_text.append(bias_table_func())
        section_text.append(perf_table_func())

        evaluation_sections.append("\n\n".join(section_text))

    # Performance across splits
    for csv_path in split_artifacts:
        if csv_path.startswith("metrics/") and csv_path.endswith("_splits.csv"):
            title = "Performance across Splits"
            table_func = make_metric_table(csv_path, title)
            registry.register("performance_by_splits_metric_table")(table_func)
            performance_section.append(table_func())

    @registry.register("evaluation_by_group_section")
    def evaluation_section():
        if not evaluation_sections:
            return f"{card.format.bold('No group evaluation metrics available')}."
        return "\n\n".join(evaluation_sections)

    @registry.register("performance_by_splits_section")
    def performance_by_splits_section():
        if not performance_section:
            return f"{card.format.bold('No performance metrics available')}."
        return "\n\n".join(performance_section)
