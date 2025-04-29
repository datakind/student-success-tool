import logging
import pandas as pd
from ..utils import utils

LOGGER = logging.getLogger(__name__)

def register_evaluation_sections(card, registry):
    evaluation_sections = []
    group_eval_artifacts = utils.list_paths_in_directory(run_id=card.run_id, directory='group_metrics')

    def make_group_metric_table(path, title):
        def group_metric_table():
            try:
                local_path = utils.safe_mlflow_download_artifacts(
                    run_id=card.run_id,
                    artifact_path=path,
                    dst_path="artifacts"
                )
                df = pd.read_csv(local_path)

                # Build markdown table
                headers = "| " + " | ".join(df.columns) + " |"
                separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
                rows = ["| " + " | ".join(str(val) for val in row) + " |" for row in df.values]

                return f"### Evaluation Metrics for {title}\n\n" + "\n".join([headers, separator] + rows)

            except Exception as e:
                LOGGER.warning(f"Could not load evaluation metrics for {title}: {str(e)}")
                return f"### Evaluation Metrics for {title}\n\nCould not load data."
        return group_metric_table

    for csv_path in group_eval_artifacts:
        if csv_path.startswith("group_metrics/test_") and csv_path.endswith("_metrics.csv"):
            group_name = csv_path.replace("group_metrics/test_", "").replace("_metrics.csv", "")
            group_title = group_name.replace("_", " ").title()

            group_table_func = make_group_metric_table(csv_path, group_title)
            registry.register(f"group_metric_table_{group_name}")(group_table_func)
            evaluation_sections.append(group_table_func())

    @registry.register("evaluation_by_group_section")
    def evaluation_section():
        if not evaluation_sections:
            return "No group evaluation metrics available."
        return "\n\n".join(evaluation_sections)
