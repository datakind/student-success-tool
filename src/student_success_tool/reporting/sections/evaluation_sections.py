import logging
import pandas as pd
from ... import utils

LOGGER = logging.getLogger(__name__)

def register_evaluation_sections(card, registry):
    evaluation_sections = []
    group_eval_artifacts = utils.list_artifacts(run_id=card.run_id, folder="group_metrics")

    for csv_path in group_eval_artifacts:
        group_name = csv_path.replace("group_metrics/test_", "").replace("_metrics.csv", "")  # "ethnicity", "gender", etc.
        group_title = group_name.replace("_", " ").title()  # clean human-friendly name

        @registry.register(f"group_metric_table_{group_name}")
        def group_metric_table(path=csv_path, title=group_title):
            try:
                local_path = utils.safe_mlflow_download_artifacts(
                    run_id=card.run_id,
                    artifact_path=f"group_metrics/{path}",
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

        evaluation_sections.append(f"{{group_metric_table_{group_name}}}")

    @registry.register("evaluation_by_group_section")
    def evaluation_section():
        if not evaluation_sections:
            return "No group evaluation metrics available."
        return "\n\n".join(evaluation_sections)