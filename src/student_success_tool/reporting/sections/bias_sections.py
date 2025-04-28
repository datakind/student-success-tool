import pandas as pd
from ... import utils

def register_bias_sections(card, registry):
    bias_sections = []

    try:
        bias_csvs = card.list_bias_group_artifacts()
    except Exception as e:
        bias_csvs = []
        LOGGER.warning(f"Could not list bias artifacts: {str(e)}")

    for csv_path in bias_csvs:
        group_name = csv_path.replace("bias_", "").replace(".csv", "").replace("_", " ").title()
        short_group_key = csv_path.replace("bias_", "").replace(".csv", "")

        @registry.register(f"bias_table_{short_group_key}")
        def bias_table(group=group_name, path=csv_path):
            try:
                local_path = utils.safe_mlflow_download_artifacts(
                    run_id=card.run_id,
                    artifact_path=f"evaluation/{path}",
                    dst_path="artifacts"
                )
                df = pd.read_csv(local_path)

                # Build markdown table
                headers = "| " + " | ".join(df.columns) + " |"
                separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
                rows = ["| " + " | ".join(str(val) for val in row) + " |" for row in df.values]
                markdown_table = "\n".join([headers, separator] + rows)

                return f"### Bias Evaluation: {group}\n\n{markdown_table}\n"

            except Exception as e:
                return f"⚠️ Could not load bias group table for {group}: {str(e)}\n"

        bias_sections.append(f"{{bias_table_{short_group_key}}}")

    @registry.register("evaluation_by_group_section")
    def group_evaluation_section():
        if not bias_sections:
            return "No bias evaluations were conducted or available for this model."
        return "\n\n".join(bias_sections)
