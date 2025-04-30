import os
import logging
import pandas as pd
from ..utils import utils

LOGGER = logging.getLogger(__name__)

def register_bias_sections(card, registry):
    bias_levels = ['high', 'moderate', 'low']
    bias_flags = {}
    all_blocks = []

    def generate_description(group, subgroups, diff, stat_summary):
        group_label = group.replace("_", " ").title()
        try:
            subgroup_1, subgroup_2 = [s.strip() for s in subgroups.split("vs")]
        except ValueError:
            return "Could not parse subgroup comparison."
        
        sg1 = card.format.bold(card.format.italic(subgroup_1))
        sg2 = card.format.bold(card.format.italic(subgroup_2))
        percent_higher = card.format.bold(f"{int(round(float(diff) * 100))}% higher")

        return (
            f"{card.format.indent_level(2)}- {sg1} students have a {percent_higher} False Negative Rate (FNR) than {sg2} students. "
            f"{card.format.indent_level(2)}- Statistical analysis indicates: {stat_summary}."
        )


    # Load bias flag CSVs and filter for test split
    for level in bias_levels:
        try:
            bias_path = f"bias_flags/{level}_bias_flags.csv"
            local_path = utils.download_artifact(
                run_id=card.run_id,
                local_folder=card.assets_folder,
                artifact_path=bias_path,
            )
            df = pd.read_csv(local_path)

            if 'split_name' in df.columns:
                df = df[df['split_name'] == 'test']

            bias_flags[level] = df
        except Exception as e:
            LOGGER.warning(f"Could not load {level} bias flags: {str(e)}")
            bias_flags[level] = pd.DataFrame()

    # Build markdown blocks
    for level in bias_levels:
        df = bias_flags.get(level, pd.DataFrame())

        for _, row in df.iterrows():
            group_name = row['group']
            subgroups = row['subgroups']
            fnpr_diff = row['fnpr_percentage_difference']
            stat_type = row['type']

            description = generate_description(group_name, subgroups, fnpr_diff, stat_type)

            # Find plot
            normalized_name = group_name.lower().replace(' ', '_')
            plot_artifact_path = f"fnr_plots/test_{normalized_name}_fnr.png"

            try:
                plot_md = utils.safe_mlflow_download_artifacts(
                    run_id=card.run_id,
                    local_folder=card.assets_folder,
                    artifact_path=plot_artifact_path,
                    description=f"False Negative Parity Rate for {group_name} on Test Data",
                    width=450
                )
            except Exception as e:
                LOGGER.warning(f"Could not load plot for {group_name}: {str(e)}")
                plot_md = "*Plot not available.*\n\n"

            block = f"{card.format.indent_level(1)}{group_name.replace('_', ' ').title()}\n\n"
            block += f"{description}\n\n"
            block += plot_md
            all_blocks.append(block)

    @registry.register("bias_summary_section")
    def bias_summary_section():
        if not all_blocks:
            return "No flagged bias detected in test group evaluations."

        section_header = f"{card.format.bold('Disparities by Student Group')}\n\n"
        
        return section_header + "\n\n".join(all_blocks)
