import os
import logging
import pandas as pd

from ..utils import utils

LOGGER = logging.getLogger(__name__)


def register_bias_sections(card, registry):
    """
    Registers the bias section of the model card with FNR group plots and
    subgroup disparity summaries. We focus on test data specifically for our reporting.

    This information is gathered from mlflow artifacts.
    """
    bias_levels = ["high", "moderate", "low"]
    bias_flags = {}
    group_disparities = {}

    def generate_description(
        group: str, subgroups: str, diff: str, stat_summary: str
    ) -> str:
        """
        Filters description for a given bias flag under 'bias_flags' directory in mlflow run
        artifacts. The FNR difference will be multiplied by 100 to represent a percentage and
        then rounded to an integer.

        Args:
            group: student group
            subgroups: student subgroups (e.g. "s1 vs. s2")
            diff: FNR (False Negative Rate) difference decimal (e.g. "0.1451")
            stat_summary: statistical summary with confidence interval & p-value information
        Returns:
            Description for a given bias flag
        """
        group_label = card.format.friendly_case(group)
        try:
            subgroup_1, subgroup_2 = [s.strip() for s in subgroups.split("vs")]
        except ValueError:
            LOGGER.warning(f"Could not parse subgroups for {group}: {subgroups}")
            return f"{card.format.bold('Could not parse subgroup comparison')}"

        sg1 = card.format.bold(card.format.italic(subgroup_1))
        sg2 = card.format.bold(card.format.italic(subgroup_2))
        percent_higher = card.format.bold(
            f"{int(round(float(diff) * 100))}% difference"
        )

        return f"- {sg1} students have a {percent_higher} in False Negative Rate (FNR) than {sg2} students. Statistical analysis indicates {stat_summary}."

    # Load bias flag CSVs and filter for test split
    for level in bias_levels:
        try:
            bias_path = f"bias_flags/{level}_bias_flags.csv"
            local_path = utils.download_artifact(
                run_id=card.run_id,
                local_folder=card.assets_folder,
                artifact_path=bias_path,
            )
            if not os.path.exists(local_path):
                LOGGER.warning(
                    f"{level} bias flags file does not exist. Bias evaluation likely has not run."
                )
                continue

            df = pd.read_csv(local_path)

            if df.empty or "split_name" not in df.columns:
                LOGGER.warning(
                    f"{level} bias flags file exists but has no data or missing 'split_name' column."
                )
                continue

            df = df[df["split_name"] == "test"]
            if df.empty:
                LOGGER.info(f"{level} bias flags file has no rows for test split.")
                continue

            # Group descriptions by group
            for _, row in df.iterrows():
                group = row["group"]
                desc = generate_description(
                    group,
                    row["subgroups"],
                    row["fnr_percentage_difference"],
                    row["type"],
                )
                group_disparities.setdefault(group, []).append(desc)
        except Exception as e:
            LOGGER.warning(
                f"Could not load {level} bias flags: [{type(e).__name__}] {str(e)}"
            )

    all_blocks = []

    for group_name, descriptions in group_disparities.items():
        normalized_name = group_name.lower().replace(" ", "_")
        plot_artifact_path = f"fnr_plots/test_{normalized_name}_fnr.png"

        try:
            plot_md = utils.download_artifact(
                run_id=card.run_id,
                local_folder=card.assets_folder,
                artifact_path=plot_artifact_path,
                description=f"False Negative Parity Rate for {group_name} on Test Data",
            )
        except Exception as e:
            LOGGER.warning(f"Could not load plot for {group_name}: {str(e)}")
            plot_md = (
                f"{card.format.bold(f'Unable to retrieve plot for {group_name}')}\n"
            )

        header = (
            f"{card.format.header_level(5)}{card.format.friendly_case(group_name)}\n\n"
        )
        text_block = "\n\n".join(descriptions)
        all_blocks.append(header + text_block + "\n\n" + plot_md)

    @registry.register("bias_groups_section")
    def bias_groups_section():
        """
        Returns just a filler text that can be used across any platform.
        """
        intro = f"{card.format.indent_level(1)}- Our assessment for FNR Parity was conducted across several student groups.\n"
        return intro

    @registry.register("bias_summary_section")
    def bias_summary_section():
        """
        Returns a markdown string containing the bias summary section of the model card.
        """
        if not all_blocks:
            LOGGER.warning(
                "No disparities found or bias evaluation artifacts missing. Skipping bias summary section."
            )
            return f"{card.format.italic('No statistically significant disparities were found on test dataset across groups.')}"

        section_header = (
            f"{card.format.header_level(4)}Disparities by Student Group\n\n"
        )
        return section_header + "\n\n".join(all_blocks)
