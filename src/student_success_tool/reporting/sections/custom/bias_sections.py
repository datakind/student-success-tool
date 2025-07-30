import os
import logging
import pandas as pd

from .. import bias_sections as base_bias_sections
from ...utils import utils

LOGGER = logging.getLogger(__name__)


def resolve_student_group_label(card, group_key):
    try:
        alias_dict = getattr(card.cfg, "student_group_aliases", {})
        if not isinstance(alias_dict, dict):
            LOGGER.warning(
                f"[resolve_student_group_label] 'student_group_aliases' is not a dict: {type(alias_dict).__name__}"
            )
            return card.format.friendly_case(group_key)
        return alias_dict.get(group_key, card.format.friendly_case(group_key))
    except Exception as e:
        LOGGER.warning(
            f"[resolve_student_group_label] Error resolving alias for '{group_key}': {e}"
        )
        return card.format.friendly_case(group_key)


def register_bias_sections(card, registry):
    # Register base sections
    base_bias_sections.register_bias_sections(card, registry)

    bias_levels = ["high", "moderate", "low"]
    group_disparities = {}

    def generate_description(group, subgroups, diff, stat_summary):
        group_label = resolve_student_group_label(card, group)

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

        return (
            f"- {sg1} students have a {percent_higher} in False Negative Rate (FNR) than {sg2} students. "
            f"Statistical analysis indicates {stat_summary}."
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

    # Build bias summary content using student group aliases
    all_blocks = []

    for group_name, descriptions in group_disparities.items():
        label = resolve_student_group_label(card, group_name)
        normalized_name = group_name.lower().replace(" ", "_")

        plot_artifact_path = f"fnr_plots/test_{normalized_name}_fnr.png"

        try:
            plot_md = utils.download_artifact(
                run_id=card.run_id,
                local_folder=card.assets_folder,
                artifact_path=plot_artifact_path,
                description=f"False Negative Parity Rate for {label} on Test Data",
            )
        except Exception as e:
            LOGGER.warning(f"Could not load plot for {group_name}: {str(e)}")
            plot_md = f"{card.format.bold(f'Unable to retrieve plot for {label}')}\n"

        header = f"{card.format.header_level(5)}{label}\n\n"
        text_block = "\n\n".join(descriptions)
        all_blocks.append(header + text_block + "\n\n" + plot_md)

    @registry.register("bias_groups_section")
    def bias_groups_section():
        """
        Returns bias groups for custom schools using aliases.
        """
        intro = f"{card.format.indent_level(1)}- Our assessment for FNR Parity was conducted across the following student groups.\n"

        try:
            alias_dict = card.cfg.student_group_aliases
            assert isinstance(alias_dict, dict), (
                "student_group_aliases must be a dictionary"
            )

            group_labels = list(alias_dict.values())
            nested = [
                f"{card.format.indent_level(2)}- {card.format.friendly_case(label)}\n"
                for label in group_labels
            ]
            return intro + "".join(nested)

        except (AttributeError, AssertionError, TypeError) as e:
            LOGGER.warning(
                f"[bias_groups_section] Failed to extract student groups: {e}"
            )
            fallback = (
                f"{card.format.indent_level(2)}- Unable to extract student groups\n"
            )
            return intro + fallback

    @registry.register("bias_summary_section")
    def bias_summary_section():
        """
        Returns a markdown string containing the bias summary section of the model card.
        """
        if not all_blocks:
            LOGGER.warning(
                "No disparities found or bias evaluation artifacts missing. Skipping bias summary section."
            )
            return card.format.italic(
                "No statistically significant disparities were found on test dataset across groups."
            )

        section_header = (
            f"{card.format.header_level(4)}Disparities by Student Group\n\n"
        )
        return section_header + "\n\n".join(all_blocks)
