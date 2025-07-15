import logging
from .. import (
    bias_sections as base_bias_sections,
)

LOGGER = logging.getLogger(__name__)


def register_bias_sections(card, registry):
    base_bias_sections.register_bias_sections(card, registry)

    @registry.register("bias_groups_section")
    def bias_groups_section():
        """
        Returns bias groups for custom schools. These groups will
        be specified in aliases (part of custom configs) for each school.
        """
        intro = f"{card.format.indent_level(1)}- Our assessment for FNR Parity was conducted across the following student groups.\n"

        try:
            alias_dict = card.cfg.student_group_aliases
            assert isinstance(alias_dict, dict), (
                "student_group_aliases must be a dictionary"
            )

            group_labels = list(alias_dict.values())  # Extract the human-readable names
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
