from .. import (
    bias_sections as base_bias_sections,
)

import logging

LOGGER = logging.getLogger(__name__)


def register_bias_sections(card, registry):
    base_bias_sections.register_bias_sections(card, registry)

    LOGGER.info("Starting custom attribute section creation")

    @registry.register("bias_groups_section")
    def bias_groups_section():
        """
        Returns bias groups for PDP. These groups will be static across
        all institutions in PDP.
        """
        intro = f"{card.format.indent_level(1)}- Our assessment for FNR Parity was conducted across the following student groups.\n"
        groups = [
            "Ethnicity",
            "First Generation Status",
            "Gender",
            "Race",
            "Student Age",
        ]
        nested = [f"{card.format.indent_level(2)}- {group}\n" for group in groups]
        return intro + "".join(nested)
