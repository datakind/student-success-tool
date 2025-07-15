from .. import (
    register_metric_sections,
    register_data_sections,
)
from .attribute_sections import (
    register_attribute_sections as register_custom_attribute_sections,
)
from .bias_sections import (
    register_bias_sections as register_custom_bias_sections,
)
from .evaluation_sections import (
    register_evaluation_sections as register_custom_evaluation_sections,
)


def register_sections(card, registry):
    # Reuse base sections
    register_metric_sections(card, registry)
    register_data_sections(card, registry)

    # Override sections
    register_custom_attribute_sections(card, registry)
    register_custom_bias_sections(card, registry)
    register_custom_evaluation_sections(card, registry)
