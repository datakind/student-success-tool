from .metric_sections import register_metric_sections
from .attribute_sections import register_attribute_sections
from .data_sections import register_data_sections
from .evaluation_sections import register_evaluation_sections
from .bias_sections import register_bias_sections


def register_sections(card, registry):
    register_metric_sections(card, registry)
    register_attribute_sections(card, registry)
    register_data_sections(card, registry)
    register_evaluation_sections(card, registry)
    register_bias_sections(card, registry)
