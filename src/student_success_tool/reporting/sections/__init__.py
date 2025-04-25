from .metrics import register_metric_sections
from .attributes import register_attribute_sections
from .tables import register_table_sections

def register_sections(card, registry):
    register_metric_sections(card, registry)
    register_attribute_sections(card, registry)
    register_table_sections(card, registry)