import pytest
import pandas as pd
from unittest.mock import MagicMock
from student_success_tool.reporting.sections.registry import SectionRegistry
from student_success_tool.reporting.sections.metric_sections import (
    register_metric_sections,
)


@pytest.fixture
def mock_card():
    card = MagicMock()
    card.cfg.modeling.training.primary_metric = "recall"
    card.context = {"num_runs_in_experiment": 12}
    card.training_data = pd.DataFrame(
        {"feature": [1, 2], "sample_weight_1": [0.2, 0.8]}
    )
    card.format.indent_level.side_effect = lambda level: "  " * level
    return card


def test_register_metric_sections_primary_metric(mock_card):
    registry = SectionRegistry()
    register_metric_sections(mock_card, registry)

    rendered = registry.render_all()
    assert "recall" in rendered["primary_metric_section"]
    assert (
        "- Our primary metric for training was recall"
        in rendered["primary_metric_section"]
    )


def test_register_metric_sections_sample_weight_used(mock_card):
    registry = SectionRegistry()
    register_metric_sections(mock_card, registry)

    rendered = registry.render_all()
    assert "Sample weights were used" in rendered["sample_weight_section"]
    assert "MLOps pipeline" in rendered["sample_weight_section"]


def test_register_metric_sections_sample_weight_not_used(mock_card):
    mock_card.training_data = pd.DataFrame({"feature": [1, 2]})  # no sample_weight col
    registry = SectionRegistry()
    register_metric_sections(mock_card, registry)

    rendered = registry.render_all()
    assert "Sample weights were used" not in rendered["sample_weight_section"]
    assert "MLOps pipeline" in rendered["sample_weight_section"]
