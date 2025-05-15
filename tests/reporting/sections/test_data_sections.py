import pytest
from unittest.mock import MagicMock
import pandas as pd
from student_success_tool.reporting.sections.registry import SectionRegistry
from student_success_tool.reporting.sections.data_sections import register_data_sections


@pytest.fixture
def mock_card_with_splits():
    card = MagicMock()
    card.cfg.preprocessing.splits = {"train": 0.7, "validate": 0.15, "test": 0.15}
    card.modeling_data = pd.DataFrame({"x": range(100)})
    card.format.bold.side_effect = lambda x: f"**{x}**"
    return card


@pytest.fixture
def mock_card_no_splits():
    card = MagicMock()
    card.cfg.preprocessing.splits = {}
    card.modeling_data = pd.DataFrame({"x": range(100)})
    card.format.bold.side_effect = lambda x: f"**{x}**"
    return card


def test_data_split_table_renders_correctly(mock_card_with_splits):
    registry = SectionRegistry()
    register_data_sections(mock_card_with_splits, registry)

    rendered = registry.render_all()
    output = rendered["data_split_table"]
    assert "| Split      | Students | Percentage |" in output
    assert "| Training   | 70       | 70%       |" in output
    assert "| Validation | 15       | 15%       |" in output
    assert "| Test       | 15       | 15%       |" in output


def test_data_split_table_missing_splits(mock_card_no_splits):
    registry = SectionRegistry()
    register_data_sections(mock_card_no_splits, registry)

    rendered = registry.render_all()
    assert rendered["data_split_table"] == "**Could not parse data split**"
