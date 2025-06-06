import pytest
from unittest.mock import patch
from student_success_tool.reporting.model_card.base import ModelCard
from student_success_tool.reporting.model_card.pdp import PDPModelCard
from student_success_tool.configs.pdp import PDPProjectConfig
from student_success_tool.reporting.sections import registry

# Dummy config for base ModelCard
class DummyModelingConfig:
    def __init__(self):
        self.feature_selection = {
            "collinear_threshold": 10.0,
            "low_variance_threshold": 0.0,
            "incomplete_threshold": 0.5,
        }
        self.training = {
            "primary_metric": "log_loss",
            "timeout_minutes": 10,
        }

class DummyPreprocessingConfig:
    def __init__(self):
        self.selection = {"student_criteria": {"status": "active"}}
        self.checkpoint = {"name": "credit", "params": {"min_num_credits": 30}}
        self.target = {
            "name": "retention",
            "type_": "retention",
            "max_academic_year": "2025-26",
            "intensity_time_limits": {"FULL-TIME": {"years": 2}},
            "max_term_rank": 6,
            "years_to_degree_col": "years_to_grad",
            "min_num_credits": 24
        }

class DummyConfig:
    def __init__(self):
        self.institution_name = "Test University"
        self.modeling = DummyModelingConfig()
        self.preprocessing = DummyPreprocessingConfig()

# Valid PDPProjectConfig
def make_pdp_config() -> PDPProjectConfig:
    return PDPProjectConfig(
        institution_id="inst_id",
        institution_name="Inst Name",
        model={"experiment_id": "exp123", "run_id": "abc", "framework": "sklearn"},
        datasets={
            "bronze": {
                "raw_course": {"file_path": "dummy.csv"},
                "raw_cohort": {"file_path": "dummy.csv"},
            },
            "silver": {
                "preprocessed": {"table_path": "dummy"},
                "modeling": {"table_path": "dummy"},
            },
            "gold": {"advisor_output": {"table_path": "dummy"}},
        },
        preprocessing={
            "selection": {"student_criteria": {"status": "active"}},
            "checkpoint": {
                "name": "checkpoint_nth",
                "type_": "nth",
                "n": 4,
                "params": {"min_num_credits": 30},
            },
            "target": {
                "name": "retention",
                "type_": "retention",
                "max_academic_year": "2025-26",
                "intensity_time_limits": {"FULL-TIME": {"years": 2}},
                "max_term_rank": 6,
                "years_to_degree_col": "years_to_grad",
                "min_num_credits": 24
            },
            "features": {
                "min_passing_grade": 1.0,
                "min_num_credits_full_time": 12,
                "course_level_pattern": "abc",
                "key_course_subject_areas": ["24"],
                "key_course_ids": ["ENGL101"],
            },
        },
        modeling={
            "feature_selection": {
                "collinear_threshold": 10.0,
                "low_variance_threshold": 0.0,
                "incomplete_threshold": 0.5,
            },
            "training": {"primary_metric": "log_loss", "timeout_minutes": 10}
        },
        split_col=None,
    )

# Main test
@patch("student_success_tool.reporting.model_card.base.ModelCard.load_model")
@patch("student_success_tool.reporting.model_card.base.ModelCard.extract_training_data")
@patch("student_success_tool.reporting.model_card.base.ModelCard.find_model_version")
@pytest.mark.parametrize("card_class", [ModelCard, PDPModelCard])
def test_template_placeholders_are_in_context(
    mock_find_version, mock_extract_data, mock_load_model, card_class
):
    if card_class.__name__ == "PDPModelCard":
        config = make_pdp_config()
    else:
        config = DummyConfig()

    card = card_class(config=config, catalog="demo", model_name="test_model")

    # Patch side effects for load_model
    mock_load_model.side_effect = lambda: (
        setattr(card, "run_id", "dummy_run_id")
        or setattr(card, "experiment_id", "dummy_experiment_id")
        or setattr(card, "model", object())
    )

    # Minimal context & registry population
    card.load_model()
    card.find_model_version()
    card.extract_training_data()
    card._register_sections()

    # Build context for template check
    card.context.update(card.get_basic_context())
    card.context.update(card.section_registry.render_all())

    # Validate placeholders exist in context
    template_path = card.template_path
    with open(template_path, "r") as f:
        template = f.read()

    import re
    matches = set(re.findall(r"{([\w_]+)}", template))
    missing = matches - card.context.keys()
    assert not missing, f"Missing context keys for template: {missing}"
