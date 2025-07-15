import pytest
import pandas as pd
import re
from unittest.mock import patch
from student_success_tool.reporting.model_card.base import ModelCard
from student_success_tool.reporting.model_card.pdp import PDPModelCard
from student_success_tool.configs.pdp import PDPProjectConfig


class DummyTrainingConfig:
    def __init__(self):
        self.primary_metric = "log_loss"
        self.timeout_minutes = 10


class DummyModelingConfig:
    def __init__(self):
        self.feature_selection = {
            "collinear_threshold": 10.0,
            "low_variance_threshold": 0.0,
            "incomplete_threshold": 0.5,
        }
        self.training = DummyTrainingConfig()


class DummyTargetConfig:
    def __init__(self):
        self.name = "retention"
        self.type_ = "retention"
        self.max_academic_year = "2025-26"
        self.intensity_time_limits = {"FULL-TIME": {"years": 2}}
        self.max_term_rank = 6
        self.years_to_degree_col = "years_to_grad"
        self.min_num_credits = 24


class DummyCheckpointConfig:
    def __init__(self):
        self.name = "checkpoint_nth"
        self.type_ = "nth"
        self.n = 4
        self.params = {"min_num_credits": 30}


class DummySelectionConfig:
    def __init__(self):
        self.student_criteria = {"status": "active"}


class DummyFeaturesConfig:
    def __init__(self):
        self.min_passing_grade = 1.0
        self.min_num_credits_full_time = 12
        self.course_level_pattern = "abc"
        self.key_course_subject_areas = ["24"]
        self.key_course_ids = ["ENGL101"]


class DummyPreprocessingConfig:
    def __init__(self):
        self.selection = DummySelectionConfig()
        self.checkpoint = DummyCheckpointConfig()
        self.target = DummyTargetConfig()
        self.features = DummyFeaturesConfig()


# Dummy config for base ModelCard
class DummyConfig:
    def __init__(self):
        self.institution_id = "test_uni"
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
                "min_num_credits": 24,
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
            "training": {"primary_metric": "log_loss", "timeout_minutes": 10},
        },
        split_col=None,
    )


@pytest.mark.parametrize("card_class", [ModelCard, PDPModelCard])
@patch("student_success_tool.reporting.sections.registry.SectionRegistry.render_all")
@patch("student_success_tool.reporting.model_card.base.ModelCard.collect_metadata")
@patch("student_success_tool.reporting.model_card.base.ModelCard.load_model")
@patch("student_success_tool.reporting.model_card.base.ModelCard.extract_training_data")
@patch("student_success_tool.reporting.model_card.base.ModelCard.find_model_version")
def test_template_placeholders_are_in_context(
    mock_find_version,
    mock_extract_data,
    mock_load_model,
    mock_collect_metadata,
    mock_render_all,
    card_class,
):
    if card_class.__name__ == "PDPModelCard":
        config = make_pdp_config()
    else:
        config = DummyConfig()

    card = card_class(config=config, catalog="demo", model_name="test_model")

    mock_load_model.side_effect = lambda: (
        setattr(card, "run_id", "dummy_run_id")
        or setattr(card, "experiment_id", "dummy_experiment_id")
        or setattr(card, "model", object())
        or setattr(card, "training_data", pd.DataFrame(columns=["sample_weight"]))
        or setattr(card, "modeling_data", pd.DataFrame({"student_id": []}))
    )

    mock_collect_metadata.side_effect = lambda: card.context.update(
        {
            "model_version": "12",
            "artifact_path": "dummy/path",
            "training_dataset_size": 100,
            "number_of_features": 20,
            "feature_importances_by_shap_plot": "![shap](shap.png)",
            "test_confusion_matrix": "confusion_matrix.png",
            "test_roc_curve": "roc_curve.png",
            "test_calibration_curve": "calibration_curve.png",
            "test_histogram": "histogram.png",
            "model_comparison_plot": "comparison.png",
            "collinearity_threshold": 10.0,
            "low_variance_threshold": 0.0,
            "incomplete_threshold": 0.5,
        }
    )

    mock_render_all.return_value = {
        "primary_metric_section": "Primary metric content",
        "checkpoint_section": "Checkpoint content",
        "bias_summary_section": "Bias summary",
        "performance_by_splits_section": "Performance content",
        "evaluation_by_group_section": "Group evaluation",
        "logo": "logo.png",
        "target_population_section": "Population info",
        "institution_name": "Test University",
        "sample_weight_section": "Sample weight info",
        "data_split_table": "Data split table",
        "bias_groups_section": "Bias groups",
        "selected_features_ranked_by_shap": "Feature list",
        "development_note_section": "Dev note",
        "outcome_section": "Outcome explanation",
    }

    card.load_model()
    card.find_model_version()
    card.extract_training_data()
    card._register_sections()
    card.collect_metadata()

    # Add section content to the context
    card.context.update(card.section_registry.render_all())

    # Validate placeholders in template
    with open(card.template_path, "r") as f:
        template = f.read()

    matches = set(re.findall(r"{([\w_]+)}", template))
    missing = matches - card.context.keys()
    assert not missing, f"Missing context keys for template: {missing}"
