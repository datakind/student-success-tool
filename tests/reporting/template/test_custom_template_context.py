import pytest
import pandas as pd
import re
from unittest.mock import patch
from student_success_tool.reporting.model_card.custom import CustomModelCard
from student_success_tool.configs.custom import CustomProjectConfig


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
        self.category = "graduation"
        self.unit = "pct_completion"
        self.value = 150
        self.params = {
            "intensity_time_limits": {
                "FULL-TIME": [3.0, "year"],
                "PART-TIME": [6.0, "year"],
            }
        }


class DummyCheckpointConfig:
    def __init__(self):
        self.unit = "credit"
        self.value = 30
        self.params = {
            "min_num_credits": 30.0,
            "num_credits_col": "cumulative_credits_earned",
        }


class DummySelectionConfig:
    def __init__(self):
        self.student_criteria_aliases = {
            "enrollment_type": "Enrollment Type",
            "credential_type_sought_year_1": "Type of Credential Sought in Year 1",
        }


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


class DummyDatasetsConfig:
    def __init__(self):
        self.bronze = {
            "raw_cohort": {"train_file_path": "dummy.csv"},
            "raw_course": {"train_file_path": "dummy.csv"},
        }
        self.silver = {
            "modeling": {"train_table_path": "dummy"},
            "model_features": {"predict_table_path": "dummy"},
        }
        self.gold = {
            "advisor_output": {"predict_table_path": "dummy"},
        }


def make_custom_project_config():
    return CustomProjectConfig(
        institution_id="custom_inst_id",
        institution_name="Custom Institution",
        student_id_col="student_id",
        target_col="target",
        split_col="split",
        sample_weight_col="sample_weight",
        pred_col="pred",
        pred_prob_col="pred_prob",
        pos_label=True,
        random_state=12345,
        student_group_cols=["firstgenflag", "gender"],
        student_group_aliases={
            "firstgenflag": "First-Generation Status",
            "gender": "Gender",
        },
        preprocessing={
            "target": {
                "category": "graduation",
                "unit": "pct_completion",
                "value": 150,
                "params": {
                    "intensity_time_limits": {
                        "FULL-TIME": [3.0, "year"],
                        "PART-TIME": [6.0, "year"],
                    }
                },
            },
            "checkpoint": {
                "unit": "credit",
                "value": 30,
                "params": {
                    "min_num_credits": 30.0,
                    "num_credits_col": "cumulative_credits_earned",
                },
            },
            "selection": {
                "student_criteria_aliases": {
                    "enrollment_type": "Enrollment Type",
                    "credential_type_sought_year_1": "Type of Credential Sought in Year 1",
                }
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
            "training": {
                "primary_metric": "log_loss",
                "timeout_minutes": 10,
            },
        },
        datasets={
            "bronze": {
                "raw_cohort": {"train_file_path": "dummy.csv"},
                "raw_course": {"train_file_path": "dummy.csv"},
            },
            "silver": {
                "modeling": {"train_table_path": "dummy"},
                "model_features": {"predict_table_path": "dummy"},
            },
            "gold": {
                "advisor_output": {"predict_table_path": "dummy"},
            },
        },
    )


@pytest.fixture
def dummy_custom_config():
    return make_custom_project_config()


@patch("student_success_tool.reporting.sections.registry.SectionRegistry.render_all")
@patch(
    "student_success_tool.reporting.model_card.custom.CustomModelCard.collect_metadata"
)
@patch("student_success_tool.reporting.model_card.custom.CustomModelCard.load_model")
@patch(
    "student_success_tool.reporting.model_card.custom.CustomModelCard.extract_training_data"
)
@patch(
    "student_success_tool.reporting.model_card.custom.CustomModelCard.find_model_version"
)
def test_custom_school_model_card_template_placeholders_filled(
    mock_find_version,
    mock_extract_data,
    mock_load_model,
    mock_collect_metadata,
    mock_render_all,
    dummy_custom_config,
):
    card = CustomModelCard(
        config=dummy_custom_config, catalog="demo", model_name="custom_model"
    )

    mock_load_model.side_effect = lambda: (
        setattr(card, "run_id", "dummy_run_id")
        or setattr(card, "experiment_id", "dummy_experiment_id")
        or setattr(card, "model", object())
        or setattr(card, "training_data", pd.DataFrame(columns=["sample_weight"]))
        or setattr(card, "modeling_data", pd.DataFrame({"student_id": []}))
    )

    mock_collect_metadata.side_effect = lambda: card.context.update(
        {
            "model_version": "42",
            "artifact_path": "custom/path",
            "training_dataset_size": 200,
            "number_of_features": 25,
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
        "checkpoint_section": "Checkpoint: 30 credits",
        "bias_summary_section": "Bias summary",
        "performance_by_splits_section": "Performance",
        "evaluation_by_group_section": "Group eval",
        "logo": "logo.png",
        "target_population_section": "Target pop details",
        "institution_name": "Custom Institution",
        "sample_weight_section": "Sample weighting",
        "data_split_table": "Split details",
        "bias_groups_section": "Bias groups",
        "selected_features_ranked_by_shap": "Ranked features",
        "development_note_section": "Model developed in 2025",
        "outcome_section": "Graduation outcome",
    }

    card.load_model()
    card.find_model_version()
    card.extract_training_data()
    card._register_sections()
    card.collect_metadata()

    card.context.update(card.section_registry.render_all())

    with open(card.template_path, "r") as f:
        template = f.read()

    placeholders = set(re.findall(r"{([\w_]+)}", template))
    missing = placeholders - card.context.keys()

    assert not missing, f"Missing context keys for template: {missing}"
