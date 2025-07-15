import pytest
import pandas as pd
import re
from unittest.mock import patch
from student_success_tool.reporting.model_card.base import ModelCard
from student_success_tool.configs.custom import CustomProjectConfig

@pytest.fixture
def custom_config():
    return CustomProjectConfig(
        institution_id="custom_inst_id",
        institution_name="Custom Institution Name",
        student_id_col="student_id",
        target_col="target",
        split_col="split",
        sample_weight_col="sample_weight",
        student_group_cols=["firstgenflag", "agegroup", "gender", "ethnicity", "disabilityflag"],
        pred_col="pred",
        pred_prob_col="pred_prob",
        pos_label=True,
        random_state=12345,
        student_group_aliases={
            "firstgenflag": "First-Generation Status",
            "agegroup": "Age",
            "gender": "Gender",
            "ethnicity": "Ethnicity",
            "disabilityflag": "Disability Status",
        },
        preprocessing={
            "target": {
                "category": "graduation",
                "unit": "pct_completion",
                "value": 150,
                "params": {
                    "intensity_time_limits": {
                        "FULL-TIME": [3.0, "year"],
                        "PART-TIME": [6.0, "year"]
                    }
                }
            },
            "checkpoint": {
                "unit": "credit",
                "value": 30,
                "params": {
                    "min_num_credits": 30.0,
                    "num_credits_col": "cumulative_credits_earned"
                }
            },
            "selection": {
                "student_criteria_aliases": {
                    "enrollment_type": "Enrollment Type",
                    "credential_type_sought_year_1": "Type of Credential Sought in Year 1"
                }
            }
        },
        modeling={
            "feature_selection": {
                "incomplete_threshold": 0.5,
                "low_variance_threshold": 0.0,
                "collinear_threshold": 10.0
            },
            "training": {
                "primary_metric": "log_loss",
                "timeout_minutes": 10
            }
        }
    )

@patch("student_success_tool.reporting.sections.registry.SectionRegistry.render_all")
@patch("student_success_tool.reporting.model_card.base.ModelCard.collect_metadata")
@patch("student_success_tool.reporting.model_card.base.ModelCard.load_model")
@patch("student_success_tool.reporting.model_card.base.ModelCard.extract_training_data")
@patch("student_success_tool.reporting.model_card.base.ModelCard.find_model_version")
def test_custom_school_model_card_context_population(
    mock_find_version,
    mock_extract_data,
    mock_load_model,
    mock_collect_metadata,
    mock_render_all,
    custom_config,
):
    card = ModelCard(config=custom_config, catalog="demo", model_name="custom_model")

    mock_load_model.side_effect = lambda: (
        setattr(card, "run_id", "dummy_run_id")
        or setattr(card, "experiment_id", "dummy_experiment_id")
        or setattr(card, "model", object())
        or setattr(card, "training_data", pd.DataFrame(columns=["sample_weight"]))
        or setattr(card, "modeling_data", pd.DataFrame({"student_id": []}))
    )

    mock_collect_metadata.side_effect = lambda: card.context.update({
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
    })

    mock_render_all.return_value = {
        "primary_metric_section": "Primary metric content",
        "checkpoint_section": "Checkpoint: 30 credits",
        "bias_summary_section": "Bias summary",
        "performance_by_splits_section": "Performance",
        "evaluation_by_group_section": "Group eval",
        "logo": "logo.png",
        "target_population_section": "Target pop details",
        "institution_name": "Custom Institution Name",
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
