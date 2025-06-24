try:
    import tomllib  # noqa
except ImportError:  # => PY3.10
    import tomli as tomllib  # noqa

import pydantic as pyd
import pytest

from student_success_tool.configs import custom


# NOTE: if template config file changes, you should update it here as well
@pytest.fixture(scope="module")
def template_cfg_str():
    return """
    institution_id = "custom_inst_id"
    institution_name = "Custom Institution Name"

    student_id_col = "student_id"
    target_col = "target"
    split_col = "split"
    sample_weight_col = "sample_weight"
    student_group_cols = ["firstgenflag", "agegroup", "gender", "ethnicity", "disabilityflag"]
    pred_col = "pred"
    pred_prob_col = "pred_prob"
    pos_label = true
    random_state = 12345

    [student_group_aliases]
    firstgenflag = "First-Generation Status"
    agegroup = "Age"
    gender = "Gender"
    ethnicity = "Ethnicity"
    disabilityflag = "Disability Status"

    [datasets.bronze]
    raw_cohort = { 
    primary_keys = ["student_id"],
    non_null_cols = ["acad_year"],
    train = { file_path = "/Volumes/CATALOG/INST_ID_bronze/.../FILE_NAME_cohort_train.csv" }, 
    inference = { file_path = "/Volumes/CATALOG/INST_ID_bronze/.../FILE_NAME_cohort_inference.csv" } 
    }
    raw_course = { 
    primary_keys = ["student_id", "course_id"],
    drop_cols = ["timestamp"],
    train = { file_path = "/Volumes/CATALOG/INST_ID_bronze/.../FILE_NAME_course_train.csv" }, 
    inference = { file_path = "/Volumes/CATALOG/INST_ID_bronze/.../FILE_NAME_course_inference.csv" } 
    }
    raw_semester = { 
    train = { file_path = "/Volumes/CATALOG/INST_ID_bronze/.../FILE_NAME_semester_train.csv" }, 
    inference = { file_path = "/Volumes/CATALOG/INST_ID_bronze/.../FILE_NAME_semester_inference.csv" } 
    }

    [datasets.silver]
    cohort = { 
    train = { table_path = "CATALOG.INST_ID_silver.DATASET_NAME_cohort_train" }, 
    inference = { table_path = "CATALOG.INST_ID_silver.DATASET_NAME_preprocessed_inference" }
    }
    course = { 
    train = { table_path = "CATALOG.INST_ID_silver.DATASET_NAME_course_train" }, 
    inference = { table_path = "CATALOG.INST_ID_silver.DATASET_NAME_course_inference" }
    }
    semester = { 
    train = { table_path = "CATALOG.INST_ID_silver.DATASET_NAME_semester_train" }, 
    inference = { table_path = "CATALOG.INST_ID_silver.DATASET_NAME_semester_inference" }
    }
    modeling = {
    primary_keys = ["student_id"],
    train = { table_path = "CATALOG.INST_ID_silver.DATASET_NAME_modeling_train" }, 
    }
    model_features = {
    primary_keys = ["student_id"],
    inference = { table_path = "CATALOG.INST_ID_silver.DATASET_NAME_model_features_inference" }, 
    }

    [datasets.gold]
    sample_advisor_output = { 
    train = { table_path = "CATALOG.INST_ID_gold.INST_ID_sample_advisor_output" }, 
    }
    advisor_output = { 
    inference = { table_path = "CATALOG.INST_ID_gold.INST_ID_predictions" }, 
    }

    [model]
    experiment_id = "EXPERIMENT_ID"
    run_id = "RUN_ID"
    framework = "sklearn"

    [preprocessing]
    splits = { train = 0.6, test = 0.2, validate = 0.2 }
    sample_class_weight = "balanced"

    [preprocessing.selection]
    student_criteria = { enrollment_type = "FIRST-TIME", credential_type_sought_year_1 = "Bachelor's Degree" }

    [preprocessing.selection.student_criteria_aliases]
    enrollment_type = "Enrollment Type"
    credential_type_sought_year_1 = "Type of Credential Sought in Year 1"

    [preprocessing.checkpoint]
    name = "30_credits_earned"
    params = { min_num_credits = 30.0, num_credits_col = "cumulative_credits_earned" }
    unit = "credit" 
    value = 30

    [preprocessing.target]
    name = "graduation_150pct_time"
    category = "graduation"
    unit = "pct_completion"
    value = 150
    params = { intensity_time_limits = { FULL-TIME = [3.0, "year"], PART-TIME = [6.0, "year"] } }

    [modeling.feature_selection]
    incomplete_threshold = 0.5
    low_variance_threshold = 0.0
    collinear_threshold = 10.0
    # force_include_cols = []

    [modeling.training]
    primary_metric = "log_loss"
    timeout_minutes = 10
    # exclude_frameworks = ["xgboost", "lightgbm"]
    # exclude_cols = []

    [modeling.evaluation]
    topn_runs_included = 3

    [inference]
    num_top_features = 5
    min_prob_pos_label = 0.5
    """


def test_template_custom_cfgs(template_cfg_str):
    cfg = tomllib.loads(template_cfg_str)
    result = custom.CustomProjectConfig.model_validate(cfg)
    assert isinstance(result, pyd.BaseModel)


@pytest.mark.parametrize(
    ["cfg_str", "context"],
    [
        (
            'institution_id = "custom_inst_id"',
            pytest.raises(pyd.ValidationError),
        ),
        (
            """
            institution_id = "INVALID_IDENTIFIER!"
            institution_name = "Custom Institution Name"
            """,
            pytest.raises(pyd.ValidationError),
        ),
        (
            """
            institution_id = "custom_inst_id"
            institution_name = "Custom Institution Name"
            [datasets.bronze]
            raw_cohort = { 
            primary_keys = ["student_id"],
            non_null_cols = ["acad_year"],
            train = { file_path = "/Volumes/CATALOG/INST_ID_bronze/.../FILE_NAME_cohort_train.csv" }, 
            inference = { file_path = "/Volumes/CATALOG/INST_ID_bronze/.../FILE_NAME_cohort_inference.csv" } 
            }
            """,
            pytest.raises(pyd.ValidationError),
        ),
        (
            """
            institution_id = "inst_id"
            institution_name = "Inst Name"

            [model]
            experiment_id = "EXPERIMENT_ID"
            run_id = "RUN_ID"
            framework = "sklearn"
            """,
            pytest.raises(pyd.ValidationError),
        ),
    ],
)
def test_bad_custom_cfgs(cfg_str, context):
    cfg = tomllib.loads(cfg_str)
    with context:
        result = custom.CustomProjectConfig.model_validate(cfg)
        assert isinstance(result, pyd.BaseModel)
