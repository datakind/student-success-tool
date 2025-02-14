try:
    import tomllib  # noqa
except ImportError:  # => PY3.10
    import tomli as tomllib  # noqa

import pydantic as pyd
import pytest

from student_success_tool.schemas.pdp import project


# NOTE: if template config file changes, you should update it here as well
@pytest.fixture(scope="module")
def template_cfg_str():
    return """
    institution_id = "INST_ID"
    institution_name = "INST NAME"

    student_id_col = "student_guid"
    target_col = "target"
    split_col = "split"
    sample_weight_col = "sample_weight"
    student_group_cols = ["student_age", "race", "ethnicity", "gender", "first_gen"]
    pred_col = "pred"
    pred_prob_col = "pred_prob"
    pos_label = true
    random_state = 12345

    [datasets.labeled]
    raw_course = { file_path = "/Volumes/CATALOG/INST_NAME_bronze/INST_NAME_bronze_file_volume/FILE_NAME_COURSE.csv" }
    raw_cohort = { file_path = "/Volumes/CATALOG/INST_NAME_bronze/INST_NAME_bronze_file_volume/FILE_NAME_COHORT.csv" }
    preprocessed = { table_path = "CATALOG.SCHEMA.TABLE_NAME" }

    [preprocessing]
    splits = { train = 0.6, test = 0.2, validate = 0.2 }
    sample_class_weight = "balanced"

    [preprocessing.features]
    min_passing_grade = 1.0
    min_num_credits_full_time = 12
    # NOTE: single quotes *required* here; it's TOML syntax for literal strings
    course_level_pattern = 'asdf'
    key_course_subject_areas = ["24", "51"]
    key_course_ids = ["ENGL101", "MATH101"]

    [preprocessing.target]
    params = { min_num_credits_checkin = 30.0, min_num_credits_target = 60.0 }
    student_criteria = { enrollment_type = "FIRST-TIME", credential_type_sought_year_1 = "Bachelor's Degree" }

    [modeling.feature_selection]
    incomplete_threshold = 0.5
    low_variance_threshold = 0.0
    collinear_threshold = 10.0

    [modeling.training]
    # exclude_frameworks = ["xgboost", "lightgbm"]
    primary_metric = "log_loss"
    timeout_minutes = 10

    [models.graduation]
    experiment_id = "EXPERIMENT_ID"
    run_id = "RUN_ID"
    framework = "sklearn"

    [inference]
    num_top_features = 5
    min_prob_pos_label = 0.5
    """


def test_template_pdp_cfgs(template_cfg_str):
    cfg = tomllib.loads(template_cfg_str)
    result = project.PDPProjectConfig.model_validate(cfg)
    assert isinstance(result, pyd.BaseModel)


@pytest.mark.parametrize(
    ["cfg_str", "context"],
    [
        (
            'institution_id = "inst_id"',
            pytest.raises(pyd.ValidationError),
        ),
        (
            """
            institution_id = "INVALID_IDENTIFIER!"
            institution_name = "Inst Name"
            """,
            pytest.raises(pyd.ValidationError),
        ),
        (
            """
            institution_id = "inst_id"
            institution_name = "Inst Name"

            [datasets.labeled]
            foo = { "table_path" = "CATALOG.SCHEMA.TABLE_NAME" }
            """,
            pytest.raises(pyd.ValidationError),
        ),
    ],
)
def test_bad_pdp_v2_cfgs(cfg_str, context):
    cfg = tomllib.loads(cfg_str)
    with context:
        result = project.PDPProjectConfig.model_validate(cfg)
        assert isinstance(result, pyd.BaseModel)
