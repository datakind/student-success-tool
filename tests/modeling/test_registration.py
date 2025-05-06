import pytest
from student_success_tool.modeling.registration import get_model_name

@pytest.mark.parametrize(
    "institution_id, target_config, checkpoint_config, expected",
    [
        (
            "inst01",
            {"category": "graduation", "unit": "term", "value": "4"},
            {"unit": "credit", "value": "90", "optional_desc": "after_1term"},
            "inst01_graduation_T_4term_C_90credit_after_1term",
        ),
        (
            "inst02",
            {"category": "graduation", "unit": "pct_completion", "value": "150"},
            {"unit": "semester", "value": "2"},
            "inst02_graduation_T_150pct_completion_C_2semester",
        ),
        (
            "inst03",
            {"category": "retention"},
            {"unit": "term", "value": "1"},
            "inst03_retention_C_1term",
        ),
    ]
)
def test_model_name_variants(institution_id, target_config, checkpoint_config, expected):
    assert get_model_name(
        institution_id=institution_id,
        target_config=target_config,
        checkpoint_config=checkpoint_config
    ) == expected
