import pytest
from student_success_tool.modeling.registration import get_model_name

@pytest.mark.parametrize(
    "institution_id, target_config, checkpoint_config, expected",
    [
        (
            "inst01",
            {"category": "graduation", "unit": "term", "value": "4"},
            {"unit": "credit", "value": "90", "optional_desc": "after_1_term"},
            "inst01_graduation_T_4term_C_90credit_after_1_term",
        ),
        (
            "inst02",
            {"category": "graduation", "unit": "completion_pct", "value": "150"},
            {"unit": "semester", "value": "2"},
            "inst03_graduation_T_150completion_pct_C_2semester",
        ),
        (
            "inst03",
            {"category": "retention"},
            {"unit": "1", "value": "term"},
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
