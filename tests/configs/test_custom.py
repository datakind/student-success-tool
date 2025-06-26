try:
    import tomllib  # noqa
except ImportError:  # => PY3.10
    import tomli as tomllib  # noqa

import pydantic as pyd
import pathlib
import pytest

from student_success_tool.configs import custom


SRC_ROOT = pathlib.Path(__file__).parents[2] / "pipelines" / "custom" / "institution_id"


@pytest.fixture(scope="module")
def template_cfg_dict():
    config_path = SRC_ROOT / "config-TEMPLATE.toml"
    with config_path.open("rb") as f:
        return tomllib.load(f)


def test_template_pdp_cfgs(template_cfg_dict):
    result = custom.CustomProjectConfig.model_validate(template_cfg_dict)
    print(result)
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

            [datasets.bronze.raw_cohort]
            primary_keys = ["student_id"]
            non_null_cols = ["acad_year"]
            train_file_path = "/Volumes/CATALOG/INST_ID_bronze/.../FILE_NAME_cohort_train.csv"
            predict_file_path = "/Volumes/CATALOG/INST_ID_bronze/.../FILE_NAME_cohort_inference.csv"
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
