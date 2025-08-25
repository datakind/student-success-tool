try:
    import tomllib  # noqa
except ImportError:  # => PY3.10
    import tomli as tomllib  # noqa

import pydantic as pyd
import pathlib
import pytest

from student_success_tool.configs import pdp

SRC_ROOT = pathlib.Path(__file__).parents[2] / "pipelines" / "pdp" / "institution_id"

CONFIG_FILES = [
    "config-CREDITS_EARNED_TEMPLATE.toml",
    "config-GRADUATION_TEMPLATE.toml",
    "config-RETENTION_TEMPLATE.toml",
]

@pytest.mark.parametrize("config_file", CONFIG_FILES)
def test_template_pdp_cfgs(config_file):
    """
    Validates that each PDP config template loads correctly and matches
    the PDPProjectConfig schema.
    """
    config_path = SRC_ROOT / config_file
    with config_path.open("rb") as f:
        cfg = tomllib.load(f)

    result = pdp.PDPProjectConfig.model_validate(cfg)
    print(result)
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
        (
            """
            institution_id = "inst_id"
            institution_name = "Inst Name"

            [models.foo]
            experiment_id = "EXPERIMENT_ID"
            """,
            pytest.raises(pyd.ValidationError),
        ),
    ],
)
def test_bad_pdp_cfgs(cfg_str, context):
    cfg = tomllib.loads(cfg_str)
    with context:
        result = pdp.PDPProjectConfig.model_validate(cfg)
        assert isinstance(result, pyd.BaseModel)
