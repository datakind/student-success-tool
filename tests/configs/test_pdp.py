try:
    import tomllib  # noqa
except ImportError:  # => PY3.10
    import tomli as tomllib  # noqa

import pydantic as pyd
import pathlib
import pytest

from student_success_tool.configs import pdp

SRC_ROOT = pathlib.Path(__file__).parents[2] / "pipelines" / "pdp" / "institution_id"


@pytest.fixture(scope="module")
def template_cfg_dict():
    config_path = SRC_ROOT / "config-TEMPLATE.toml"
    with config_path.open("rb") as f:
        return tomllib.load(f)


def test_template_pdp_cfgs(template_cfg_dict):
    result = pdp.PDPProjectConfig.model_validate(template_cfg_dict)
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
