import os
from contextlib import nullcontext as does_not_raise

import pydantic as pyd
import pytest

from student_success_tool import configs

FIXTURES_PATH = "tests/fixtures"


class BadProjectConfig(pyd.BaseModel):
    is_bad: bool = pyd.Field(...)


@pytest.mark.parametrize(
    ["file_name", "schema", "context"],
    [
        ("project_config.toml", configs.PDPProjectConfig, does_not_raise()),
        ("project_config.toml", BadProjectConfig, pytest.raises(pyd.ValidationError)),
    ],
)
def test_load_config(file_name, schema, context):
    file_path = os.path.join(FIXTURES_PATH, file_name)
    with context:
        result = configs.load_config(file_path, schema=schema)
        assert isinstance(result, pyd.BaseModel)
