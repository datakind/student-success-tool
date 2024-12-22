import logging
import pathlib

import pydantic as pyd

try:
    import tomllib  # noqa
except ImportError:  # => PY3.10
    import tomli as tomllib  # noqa


LOGGER = logging.getLogger(__name__)


def load_config(file_path: str, schema: pyd.BaseModel) -> pyd.BaseModel:
    fpath = pathlib.Path(file_path).resolve()
    with fpath.open(mode="rb") as f:
        config = tomllib.load(f)
    LOGGER.info("loaded config from '%s'", fpath)
    assert isinstance(config, dict)  # type guard
    return schema.model_validate(config)
