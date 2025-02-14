import typing as t

import pydantic as pyd

from . import read

S = t.TypeVar("S", bound=pyd.BaseModel)


def read_config(file_path: str, *, schema: type[S]) -> S:
    """
    Read config from ``file_path`` and validate it using ``schema`` ,
    returning an instance with parameters accessible by attribute.
    """
    cfg = read.from_toml_file(file_path)
    return schema.model_validate(cfg)
