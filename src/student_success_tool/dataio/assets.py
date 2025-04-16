import logging
import pathlib
import typing as t

import pydantic as pyd

from . import read

LOGGER = logging.getLogger(__name__)

S = t.TypeVar("S", bound=pyd.BaseModel)


def read_config(file_path: str, *, schema: type[S]) -> S:
    """
    Read config from ``file_path`` and validate it using ``schema`` ,
    returning an instance with parameters accessible by attribute.
    """
    cfg = read.from_toml_file(file_path)
    return schema.model_validate(cfg)


def read_features_table(file_path: str) -> dict[str, dict[str, str]]:
    """
    Read a features table mapping columns to readable names and (optionally) descriptions
    from a TOML file located at ``fpath``, which can either refer to a relative path in this
    package or an absolute path loaded from local disk.

    Args:
        file_path: Path to features table TOML file relative to package root or absolute;
            for example: "assets/pdp/features_table.toml" or "/path/to/features_table.toml".
    """
    pkg_root_dir = next(
        p
        for p in pathlib.Path(__file__).parents
        if p.parts[-1] == "student_success_tool"
    )
    fpath = (
        pathlib.Path(file_path)
        if pathlib.Path(file_path).is_absolute()
        else pkg_root_dir / file_path
    )
    features_table = read.from_toml_file(str(fpath))
    LOGGER.info("loaded features table from '%s'", fpath)
    return features_table  # type: ignore
