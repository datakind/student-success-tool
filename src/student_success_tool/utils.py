import logging
import re
import typing as t
from collections.abc import Collection, Iterable

LOGGER = logging.getLogger(__name__)

RE_VARIOUS_PUNCTS = re.compile(r"[!()*+\,\-./:;<=>?[\]^_{|}~]")
RE_QUOTATION_MARKS = re.compile(r"[\'\"\`]")


def to_list(value: t.Any) -> list:
    """Cast ``value`` into a list, regardless of its type."""
    if isinstance(value, list):
        return value
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return list(value)
    else:
        return [value]


def is_collection_but_not_string(value: t.Any) -> bool:
    """Check if ``value`` is a collection, excluding strings (and bytes)."""
    return isinstance(value, Collection) and not isinstance(value, (str, bytes))


def unique_elements_in_order(eles: Iterable) -> Iterable:
    """Get unique elements from an iterable, in order of appearance."""
    seen = set()  # type: ignore
    seen_add = seen.add
    for ele in eles:
        if ele not in seen:
            seen_add(ele)
            yield ele


def convert_to_snake_case(col: str) -> str:
    """Convert column name into snake case, without punctuation."""
    col = RE_VARIOUS_PUNCTS.sub(" ", col)
    col = RE_QUOTATION_MARKS.sub("", col)
    # TODO: *pretty sure* this could be cleaner and more performant, but shrug
    words = re.sub(
        r"([A-Z][a-z]+)", r" \1", re.sub(r"([A-Z]+|[0-9]+|\W+)", r" \1", col)
    ).split()
    return "_".join(words).lower()


def get_db_widget_param(name: str, *, default: t.Optional[object] = None) -> object:
    """
    Get a Databricks widget parameter by ``name``,
    returning a ``default`` value if not found.

    References:
        - https://docs.databricks.com/en/dev-tools/databricks-utils.html#dbutils-widgets-get
    """
    # these only work in a databricks env, so hide them here
    from databricks.sdk.runtime import dbutils
    from py4j.protocol import Py4JJavaError

    try:
        return dbutils.widgets.get(name)
    except Py4JJavaError:
        LOGGER.warning(
            "no db widget found with name=%s; returning default=%s", name, default
        )
        return default


def convert_intensity_time_limits(
    unit: t.Literal["term", "year"],
    intensity_time_limits: dict[str, tuple[float, t.Literal["term", "year"]]],
    *,
    num_terms_in_year: int,
) -> dict[str, float]:
    """
    Convert enrollment intensity-specific time limits into a particular ``unit`` ,
    whether input limits were given in units of years or terms.

    Args:
        unit: The time unit into which inputs are converted, either "term" or "year".
        intensity_time_limits: Mapping of enrollment intensity value (e.g. "FULL-TIME")
            to the maximum number of years or terms (e.g. [4.0, "year"], [12.0, "term"])
            considered "success" for a school in their particular use case.
        num_terms_in_year: Number of academic terms in one academic year,
            used to convert between term- and year-based time limits;
            for example: 4 => FALL, WINTER, SPRING, and SUMMER terms.
    """
    if unit == "year":
        intensity_nums = {
            intensity: num if unit == "year" else num / num_terms_in_year
            for intensity, (num, unit) in intensity_time_limits.items()
        }
    else:
        intensity_nums = {
            intensity: num if unit == "term" else num * num_terms_in_year
            for intensity, (num, unit) in intensity_time_limits.items()
        }
    return intensity_nums


def mock_pandera():
    import sys
    import types

    m1 = types.ModuleType("pandera")
    m2 = types.ModuleType("pandera.typing")

    class DataFrameModel: ...

    def Field(): ...

    class Series:
        def __getitem__(self, item): ...

    m1.DataFrameModel = DataFrameModel  # type: ignore
    m1.Field = Field  # type: ignore
    m2.Series = Series  # type: ignore

    sys.modules[m1.__name__] = m1
    sys.modules[m2.__name__] = m2
