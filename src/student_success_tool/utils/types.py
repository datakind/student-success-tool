import typing as t
from collections.abc import Collection, Iterable

import pandas as pd

TermType = t.Literal["FALL", "WINTER", "SPRING", "SUMMER"]
IntensityTimeLimitsType = dict[str, tuple[float, t.Literal["year", "term"]]]
CheckpointType = t.Union[pd.DataFrame, t.Callable[[pd.DataFrame], pd.DataFrame]]
CheckpointTypeType = t.Literal[
    "nth",
    "first",
    "last",
    "first_at_num_credits_earned",
    "first_within_cohort",
    "last_in_enrollment_year",
]
TargetTypeType = t.Literal["graduation", "retention", "credits_earned"]


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
