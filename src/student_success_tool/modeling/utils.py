import pathlib
import typing as t
from collections.abc import Sequence

import numpy as np
import pandas as pd
import sklearn.utils

try:
    import tomllib  # noqa
except ImportError:  # => PY3.10
    import tomli as tomllib  # noqa


def compute_dataset_splits(
    df: pd.DataFrame,
    *,
    labels: Sequence[str] = ("train", "test", "valid"),
    fracs: Sequence[float] = (0.6, 0.2, 0.2),
    shuffle: bool = True,
    seed: t.Optional[int] = None,
) -> pd.Series:
    """
    Split input dataset into random subsets with configurable proportions;
    by default, Databricks' standard train/test/valid splits are generated.

    Args:
        df
        labels: Labels for each subset into which ``df`` is split.
        fracs: Approximate proportions of each subset into which ``df`` is split;
            corresponds 1:1 with each label in ``labels`` .
        shuffle: Whether or not to shuffle the data before splitting.
        seed: Optional integer used to set state for the underlying random generator;
            specify a value for reproducible splits, otherwise each call is unique.

    See Also:
        - :func:`sklearn.model_selection.train_test_split()`
    """
    if len(labels) != len(fracs):
        raise ValueError(
            f"the number of specified labels ({len(labels)}) and fracs {len(fracs)} "
            "must be the same"
        )

    rng = np.random.default_rng(seed=seed)
    return pd.Series(
        data=rng.choice(labels, size=len(df), p=fracs, shuffle=shuffle),
        index=df.index,
        dtype="string",
        name="split",
    )


def compute_sample_weights(
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    class_weight: t.Literal["balanced"] | dict[object, int] = "balanced",
) -> pd.Series:
    """
    Estimate sample weights by class for imbalanced datasets.

    Args:
        df
        target_col: Name of column in ``df`` containing class label values
            i.e. "targets" to be predicted.
        class_weight: Weights associated with classes in the form ``{class_label: weight}``
            or "balanced" to automatically adjust weights inversely proportional to
            class frequencies in the input data.

    See Also:
        - :func:`sklearn.utils.class_weight.compute_sample_weight()`
    """
    return pd.Series(
        data=sklearn.utils.class_weight.compute_sample_weight(
            class_weight, df[target_col]
        ),
        index=df.index,
        dtype="float32",
        name="sample_weight",
    )


def load_features_table(rel_path: str) -> dict[str, dict[str, str]]:
    pkg_root_dir = next(
        p
        for p in pathlib.Path(__file__).parents
        if p.parts[-1] == "student_success_tool"
    )
    print(f"{pkg_root_dir=}")
    file_path = pkg_root_dir / rel_path
    with file_path.open(mode="rb") as f:
        features_table = tomllib.load(f)
    assert isinstance(features_table, dict)  # type guard
    return features_table
