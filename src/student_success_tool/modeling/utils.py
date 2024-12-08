import typing as t
from collections.abc import Sequence

import numpy as np
import pandas as pd


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
        - :func:`sklearn.model_selection.train_test_split()
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
