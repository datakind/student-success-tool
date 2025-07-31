import logging
import typing as t
import pathlib
import tomlkit
from tomlkit.items import Table


import numpy as np
import pandas as pd
import sklearn.utils

import pydantic as pyd

LOGGER = logging.getLogger(__name__)

_DEFAULT_SPLIT_LABEL_FRACS = {"train": 0.6, "test": 0.2, "validate": 0.2}

S = t.TypeVar("S", bound=pyd.BaseModel)


def compute_dataset_splits(
    df: pd.DataFrame,
    *,
    label_fracs: t.Optional[dict[str, float]] = None,
    shuffle: bool = True,
    seed: t.Optional[int] = None,
) -> pd.Series:
    """
    Split input dataset into random subsets with configurable proportions;
    by default, Databricks' standard train/test/validate splits are generated.

    Args:
        df
        label_fracs: Mapping of subset label to the (approximate) proportion of ``df``
            that gets split into it.
        shuffle: Whether or not to shuffle the data before splitting.
        seed: Optional integer used to set state for the underlying random generator;
            specify a value for reproducible splits, otherwise each call is unique.

    See Also:
        - :func:`sklearn.model_selection.train_test_split()`
    """
    if label_fracs is None:
        label_fracs = _DEFAULT_SPLIT_LABEL_FRACS

    labels = list(label_fracs.keys())
    fracs = list(label_fracs.values())
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


def update_run_metadata_in_toml(
    config_path: str, run_id: str, experiment_id: str
) -> None:
    """
    Update the 'model.run_id' and 'model.experiment_id' fields in a TOML config file,
    preserving the original formatting and structure using tomlkit.

    Args:
        config_path: Path to the TOML config file.
        run_id: The run ID to set.
        experiment_id: The experiment ID to set.
    """
    path = pathlib.Path(config_path).resolve()  # Absolute path

    try:
        doc = tomlkit.parse(path.read_text())

        if "model" not in doc:
            doc["model"] = tomlkit.table()

        model_table = t.cast(Table, doc["model"])
        model_table["run_id"] = run_id
        model_table["experiment_id"] = experiment_id

        path.write_text(tomlkit.dumps(doc))

        LOGGER.info(
            "Updated TOML config at %s with run_id=%s and experiment_id=%s",
            path,
            run_id,
            experiment_id,
        )

        # Re-read to confirm
        confirmed = tomlkit.parse(path.read_text())
        confirmed_model = t.cast(Table, confirmed["model"])
        LOGGER.info("Confirmed run_id = %s", confirmed_model.get("run_id"))
        LOGGER.info(
            "Confirmed experiment_id = %s", confirmed_model.get("experiment_id")
        )

    except Exception as e:
        LOGGER.error("Failed to update TOML config at %s: %s", path, e)
        raise
