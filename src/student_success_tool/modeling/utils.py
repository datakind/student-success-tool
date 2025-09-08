import logging
import typing as t
import pathlib
import tomlkit
from tomlkit.items import Table


import pandas as pd
import sklearn.utils
from sklearn.model_selection import train_test_split

import pydantic as pyd

LOGGER = logging.getLogger(__name__)

_DEFAULT_SPLIT_LABEL_FRACS = {"train": 0.6, "test": 0.2, "validate": 0.2}

S = t.TypeVar("S", bound=pyd.BaseModel)


def compute_dataset_splits(
    df: pd.DataFrame,
    *,
    stratify_col: t.Optional[str] = None,
    label_fracs: t.Optional[dict[str, float]] = None,
    seed: t.Optional[int] = None,
    shuffle: bool = True,
) -> pd.Series:
    """
    Assign each row of a DataFrame to "train", "validate", or "test"
    according to user-specified fractions. Wraps sklearn's
    ``train_test_split`` to ensure stratification, reproducibility,
    and exact partitioning with no leftover rows.

    Parameters:
    df: The input dataset to split.
    stratify_col: Column name to use for stratified sampling (e.g., the target label).
    label_fracs: Mapping of {"train": x, "validate": y, "test": z}.
    seed: Random seed passed to sklearn for reproducibility.
    shuffle: Whether to shuffle rows before splitting.

    Returns:
        pd.Series: A Series of dtype "string" aligned to ``df.index`` with values
        in {"train", "validate", "test"}.
    """
    if label_fracs is None:
        label_fracs = _DEFAULT_SPLIT_LABEL_FRACS

    # Normalize fractions
    total = sum(label_fracs.values())
    fracs = {k: v / total for k, v in label_fracs.items()}

    train_frac = fracs["train"]
    valid_frac = fracs["validate"]
    test_frac = fracs["test"]

    stratify_vec = df[stratify_col] if stratify_col else None

    # train vs temp
    df_train, df_temp = train_test_split(
        df,
        test_size=(1 - train_frac),
        stratify=stratify_vec,
        random_state=seed,
        shuffle=shuffle,
    )

    # validate vs test (within temp)
    valid_size = valid_frac / (valid_frac + test_frac)
    stratify_vec_temp = df_temp[stratify_col] if stratify_col else None
    df_valid, df_test = train_test_split(
        df_temp,
        test_size=(1 - valid_size),
        stratify=stratify_vec_temp,
        random_state=seed,
        shuffle=shuffle,
    )

    # Build final split Series
    split = pd.Series(index=df.index, dtype="string", name="split")
    split.loc[df_train.index] = "train"
    split.loc[df_valid.index] = "validate"
    split.loc[df_test.index] = "test"
    return split


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
