import pandas as pd
import pytest

from student_success_tool.modeling import utils


@pytest.mark.parametrize(
    ["df", "labels", "fracs", "shuffle", "seed"],
    [
        (
            pd.DataFrame(data=list(range(1000))),
            ["train", "test"],
            [0.5, 0.5],
            True,
            None,
        ),
        (
            pd.DataFrame(data=list(range(1000))),
            ["train", "test", "valid"],
            [0.6, 0.2, 0.2],
            False,
            None,
        ),
        (
            pd.DataFrame(data=list(range(1000))),
            ["train", "test"],
            [0.5, 0.5],
            True,
            42,
        ),
    ],
)
def test_compute_dataset_splits(df, labels, fracs, shuffle, seed):
    obs = utils.compute_dataset_splits(
        df, labels=labels, fracs=fracs, shuffle=shuffle, seed=seed
    )
    assert isinstance(obs, pd.Series)
    assert len(obs) == len(df)
    obs_value_counts = obs.value_counts(normalize=True)
    exp_value_counts = pd.Series(
        data=list(fracs),
        index=pd.Index(list(labels), dtype="string", name="split"),
        name="proportion",
        dtype="Float64",
    )
    assert (
        pd.testing.assert_series_equal(
            obs_value_counts, exp_value_counts, rtol=0.1, check_like=True
        )
        is None
    )
    if seed is not None:
        obs2 = utils.compute_dataset_splits(
            df, labels=labels, fracs=fracs, shuffle=shuffle, seed=seed
        )
        assert obs.equals(obs2)


@pytest.mark.parametrize(
    ["df", "target_col", "class_weight", "exp"],
    [
        (
            pd.DataFrame({"target": [1, 1, 1, 0]}),
            "target",
            "balanced",
            pd.Series(
                [0.667, 0.667, 0.667, 2.0], dtype="float32", name="sample_weight"
            ),
        ),
        (
            pd.DataFrame({"target": [1, 1, 1, 0]}),
            "target",
            {1: 2, 0: 0.5},
            pd.Series([2.0, 2.0, 2.0, 0.5], dtype="float32", name="sample_weight"),
        ),
    ],
)
def test_compute_sample_weights(df, target_col, class_weight, exp):
    obs = utils.compute_sample_weights(
        df, target_col=target_col, class_weight=class_weight
    )
    assert isinstance(obs, pd.Series)
    assert len(obs) == len(df)
    assert pd.testing.assert_series_equal(obs, exp, rtol=0.01) is None
