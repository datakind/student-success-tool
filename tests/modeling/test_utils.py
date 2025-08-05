import pandas as pd
import pytest
import tomlkit

from student_success_tool.modeling import utils


@pytest.mark.parametrize(
    ["df", "label_fracs", "shuffle", "seed"],
    [
        (
            pd.DataFrame(data=list(range(1000))),
            {"train": 0.5, "test": 0.5},
            True,
            10,
        ),
        (
            pd.DataFrame(data=list(range(1000))),
            {"train": 0.6, "test": 0.2, "valid": 0.2},
            False,
            11,
        ),
        (
            pd.DataFrame(data=list(range(1000))),
            {"train": 0.5, "test": 0.5},
            True,
            42,
        ),
    ],
)
def test_compute_dataset_splits(df, label_fracs, shuffle, seed):
    obs = utils.compute_dataset_splits(
        df, label_fracs=label_fracs, shuffle=shuffle, seed=seed
    )
    assert isinstance(obs, pd.Series)
    assert len(obs) == len(df)
    labels = list(label_fracs.keys())
    fracs = list(label_fracs.values())
    obs_value_counts = obs.value_counts(normalize=True)
    exp_value_counts = pd.Series(
        data=fracs,
        index=pd.Index(labels, dtype="string", name="split"),
        name="proportion",
        dtype="Float64",
    )
    assert (
        pd.testing.assert_series_equal(
            obs_value_counts, exp_value_counts, rtol=0.15, check_like=True
        )
        is None
    )
    if seed is not None:
        obs2 = utils.compute_dataset_splits(
            df, label_fracs=label_fracs, shuffle=shuffle, seed=seed
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


# --- Sample config classes ---


def test_update_run_metadata_in_toml(tmp_path):
    # Create a dummy TOML config
    config_path = tmp_path / "config.toml"
    original_doc = tomlkit.document()
    original_doc["institution_id"] = "TEST_INST"
    original_doc["model"] = tomlkit.table()
    original_doc["model"]["framework"] = "sklearn"

    config_path.write_text(tomlkit.dumps(original_doc))

    # Update with new values
    run_id = "new_run_123"
    experiment_id = "exp_456"
    utils.update_run_metadata_in_toml(str(config_path), run_id, experiment_id)

    # Re-read and validate
    updated = tomlkit.parse(config_path.read_text())
    assert updated["model"]["run_id"] == run_id
    assert updated["model"]["experiment_id"] == experiment_id
    assert updated["model"]["framework"] == "sklearn"  # Existing field preserved
    assert updated["institution_id"] == "TEST_INST"  # Other sections untouched
