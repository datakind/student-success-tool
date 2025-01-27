import pandas as pd
import pytest
from contextlib import nullcontext as does_not_raise

try:
    import tomllib  # noqa
except ImportError:  # => PY3.10
    import tomli as tomllib  # noqa

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
            obs_value_counts, exp_value_counts, rtol=0.15, check_like=True
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

@pytest.mark.parametrize(
    "toml_content, expected_output, expect_exception",
    [
        (
            """
            academic_term = { name = "academic term" }
            term_in_peak_covid = { name = "term occurred in 'peak' COVID" }
            num_courses = { name = "number of courses taken this term" }
            """,
            {
                "academic_term": {"name": "academic term"},
                "term_in_peak_covid": {"name": "term occurred in 'peak' COVID"},
                "num_courses": {"name": "number of courses taken this term"}
            },
            does_not_raise(),  
        ),
        (
            """
            academic_term = { name = "academic term" }
            term_in_peak_covid = { name = "term occurred in 'peak' COVID" }
            num_courses = { name = "number of courses taken this term"
            """,
            None,  
            tomllib.TOMLDecodeError, 
        ),
        (
            "",
            None, 
            FileNotFoundError, 
        ),
    ]
)
def test_load_features_table(tmpdir, toml_content, expected_output, expect_exception):
    if toml_content:
        toml_file = tmpdir.join("features_table.toml")
        toml_file.write(toml_content)
        
        file_path = str(toml_file) 
    else:
        file_path = "non_existent_path/features_table.toml"
    
    with expect_exception:
        features_table = utils.load_features_table(file_path)
        assert isinstance(features_table, dict)
        for key, value in expected_output.items():
            assert key in features_table
            assert features_table[key] == value