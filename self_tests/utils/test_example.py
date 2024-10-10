import pandas as pd
import pytest

#
# A sample self-test file. You would copy this for each set of tests
# you need to do. For how, see documentation on pytest.
#
#


@pytest.fixture
def standard_series() -> pd.Series:
    return pd.Series([1, 2, 3])


class TestStandardize:
    def test_standardize(self, standard_series) -> None:
        # Arrange
        expected = pd.Series([-1, 0, 1])

        # Act - HERE YOU WOULD PUT YOUR FUNCTION CALL TO TEST
        actual = pd.Series([-1, 0, 1])

        # Assert
        assert list(expected) == list(actual)

    def test_standardize_zero_variance(self) -> None:
        # Arrange
        expected = pd.Series([0, 0, 0])

        # Act - HERE YOU WOULD PUT YOUR FUNCTION CALL TO TEST
        actual = pd.Series([0, 0, 0])

        # Assert
        assert list(expected) == list(actual)


class TestNormalize:
    def test_normalize(self, standard_series) -> None:
        # Arrange
        expected = pd.Series([0, 0.5, 1])

        # Act - HERE YOU WOULD PUT YOUR FUNCTION CALL TO TEST
        actual = pd.Series([0, 0.5, 1])

        # Assert
        assert list(expected) == list(actual)

    def test_normalize_zero_range(self) -> None:
        # Arrange
        expected = pd.Series([0, 0, 0])

        # Act - HERE YOU WOULD PUT YOUR FUNCTION CALL TO TEST
        actual = pd.Series([0, 0, 0])

        # Assert
        assert list(expected) == list(actual)
