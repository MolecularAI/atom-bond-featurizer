"""Test functions for the ``bonafide.utils.helper_functions_output`` module."""

import logging
from typing import List

import numpy as np
import pandas as pd
import pytest

from bonafide.utils.helper_functions_output import (
    get_energy_based_reduced_features,
    get_non_energy_based_reduced_features,
)


@pytest.mark.helper_functions_output_non_energy_based
class TestGetNonEnergyBasedReducedFeatures:
    """Test suite for get_non_energy_based_reduced_features function."""

    @pytest.fixture
    def sample_numeric_dataframe(self) -> pd.DataFrame:
        """Create a smaller sample DataFrame with numeric features and unique conformer IDs."""
        data = {
            "feature1": [1.0, 3.0],
            "feature2": [0.5, 2.5],
            "conformer_name": ["conf_0", "conf_1"],
            "conformer_energy": [10.5, 12.3],
            "boltzmann_weight": [0.7, 0.3],
        }
        df = pd.DataFrame(data, index=[0, 0])  # Same atom index, different conformers
        df.index.name = "ATOM_INDEX"
        return df

    @pytest.fixture
    def sample_mixed_dataframe(self) -> pd.DataFrame:
        """Create a smaller DataFrame with mixed numeric and non-numeric features."""
        data = {
            "numeric_feature": [1.0, 3.0],
            "string_feature": ["[1,2,3]", "[7,8,9]"],
            "mixed_feature": [1.0, "invalid"],
            "conformer_name": ["conf_0", "conf_1"],
            "conformer_energy": [10.5, 12.3],
        }
        df = pd.DataFrame(data, index=[0, 0])
        df.index.name = "BOND_INDEX"
        return df

    @pytest.fixture
    def sample_object_dtype_numeric(self) -> pd.DataFrame:
        """Create a smaller DataFrame with object dtype that contains numeric values."""
        data = {
            "object_numeric": ["1.5", "3.5"],
            "true_numeric": [1.0, 3.0],
            "conformer_name": ["conf_0", "conf_1"],
        }
        df = pd.DataFrame(data, index=[0, 0])
        df.index.name = "ATOM_INDEX"
        # Ensure object_numeric is object dtype
        df["object_numeric"] = df["object_numeric"].astype("object")
        return df

    @pytest.fixture
    def multi_atom_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with multiple atoms and conformers."""
        data = {
            "feature1": [1.0, 3.0, 2.0, 4.0],  # atom 0: [1.0, 3.0], atom 1: [2.0, 4.0]
            "feature2": [0.5, 2.5, 1.5, 3.5],  # atom 0: [0.5, 2.5], atom 1: [1.5, 3.5]
            "conformer_name": ["conf_0", "conf_1", "conf_0", "conf_1"],
            "conformer_energy": [10.5, 12.3, 10.5, 12.3],
        }
        df = pd.DataFrame(data, index=[0, 0, 1, 1])
        df.index.name = "ATOM_INDEX"
        return df

    @pytest.fixture
    def multi_atom_mixed_validity_features_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with multiple atoms and conformers, with some invalid features."""
        data = {
            "feature1": [1.0, 3.0, None, 4.0],  # atom 0: [1.0, 3.0], atom 1: [2.0, 4.0]
            "feature2": [0.5, 2.5, 1.5, 3.5],  # atom 0: [0.5, 2.5], atom 1: [1.5, 3.5]
            "conformer_name": ["conf_0", "conf_1", "conf_0", "conf_1"],
            "conformer_energy": [10.5, 12.3, 10.5, 12.3],
        }
        df = pd.DataFrame(data, index=[0, 0, 1, 1])
        df.index.name = "ATOM_INDEX"
        return df

    @pytest.fixture
    def sample_none_values_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with None values to test the new None value warning."""
        data = {
            "feature_with_none": [1.0, None],
            "feature_with_mixed_none": [None, 2.5],
            "feature_all_none": [None, None],
            "feature_normal": [1.0, 3.0],
            "conformer_name": ["conf_0", "conf_1"],
            "conformer_energy": [10.5, 12.3],
        }
        df = pd.DataFrame(data, index=[0, 0])
        df.index.name = "ATOM_INDEX"
        return df

    @pytest.fixture
    def multi_atom_none_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with multiple atoms having None values."""
        data = {
            "feature1": [1.0, None, 2.0, 4.0],  # atom 0: [1.0, None], atom 1: [2.0, 4.0]
            "feature2": [0.5, 2.5, None, None],  # atom 0: [0.5, 2.5], atom 1: [None, None]
            "conformer_name": ["conf_0", "conf_1", "conf_0", "conf_1"],
            "conformer_energy": [10.5, 12.3, 10.5, 12.3],
        }
        df = pd.DataFrame(data, index=[0, 0, 1, 1])
        df.index.name = "ATOM_INDEX"
        return df

    @pytest.mark.parametrize("feature_type", ["atom", "bond"])
    def test_basic_functionality_numeric_data(
        self, sample_numeric_dataframe: pd.DataFrame, feature_type: str
    ) -> None:
        """Test basic functionality with purely numeric data."""
        exclude_cols = ["conformer_name", "conformer_energy", "boltzmann_weight"]

        mean_df, min_df, max_df = get_non_energy_based_reduced_features(
            df=sample_numeric_dataframe,
            exclude_cols=exclude_cols,
            feature_type=feature_type,
            _namespace="TEST",
            _loc="test",
        )

        # Check return types
        assert isinstance(mean_df, pd.DataFrame)
        assert isinstance(min_df, pd.DataFrame)
        assert isinstance(max_df, pd.DataFrame)

        # Check column naming
        assert all(col.startswith("MEAN__") for col in mean_df.columns)
        assert all(col.startswith("MIN__") for col in min_df.columns)
        assert all(col.startswith("MAX__") for col in max_df.columns)

        # Check dimensions
        assert len(mean_df) == 1  # One unique index (0)
        assert len(min_df) == 1
        assert len(max_df) == 1

        # Check specific calculations for index 0 using pytest.approx
        assert mean_df.loc[0, "MEAN__feature1"] == pytest.approx(2.0)  # (1.0 + 3.0) / 2
        assert min_df.loc[0, "MIN__feature1"] == pytest.approx(1.0)
        assert max_df.loc[0, "MAX__feature1"] == pytest.approx(3.0)

        assert mean_df.loc[0, "MEAN__feature2"] == pytest.approx(1.5)  # (0.5 + 2.5) / 2
        assert min_df.loc[0, "MIN__feature2"] == pytest.approx(0.5)
        assert max_df.loc[0, "MAX__feature2"] == pytest.approx(2.5)

    def test_multi_atom_calculations(self, multi_atom_dataframe: pd.DataFrame) -> None:
        """Test calculations with multiple atoms."""
        exclude_cols = ["conformer_name", "conformer_energy"]

        mean_df, min_df, max_df = get_non_energy_based_reduced_features(
            multi_atom_dataframe, exclude_cols, "atom", _namespace="TEST", _loc="test"
        )

        # Check dimensions
        assert len(mean_df) == 2  # Two unique indices (0, 1)

        # Check calculations for atom 0: feature1 values [1.0, 3.0]
        assert mean_df.loc[0, "MEAN__feature1"] == pytest.approx(2.0)
        assert min_df.loc[0, "MIN__feature1"] == pytest.approx(1.0)
        assert max_df.loc[0, "MAX__feature1"] == pytest.approx(3.0)

        # Check calculations for atom 1: feature1 values [2.0, 4.0]
        assert mean_df.loc[1, "MEAN__feature1"] == pytest.approx(3.0)
        assert min_df.loc[1, "MIN__feature1"] == pytest.approx(2.0)
        assert max_df.loc[1, "MAX__feature1"] == pytest.approx(4.0)

        # Check calculations for atom 0: feature2 values [0.5, 2.5]
        assert mean_df.loc[0, "MEAN__feature2"] == pytest.approx(1.5)
        assert min_df.loc[0, "MIN__feature2"] == pytest.approx(0.5)
        assert max_df.loc[0, "MAX__feature2"] == pytest.approx(2.5)

    def test_exclude_columns_functionality(self, sample_numeric_dataframe: pd.DataFrame) -> None:
        """Test that exclude_cols parameter works correctly."""
        exclude_cols = ["conformer_name", "conformer_energy", "boltzmann_weight", "feature2"]

        mean_df, _, _ = get_non_energy_based_reduced_features(
            sample_numeric_dataframe, exclude_cols, "atom", _namespace="TEST", _loc="test"
        )

        # Check that excluded columns are not present
        assert "MEAN__conformer_name" not in mean_df.columns
        assert "MEAN__conformer_energy" not in mean_df.columns
        assert "MEAN__boltzmann_weight" not in mean_df.columns
        assert "MEAN__feature2" not in mean_df.columns

        # Check that non-excluded columns are present
        assert "MEAN__feature1" in mean_df.columns

    def test_object_dtype_numeric_conversion(
        self, sample_object_dtype_numeric: pd.DataFrame
    ) -> None:
        """Test conversion of object dtype columns containing numeric values."""
        exclude_cols = ["conformer_name"]

        mean_df, min_df, max_df = get_non_energy_based_reduced_features(
            sample_object_dtype_numeric, exclude_cols, "atom", _namespace="TEST", _loc="test"
        )

        # Check that object dtype numeric column was converted and calculated
        assert "MEAN__object_numeric" in mean_df.columns
        assert mean_df.loc[0, "MEAN__object_numeric"] == pytest.approx(2.5)  # (1.5 + 3.5) / 2
        assert min_df.loc[0, "MIN__object_numeric"] == pytest.approx(1.5)
        assert max_df.loc[0, "MAX__object_numeric"] == pytest.approx(3.5)

    def test_non_numeric_columns_warning(
        self, sample_mixed_dataframe: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that warnings are logged for non-numeric columns."""
        exclude_cols = ["conformer_name", "conformer_energy"]

        with caplog.at_level(logging.WARNING):
            get_non_energy_based_reduced_features(
                sample_mixed_dataframe, exclude_cols, "bond", _namespace="TEST", _loc="test"
            )

        # Check that warnings were logged for non-numeric columns
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]

        # Should have warnings for string_feature and mixed_feature
        string_feature_warnings = [msg for msg in warning_messages if "string_feature" in msg]
        mixed_feature_warnings = [msg for msg in warning_messages if "mixed_feature" in msg]

        assert len(string_feature_warnings) >= 1
        assert len(mixed_feature_warnings) >= 1

        # Check that warning contains expected information
        assert any("bond with index" in msg for msg in warning_messages)
        assert any("non-numeric values" in msg for msg in warning_messages)

    def test_numeric_only_columns_included(self, sample_mixed_dataframe: pd.DataFrame) -> None:
        """Test that only numeric columns are included in results."""
        exclude_cols = ["conformer_name", "conformer_energy"]

        mean_df, _, _ = get_non_energy_based_reduced_features(
            sample_mixed_dataframe, exclude_cols, "bond", _namespace="TEST", _loc="test"
        )

        # Only numeric_feature should be included (mixed_feature and string_feature should be excluded)
        assert "MEAN__numeric_feature" in mean_df.columns
        assert "MEAN__string_feature" not in mean_df.columns
        assert "MEAN__mixed_feature" not in mean_df.columns

    @pytest.mark.parametrize("index_values,expected_length", [([0, 1], 2), ([0, 0], 1), ([5], 1)])
    def test_different_index_configurations(
        self, index_values: List[int], expected_length: int
    ) -> None:
        """Test function with different index configurations."""
        data = {
            "feature1": [1.0] * len(index_values),
            "feature2": [2.0] * len(index_values),
            "conformer_name": [f"conf_{i}" for i in range(len(index_values))],
        }
        df = pd.DataFrame(data, index=index_values)
        df.index.name = "TEST_INDEX"

        exclude_cols = ["conformer_name"]

        mean_df, min_df, max_df = get_non_energy_based_reduced_features(
            df, exclude_cols, "atom", _namespace="TEST", _loc="test"
        )

        assert len(mean_df) == expected_length
        assert len(min_df) == expected_length
        assert len(max_df) == expected_length

    def test_empty_dataframe(self) -> None:
        """Test function behavior with empty DataFrame."""
        df = pd.DataFrame()
        df.index.name = "EMPTY_INDEX"

        mean_df, min_df, max_df = get_non_energy_based_reduced_features(
            df, [], "atom", _namespace="TEST", _loc="test"
        )

        assert len(mean_df) == 0
        assert len(min_df) == 0
        assert len(max_df) == 0
        assert isinstance(mean_df, pd.DataFrame)
        assert isinstance(min_df, pd.DataFrame)
        assert isinstance(max_df, pd.DataFrame)

    def test_single_conformer_per_index(self) -> None:
        """Test function with single conformer per index."""
        data = {
            "feature1": [1.0, 2.0],
            "feature2": [0.5, 1.5],
            "conformer_name": ["conf_0", "conf_1"],
        }
        df = pd.DataFrame(data, index=[0, 1])
        df.index.name = "ATOM_INDEX"

        exclude_cols = ["conformer_name"]

        mean_df, min_df, max_df = get_non_energy_based_reduced_features(
            df, exclude_cols, "atom", _namespace="TEST", _loc="test"
        )

        # With single conformer per index, mean, min, and max should be the same
        assert mean_df.loc[0, "MEAN__feature1"] == pytest.approx(1.0)
        assert min_df.loc[0, "MIN__feature1"] == pytest.approx(1.0)
        assert max_df.loc[0, "MAX__feature1"] == pytest.approx(1.0)

    def test_none_values_warning_and_handling(
        self, sample_none_values_dataframe: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that warnings are logged for None values and calculations handle them correctly."""
        exclude_cols = ["conformer_name", "conformer_energy"]

        with caplog.at_level(logging.WARNING):
            mean_df, min_df, max_df = get_non_energy_based_reduced_features(
                sample_none_values_dataframe, exclude_cols, "atom", _namespace="TEST", _loc="test"
            )

        # Check that warnings were logged for columns containing None values
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]

        # Should have warnings for features containing None values
        none_warnings = [msg for msg in warning_messages if "None values" in msg]
        assert (
            len(none_warnings) >= 3
        )  # Should warn for feature_with_none, feature_with_mixed_none, feature_all_none

        # Check that warnings contain expected information
        feature_with_none_warnings = [msg for msg in none_warnings if "feature_with_none" in msg]
        feature_with_mixed_none_warnings = [
            msg for msg in none_warnings if "feature_with_mixed_none" in msg
        ]
        feature_all_none_warnings = [msg for msg in none_warnings if "feature_all_none" in msg]

        assert len(feature_with_none_warnings) >= 1
        assert len(feature_with_mixed_none_warnings) >= 1
        assert len(feature_all_none_warnings) >= 1

        # Check that warnings contain expected text
        assert any("atom with index 0" in msg for msg in none_warnings)
        assert any("ignored during the calculation" in msg for msg in none_warnings)
        assert any("Check the unreduced features" in msg for msg in none_warnings)

        # Check calculations handle None values correctly
        # feature_with_none: [1.0, None] -> mean should be 1.0 (ignoring None)
        assert mean_df.loc[0, "MEAN__feature_with_none"] == pytest.approx(1.0)
        assert min_df.loc[0, "MIN__feature_with_none"] == pytest.approx(1.0)
        assert max_df.loc[0, "MAX__feature_with_none"] == pytest.approx(1.0)

        # feature_with_mixed_none: [None, 2.5] -> mean should be 2.5 (ignoring None)
        assert mean_df.loc[0, "MEAN__feature_with_mixed_none"] == pytest.approx(2.5)
        assert min_df.loc[0, "MIN__feature_with_mixed_none"] == pytest.approx(2.5)
        assert max_df.loc[0, "MAX__feature_with_mixed_none"] == pytest.approx(2.5)

        # feature_all_none: [None, None] -> should result in NaN
        assert pd.isna(mean_df.loc[0, "MEAN__feature_all_none"])
        assert pd.isna(min_df.loc[0, "MIN__feature_all_none"])
        assert pd.isna(max_df.loc[0, "MAX__feature_all_none"])

        # feature_normal: [1.0, 3.0] -> should calculate normally (no None values)
        assert mean_df.loc[0, "MEAN__feature_normal"] == pytest.approx(2.0)
        assert min_df.loc[0, "MIN__feature_normal"] == pytest.approx(1.0)
        assert max_df.loc[0, "MAX__feature_normal"] == pytest.approx(3.0)

    def test_multi_atom_none_values_warning(
        self, multi_atom_none_dataframe: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test None value warnings with multiple atoms."""
        exclude_cols = ["conformer_name", "conformer_energy"]

        with caplog.at_level(logging.WARNING):
            mean_df, min_df, max_df = get_non_energy_based_reduced_features(
                multi_atom_none_dataframe, exclude_cols, "atom", _namespace="TEST", _loc="test"
            )

        # Check that warnings were logged for both atoms
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        none_warnings = [msg for msg in warning_messages if "None values" in msg]

        # Should have warnings for both atoms
        atom0_warnings = [msg for msg in none_warnings if "atom with index 0" in msg]
        atom1_warnings = [msg for msg in none_warnings if "atom with index 1" in msg]

        assert len(atom0_warnings) >= 1  # atom 0 has None in feature1
        assert len(atom1_warnings) >= 1  # atom 1 has None in feature2

        # Check calculations
        # Atom 0: feature1 [1.0, None] -> mean = 1.0
        assert mean_df.loc[0, "MEAN__feature1"] == pytest.approx(1.0)

        # Atom 1: feature1 [2.0, 4.0] -> mean = 3.0 (no None values)
        assert mean_df.loc[1, "MEAN__feature1"] == pytest.approx(3.0)

        # Atom 0: feature2 [0.5, 2.5] -> mean = 1.5 (no None values)
        assert mean_df.loc[0, "MEAN__feature2"] == pytest.approx(1.5)

        # Atom 1: feature2 [None, None] -> mean = NaN (all None values)
        assert pd.isna(mean_df.loc[1, "MEAN__feature2"])

    def test_none_values_no_warning_when_no_none(
        self, sample_numeric_dataframe: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that no None value warnings are logged when there are no None values."""
        exclude_cols = ["conformer_name", "conformer_energy", "boltzmann_weight"]

        with caplog.at_level(logging.WARNING):
            get_non_energy_based_reduced_features(
                sample_numeric_dataframe, exclude_cols, "atom", _namespace="TEST", _loc="test"
            )

        # Check that no None value warnings were logged
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        none_warnings = [msg for msg in warning_messages if "None values" in msg]

        assert len(none_warnings) == 0

    @pytest.mark.parametrize(
        "none_positions,expected_mean,expected_min,expected_max",
        [
            ([True, False], 3.0, 3.0, 3.0),  # [None, 3.0] -> only 3.0 is valid
            ([False, True], 1.0, 1.0, 1.0),  # [1.0, None] -> only 1.0 is valid
            ([True, True], np.nan, np.nan, np.nan),  # [None, None] -> all NaN
            ([False, False], 2.0, 1.0, 3.0),  # [1.0, 3.0] -> normal calculation
        ],
    )
    def test_none_values_different_positions(
        self,
        none_positions: List[bool],
        expected_mean: float,
        expected_min: float,
        expected_max: float,
    ) -> None:
        """Test None value handling in different positions within the data."""
        feature_values = [1.0, 3.0]

        # Replace values with None based on none_positions
        for i, is_none in enumerate(none_positions):
            if is_none:
                feature_values[i] = None

        data = {"test_feature": feature_values, "conformer_name": ["conf_0", "conf_1"]}
        df = pd.DataFrame(data, index=[0, 0])
        df.index.name = "ATOM_INDEX"

        exclude_cols = ["conformer_name"]

        mean_df, min_df, max_df = get_non_energy_based_reduced_features(
            df, exclude_cols, "atom", _namespace="TEST", _loc="test"
        )

        # Check results
        if pd.isna(expected_mean):
            assert pd.isna(mean_df.loc[0, "MEAN__test_feature"])
            assert pd.isna(min_df.loc[0, "MIN__test_feature"])
            assert pd.isna(max_df.loc[0, "MAX__test_feature"])
        else:
            assert mean_df.loc[0, "MEAN__test_feature"] == pytest.approx(expected_mean)
            assert min_df.loc[0, "MIN__test_feature"] == pytest.approx(expected_min)
            assert max_df.loc[0, "MAX__test_feature"] == pytest.approx(expected_max)

    def test_none_and_nan_values_combined(self) -> None:
        """Test handling of both None and NaN values in the same dataset."""
        data = {
            "feature_none_nan": [None, np.nan],
            "feature_none_value": [None, 2.0],
            "feature_nan_value": [np.nan, 3.0],
            "conformer_name": ["conf_0", "conf_1"],
        }
        df = pd.DataFrame(data, index=[0, 0])
        df.index.name = "ATOM_INDEX"

        exclude_cols = ["conformer_name"]

        mean_df, min_df, max_df = get_non_energy_based_reduced_features(
            df, exclude_cols, "atom", _namespace="TEST", _loc="test"
        )

        # feature_none_nan: [None, NaN] -> should result in NaN
        assert pd.isna(mean_df.loc[0, "MEAN__feature_none_nan"])

        # feature_none_value: [None, 2.0] -> should use only 2.0
        assert mean_df.loc[0, "MEAN__feature_none_value"] == pytest.approx(2.0)

        # feature_nan_value: [NaN, 3.0] -> should use only 3.0
        assert mean_df.loc[0, "MEAN__feature_nan_value"] == pytest.approx(3.0)


@pytest.mark.helper_functions_output_energy_based
class TestGetEnergyBasedReducedFeatures:
    """Test suite for get_energy_based_reduced_features function."""

    @pytest.fixture
    def sample_energy_dataframe(self) -> pd.DataFrame:
        """Create a smaller sample DataFrame with energy and Boltzmann weight data."""
        data = {
            "feature1": [1.0, 3.0],
            "feature2": [0.5, 2.5],
            "conformer_name": ["conf_0", "conf_1"],
            "conformer_energy": [10.5, 8.3],  # conf_1 has lower energy
            "boltzmann_weight": [0.3, 0.7],
        }
        df = pd.DataFrame(data, index=[0, 0])
        df.index.name = "ATOM_INDEX"
        return df

    @pytest.fixture
    def multi_atom_energy_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with multiple atoms and energy data."""
        data = {
            "feature1": [1.0, 3.0, 2.0, 4.0],  # atom 0: [1.0, 3.0], atom 1: [2.0, 4.0]
            "feature2": [0.5, 2.5, 1.5, 3.5],  # atom 0: [0.5, 2.5], atom 1: [1.5, 3.5]
            "conformer_name": ["conf_0", "conf_1", "conf_0", "conf_1"],
            "conformer_energy": [10.5, 8.3, 10.5, 8.3],  # conf_1 has lower energy
            "boltzmann_weight": [0.3, 0.7, 0.3, 0.7],
        }
        df = pd.DataFrame(data, index=[0, 0, 1, 1])
        df.index.name = "ATOM_INDEX"
        return df

    @pytest.fixture
    def sample_inaccessible_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with '_inaccessible' values."""
        data = {
            "feature1": [1.0, 3.0],
            "feature2": ["_inaccessible", 2.5],
            "conformer_name": ["conf_0", "conf_1"],
            "conformer_energy": [10.5, 8.3],
            "boltzmann_weight": [0.3, 0.7],
        }
        df = pd.DataFrame(data, index=[0, 0])
        df.index.name = "BOND_INDEX"
        return df

    @pytest.fixture
    def sample_non_numeric_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with non-numeric features."""
        data = {
            "numeric_feature": [1.0, 3.0],
            "string_feature": ["[1,2,3]", "[7,8,9]"],
            "conformer_name": ["conf_0", "conf_1"],
            "conformer_energy": [10.5, 8.3],
            "boltzmann_weight": [0.3, 0.7],
        }
        df = pd.DataFrame(data, index=[0, 0])
        df.index.name = "ATOM_INDEX"
        return df

    @pytest.fixture
    def sample_missing_boltzmann_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with missing Boltzmann weights."""
        data = {
            "feature1": [1.0, 3.0],
            "feature2": [0.5, 2.5],
            "conformer_name": ["conf_0", "conf_1"],
            "conformer_energy": [10.5, 8.3],
            "boltzmann_weight": [0.3, np.nan],  # Missing weight for conf_1
        }
        df = pd.DataFrame(data, index=[0, 0])
        df.index.name = "ATOM_INDEX"
        return df

    @pytest.mark.parametrize("feature_type", ["atom", "bond"])
    def test_basic_functionality(
        self, sample_energy_dataframe: pd.DataFrame, feature_type: str
    ) -> None:
        """Test basic functionality with energy-based features."""
        exclude_cols = ["conformer_name", "conformer_energy", "boltzmann_weight"]

        min_e_df, max_e_df, boltzmann_df = get_energy_based_reduced_features(
            sample_energy_dataframe, exclude_cols, feature_type, _namespace="TEST", _loc="test"
        )

        # Check return types
        assert isinstance(min_e_df, pd.DataFrame)
        assert isinstance(max_e_df, pd.DataFrame)
        assert isinstance(boltzmann_df, pd.DataFrame)

        # Check column naming
        assert all(col.startswith("LOWEST_ENERGY__") for col in min_e_df.columns)
        assert all(col.startswith("HIGHEST_ENERGY__") for col in max_e_df.columns)
        assert all(col.startswith("BOLTZMANN__") for col in boltzmann_df.columns)

        # Check index name preservation
        assert boltzmann_df.index.name == "ATOM_INDEX"

    def test_lowest_highest_energy_selection(self, sample_energy_dataframe: pd.DataFrame) -> None:
        """Test that lowest- and highest-energy conformers are correctly selected."""
        exclude_cols = ["conformer_name", "conformer_energy", "boltzmann_weight"]

        min_e_df, max_e_df, _ = get_energy_based_reduced_features(
            sample_energy_dataframe, exclude_cols, "atom", _namespace="TEST", _loc="test"
        )

        # conf_1 has energy 8.3 (lowest), conf_0 has energy 10.5 (highest)
        # conf_1 has feature1=3.0, conf_0 has feature1=1.0
        assert min_e_df.loc[0, "LOWEST_ENERGY__feature1"] == pytest.approx(
            3.0
        )  # From lowest-energy conformer
        assert max_e_df.loc[0, "HIGHEST_ENERGY__feature1"] == pytest.approx(
            1.0
        )  # From highest-energy conformer

        # conf_1 has feature2=2.5, conf_0 has feature2=0.5
        assert min_e_df.loc[0, "LOWEST_ENERGY__feature2"] == pytest.approx(2.5)
        assert max_e_df.loc[0, "HIGHEST_ENERGY__feature2"] == pytest.approx(0.5)

    def test_multi_atom_energy_selection(self, multi_atom_energy_dataframe: pd.DataFrame) -> None:
        """Test energy-based selection with multiple atoms."""
        exclude_cols = ["conformer_name", "conformer_energy", "boltzmann_weight"]

        min_e_df, max_e_df, boltzmann_df = get_energy_based_reduced_features(
            multi_atom_energy_dataframe, exclude_cols, "atom", _namespace="TEST", _loc="test"
        )

        # Check dimensions
        assert len(min_e_df) == 2
        assert len(max_e_df) == 2
        assert len(boltzmann_df) == 2

        # For both atoms, conf_1 (energy 8.3) should be lowest, conf_0 (energy 10.5) should be highest
        # Atom 0: conf_1 has feature1=3.0, conf_0 has feature1=1.0
        assert min_e_df.loc[0, "LOWEST_ENERGY__feature1"] == pytest.approx(3.0)
        assert max_e_df.loc[0, "HIGHEST_ENERGY__feature1"] == pytest.approx(1.0)

        # Atom 1: conf_1 has feature1=4.0, conf_0 has feature1=2.0
        assert min_e_df.loc[1, "LOWEST_ENERGY__feature1"] == pytest.approx(4.0)
        assert max_e_df.loc[1, "HIGHEST_ENERGY__feature1"] == pytest.approx(2.0)

    def test_boltzmann_weighted_average(self, sample_energy_dataframe: pd.DataFrame) -> None:
        """Test Boltzmann-weighted average calculation."""
        exclude_cols = ["conformer_name", "conformer_energy", "boltzmann_weight"]

        _, _, boltzmann_df = get_energy_based_reduced_features(
            sample_energy_dataframe, exclude_cols, "atom", _namespace="TEST", _loc="test"
        )

        # For index 0: feature1 values are 1.0 (weight 0.3) and 3.0 (weight 0.7)
        # Boltzmann average = (1.0 * 0.3 + 3.0 * 0.7) / (0.3 + 0.7) = 2.4
        expected_boltzmann_feature1 = (1.0 * 0.3 + 3.0 * 0.7) / (0.3 + 0.7)
        assert boltzmann_df.loc[0, "BOLTZMANN__feature1"] == pytest.approx(
            expected_boltzmann_feature1
        )

        # For index 0: feature2 values are 0.5 (weight 0.3) and 2.5 (weight 0.7)
        # Boltzmann average = (0.5 * 0.3 + 2.5 * 0.7) / (0.3 + 0.7) = 1.9
        expected_boltzmann_feature2 = (0.5 * 0.3 + 2.5 * 0.7) / (0.3 + 0.7)
        assert boltzmann_df.loc[0, "BOLTZMANN__feature2"] == pytest.approx(
            expected_boltzmann_feature2
        )

    def test_multi_atom_boltzmann_calculation(
        self, multi_atom_energy_dataframe: pd.DataFrame
    ) -> None:
        """Test Boltzmann calculation with multiple atoms."""
        exclude_cols = ["conformer_name", "conformer_energy", "boltzmann_weight"]

        _, _, boltzmann_df = get_energy_based_reduced_features(
            multi_atom_energy_dataframe, exclude_cols, "atom", _namespace="TEST", _loc="test"
        )

        # Atom 0: feature1 values [1.0, 3.0] with weights [0.3, 0.7]
        expected_atom0_feature1 = (1.0 * 0.3 + 3.0 * 0.7) / (0.3 + 0.7)
        assert boltzmann_df.loc[0, "BOLTZMANN__feature1"] == pytest.approx(expected_atom0_feature1)

        # Atom 1: feature1 values [2.0, 4.0] with weights [0.3, 0.7]
        expected_atom1_feature1 = (2.0 * 0.3 + 4.0 * 0.7) / (0.3 + 0.7)
        assert boltzmann_df.loc[1, "BOLTZMANN__feature1"] == pytest.approx(expected_atom1_feature1)

    def test_inaccessible_values_replacement(
        self, sample_inaccessible_dataframe: pd.DataFrame
    ) -> None:
        """Test that '_inaccessible' values are replaced with NaN."""
        exclude_cols = ["conformer_name", "conformer_energy", "boltzmann_weight"]

        min_e_df, max_e_df, _ = get_energy_based_reduced_features(
            sample_inaccessible_dataframe, exclude_cols, "bond", _namespace="TEST", _loc="test"
        )

        # Check that '_inaccessible' values were converted to NaN
        # conf_1 (lowest-energy) has feature1=3.0, feature2=2.5
        assert min_e_df.loc[0, "LOWEST_ENERGY__feature1"] == pytest.approx(3.0)
        assert min_e_df.loc[0, "LOWEST_ENERGY__feature2"] == pytest.approx(2.5)

        # conf_0 (highest-energy) has feature1=1.0, feature2='_inaccessible' -> NaN
        assert max_e_df.loc[0, "HIGHEST_ENERGY__feature1"] == pytest.approx(1.0)
        assert pd.isna(max_e_df.loc[0, "HIGHEST_ENERGY__feature2"])

    def test_non_numeric_features_warning(
        self, sample_non_numeric_dataframe: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that warnings are logged for non-numeric features during Boltzmann calculation."""
        exclude_cols = ["conformer_name", "conformer_energy", "boltzmann_weight"]

        with caplog.at_level(logging.WARNING):
            get_energy_based_reduced_features(
                sample_non_numeric_dataframe, exclude_cols, "atom", _namespace="TEST", _loc="test"
            )

        # Check that warnings were logged for non-numeric columns
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]

        string_feature_warnings = [msg for msg in warning_messages if "string_feature" in msg]
        assert len(string_feature_warnings) >= 1

        # Check that warning contains expected information
        assert any("Boltzmann-weighted average" in msg for msg in warning_messages)
        assert any("non-numeric values" in msg for msg in warning_messages)
        assert any("atom with index" in msg for msg in warning_messages)

    def test_non_numeric_features_set_to_none(
        self, sample_non_numeric_dataframe: pd.DataFrame
    ) -> None:
        """Test that non-numeric features are set to None in Boltzmann calculation."""
        exclude_cols = ["conformer_name", "conformer_energy", "boltzmann_weight"]

        _, _, boltzmann_df = get_energy_based_reduced_features(
            sample_non_numeric_dataframe, exclude_cols, "atom", _namespace="TEST", _loc="test"
        )

        # Non-numeric features should result in None/NaN in Boltzmann results
        assert pd.isna(boltzmann_df.loc[0, "BOLTZMANN__string_feature"])

        # Numeric features should be calculated normally
        assert not pd.isna(boltzmann_df.loc[0, "BOLTZMANN__numeric_feature"])
        expected_numeric = (1.0 * 0.3 + 3.0 * 0.7) / (0.3 + 0.7)
        assert boltzmann_df.loc[0, "BOLTZMANN__numeric_feature"] == pytest.approx(expected_numeric)

    def test_missing_boltzmann_weights(
        self, sample_missing_boltzmann_dataframe: pd.DataFrame
    ) -> None:
        """Test behavior when some Boltzmann weights are missing."""
        exclude_cols = ["conformer_name", "conformer_energy", "boltzmann_weight"]

        _, _, boltzmann_df = get_energy_based_reduced_features(
            sample_missing_boltzmann_dataframe, exclude_cols, "atom", _namespace="TEST", _loc="test"
        )

        # Should only use conformers with valid Boltzmann weights
        # Only conf_0 has valid weight (0.3), so Boltzmann = feature value of conf_0
        assert boltzmann_df.loc[0, "BOLTZMANN__feature1"] == pytest.approx(1.0)  # From conf_0
        assert boltzmann_df.loc[0, "BOLTZMANN__feature2"] == pytest.approx(0.5)  # From conf_0

    def test_exclude_columns_functionality(self, sample_energy_dataframe: pd.DataFrame) -> None:
        """Test that exclude_cols parameter works correctly."""
        exclude_cols = ["conformer_name", "conformer_energy", "boltzmann_weight", "feature2"]

        min_e_df, _, boltzmann_df = get_energy_based_reduced_features(
            sample_energy_dataframe, exclude_cols, "atom", _namespace="TEST", _loc="test"
        )

        # Check that excluded columns are not present
        assert "LOWEST_ENERGY__conformer_name" not in min_e_df.columns
        assert "LOWEST_ENERGY__feature2" not in min_e_df.columns
        assert "BOLTZMANN__feature2" not in boltzmann_df.columns

        # Check that non-excluded columns are present
        assert "LOWEST_ENERGY__feature1" in min_e_df.columns
        assert "BOLTZMANN__feature1" in boltzmann_df.columns

    def test_single_conformer_per_index(self) -> None:
        """Test function with single conformer per index."""
        data = {
            "feature1": [1.0, 2.0],
            "feature2": [0.5, 1.5],
            "conformer_name": ["conf_0", "conf_1"],
            "conformer_energy": [10.5, 8.3],
            "boltzmann_weight": [0.3, 0.5],
        }
        df = pd.DataFrame(data, index=[0, 1])
        df.index.name = "ATOM_INDEX"

        exclude_cols = ["conformer_name", "conformer_energy", "boltzmann_weight"]

        min_e_df, max_e_df, boltzmann_df = get_energy_based_reduced_features(
            df, exclude_cols, "atom", _namespace="TEST", _loc="test"
        )

        # With single conformer per index, all values should be the same
        assert min_e_df.loc[0, "LOWEST_ENERGY__feature1"] == pytest.approx(1.0)
        assert max_e_df.loc[0, "HIGHEST_ENERGY__feature1"] == pytest.approx(1.0)
        assert boltzmann_df.loc[0, "BOLTZMANN__feature1"] == pytest.approx(1.0)

    def test_empty_dataframe(self) -> None:
        """Test function behavior with empty DataFrame."""
        df = pd.DataFrame()
        df.index.name = "EMPTY_INDEX"

        min_e_df, max_e_df, boltzmann_df = get_energy_based_reduced_features(
            df, [], "atom", _namespace="TEST", _loc="test"
        )

        assert len(min_e_df) == 0
        assert len(max_e_df) == 0
        assert len(boltzmann_df) == 0
        assert isinstance(min_e_df, pd.DataFrame)
        assert isinstance(max_e_df, pd.DataFrame)
        assert isinstance(boltzmann_df, pd.DataFrame)

    @pytest.mark.parametrize(
        "energy_values,weights,expected_min_feature,expected_max_feature",
        [
            (
                [10.0, 8.0],
                [0.3, 0.7],
                3.0,
                1.0,
            ),  # Lower energy (8.0) -> feature=3.0, higher energy (10.0) -> feature=1.0
            ([5.0, 5.0], [0.5, 0.5], 1.0, 1.0),  # Same energies -> first conformer for both
            (
                [15.0, 10.0],
                [0.2, 0.8],
                3.0,
                1.0,
            ),  # Lower energy (10.0) -> feature=3.0, higher energy (15.0) -> feature=1.0
        ],
    )
    def test_different_energy_configurations(
        self,
        energy_values: List[float],
        weights: List[float],
        expected_min_feature: float,
        expected_max_feature: float,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test function with different energy configurations."""
        data = {
            "feature1": [1.0, 3.0],
            "conformer_name": ["conf_0", "conf_1"],
            "conformer_energy": energy_values,
            "boltzmann_weight": weights,
        }
        df = pd.DataFrame(data, index=[0, 0])
        df.index.name = "ATOM_INDEX"

        exclude_cols = ["conformer_name", "conformer_energy", "boltzmann_weight"]

        # Capture warnings emitted by the function (e.g. when multiple conformers share the same energy)
        with caplog.at_level(logging.WARNING):
            min_e_df, max_e_df, boltzmann_df = get_energy_based_reduced_features(
                df, exclude_cols, "atom", _namespace="TEST", _loc="test"
            )

        # Collect warning messages
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]

        # Verify that the correct conformers were selected based on energy
        if len(np.unique(energy_values)) < len(energy_values):
            # If energies are the same (or duplicated), the function returns values for all conformers
            # that share the same min/max energy and logs a warning. Ensure warnings were emitted.
            assert any(
                "Multiple conformers have the same lowest energy" in msg for msg in warning_messages
            )
            assert any(
                "Multiple conformers have the same highest energy" in msg
                for msg in warning_messages
            )

            # min_e_df and max_e_df may contain multiple rows for the same atom index when energies are tied.
            # Gather all returned values for index 0 and ensure both feature values are present.
            min_vals = min_e_df.loc[min_e_df.index == 0, "LOWEST_ENERGY__feature1"].values
            max_vals = max_e_df.loc[max_e_df.index == 0, "HIGHEST_ENERGY__feature1"].values

            assert set(min_vals) == set([1.0, 3.0])
            assert set(max_vals) == set([1.0, 3.0])
        else:
            assert min_e_df.loc[0, "LOWEST_ENERGY__feature1"] == pytest.approx(expected_min_feature)
            assert max_e_df.loc[0, "HIGHEST_ENERGY__feature1"] == pytest.approx(
                expected_max_feature
            )

        # Verify Boltzmann calculation
        expected_boltzmann = (1.0 * weights[0] + 3.0 * weights[1]) / sum(weights)
        assert boltzmann_df.loc[0, "BOLTZMANN__feature1"] == pytest.approx(expected_boltzmann)
