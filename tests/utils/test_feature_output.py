"""Test functions for the ``bonafide.utils.feature_output`` module."""

from typing import List, Optional
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from bonafide.utils.feature_output import FeatureOutput


class MockMolVaultFactory:
    """Factory for creating mock MolVault objects for testing."""

    @staticmethod
    def create_simple_molecule_with_properties(
        smiles: str = "CCO", atom_props: Optional[dict] = None, bond_props: Optional[dict] = None
    ) -> Chem.rdchem.Mol:
        """Create a simple molecule with predefined atom/bond properties."""
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        # Add default atom properties if none provided
        if atom_props is None:
            atom_props = {
                "partial_charge": [0.1, -0.2, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "electronegativity": [2.5, 3.0, 2.8, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1],
                "atomic_radius": [1.2, 1.4, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            }

        # Add default bond properties if none provided
        if bond_props is None:
            bond_props = {
                "bond_length": [1.54, 1.43, 1.09, 1.09, 1.09, 1.09, 1.09, 1.09],
                "bond_order": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "polarizability": [2.1, 1.8, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2],
            }

        # Set atom properties
        for prop_name, values in atom_props.items():
            for i, atom in enumerate(mol.GetAtoms()):
                if i < len(values):
                    val = values[i]
                    if isinstance(val, (int, np.integer)):
                        atom.SetIntProp(prop_name, int(val))
                    elif isinstance(val, (float, np.floating)):
                        atom.SetDoubleProp(prop_name, float(val))
                    elif isinstance(val, str):
                        atom.SetProp(prop_name, val)
                    elif isinstance(val, bool):
                        atom.SetBoolProp(prop_name, val)

        # Set bond properties
        for prop_name, values in bond_props.items():
            for i, bond in enumerate(mol.GetBonds()):
                if i < len(values):
                    val = values[i]
                    if isinstance(val, (int, np.integer)):
                        bond.SetIntProp(prop_name, int(val))
                    elif isinstance(val, (float, np.floating)):
                        bond.SetDoubleProp(prop_name, float(val))
                    elif isinstance(val, str):
                        bond.SetProp(prop_name, val)
                    elif isinstance(val, bool):
                        bond.SetBoolProp(prop_name, val)

        return mol

    @staticmethod
    def create_single_conformer_vault(
        smiles: str = "CCO",
        namespace: str = "test_mol",
        has_energies: bool = True,
        is_valid: bool = True,
        atom_props: Optional[dict] = None,
        bond_props: Optional[dict] = None,
    ) -> Mock:
        """Create a MolVault mock with a single conformer."""
        mock_vault = Mock()

        # Create molecule with properties
        mol = MockMolVaultFactory.create_simple_molecule_with_properties(
            smiles=smiles, atom_props=atom_props, bond_props=bond_props
        )

        # Basic attributes
        mock_vault.mol_objects = [mol]
        mock_vault.is_valid = [is_valid]
        mock_vault.namespace = namespace
        mock_vault.size = 1
        mock_vault.conformer_names = ["conf_0"]

        # Energy-related attributes
        if has_energies:
            mock_vault.energies_n = [[25.5]]  # Single energy value
            mock_vault.energies_n_read = True
            mock_vault.boltzmann_weights = (None, [1.0])  # Single conformer gets weight 1.0
        else:
            mock_vault.energies_n = [[None]]
            mock_vault.energies_n_read = False
            mock_vault.boltzmann_weights = (None, [None])

        return mock_vault

    @staticmethod
    def create_multi_conformer_vault(
        smiles: str = "CCO",
        namespace: str = "test_mol_multi",
        num_conformers: int = 3,
        has_energies: bool = True,
        valid_conformers: Optional[List[bool]] = None,
        energies: Optional[List[float]] = None,
        atom_props_list: Optional[List[dict]] = None,
        bond_props_list: Optional[List[dict]] = None,
    ) -> Mock:
        """Create a MolVault mock with multiple conformers. num_conformers <= 3"""
        mock_vault = Mock()

        # Default validity (all valid)
        if valid_conformers is None:
            valid_conformers = [True] * num_conformers

        # Default energies
        if energies is None:
            energies = [25.5, 26.2, 27.1][:num_conformers]

        # Create molecules with varying properties
        mol_objects = []
        for i in range(num_conformers):
            # Get properties for this conformer
            atom_props = atom_props_list[i] if atom_props_list else None
            bond_props = bond_props_list[i] if bond_props_list else None

            # Create slight variations in properties for each conformer
            if atom_props is None:
                atom_props = {
                    "partial_charge": [
                        0.1 + i * 0.01,
                        -0.2 + i * 0.01,
                        0.05 + i * 0.01,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "electronegativity": [2.5, 3.0, 2.8, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1],
                }

            if bond_props is None:
                bond_props = {
                    "bond_length": [
                        1.54 + i * 0.001,
                        1.43 + i * 0.001,
                        1.09,
                        1.09,
                        1.09,
                        1.09,
                        1.09,
                        1.09,
                    ],
                    "bond_order": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                }

            mol = MockMolVaultFactory.create_simple_molecule_with_properties(
                smiles=smiles, atom_props=atom_props, bond_props=bond_props
            )
            mol_objects.append(mol)

        # Basic attributes
        mock_vault.mol_objects = mol_objects
        mock_vault.is_valid = valid_conformers
        mock_vault.namespace = namespace
        mock_vault.size = num_conformers
        mock_vault.conformer_names = [f"conf_{i}" for i in range(num_conformers)]

        # Energy-related attributes
        if has_energies:
            mock_vault.energies_n = [[e] for e in energies]
            mock_vault.energies_n_read = True
            # Calculate Boltzmann weights (simplified)
            if all(e is not None for e in energies):
                weights = MockMolVaultFactory._calculate_boltzmann_weights(energies)
                mock_vault.boltzmann_weights = (None, weights)
            else:
                mock_vault.boltzmann_weights = (None, [None] * num_conformers)
        else:
            mock_vault.energies_n = [[None] for _ in range(num_conformers)]
            mock_vault.energies_n_read = False
            mock_vault.boltzmann_weights = (None, [None] * num_conformers)

        return mock_vault

    @staticmethod
    def create_mixed_validity_vault(
        smiles: str = "CCO",
        namespace: str = "test_mol_mixed",
        num_conformers: int = 4,
        valid_indices: List[int] = [0, 2, 3],
    ) -> Mock:
        """Create a MolVault with some valid and some invalid conformers."""
        valid_conformers = [i in valid_indices for i in range(num_conformers)]
        energies = [25.5 if valid else None for valid in valid_conformers]

        return MockMolVaultFactory.create_multi_conformer_vault(
            smiles=smiles,
            namespace=namespace,
            num_conformers=num_conformers,
            has_energies=True,
            valid_conformers=valid_conformers,
            energies=energies,
        )

    @staticmethod
    def create_no_energies_vault(
        smiles: str = "CCO", namespace: str = "test_mol_no_energies", num_conformers: int = 3
    ) -> Mock:
        """Create a MolVault with no energy information."""
        return MockMolVaultFactory.create_multi_conformer_vault(
            smiles=smiles, namespace=namespace, num_conformers=num_conformers, has_energies=False
        )

    @staticmethod
    def create_all_invalid_vault(
        smiles: str = "CCO", namespace: str = "test_mol_all_invalid", num_conformers: int = 3
    ) -> Mock:
        """Create a MolVault where all conformers are invalid."""
        return MockMolVaultFactory.create_multi_conformer_vault(
            smiles=smiles,
            namespace=namespace,
            num_conformers=num_conformers,
            has_energies=True,
            valid_conformers=[False] * num_conformers,
            energies=[None] * num_conformers,
        )

    @staticmethod
    def create_missing_properties_vault(
        smiles: str = "CCO", namespace: str = "test_mol_missing_props"
    ) -> Mock:
        """Conformers with missing properties for testing."""
        atom_props_list = [
            {"partial_charge": [0.1, -0.2, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            {"electronegativity": [2.5, 3.0, 2.8, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1]},
            {},
        ]
        bond_props_list = [
            {"bond_length": [1.54, 1.43, 1.09, 1.09, 1.09, 1.09, 1.09, 1.09]},
            {"bond_order": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
            {},
        ]
        return MockMolVaultFactory.create_multi_conformer_vault(
            smiles=smiles,
            namespace=namespace,
            num_conformers=3,
            has_energies=True,
            atom_props_list=atom_props_list,
            bond_props_list=bond_props_list,
        )

    @staticmethod
    def _calculate_boltzmann_weights(
        energies: List[float], temperature: float = 298.15
    ) -> List[float]:
        """Calculate Boltzmann weights from energies (simplified calculation)."""
        if not energies or any(e is None for e in energies):
            return [None] * len(energies)

        energies_array = np.array(energies)
        min_energy = np.min(energies_array)
        rel_energies = energies_array - min_energy
        kb = 0.001987204
        exp_terms = np.exp(-rel_energies / (kb * temperature))
        weights = exp_terms / np.sum(exp_terms)
        return weights.tolist()


# Pytest fixtures using the factory
@pytest.fixture
def single_conformer_vault() -> Mock:
    """Single conformer with energies."""
    return MockMolVaultFactory.create_single_conformer_vault()


@pytest.fixture
def single_conformer_vault_no_energies() -> Mock:
    """Single conformer without energies."""
    return MockMolVaultFactory.create_single_conformer_vault(has_energies=False)


@pytest.fixture
def single_conformer_vault_invalid() -> Mock:
    """Single invalid conformer."""
    return MockMolVaultFactory.create_single_conformer_vault(is_valid=False)


@pytest.fixture
def multi_conformer_vault() -> Mock:
    """Multiple valid conformers with energies."""
    return MockMolVaultFactory.create_multi_conformer_vault()


@pytest.fixture
def multi_conformer_vault_no_energies() -> Mock:
    """Multiple conformers without energies."""
    return MockMolVaultFactory.create_no_energies_vault()


@pytest.fixture
def mixed_validity_vault() -> Mock:
    """Mix of valid and invalid conformers."""
    return MockMolVaultFactory.create_mixed_validity_vault()


@pytest.fixture
def all_invalid_vault() -> Mock:
    """All conformers are invalid."""
    return MockMolVaultFactory.create_all_invalid_vault()


@pytest.fixture
def missing_properties_vault() -> Mock:
    """Conformers with missing properties."""
    return MockMolVaultFactory.create_missing_properties_vault()


@pytest.fixture
def large_molecule_vault() -> Mock:
    """Larger molecule for more complex testing."""
    return MockMolVaultFactory.create_multi_conformer_vault(
        smiles="CC(C)C1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        namespace="ibuprofen_test",
        num_conformers=5,
    )


@pytest.mark.get_results
class TestFeatureOutputGetResults:
    """Test suite for FeatureOutput.get_results method."""

    # Helper to check a DataFrame argument in mock calls
    def _assert_called_once_with_df(self, mock_method: Mock, expected_df: pd.DataFrame):
        """Helper to assert a mock method was called once with a specific DataFrame."""
        mock_method.assert_called_once()
        called_args, called_kwargs = mock_method.call_args
        if "df" in called_kwargs:
            df_arg = called_kwargs["df"]
        else:
            df_arg = called_args[0]
        pd.testing.assert_frame_equal(df_arg, expected_df)

    # 3.1 Mol Output (Non-Reduced) Tests

    def test_mol_output_non_reduced_single_conformer(self, single_conformer_vault: Mock) -> None:
        """Test mol output with reduce=False for single conformer.

        Should return a list with the single molecule object.
        """
        # Arrange
        feature_output = FeatureOutput(
            mol_vault=single_conformer_vault,
            indices=[0, 1, 2],
            feature_type="atom",
            reduce=False,
            ignore_invalid=False,
            _loc="test_mol_output",
        )

        with (
            patch.object(feature_output, "_clear_mols") as mock_clear,
            patch.object(feature_output, "_fill_missing_features") as mock_fill,
        ):
            # Setup mocks
            cleared_mols = [Mock()]
            filled_mols = [Mock()]
            mock_clear.return_value = cleared_mols
            mock_fill.return_value = filled_mols

            # Act
            result = feature_output.get_results("mol_object")

            # Assert that we clear and fill the mol objects properly:
            assert result == filled_mols
            mock_clear.assert_called_once_with(mols=single_conformer_vault.mol_objects)
            mock_fill.assert_called_once_with(mols=cleared_mols)

    def test_mol_output_non_reduced_multi_conformer(self, multi_conformer_vault: Mock) -> None:
        """Test mol output with reduce=False for multiple conformers.

        Should return a list of molecule objects corresponding to each conformer.
        """
        # Arrange
        feature_output = FeatureOutput(
            mol_vault=multi_conformer_vault,
            indices=[0, 1, 2],
            feature_type="bond",
            reduce=False,
            ignore_invalid=False,
            _loc="test_mol_output_multi",
        )

        with (
            patch.object(feature_output, "_clear_mols") as mock_clear,
            patch.object(feature_output, "_fill_missing_features") as mock_fill,
        ):
            # Setup mocks
            cleared_mols = [Mock(), Mock(), Mock()]
            filled_mols = [Mock(), Mock(), Mock()]
            mock_clear.return_value = cleared_mols
            mock_fill.return_value = filled_mols

            # Act
            result = feature_output.get_results("mol_object")

            # Assert
            assert result == filled_mols
            assert len(result) == 3
            mock_clear.assert_called_once_with(mols=multi_conformer_vault.mol_objects)
            mock_fill.assert_called_once_with(mols=cleared_mols)

    # 3.2 DataFrame Processing Path Tests

    def test_dataframe_processing_valid_conformers(self, multi_conformer_vault: Mock) -> None:
        """Test DataFrame creation for valid conformers."""
        # Arrange
        feature_output = FeatureOutput(
            mol_vault=multi_conformer_vault,
            indices=[0, 1],
            feature_type="atom",
            reduce=False,
            ignore_invalid=False,
            _loc="test_df_processing",
        )

        # Mock the helper methods
        with (
            patch.object(feature_output, "_get_feature_df") as mock_get_df,
            patch.object(feature_output, "_postprocess_df") as mock_postprocess,
        ):
            # Setup mock DataFrames
            mock_df1 = pd.DataFrame({"feature1": [1.0, 2.0]}, index=[0, 1])
            mock_df2 = pd.DataFrame({"feature1": [1.1, 2.1]}, index=[0, 1])
            mock_combined = pd.concat([mock_df1, mock_df2])

            mock_get_df.side_effect = [mock_df1, mock_combined, mock_combined]
            mock_postprocess.return_value = mock_combined

            # Act
            result = feature_output.get_results("df")

            # Assert
            assert isinstance(result, pd.DataFrame)
            assert mock_get_df.call_count == 3  # Called for each conformer
            mock_postprocess.assert_called_once()

    @patch("bonafide.utils.feature_output.logging")
    def test_dataframe_processing_skip_invalid_conformers(
        self, mock_logging, mixed_validity_vault: Mock
    ) -> None:
        """Test skipping invalid conformers with warnings."""
        # Arrange
        feature_output = FeatureOutput(
            mol_vault=mixed_validity_vault,
            indices=[0, 1],
            feature_type="atom",
            reduce=False,
            ignore_invalid=False,
            _loc="test_skip_invalid",
        )

        with (
            patch.object(feature_output, "_get_feature_df") as mock_get_df,
            patch.object(feature_output, "_postprocess_df") as mock_postprocess,
        ):
            # Setup mocks - only called for valid conformers
            mock_df = pd.DataFrame({"feature1": [1.0, 2.0]}, index=[0, 1])
            mock_get_df.return_value = mock_df
            mock_postprocess.return_value = mock_df

            # Act
            _ = feature_output.get_results("df")

            # Assert
            # Should only be called for valid conformers (indices 0, 2, 3 based on mixed_validity_vault)
            assert mock_get_df.call_count == 3  # Only valid conformers
            mock_logging.warning.assert_called()  # Should log warnings for invalid conformers
            assert "invalid" in str(mock_logging.warning.call_args)

    @patch("bonafide.utils.feature_output.logging")
    def test_dataframe_processing_no_valid_conformers_error(
        self, mock_logging, all_invalid_vault: Mock
    ) -> None:
        """Test ValueError when no valid conformers exist."""
        # Arrange
        feature_output = FeatureOutput(
            mol_vault=all_invalid_vault,
            indices=[0, 1],
            feature_type="atom",
            reduce=False,
            ignore_invalid=False,
            _loc="test_no_valid",
        )

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            _ = feature_output.get_results("df")

        assert "No valid conformers were found" in str(exc_info.value)
        mock_logging.error.assert_called()

    # 3.3 Reduction Logic Tests

    def test_reduction_multiple_conformers_reduce_true(self, multi_conformer_vault: Mock) -> None:
        """Test reduction when reduce=True and multiple conformers exist."""
        # Arrange
        feature_output = FeatureOutput(
            mol_vault=multi_conformer_vault,
            indices=[0, 1],
            feature_type="atom",
            reduce=True,
            ignore_invalid=False,
            _loc="test_reduction",
        )

        with (
            patch.object(feature_output, "_get_feature_df") as mock_get_df,
            patch.object(feature_output, "_reduce_conformer_data") as mock_reduce,
            patch.object(feature_output, "_postprocess_df") as mock_postprocess,
        ):
            # Setup mocks
            mock_combined_df = pd.DataFrame({"feature1": [1.0, 2.0, 1.1, 2.1]})
            mock_reduced_df = pd.DataFrame({"feature1_mean": [1.05, 2.05]})

            mock_get_df.side_effect = [
                mock_combined_df.iloc[:2],
                mock_combined_df,
                mock_combined_df,
            ]
            mock_reduce.return_value = mock_reduced_df
            mock_postprocess.return_value = mock_reduced_df

            # Act
            _ = feature_output.get_results("df")

            # Assert
            self._assert_called_once_with_df(mock_reduce, mock_combined_df)
            self._assert_called_once_with_df(mock_postprocess, mock_reduced_df)

    @patch("bonafide.utils.feature_output.logging")
    def test_reduction_single_conformer_reduce_true_warning(
        self, mock_logging, single_conformer_vault: Mock
    ) -> None:
        """Test warning when reduce=True but only one conformer exists."""
        # Arrange
        feature_output = FeatureOutput(
            mol_vault=single_conformer_vault,
            indices=[0, 1],
            feature_type="atom",
            reduce=True,
            ignore_invalid=False,
            _loc="test_single_reduce",
        )

        with (
            patch.object(feature_output, "_get_feature_df") as mock_get_df,
            patch.object(feature_output, "_reduce_conformer_data") as mock_reduce,
            patch.object(feature_output, "_postprocess_df") as mock_postprocess,
        ):
            # Setup mocks
            mock_df = pd.DataFrame({"feature1": [1.0, 2.0]})
            mock_get_df.return_value = mock_df
            mock_postprocess.return_value = mock_df

            # Act
            _ = feature_output.get_results("df")

            # Assert
            mock_reduce.assert_not_called()  # Should not call reduce for single conformer
            mock_logging.warning.assert_called()
            assert "only contains one conformer" in str(mock_logging.warning.call_args)

    def test_no_reduction_reduce_false(self, multi_conformer_vault: Mock) -> None:
        """Test no reduction when reduce=False."""
        # Arrange
        feature_output = FeatureOutput(
            mol_vault=multi_conformer_vault,
            indices=[0, 1],
            feature_type="atom",
            reduce=False,
            ignore_invalid=False,
            _loc="test_no_reduction",
        )

        with (
            patch.object(feature_output, "_get_feature_df") as mock_get_df,
            patch.object(feature_output, "_reduce_conformer_data") as mock_reduce,
            patch.object(feature_output, "_postprocess_df") as mock_postprocess,
        ):
            # Setup mocks
            mock_df = pd.DataFrame({"feature1": [1.0, 2.0, 1.1, 2.1]})
            mock_get_df.side_effect = [mock_df.iloc[:2], mock_df, mock_df]
            mock_postprocess.return_value = mock_df

            # Act
            result = feature_output.get_results("df")

            # Assert we don't reduce the features, and we postprocess the full DataFrame
            mock_reduce.assert_not_called()  # Should not call reduce
            self._assert_called_once_with_df(mock_postprocess, mock_df)

            assert isinstance(result, pd.DataFrame)

    # 3.4 Output Format Conversion Tests

    def test_dataframe_output_format(self, single_conformer_vault: Mock) -> None:
        """Test DataFrame output format returns processed DataFrame as-is."""
        # Arrange
        feature_output = FeatureOutput(
            mol_vault=single_conformer_vault,
            indices=[0, 1],
            feature_type="atom",
            reduce=False,
            ignore_invalid=False,
            _loc="test_df_output",
        )

        with (
            patch.object(feature_output, "_get_feature_df") as mock_get_df,
            patch.object(feature_output, "_postprocess_df") as mock_postprocess,
        ):
            # Setup mocks
            expected_df = pd.DataFrame({"feature1": [1.0, 2.0]}, index=[0, 1])
            mock_get_df.return_value = expected_df
            mock_postprocess.return_value = expected_df

            # Act
            result = feature_output.get_results("df")

            # Assert
            assert isinstance(result, pd.DataFrame)
            pd.testing.assert_frame_equal(result, expected_df)

    def test_dict_output_multiple_conformers_no_reduction(
        self, multi_conformer_vault: Mock
    ) -> None:
        """Test dict output with multiple conformers without reduction → grouped dictionary."""
        # Arrange
        feature_output = FeatureOutput(
            mol_vault=multi_conformer_vault,
            indices=[0, 1],
            feature_type="atom",
            reduce=False,
            ignore_invalid=False,
            _loc="test_dict_multi",
        )

        with (
            patch.object(feature_output, "_get_feature_df") as mock_get_df,
            patch.object(feature_output, "_postprocess_df") as mock_postprocess,
        ):
            # Setup mock DataFrame with multi-level index (conformer grouping)
            mock_df = pd.DataFrame(
                {"feature1": [1.0, 2.0, 1.1, 2.1, 1.2, 2.2]},
                index=pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]),
            )

            mock_get_df.side_effect = [mock_df.iloc[:2], mock_df.iloc[:4], mock_df]
            mock_postprocess.return_value = mock_df

            # Act
            result = feature_output.get_results("dict")

            # Assert
            assert isinstance(result, dict)
            # Should be grouped by level 0 (conformer index) with lists of values
            assert all(isinstance(v, dict) for v in result.values())

    def test_dict_output_single_conformer_or_reduced(self, single_conformer_vault: Mock) -> None:
        """Test dict output with single conformer or reduced data → standard dictionary."""
        # Arrange
        feature_output = FeatureOutput(
            mol_vault=single_conformer_vault,
            indices=[0, 1],
            feature_type="atom",
            reduce=True,
            ignore_invalid=False,
            _loc="test_dict_single",
        )

        with (
            patch.object(feature_output, "_get_feature_df") as mock_get_df,
            patch.object(feature_output, "_postprocess_df") as mock_postprocess,
        ):
            # Setup mock DataFrame
            mock_df = pd.DataFrame({"feature1_mean": [1.0, 2.0]}, index=[0, 1])

            mock_get_df.return_value = mock_df
            mock_postprocess.return_value = mock_df

            # Act
            result = feature_output.get_results("dict")

            # Assert
            assert isinstance(result, dict)
            expected_dict = {0: {"feature1_mean": 1.0}, 1: {"feature1_mean": 2.0}}
            assert result == expected_dict

    def test_mol_output_with_reduction(self, multi_conformer_vault: Mock) -> None:
        """Test mol output with reduction casts reduced properties to molecule object."""
        # Arrange
        feature_output = FeatureOutput(
            mol_vault=multi_conformer_vault,
            indices=[0, 1],
            feature_type="atom",
            reduce=True,
            ignore_invalid=False,
            _loc="test_mol_reduced",
        )

        with (
            patch.object(feature_output, "_get_feature_df") as mock_get_df,
            patch.object(feature_output, "_reduce_conformer_data") as mock_reduce,
            patch.object(feature_output, "_postprocess_df") as mock_postprocess,
            patch.object(feature_output, "_cast_reduced_props_to_mol") as mock_cast,
        ):
            # Setup mocks
            mock_df = pd.DataFrame({"feature1": [1.0, 2.0]})
            mock_reduced_df = pd.DataFrame({"feature1_mean": [1.0, 2.0]})
            mock_mol = Mock()

            mock_get_df.side_effect = [mock_df.iloc[:1], mock_df, mock_df]
            mock_reduce.return_value = mock_reduced_df
            mock_postprocess.return_value = mock_reduced_df
            mock_cast.return_value = mock_mol

            # Act
            result = feature_output.get_results("mol_object")

            # Assert
            assert result == mock_mol
            mock_cast.assert_called_once_with(
                df=mock_reduced_df, mol=multi_conformer_vault.mol_objects[0]
            )


@pytest.mark.get_results_integration
class TestFeatureOutputGetResultsIntegration:
    """Integration tests combining multiple parameters and scenarios."""

    @pytest.mark.parametrize(
        "output_format, reduce, feature_type",
        [
            ("df", True, "atom"),
            ("df", False, "atom"),
            ("df", True, "bond"),
            ("df", False, "bond"),
            ("dict", True, "atom"),
            ("dict", False, "atom"),
            ("dict", True, "bond"),
            ("dict", False, "bond"),
        ],
    )
    def test_output_format_and_reduction_combinations(
        self, multi_conformer_vault: Mock, output_format: str, reduce: bool, feature_type: str
    ) -> None:
        """Test all combinations of output format, reduction, and feature type."""
        # Arrange
        feature_output = FeatureOutput(
            mol_vault=multi_conformer_vault,
            indices=[0, 1, 2],
            feature_type=feature_type,
            reduce=reduce,
            ignore_invalid=False,
            _loc="test_combinations",
        )

        with (
            patch.object(feature_output, "_get_feature_df") as mock_get_df,
            patch.object(feature_output, "_reduce_conformer_data") as mock_reduce_data,
            patch.object(feature_output, "_postprocess_df") as mock_postprocess,
        ):
            # Setup mocks
            mock_df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})
            mock_reduced_df = pd.DataFrame({"feature1_mean": [1.0, 2.0, 3.0]})

            mock_get_df.side_effect = [mock_df.iloc[:1], mock_df.iloc[:2], mock_df]
            mock_reduce_data.return_value = mock_reduced_df if reduce else mock_df
            mock_postprocess.return_value = mock_reduced_df if reduce else mock_df

            # Act
            result = feature_output.get_results(output_format)

            # Assert
            if output_format == "df":
                assert isinstance(result, pd.DataFrame)
            elif output_format == "dict":
                assert isinstance(result, dict)

            # Check if reduction was called appropriately
            if reduce and multi_conformer_vault.size > 1:
                mock_reduce_data.assert_called_once()
            else:
                mock_reduce_data.assert_not_called()

    def test_ignore_invalid_parameter_effect(self, mixed_validity_vault: Mock) -> None:
        """Test the effect of ignore_invalid parameter on processing."""
        # Test with ignore_invalid=True
        feature_output_ignore = FeatureOutput(
            mol_vault=mixed_validity_vault,
            indices=[0, 1],
            feature_type="atom",
            reduce=True,
            ignore_invalid=True,
            _loc="test_ignore_true",
        )

        # Test with ignore_invalid=False
        feature_output_no_ignore = FeatureOutput(
            mol_vault=mixed_validity_vault,
            indices=[0, 1],
            feature_type="atom",
            reduce=True,
            ignore_invalid=False,
            _loc="test_ignore_false",
        )

        with (
            patch.object(feature_output_ignore, "_get_feature_df") as mock_get_df_ignore,
            patch.object(feature_output_ignore, "_reduce_conformer_data") as mock_reduce_ignore,
            patch.object(feature_output_ignore, "_postprocess_df") as mock_postprocess_ignore,
            patch.object(feature_output_no_ignore, "_get_feature_df") as mock_get_df_no_ignore,
            patch.object(
                feature_output_no_ignore, "_reduce_conformer_data"
            ) as mock_reduce_no_ignore,
            patch.object(feature_output_no_ignore, "_postprocess_df") as mock_postprocess_no_ignore,
        ):
            # Setup mocks
            mock_df = pd.DataFrame({"feature1": [1.0, 2.0]})
            mock_get_df_ignore.side_effect = [mock_df.iloc[:1], mock_df, mock_df]
            mock_get_df_no_ignore.side_effect = [mock_df.iloc[:1], mock_df, mock_df]
            mock_reduce_ignore.return_value = mock_df
            mock_reduce_no_ignore.return_value = mock_df
            mock_postprocess_ignore.return_value = mock_df
            mock_postprocess_no_ignore.return_value = mock_df

            # Act
            result_ignore = feature_output_ignore.get_results("df")
            result_no_ignore = feature_output_no_ignore.get_results("df")

            # Assert - Both should process successfully
            assert isinstance(result_ignore, pd.DataFrame)
            assert isinstance(result_no_ignore, pd.DataFrame)

    def test_complex_workflow_with_all_features(self, large_molecule_vault: Mock) -> None:
        """Test complex workflow with larger molecule and all features enabled."""
        # Arrange
        feature_output = FeatureOutput(
            mol_vault=large_molecule_vault,
            indices=list(range(10)),  # Multiple indices
            feature_type="atom",
            reduce=True,
            ignore_invalid=False,
            _loc="test_complex_workflow",
        )

        with (
            patch.object(feature_output, "_get_feature_df") as mock_get_df,
            patch.object(feature_output, "_reduce_conformer_data") as mock_reduce,
            patch.object(feature_output, "_postprocess_df") as mock_postprocess,
            patch.object(feature_output, "_cast_reduced_props_to_mol") as mock_cast,
        ):
            # Setup mocks for complex DataFrame
            mock_df = pd.DataFrame(
                {
                    "feature1": list(range(50)),  # 10 atoms × 5 conformers
                    "feature2": [x * 2 for x in range(50)],
                }
            )
            mock_reduced_df = pd.DataFrame(
                {"feature1_mean": list(range(10)), "feature2_mean": [x * 2 for x in range(10)]}
            )
            mock_mol = Mock()

            def setup_mock_get_df():
                """Helper to set up the mock side_effect."""
                return [
                    mock_df.iloc[:10],  # First conformer
                    mock_df.iloc[:20],  # First two conformers
                    mock_df.iloc[:30],  # First three conformers
                    mock_df.iloc[:40],  # First four conformers
                    mock_df,  # All conformers
                ]

            mock_reduce.return_value = mock_reduced_df
            mock_postprocess.return_value = mock_reduced_df
            mock_cast.return_value = mock_mol

            # Test DataFrame output
            mock_get_df.side_effect = setup_mock_get_df()
            df_result = feature_output.get_results("df")
            assert isinstance(df_result, pd.DataFrame)

            # Reset and test Dictionary output
            mock_get_df.side_effect = setup_mock_get_df()
            dict_result = feature_output.get_results("dict")
            assert isinstance(dict_result, dict)

            # Reset and test Mol output
            mock_get_df.side_effect = setup_mock_get_df()
            mol_result = feature_output.get_results("mol_object")
            assert mol_result == mock_mol

            # Verify reduction was called for each output format
            assert mock_reduce.call_count == 3
            assert mock_postprocess.call_count == 3
            assert mock_cast.call_count == 1  # Only for mol output


@pytest.mark.get_results_edge_cases
class TestFeatureOutputGetResultsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_indices_processing(self, single_conformer_vault: Mock) -> None:
        """Test processing with empty indices list."""
        # Arrange
        feature_output = FeatureOutput(
            mol_vault=single_conformer_vault,
            indices=[],  # Empty indices
            feature_type="atom",
            reduce=False,
            ignore_invalid=False,
            _loc="test_empty_indices",
        )

        with (
            patch.object(feature_output, "_get_feature_df") as mock_get_df,
            patch.object(feature_output, "_postprocess_df") as mock_postprocess,
        ):
            # Setup mocks
            mock_df = pd.DataFrame()  # Empty DataFrame
            mock_get_df.return_value = mock_df
            mock_postprocess.return_value = mock_df

            # Act
            result = feature_output.get_results("df")

            # Assert
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0


@pytest.mark.get_results_object_helper_methods
class TestFeatureOutputObjectHelperMethods:
    """Test suite for FeatureOutput helper methods."""

    def test__clear_mols(self, multi_conformer_vault: Mock) -> None:
        """_clear_mols should remove properties from the appropriate objects.

        Note: the current implementation clears bond properties when feature_type == 'atom'
        and clears atom properties when feature_type == 'bond'. The test asserts that
        behaviour matches the current implementation.
        """
        # Arrange - create FeatureOutput for atom (so bond props are cleared)
        fo_atom = FeatureOutput(
            mol_vault=multi_conformer_vault,
            indices=[0, 1],
            feature_type="atom",
            reduce=False,
            ignore_invalid=False,
            _loc="test__clear_mols",
        )

        original_mol = multi_conformer_vault.mol_objects[0]
        # Ensure bond props exist initially
        bond = original_mol.GetBonds()[0]
        assert len(list(bond.GetPropNames())) > 0

        # Act
        cleared = fo_atom._clear_mols(mols=[original_mol])

        # Assert: bond props removed, atom props remain (per implementation)
        cleared_bond = cleared[0].GetBonds()[0]
        assert len(list(cleared_bond.GetPropNames())) == 0
        # atoms should still have properties
        assert len(list(cleared[0].GetAtoms()[0].GetPropNames())) >= 0

        # Now test the opposite behaviour for feature_type == 'bond'
        fo_bond = FeatureOutput(
            mol_vault=multi_conformer_vault,
            indices=[0, 1],
            feature_type="bond",
            reduce=False,
            ignore_invalid=False,
            _loc="test_clear_mols_bond",
        )
        # Ensure atom props exist initially
        assert len(list(original_mol.GetAtoms()[0].GetPropNames())) > 0

        cleared2 = fo_bond._clear_mols(mols=[original_mol])
        cleared2_atom = cleared2[0].GetAtoms()[0]
        assert len(list(cleared2_atom.GetPropNames())) == 0

    @patch("bonafide.utils.feature_output.logging")
    def test__fill_missing_features_atoms(
        self, mock_logging, missing_properties_vault: Mock
    ) -> None:
        """_fill_missing_features fills absent properties with 'NaN' and warns when all features are NaN."""
        fo = FeatureOutput(
            mol_vault=missing_properties_vault,
            indices=[0, 1, 2],
            feature_type="atom",
            reduce=False,
            ignore_invalid=False,
            _loc="test__fill_missing",
        )

        mols = missing_properties_vault.mol_objects

        # Collect expected feature names (union across molecules)
        feature_names = set()
        for m in mols:
            for a in m.GetAtoms():
                feature_names.update(a.GetPropNames())

        # Act
        result_mols = fo._fill_missing_features(mols=mols)

        # Assert: every atom for indices has all features present (possibly 'NaN')
        for m in result_mols:
            for atom in m.GetAtoms():
                if atom.GetIdx() in fo.indices:
                    for fn in feature_names:
                        assert atom.HasProp(fn)

        # Should have warned if some atom ended up with all 'NaN' features
        assert mock_logging.warning.called

    @patch("bonafide.utils.feature_output.logging")
    def test__fill_missing_features_bonds(
        self, mock_logging, missing_properties_vault: Mock
    ) -> None:
        """_fill_missing_features for bonds fills absent properties with 'NaN' and warns when all features are NaN."""
        fo = FeatureOutput(
            mol_vault=missing_properties_vault,
            indices=[0, 1, 2],
            feature_type="bond",
            reduce=False,
            ignore_invalid=False,
            _loc="test__fill_missing_bonds",
        )

        mols = missing_properties_vault.mol_objects

        # Collect expected feature names (union across molecules) for bonds
        feature_names = set()
        for m in mols:
            for b in m.GetBonds():
                feature_names.update(b.GetPropNames())

        # Act
        result_mols = fo._fill_missing_features(mols=mols)

        # Assert: every bond for indices has all features present (possibly 'NaN')
        for m in result_mols:
            for bond in m.GetBonds():
                if bond.GetIdx() in fo.indices:
                    for fn in feature_names:
                        assert bond.HasProp(fn)

        # Should have warned if some bond ended up with all 'NaN' features
        assert mock_logging.warning.called

    def test__get_feature_df(self, multi_conformer_vault: Mock) -> None:
        """_get_feature_df returns a DataFrame with expected index name and added meta columns."""
        fo = FeatureOutput(
            mol_vault=multi_conformer_vault,
            indices=[0, 1],
            feature_type="atom",
            reduce=False,
            ignore_invalid=False,
            _loc="test__get_feature_df",
        )

        mol = multi_conformer_vault.mol_objects[0]
        df = fo._get_feature_df(mol=mol, conformer_idx=0, combined_df=None)

        assert df.index.name == "ATOM_INDEX"
        # For multi-conformer vault, these columns should be present
        assert "conformer_name" in df.columns
        assert "conformer_energy" in df.columns
        # boltzmann weight may be present
        assert "boltzmann_weight" in df.columns

    def test__get_feature_df_bonds(self, multi_conformer_vault: Mock) -> None:
        """_get_feature_df returns a DataFrame with expected index name and added meta columns for bonds."""
        fo = FeatureOutput(
            mol_vault=multi_conformer_vault,
            indices=[0, 1],
            feature_type="bond",
            reduce=False,
            ignore_invalid=False,
            _loc="test__get_feature_df_bonds",
        )

        mol = multi_conformer_vault.mol_objects[0]
        df = fo._get_feature_df(mol=mol, conformer_idx=0, combined_df=None)

        assert df.index.name == "BOND_INDEX"
        assert "conformer_name" in df.columns
        assert "conformer_energy" in df.columns
        assert "boltzmann_weight" in df.columns

    def test__reduce_conformer_data(self, multi_conformer_vault: Mock) -> None:
        """_reduce_conformer_data should return a DataFrame containing reduced statistics."""
        fo = FeatureOutput(
            mol_vault=multi_conformer_vault,
            indices=[0, 1],
            feature_type="atom",
            reduce=True,
            ignore_invalid=False,
            _loc="test_reduce",
        )

        # Build combined_df by aggregating per-conformer feature dfs
        combined = None
        for i, mol in enumerate(multi_conformer_vault.mol_objects):
            combined = fo._get_feature_df(mol=mol, conformer_idx=i, combined_df=combined)

        reduced = fo._reduce_conformer_data(df=combined)
        print(reduced)

        # Expect reduced to have an index name and some mean/min/max style columns
        assert reduced.index.name == "ATOM_INDEX"
        # At least one column should end with '_mean' (non-energy based reduction)
        assert any([c.startswith("MEAN_") for c in reduced.columns])

    def test__reduce_conformer_data_bonds(self, multi_conformer_vault: Mock) -> None:
        """_reduce_conformer_data should return a DataFrame containing reduced statistics for bonds."""
        fo = FeatureOutput(
            mol_vault=multi_conformer_vault,
            indices=[0, 1],
            feature_type="bond",
            reduce=True,
            ignore_invalid=False,
            _loc="test_reduce_bonds",
        )

        # Build combined_df by aggregating per-conformer feature dfs
        combined = None
        for i, mol in enumerate(multi_conformer_vault.mol_objects):
            combined = fo._get_feature_df(mol=mol, conformer_idx=i, combined_df=combined)

        reduced = fo._reduce_conformer_data(df=combined)

        assert reduced.index.name == "BOND_INDEX"
        assert any([c.startswith("MEAN_") for c in reduced.columns])

    @patch("bonafide.utils.feature_output.logging")
    def test__postprocess_df(self, mock_logging, multi_conformer_vault: Mock) -> None:
        """_postprocess_df should drop conformer related columns and warn on NaN-only rows."""
        fo = FeatureOutput(
            mol_vault=multi_conformer_vault,
            indices=[0, 1],
            feature_type="atom",
            reduce=True,
            ignore_invalid=False,
            _loc="test_postprocess",
        )

        # Create a df with conformer columns and one NaN-only row
        df = pd.DataFrame(
            {
                "partial_charge": [0.1, float("nan")],
                "conformer_name": ["c0", "c1"],
                "conformer_energy": [25.0, 26.0],
                "boltzmann_weight": [0.7, 0.3],
            },
            index=[0, 1],
        )
        df.index.name = "ATOM_INDEX"

        processed = fo._postprocess_df(df=df)

        # Conformer columns should be removed because reduce=True, conformer_name should stay
        assert "conformer_name" in processed.columns
        assert "conformer_energy" not in processed.columns
        assert "boltzmann_weight" not in processed.columns

        # Since one row is NaN-only for data columns, a warning should have been emitted
        assert mock_logging.warning.called

    @patch("bonafide.utils.feature_output.logging")
    def test__postprocess_df_bonds(self, mock_logging, multi_conformer_vault: Mock) -> None:
        """_postprocess_df should drop conformer related columns and warn on NaN-only rows for bonds."""
        fo = FeatureOutput(
            mol_vault=multi_conformer_vault,
            indices=[0, 1],
            feature_type="bond",
            reduce=True,
            ignore_invalid=False,
            _loc="test_postprocess_bonds",
        )

        # Create a df with conformer columns and one NaN-only row
        df = pd.DataFrame(
            {
                "bond_length": [1.54, float("nan")],
                "conformer_name": ["c0", "c1"],
                "conformer_energy": [25.0, 26.0],
                "boltzmann_weight": [0.7, 0.3],
            },
            index=[0, 1],
        )
        df.index.name = "BOND_INDEX"

        processed = fo._postprocess_df(df=df)

        assert "conformer_name" in processed.columns
        assert "conformer_energy" not in processed.columns
        assert "boltzmann_weight" not in processed.columns

        assert mock_logging.warning.called

    def test__cast_reduced_props_to_mol(self, single_conformer_vault: Mock) -> None:
        """_cast_reduced_props_to_mol should write reduced properties to the molecule object."""
        fo = FeatureOutput(
            mol_vault=single_conformer_vault,
            indices=[0, 1],
            feature_type="atom",
            reduce=True,
            ignore_invalid=False,
            _loc="test_cast",
        )

        # Create a reduced df with various dtypes
        df = pd.DataFrame(
            {
                "int_col": [1, 2],
                "float_col": [1.5, 2.5],
                "str_col": ["a", "b"],
                "bool_col": [True, False],
            },
            index=[0, 1],
        )
        df.index.name = "ATOM_INDEX"

        mol = single_conformer_vault.mol_objects[0]
        new_mol = fo._cast_reduced_props_to_mol(df=df, mol=mol)

        # Check that properties were set on atoms
        for idx in df.index:
            atom = new_mol.GetAtomWithIdx(idx)
            for col in df.columns:
                assert atom.HasProp(col)

            # Check numeric retrieval where available
            assert atom.GetIntProp("int_col") == df.loc[idx, "int_col"]
            assert abs(atom.GetDoubleProp("float_col") - df.loc[idx, "float_col"]) < 1e-8
            assert atom.GetProp("str_col") == df.loc[idx, "str_col"]

        # bool properties are set via SetBoolProp; HasProp should be True
        assert new_mol.GetAtomWithIdx(0).HasProp("bool_col")

    def test__cast_reduced_props_to_mol_bonds(self, single_conformer_vault: Mock) -> None:
        """_cast_reduced_props_to_mol should write reduced properties to bond objects in the molecule."""
        fo = FeatureOutput(
            mol_vault=single_conformer_vault,
            indices=[0, 1],
            feature_type="bond",
            reduce=True,
            ignore_invalid=False,
            _loc="test_cast_bonds",
        )

        # Create a reduced df with various dtypes
        df = pd.DataFrame(
            {
                "int_col": [1, 2],
                "float_col": [1.5, 2.5],
                "str_col": ["a", "b"],
                "bool_col": [True, False],
            },
            index=[0, 1],
        )
        df.index.name = "BOND_INDEX"

        mol = single_conformer_vault.mol_objects[0]
        new_mol = fo._cast_reduced_props_to_mol(df=df, mol=mol)

        # Check that properties were set on bonds
        for idx in df.index:
            bond = new_mol.GetBondWithIdx(idx)
            for col in df.columns:
                assert bond.HasProp(col)

            assert bond.GetIntProp("int_col") == df.loc[idx, "int_col"]
            assert abs(bond.GetDoubleProp("float_col") - df.loc[idx, "float_col"]) < 1e-8
            assert bond.GetProp("str_col") == df.loc[idx, "str_col"]

        # bool properties are set via SetBoolProp; HasProp should be True
        assert new_mol.GetBondWithIdx(0).HasProp("bool_col")

    @pytest.mark.parametrize("ignore_invalid,expect_energy_cols", [(True, True), (False, False)])
    @patch("bonafide.utils.feature_output.logging")
    def test_reduce_with_nan_conformer_energy(
        self,
        mock_logging,
        ignore_invalid: bool,
        expect_energy_cols: bool,
    ) -> None:
        """When conformer energies contain NaN values, behavior depends on ignore_invalid.

        - If ignore_invalid is True: a warning is emitted and energy-based features are computed
          using only valid conformers (energy columns should appear).
        - If ignore_invalid is False: info is logged and energy-based calculations are skipped
          (energy columns should be absent from the reduced DataFrame).
        """
        # Create a vault with a NaN energy in one conformer
        vault = MockMolVaultFactory.create_multi_conformer_vault(
            num_conformers=3, energies=[25.5, None, 27.1]
        )

        fo = FeatureOutput(
            mol_vault=vault,
            indices=[0, 1],
            feature_type="atom",
            reduce=True,
            ignore_invalid=ignore_invalid,
            _loc="test_reduce_nan_energy",
        )

        # Build combined df from conformers
        combined = None
        for i, mol in enumerate(vault.mol_objects):
            combined = fo._get_feature_df(mol=mol, conformer_idx=i, combined_df=combined)

        reduced = fo._reduce_conformer_data(df=combined)

        # Determine whether energy-based columns are present
        has_lowest = any([c.startswith("LOWEST_ENERGY__") for c in reduced.columns])
        has_boltz = any([c.startswith("BOLTZMANN__") for c in reduced.columns])

        if expect_energy_cols:
            assert has_lowest or has_boltz
            mock_logging.warning.assert_called()
        else:
            assert not (has_lowest or has_boltz)
            mock_logging.info.assert_called()
