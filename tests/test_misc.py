"""Miscellaneous tests for specific edge and corner cases."""

import logging
import os
from typing import Callable

import numpy as np
import pytest

from bonafide import AtomBondFeaturizer

#######################################################################
# Check combination of attach_smiles with attach_electronic_structure #
#######################################################################

# Copied from the xyz file
_REF_XYZ_COORDINATES = np.array(
    [
        [0.77388841613048, 0.58674010134478, -0.89274836586007],
        [2.20092369516530, 0.88785827075375, -0.54903559763341],
        [2.79229957230656, 2.06992483652472, -0.97140753697272],
        [4.11671659703431, 2.32980417576442, -0.66734492542802],
        [4.83885570051199, 3.80605176366704, -1.20326464389222],
        [4.87405155117325, 1.42169044970159, 0.05026473202767],
        [4.30274170122295, 0.23582292830632, 0.48898035856099],
        [5.10605665819679, -0.74938265059168, 1.28113608730754],
        [2.96253836495671, -0.02095004378590, 0.19049914340351],
        [2.40858203610829, -1.22546389390337, 0.60452683820302],
        [1.58989025674197, -1.14894313235419, 1.74678118575774],
        [0.25393751697626, -1.85791403563287, 1.56602336104810],
        [-0.59014280841820, -1.89112522408953, 2.42198228972300],
        [0.15135125699371, -2.43787152732264, 0.37581156667450],
        [-1.07538984299337, -3.11271251540007, 0.07459391671000],
        [-0.79517584084177, -4.23071938521842, -0.91452749498216],
        [-2.04186289706408, -2.12353328872998, -0.48611596116432],
        [-2.84254274034567, -2.28063258079753, -1.48914369557041],
        [-3.53317219164771, -1.13819372686263, -1.62548953172365],
        [-3.12262051272502, -0.32544717652426, -0.69753386235224],
        [-3.58098692861413, 1.02071813941183, -0.44556864093641],
        [-3.08679082191909, 1.74325061001735, 0.63999463330883],
        [-3.53675872123371, 3.02692178492436, 0.87777923059549],
        [-4.48052878428381, 3.57081029304782, 0.02274852579623],
        [-4.92181947224272, 4.82035315593810, 0.25314528579425],
        [-4.98699568764019, 2.86962140237914, -1.06031940956722],
        [-4.53309910498527, 1.58787206422267, -1.29374824572445],
        [-2.16343446046633, -0.90085125555943, 0.06719634044807],
        [0.47293842272976, 1.12798635577026, -1.78617112124931],
        [0.65298071859340, -0.48114949774144, -1.06077100041288],
        [0.10984552645521, 0.88295674609536, -0.07905146862138],
        [2.22023942820107, 2.78282933091196, -1.54459008037555],
        [5.90822526748410, 1.63498307095687, 0.27116406045760],
        [5.04253766484778, -0.51414127160996, 2.34408140672787],
        [4.72256365018043, -1.75468917912887, 1.12893803652074],
        [6.15201021480150, -0.71375716386631, 0.98898597504835],
        [2.09375530464267, -1.63154965972500, 2.59832428292250],
        [1.38044726696555, -0.11162810961873, 2.03435398152421],
        [-1.49021754142158, -3.50208398934074, 1.01778693449612],
        [-1.73328101387201, -4.68763975040358, -1.21165116711631],
        [-0.30763120863337, -3.82511350979344, -1.79610912411285],
        [-0.14680056662276, -4.97388058433431, -0.45989561539737],
        [-2.36123809780702, 1.29072749257986, 1.29891305938965],
        [-3.17241345713767, 3.60536504813977, 1.71176314980535],
        [-5.72344605721833, 3.33127410754090, -1.69838754213716],
        [-4.90102303018957, 1.00580902438367, -2.12289835103496],
    ]
)

# Copied from the xyz file
_REF_ELEMENTS = np.array(
    [
        "C",
        "C",
        "C",
        "C",
        "Cl",
        "C",
        "C",
        "C",
        "C",
        "O",
        "C",
        "C",
        "O",
        "O",
        "C",
        "C",
        "C",
        "N",
        "N",
        "C",
        "C",
        "C",
        "C",
        "C",
        "F",
        "C",
        "C",
        "O",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
    ]
)


@pytest.mark.miscellaneous
@pytest.mark.parametrize("align", [True, False])
def test_miscellaneous(
    caplog: pytest.LogCaptureFixture,
    fetch_data_file: Callable[[str], str],
    clean_up_logfile: Callable[[str], None],
    align: bool,
) -> None:
    """Test that the combination of attach_smiles with attach_electronic_structure works as
    expected. The atom order from the electronic structure data file must match the data
    already present in the MolVault.

    .. versionadded:: 0.2.4
    """
    try:
        # Get input file paths
        xyz_input_file_path = fetch_data_file("misc_mol.xyz")
        fchk_input_file_path = fetch_data_file("misc_mol.fchk")

        # Read all input
        f = AtomBondFeaturizer()
        f.read_input(
            input_value=xyz_input_file_path,
            namespace="irrelevant",
            input_format="file",
            output_directory="just_some_out_dir",
        )
        f.set_charge(0)

        f.attach_smiles(
            "[H]c1c([H])c(-c2nnc([C@]([H])(OC(=O)C([H])([H])Oc3c(C([H])([H])[H])c([H])c(Cl)c([H])"
            "c3C([H])([H])[H])C([H])([H])[H])o2)c([H])c([H])c1F",
            align=align,
        )

        if align is True:
            f.attach_electronic_structure(fchk_input_file_path)
            coords = f.mol_vault.mol_objects[0].GetConformer().GetPositions()

            # Check values
            assert bool(np.allclose(coords, _REF_XYZ_COORDINATES))
            assert bool(np.all(f.mol_vault.elements == _REF_ELEMENTS))
            assert bool(
                np.all(
                    np.array([atom.GetSymbol() for atom in f.mol_vault.mol_objects[0].GetAtoms()])
                    == _REF_ELEMENTS
                )
            )

            # Check logs
            assert all(record.levelno == logging.INFO for record in caplog.records)

        else:
            with pytest.raises(
                ValueError,
                match="The structure sanity check of the electronic structure data file failed for "
                "conformer with index 0 for state 'n'",
            ):
                f.attach_electronic_structure(fchk_input_file_path)

            coords = f.mol_vault.mol_objects[0].GetConformer().GetPositions()

            # Check values
            assert bool(np.allclose(coords, _REF_XYZ_COORDINATES)) is False
            assert bool(np.all(f.mol_vault.elements == _REF_ELEMENTS)) is False
            assert (
                bool(
                    np.all(
                        np.array(
                            [atom.GetSymbol() for atom in f.mol_vault.mol_objects[0].GetAtoms()]
                        )
                        == _REF_ELEMENTS
                    )
                )
                is False
            )

            # Check logs
            assert any(record.levelno == logging.ERROR for record in caplog.records)

        # Check logs (general)
        assert len(caplog.records) > 0

    # Clean up
    finally:
        clean_up_logfile()
        if os.path.exists("just_some_out_dir"):
            os.rmdir(path="just_some_out_dir")
