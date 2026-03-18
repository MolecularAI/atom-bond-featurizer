"""Fuzzy space features from ``Multiwfn``."""

import os
import shutil
from typing import Dict, List, Optional, Union

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.constants import PROGRAM_ENVIRONMENT_VARIABLES
from bonafide.utils.driver import multiwfn_driver
from bonafide.utils.helper_functions import matrix_parser


class _Multiwfn3DFuzzySpace(BaseFeaturizer):
    """Parent feature factory for the 3D atom and bond Multiwfn fuzzy space features.

    For details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last accessed
    on 12.09.2025).
    """

    exclude_atoms: List[int]
    integration_grid: int
    n_iterations_becke_partition: int
    partitioning_scheme: int
    radius_becke_partition: int
    real_space_function: int

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def _run_multiwfn(
        self,
        command_list: List[Union[int, float, str]],
        output_file_prefix: Optional[str] = None,
        select_scheme: bool = True,
        enforce_closed_shell: Optional[str] = None,
        enforce_open_shell: bool = False,
    ) -> bool:
        """Run Multiwfn.

        Parameters
        ----------
        command_list : List[Union[int, float, str]]
            List of commands to be passed to Multiwfn to select the respective fuzzy space analysis
            method.
        output_file_prefix : Optional[str], optional
            Prefix to be added to the output file name, by default ``None``. If ``None``, the
            prefix is automatically constructed from the name of the class.
        select_scheme : bool, optional
            Whether to select a partitioning scheme for the fuzzy space analysis, by default
            ``True``.
        enforce_closed_shell : Optional[str], optional
            Whether to enforce closed-shell electronic structure for the calculation, by default
            ``None``. If set to a string value, the calculation is only run if the molecule is
            closed-shell.
        enforce_open_shell : bool, optional
            Whether to enforce open-shell electronic structure for the calculation, by default
            ``False``. If set to ``True``, the calculation is only run if the molecule is
            open-shell.

        Returns
        -------
        bool
            Returns ``False`` if a feature only defined for closed-shell molecules is requested for
            an open-shell molecule. If the input is correct and Multiwfn is executed, ``True`` is
            returned.
        """
        # Is open-shell but must be closed-shell case
        _namespace = self.conformer_name[::-1].split("__", 1)[-1][::-1]
        if self.multiplicity != 1 and enforce_closed_shell is not None:
            # Set one atom/bond feature to _inaccessible, the rest is done by the base featurizer
            self.results[0] = {self.feature_name: "_inaccessible"}
            return False

        # Is closed-shell but must be open-shell case
        if self.multiplicity == 1 and enforce_open_shell is True:
            # Set one atom feature to _inaccessible, the rest is done by the base featurizer
            self.results[0] = {self.feature_name: "_inaccessible"}
            return False

        # Select fuzzy space
        multiwfn_commands: List[Union[str, int, float]]
        multiwfn_commands = [15]

        # Choose type of integration grid
        multiwfn_commands.extend([-6, self.integration_grid])

        # Exclude atoms (if requested)
        if self.exclude_atoms != []:
            multiwfn_commands.extend(
                [
                    -5,
                    ",".join(
                        [
                            str(atom.GetIdx() + 1)
                            for atom in self.mol.GetAtoms()
                            if atom.GetIdx() not in self.exclude_atoms
                        ]
                    ),
                ]
            )

        # Set the number of iterations for Becke partitioning
        multiwfn_commands.extend([-3, self.n_iterations_becke_partition])

        # Select radius definition for Becke partition
        multiwfn_commands.extend([-2, self.radius_becke_partition, 0])

        # Select method for partitioning atomic space
        if select_scheme is True:
            multiwfn_commands.extend([-1, self.partitioning_scheme])

        # Add commands from the child class to the Multiwfn command
        multiwfn_commands.extend(command_list)

        # Exit Multiwfn
        multiwfn_commands.extend([0, "q"])

        # Modify output file name
        if output_file_prefix is None:
            output_file_name = f"{self.__class__.__name__}__{self.conformer_name}"
        else:
            output_file_name = f"{output_file_prefix}__{self.conformer_name}"

        # Set up environment variables
        environment_variables = {
            var: getattr(self, var, None) for var in PROGRAM_ENVIRONMENT_VARIABLES["multiwfn"]
        }

        # Run Multiwfn
        multiwfn_driver(
            cmds=multiwfn_commands,
            input_file_path=str(self.electronic_struc_n),
            output_file_name=output_file_name,
            environment_variables=environment_variables,
            namespace=_namespace,
        )
        return True

    def _read_output_file_atom(self) -> None:
        """Read the output file from Multiwfn and write the results to the ``results`` dictionary.

        This method is used read the data for the integration descriptor features.

        Returns
        -------
        None
        """
        # Check if the output file exists
        _opath = f"{self.__class__.__name__}__{self.conformer_name}.out"
        if os.path.isfile(_opath) is False:
            self._err = (
                f"Multiwfn output file '{_opath}' not found; probably the calculation "
                "did not run. Check your input"
            )
            return

        # Open output file
        with open(_opath, "r") as f:
            multiwfn_output = f.readlines()

        # Find relevant position in the file
        start_idx = None
        for line_idx, line in enumerate(multiwfn_output):
            if all(
                [
                    "Atomic space" in line,
                    "Value" in line,
                    "% of sum" in line,
                    "% of sum abs" in line,
                ]
            ):
                start_idx = line_idx + 1

        # Check if start_idx was found
        if start_idx is None:
            self._err = (
                f"output file generated through '{self.__class__.__name__}' does not "
                "contain the requested data; probably the calculation failed. Check the "
                "output file"
            )
            return

        # Extract data and write it to the results dictionary
        for line in multiwfn_output[start_idx:]:
            if line.startswith(" Summing up above values:"):
                break

            atom_idx = int(line.split("(")[0])
            splitted = line.split()
            val = float(splitted[-3])
            rel_sum = float(splitted[-2])
            abs_rel_sum = float(splitted[-1])

            self.results[atom_idx - 1] = {
                "multiwfn3D-atom-fuzzy_space_integration_descriptor": val,
                "multiwfn3D-atom-fuzzy_space_integration_descriptor_relative": rel_sum,
                "multiwfn3D-atom-fuzzy_space_integration_descriptor_abs_relative": abs_rel_sum,
            }

    def _read_output_file_atom2(self, is_open_shell: bool) -> None:
        """Read the output file from Multiwfn and write the results to the ``results`` dictionary.

        This method is used read the data for the atomic valence and localization index features.

        Parameters
        ----------
        is_open_shell : bool
            Whether the calculation was performed for an open-shell molecule.

        Returns
        -------
        None
        """
        # Check if the output file exists
        _opath = f"{self.__class__.__name__}__{self.conformer_name}.out"
        if os.path.isfile(_opath) is False:
            self._err = (
                f"Multiwfn output file '{_opath}' not found; probably the calculation "
                "did not run. Check your input"
            )
            return

        # Open output file
        with open(_opath, "r") as f:
            multiwfn_output = f.readlines()

        # Find relevant positions in the file
        start_idx = None
        start_idx2 = None
        start_idx3 = None
        start_idx4 = None
        start_idx5 = None
        start_idx6 = None
        for line_idx, line in enumerate(multiwfn_output):
            if line.startswith(
                " **************** Delocalization index matrix for alpha spin ****************"
            ):
                start_idx = line_idx + 1

            if line.startswith(" Localization index for alpha spin:"):
                start_idx2 = line_idx + 1

            if line.startswith(
                " ***************** Delocalization index matrix for beta spin *****************"
            ):
                start_idx3 = line_idx + 1

            if line.startswith(" Localization index for beta spin:"):
                start_idx4 = line_idx + 1

            if line.startswith(
                " ********************* Total delocalization index matrix *********************"
            ):
                start_idx5 = line_idx + 1

            if line.startswith(" Localization index:"):
                start_idx6 = line_idx + 1

        # Check if all required start indices were found (depending on open-/closed-shell)
        if any(
            [
                start_idx5 is None,
                start_idx6 is None,
                (start_idx is None and is_open_shell is True),
                (start_idx2 is None and is_open_shell is True),
                (start_idx3 is None and is_open_shell is True),
                (start_idx4 is None and is_open_shell is True),
            ]
        ):
            self._err = (
                f"output file generated through '{self.__class__.__name__}' does not "
                "contain the requested data; probably the calculation failed. Check the "
                "output file"
            )
            return

        # Extract data
        delocalization_index_matrix_alpha = []
        delocalization_index_matrix_beta = []
        delocalization_index_matrix = []

        localization_index_block_alpha = []
        localization_index_block_beta = []
        localization_index_block = []

        # Data available in open- and closed-shell case
        for line in multiwfn_output[start_idx5:]:
            if line.strip() == "":
                break
            delocalization_index_matrix.append(line)

        for line in multiwfn_output[start_idx6:]:
            if line.strip() == "":
                break
            localization_index_block.append(line)

        # Data only available in open-shell case
        if is_open_shell is True:
            for line in multiwfn_output[start_idx:]:
                if line.strip() == "":
                    break
                delocalization_index_matrix_alpha.append(line)

            for line in multiwfn_output[start_idx3:]:
                if line.strip() == "":
                    break
                delocalization_index_matrix_beta.append(line)

            for line in multiwfn_output[start_idx2:]:
                if line.strip() == "":
                    break
                localization_index_block_alpha.append(line)

            for line in multiwfn_output[start_idx4:]:
                if line.strip() == "":
                    break
                localization_index_block_beta.append(line)

        # Parse data and write it to the results dictionary
        self._parse_delocalization_matrix(
            files_lines=delocalization_index_matrix,
            loc=self.__class__.__name__,
            feature_name="multiwfn3D-atom-fuzzy_space_atomic_valence",
        )

        self._parse_localization_index_block(
            files_lines=localization_index_block,
            feature_name="multiwfn3D-atom-fuzzy_space_localization_index",
        )

        if is_open_shell is True:
            self._parse_delocalization_matrix(
                files_lines=delocalization_index_matrix_alpha,
                loc=self.__class__.__name__,
                feature_name="multiwfn3D-atom-fuzzy_space_atomic_valence_alpha",
            )
            self._parse_delocalization_matrix(
                files_lines=delocalization_index_matrix_beta,
                loc=self.__class__.__name__,
                feature_name="multiwfn3D-atom-fuzzy_space_atomic_valence_beta",
            )

            self._parse_localization_index_block(
                files_lines=localization_index_block_alpha,
                feature_name="multiwfn3D-atom-fuzzy_space_localization_index_alpha",
            )
            self._parse_localization_index_block(
                files_lines=localization_index_block_beta,
                feature_name="multiwfn3D-atom-fuzzy_space_localization_index_beta",
            )

    def _read_output_file_atom3(self) -> None:
        """Read the output file from Multiwfn and write the results to the ``results`` dictionary.

        This method is used to read the data for all fuzzy space features that are not covered by
        the other methods (``_read_output_file_atom()``, ``_read_output_file_bond()``, and
        ``_read_output_file_clrk()``).

        Returns
        -------
        None
        """
        # Check if the output file exists
        _opath = f"{self.__class__.__name__}__{self.conformer_name}.out"
        if os.path.isfile(_opath) is False:
            self._err = (
                f"Multiwfn output file '{_opath}' not found; probably the calculation "
                "did not run. Check your input"
            )
            return

        # Open output file
        with open(_opath, "r") as f:
            multiwfn_output = f.readlines()

        # Extract data and write it to the results dictionary
        for line_idx, line in enumerate(multiwfn_output):
            if all(
                [
                    line.count("*****") == 2,
                    "Atom" in line,
                    line.count("(") == 1,
                ]
            ):
                atom_idx = int(line.split("Atom")[-1].split("(")[0])

                _results: Dict[str, Optional[Union[str, float]]] = {
                    "multiwfn3D-atom-fuzzy_space_monopole_moment": None,
                    "multiwfn3D-atom-fuzzy_space_dipole_moment": None,
                    "multiwfn3D-atom-fuzzy_space_dipole_moment_norm": None,
                    "multiwfn3D-atom-fuzzy_space_contribution_to_system_dipole_moment": None,
                    "multiwfn3D-atom-fuzzy_space_contribution_to_system_dipole_moment_norm": None,
                    "multiwfn3D-atom-fuzzy_space_traceless_quadrupole_moment_tensor_magnitude": None,
                    "multiwfn3D-atom-fuzzy_space_quadrupole_moment_magnitude": None,
                    "multiwfn3D-atom-fuzzy_space_octopole_moment_magnitude": None,
                    "multiwfn3D-atom-fuzzy_space_atomic_electronic_spatial_extent": None,
                }

                for line_idx_a, line_a in enumerate(multiwfn_output[line_idx + 1 :]):
                    if line_a.strip() == "":
                        break

                    line_idx_a += line_idx + 1

                    if line_a.startswith(" Atomic monopole moment (from electrons):"):
                        monopole = float(line_a.split(":")[-1])
                        _results["multiwfn3D-atom-fuzzy_space_monopole_moment"] = monopole

                    if line_a.startswith(" Atomic dipole moments:"):
                        _x_, _y_, _z_, norm_ = multiwfn_output[line_idx_a + 1].split("=")[1:]
                        _x = float(_x_.split()[0])
                        _y = float(_y_.split()[0])
                        _z = float(_z_.split()[0])
                        norm = float(norm_)
                        dipole = ",".join([str(_x), str(_y), str(_z)])
                        _results["multiwfn3D-atom-fuzzy_space_dipole_moment"] = dipole
                        _results["multiwfn3D-atom-fuzzy_space_dipole_moment_norm"] = norm

                    if line_a.startswith(" Contribution to molecular dipole moment:"):
                        _x_, _y_, _z_, contrib_norm_ = multiwfn_output[line_idx_a + 1].split("=")[
                            1:
                        ]
                        _x = float(_x_.split()[0])
                        _y = float(_y_.split()[0])
                        _z = float(_z_.split()[0])
                        contrib_norm = float(contrib_norm_)
                        contrib_dipole = ",".join([str(_x), str(_y), str(_z)])
                        _results[
                            "multiwfn3D-atom-fuzzy_space_contribution_to_system_dipole_moment"
                        ] = contrib_dipole
                        _results[
                            "multiwfn3D-atom-fuzzy_space_contribution_to_system_dipole_moment_norm"
                        ] = contrib_norm

                    if line_a.startswith(" Magnitude of the traceless quadrupole moment tensor:"):
                        traceless_quadrupole = float(line_a.split(":")[-1])
                        _results[
                            "multiwfn3D-atom-fuzzy_space_traceless_quadrupole_moment_tensor_magnitude"
                        ] = traceless_quadrupole

                    if line_a.startswith(" Magnitude: |Q_2|="):
                        quadrupole = float(line_a.split("=")[-1])
                        _results["multiwfn3D-atom-fuzzy_space_quadrupole_moment_magnitude"] = (
                            quadrupole
                        )

                    if line_a.startswith(" Magnitude: |Q_3|="):
                        octopole = float(line_a.split("=")[-1])
                        _results["multiwfn3D-atom-fuzzy_space_octopole_moment_magnitude"] = octopole

                    if line_a.startswith(" Atomic electronic spatial extent <r^2>:"):
                        spatial_extent = float(line_a.split(":")[-1])
                        _results["multiwfn3D-atom-fuzzy_space_atomic_electronic_spatial_extent"] = (
                            spatial_extent
                        )

                    self.results[atom_idx - 1] = _results

    def _read_output_file_bond(self) -> None:
        """Read the output file from Multiwfn and write the results to the ``results`` dictionary.

        This method is used read the data for all fuzzy space bond features.

        Returns
        -------
        None
        """
        # Check if the output file exists
        _opath = f"{self.__class__.__name__}__{self.conformer_name}.out"
        if os.path.isfile(_opath) is False:
            self._err = (
                f"Multiwfn output file '{_opath}' not found; probably the calculation "
                "did not run. Check your input"
            )
            return

        # Open output file
        with open(_opath, "r") as f:
            multiwfn_output = f.readlines()

        # Extract matrix blocks
        positive_block = []
        negative_block = []
        total_block = []
        for line_idx, line in enumerate(multiwfn_output):
            if line.startswith(
                " ************* Integration of positive values in overlap region *************"
            ):
                for line_a in multiwfn_output[line_idx + 1 :]:
                    if line_a.startswith(" Summing up diagonal matrix elements:"):
                        break
                    positive_block.append(line_a)

            if line.startswith(
                " ************* Integration of negative values in overlap region *************"
            ):
                for line_a in multiwfn_output[line_idx + 1 :]:
                    if line_a.startswith(" Summing up diagonal matrix elements:"):
                        break
                    negative_block.append(line_a)

            if line.startswith(
                " **************** Integration of all values in overlap region ****************"
            ):
                for line_a in multiwfn_output[line_idx + 1 :]:
                    if line_a.startswith(" Summing up diagonal matrix elements:"):
                        break
                    total_block.append(line_a)

        # Extract data for every bond and write it to the results dictionary
        positive_matrix, error_message_positive = matrix_parser(
            files_lines=positive_block,
            n_atoms=self.mol.GetNumAtoms(),
        )
        negative_matrix, error_message_negative = matrix_parser(
            files_lines=negative_block,
            n_atoms=self.mol.GetNumAtoms(),
        )
        total_matrix, error_message_total = matrix_parser(
            files_lines=total_block,
            n_atoms=self.mol.GetNumAtoms(),
        )

        _errors = [error_message_positive, error_message_negative, error_message_total]
        for e in _errors:
            if e is not None:
                self._err = (
                    f"output file generated through '{self.__class__.__name__}' does not "
                    f"contain the requested data: {e}. Check the output file"
                )
                return

        assert positive_matrix is not None  # for type checker
        assert negative_matrix is not None  # for type checker
        assert total_matrix is not None  # for type checker

        for bond in self.mol.GetBonds():
            bond_idx = bond.GetIdx()
            begin_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()

            pos = positive_matrix[begin_atom_idx][end_atom_idx]
            neg = negative_matrix[begin_atom_idx][end_atom_idx]
            tot = total_matrix[begin_atom_idx][end_atom_idx]

            self.results[bond_idx] = {
                "multiwfn3D-bond-fuzzy_space_overlap_integration_descriptor_positive": pos,
                "multiwfn3D-bond-fuzzy_space_overlap_integration_descriptor_negative": neg,
                "multiwfn3D-bond-fuzzy_space_overlap_integration_descriptor": tot,
            }

    def _read_output_file_clrk(self, feature_type: str) -> None:
        """Read the output file from Multiwfn and write the results to the ``results`` dictionary.

        This method is used read the data for the condensed linear response kernel matrix element
        features, both for atoms and bonds.

        Parameters
        ----------
        feature_type : str
            The type of the feature to calculate, either "atom" or "bond".

        Returns
        -------
        None
        """
        # Check if the output file exists
        _opath = (
            f"Multiwfn3DFuzzySpaceCondensedLinearResponseKernelMatrixElement"
            f"__{self.conformer_name}.out"
        )
        if os.path.isfile(_opath) is False:
            self._err = (
                f"Multiwfn output file '{_opath}' not found; probably the calculation "
                "did not run. Check your input"
            )
            return

        # Open output file
        with open(_opath, "r") as f:
            multiwfn_output = f.readlines()

        # Extract matrix blocks
        matrix_lines = []
        for line_idx, line in enumerate(multiwfn_output):
            if line.startswith(
                " ************** Condensed linear response kernel (CLRK) matrix **************"
            ):
                for line_a in multiwfn_output[line_idx + 1 :]:
                    if line_a.strip() == "":
                        break
                    matrix_lines.append(line_a)

        # Parse matrix
        matrix, error_message = matrix_parser(
            files_lines=matrix_lines,
            n_atoms=self.mol.GetNumAtoms(),
        )

        if error_message is not None:
            self._err = (
                f"output file generated through '{self.__class__.__name__}' does not "
                f"contain the requested data: {error_message}. Check the output file"
            )
            return

        assert matrix is not None  # for type checker

        # Write data to results dictionary for all atoms
        if feature_type == "atom":
            for atom in self.mol.GetAtoms():
                atom_idx = atom.GetIdx()
                val = matrix[atom_idx][atom_idx]
                self.results[atom_idx] = {self.feature_name: val}

        # Write data to results dictionary for all bonds
        if feature_type == "bond":
            for bond in self.mol.GetBonds():
                bond_idx = bond.GetIdx()
                begin_atom_idx = bond.GetBeginAtomIdx()
                end_atom_idx = bond.GetEndAtomIdx()
                val = matrix[begin_atom_idx][end_atom_idx]
                self.results[bond_idx] = {self.feature_name: val}

    def _clrk_exists(self) -> bool:
        """Check if the CLRK output file already exists.

        If it already exists, it is moved to the current working directory.

        Returns
        -------
        bool
            Whether the CLRK output file already exists.
        """
        base_path = os.path.dirname(os.getcwd())
        path = os.path.join(
            base_path,
            f"Multiwfn3DFuzzySpaceCondensedLinearResponseKernelMatrixElement"
            f"__{self.conformer_name}.out",
        )
        file_exists = os.path.isfile(path)
        if file_exists is True:
            shutil.move(path, os.getcwd())
        return file_exists

    def _parse_delocalization_matrix(
        self, files_lines: List[str], loc: str, feature_name: str
    ) -> None:
        """Extract the delocalization matrix data from the file format and write it to the results
        dictionary.

        Parameters
        ----------
        files_lines : List[str]
            The lines of the file containing the delocalization matrix.
        loc : str
            The location (name of the calling method) for error messages.
        feature_name : str
            The name of the feature.

        Returns
        -------
        None
        """
        matrix, error_message = matrix_parser(
            files_lines=files_lines,
            n_atoms=self.mol.GetNumAtoms(),
        )

        if error_message is not None:
            self._err = (
                f"output file generated through '{loc}' does not contain the requested data: "
                f"{error_message}. Check the output file"
            )
            return

        assert matrix is not None  # for type checker

        # Write data to results dictionary
        for atom in self.mol.GetAtoms():
            atom_idx = atom.GetIdx()
            val = matrix[atom_idx][atom_idx]

            if atom_idx not in self.results:
                self.results[atom_idx] = {}
            self.results[atom_idx][feature_name] = val

    def _parse_localization_index_block(self, files_lines: List[str], feature_name: str) -> None:
        """Extract the localization index data from the file format and write it to the results
        dictionary.

        Parameters
        ----------
        files_lines : List[str]
            The lines of the file containing the localization indices.
        feature_name : str
            The name of the feature.

        Returns
        -------
        None
        """
        for line in files_lines:
            splitted = line.split(":")
            idx_val_pairs: List[Union[int, float]] = [
                int(splitted[0].split("(")[0])
            ]  # First atom index
            for s in splitted[1:-1]:
                _val, _atom_idx = s.split("(")[0].split()
                idx_val_pairs.append(float(_val))
                idx_val_pairs.append(int(_atom_idx))
            idx_val_pairs.append(float(splitted[-1]))  # Last value

            for i in range(0, len(idx_val_pairs), 2):
                atom_idx = idx_val_pairs[i] - 1
                val = idx_val_pairs[i + 1]

                if atom_idx not in self.results:
                    self.results[int(atom_idx)] = {}
                self.results[int(atom_idx)][feature_name] = val


class Multiwfn3DAtomFuzzySpaceAtomicElectronicSpatialExtent(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D atom feature "fuzzy_space_atomic_electronic_spatial_extent",
    calculated with multiwfn.

    The index of this feature is 236 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-fuzzy_space_atomic_electronic_spatial_extent``
        feature."""
        self._run_multiwfn(command_list=[2, 1])
        self._read_output_file_atom3()


class Multiwfn3DAtomFuzzySpaceAtomicValence(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D atom feature "fuzzy_space_atomic_valence", calculated with
    multiwfn.

    The index of this feature is 237 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-fuzzy_space_atomic_valence`` feature."""
        self._run_multiwfn(command_list=[4, "n"])
        self._read_output_file_atom2(is_open_shell=True if self.multiplicity != 1 else False)


class Multiwfn3DAtomFuzzySpaceAtomicValenceAlpha(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D atom feature "fuzzy_space_atomic_valence_alpha", calculated
    with multiwfn.

    The index of this feature is 238 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-fuzzy_space_atomic_valence_alpha`` feature."""
        _read_output = self._run_multiwfn(command_list=[4, "n"], enforce_open_shell=True)
        if _read_output is True:
            self._read_output_file_atom2(is_open_shell=True)


class Multiwfn3DAtomFuzzySpaceAtomicValenceBeta(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D atom feature "fuzzy_space_atomic_valence_beta", calculated
    with multiwfn.

    The index of this feature is 239 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-fuzzy_space_atomic_valence_beta`` feature."""
        _read_output = self._run_multiwfn(command_list=[4, "n"], enforce_open_shell=True)
        if _read_output is True:
            self._read_output_file_atom2(is_open_shell=True)


class Multiwfn3DAtomFuzzySpaceCondensedLinearResponseKernelMatrixElement(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D atom feature
    "fuzzy_space_condensed_linear_response_kernel_matrix_element", calculated with multiwfn.

    The index of this feature is 240 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the
        ``multiwfn3D-atom-fuzzy_space_condensed_linear_response_kernel_matrix_element``
        feature."""
        _read_output = True
        if self._clrk_exists() is False:
            _read_output = self._run_multiwfn(
                command_list=[9, "n"],
                output_file_prefix="Multiwfn3DFuzzySpaceCondensedLinearResponseKernelMatrixElement",
                enforce_closed_shell="atoms",
            )
        if _read_output is True:
            self._read_output_file_clrk(feature_type="atom")


class Multiwfn3DAtomFuzzySpaceContributionToSystemDipoleMoment(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D atom feature
    "fuzzy_space_contribution_to_system_dipole_moment", calculated with multiwfn.

    The index of this feature is 241 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-fuzzy_space_contribution_to_system_dipole_moment``
        feature."""
        self._run_multiwfn(command_list=[2, 1])
        self._read_output_file_atom3()


class Multiwfn3DAtomFuzzySpaceContributionToSystemDipoleMomentNorm(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D atom feature
    "fuzzy_space_contribution_to_system_dipole_moment_norm", calculated with multiwfn.

    The index of this feature is 242 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the
        ``multiwfn3D-atom-fuzzy_space_contribution_to_system_dipole_moment_norm`` feature."""
        self._run_multiwfn(command_list=[2, 1])
        self._read_output_file_atom3()


class Multiwfn3DAtomFuzzySpaceDipoleMoment(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D atom feature "fuzzy_space_dipole_moment", calculated with
    multiwfn.

    The index of this feature is 243 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-fuzzy_space_dipole_moment`` feature."""
        self._run_multiwfn(command_list=[2, 1])
        self._read_output_file_atom3()


class Multiwfn3DAtomFuzzySpaceDipoleMomentNorm(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D atom feature "fuzzy_space_dipole_moment_norm", calculated with
    multiwfn.

    The index of this feature is 244 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-fuzzy_space_dipole_moment_norm`` feature."""
        self._run_multiwfn(command_list=[2, 1])
        self._read_output_file_atom3()


class Multiwfn3DAtomFuzzySpaceIntegrationDescriptor(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D atom feature "fuzzy_space_integration_descriptor", calculated
    with multiwfn.

    The index of this feature is 245 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-fuzzy_space_integration_descriptor`` feature."""
        self._run_multiwfn(command_list=[1, self.real_space_function])
        self._read_output_file_atom()


class Multiwfn3DAtomFuzzySpaceIntegrationDescriptorAbsRelative(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D atom feature
    "fuzzy_space_integration_descriptor_abs_relative", calculated with multiwfn.

    The index of this feature is 246 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-fuzzy_space_integration_descriptor_abs_relative``
        feature."""
        self._run_multiwfn(command_list=[1, self.real_space_function])
        self._read_output_file_atom()


class Multiwfn3DAtomFuzzySpaceIntegrationDescriptorRelative(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D atom feature "fuzzy_space_integration_descriptor_relative",
    calculated with multiwfn.

    The index of this feature is 247 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-fuzzy_space_integration_descriptor_relative``
        feature."""
        self._run_multiwfn(command_list=[1, self.real_space_function])
        self._read_output_file_atom()


class Multiwfn3DAtomFuzzySpaceMonopoleMoment(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D atom feature "fuzzy_space_monopole_moment", calculated with
    multiwfn.

    The index of this feature is 251 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-fuzzy_space_monopole_moment`` feature."""
        self._run_multiwfn(command_list=[2, 1])
        self._read_output_file_atom3()


class Multiwfn3DAtomFuzzySpaceLocalizationIndex(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D atom feature "fuzzy_space_localization_index", calculated with
    multiwfn.

    The index of this feature is 248 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-fuzzy_space_localization_index`` feature."""
        self._run_multiwfn(command_list=[4, "n"])
        self._read_output_file_atom2(is_open_shell=True if self.multiplicity != 1 else False)


class Multiwfn3DAtomFuzzySpaceLocalizationIndexAlpha(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D atom feature "fuzzy_space_localization_index_alpha",
    calculated with multiwfn.

    The index of this feature is 249 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-fuzzy_space_localization_index_alpha`` feature."""
        _read_output = self._run_multiwfn(command_list=[4, "n"], enforce_open_shell=True)
        if _read_output is True:
            self._read_output_file_atom2(is_open_shell=True)


class Multiwfn3DAtomFuzzySpaceLocalizationIndexBeta(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D atom feature "fuzzy_space_localization_index_beta", calculated
    with multiwfn.

    The index of this feature is 250 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-fuzzy_space_localization_index_beta`` feature."""
        _read_output = self._run_multiwfn(command_list=[4, "n"], enforce_open_shell=True)
        if _read_output is True:
            self._read_output_file_atom2(is_open_shell=True)


class Multiwfn3DAtomFuzzySpaceOctopoleMomentMagnitude(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D atom feature "fuzzy_space_octopole_moment_magnitude",
    calculated with multiwfn.

    The index of this feature is 252 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-fuzzy_space_octopole_moment_magnitude`` feature."""
        self._run_multiwfn(command_list=[2, 1])
        self._read_output_file_atom3()


class Multiwfn3DAtomFuzzySpaceQuadrupoleMomentMagnitude(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D atom feature "fuzzy_space_quadrupole_moment_magnitude",
    calculated with multiwfn.

    The index of this feature is 253 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-fuzzy_space_quadrupole_moment_magnitude`` feature."""
        self._run_multiwfn(command_list=[2, 1])
        self._read_output_file_atom3()


class Multiwfn3DAtomFuzzySpaceTracelessQuadrupoleMomentTensorMagnitude(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D atom feature
    "fuzzy_space_traceless_quadrupole_moment_tensor_magnitude", calculated with multiwfn.

    The index of this feature is 254 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the
        ``multiwfn3D-atom-fuzzy_space_traceless_quadrupole_moment_tensor_magnitude`` feature."""
        self._run_multiwfn(command_list=[2, 1])
        self._read_output_file_atom3()


class Multiwfn3DBondFuzzySpaceCondensedLinearResponseKernelMatrixElement(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D bond feature
    "fuzzy_space_condensed_linear_response_kernel_matrix_element", calculated with multiwfn.

    The index of this feature is 435 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the
        ``multiwfn3D-bond-fuzzy_space_condensed_linear_response_kernel_matrix_element``
        feature."""
        _read_output = True
        if self._clrk_exists() is False:
            _read_output = self._run_multiwfn(
                command_list=[9, "n"],
                output_file_prefix="Multiwfn3DFuzzySpaceCondensedLinearResponseKernelMatrixElement",
                enforce_closed_shell="bonds",
            )
        if _read_output is True:
            self._read_output_file_clrk(feature_type="bond")


class Multiwfn3DBondFuzzySpaceOverlapIntegrationDescriptor(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D bond feature "fuzzy_space_overlap_integration_descriptor",
    calculated with multiwfn.

    The index of this feature is 436 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-bond-fuzzy_space_overlap_integration_descriptor``
        feature."""
        self._run_multiwfn(command_list=[8, self.real_space_function, "n"], select_scheme=False)
        self._read_output_file_bond()


class Multiwfn3DBondFuzzySpaceOverlapIntegrationDescriptorNegative(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D bond feature
    "fuzzy_space_overlap_integration_descriptor_negative", calculated with multiwfn.

    The index of this feature is 437 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-bond-fuzzy_space_overlap_integration_descriptor_negative``
        feature."""
        self._run_multiwfn(command_list=[8, self.real_space_function, "n"], select_scheme=False)
        self._read_output_file_bond()


class Multiwfn3DBondFuzzySpaceOverlapIntegrationDescriptorPositive(_Multiwfn3DFuzzySpace):
    """Feature factory for the 3D bond feature
    "fuzzy_space_overlap_integration_descriptor_positive", calculated with multiwfn.

    The index of this feature is 438 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.fuzzy" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-bond-fuzzy_space_overlap_integration_descriptor_positive``
        feature."""
        self._run_multiwfn(command_list=[8, self.real_space_function, "n"], select_scheme=False)
        self._read_output_file_bond()
