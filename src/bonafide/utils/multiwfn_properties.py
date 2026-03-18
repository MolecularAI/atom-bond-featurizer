"""Extraction of the Multiwfn real space properties."""

from typing import Any, Dict, List, Optional, Tuple, Union


def read_prop_file(
    file_content: List[str], prefix: str = ""
) -> List[Dict[str, Optional[Union[str, float, int, Tuple[int, int], List[str]]]]]:
    """Read the Multiwfn real space properties.

    Parameters
    ----------
    file_content : List[str]
        The content of the Multiwfn output file as a list of the individual lines of the file.
    prefix : str, optional
        A prefix to add to all property names, by default "".

    Returns
    -------
    List[Dict[str, Optional[Union[str, float, int, Tuple[int, int], List[str]]]]]
        A list of dictionaries containing the extracted properties for each data block.
    """
    props = [
        "electron_density",
        "electron_density_alpha",
        "electron_density_beta",
        "spin_density",
        "lagrangian_kinetic_energy",
        "hamiltonian_kinetic_energy",
        "potential_energy_density",
        "energy_density",
        "laplacian_of_electron_density",
        "electron_localization_function",
        "localized_orbital_locator",
        "local_information_entropy",
        "interaction_region_indicator",
        "reduced_density_gradient",
        "reduced_density_gradient_promolecular",
        "sign_second_largest_eigenvalue_electron_density_hessian",
        "sign_second_largest_eigenvalue_electron_density_hessian_promolecular",
        "average_local_ionization_energy",
        "vdw_potential",
        "delta_g_promolecular",
        "delta_g_hirshfeld",
        "electrostatic_potential_nuclear_charges",
        "electrostatic_potential_electrons",
        "electrostatic_potential",
        "gradient_components_x_y_z",
        "gradient_norm",
        "laplacian_components_x_y_z",
        "laplacian_total",
        "hessian_eigenvalues",
        "hessian_determinant",
        "electron_density_ellipticity",
        "eta_index",
    ]

    if prefix == "bcp_":
        props.append("coordinates")
        props = [f"{prefix}{p}" for p in props]
        props.extend(
            [
                "_atoms",
                "cp_index",
                "cp_type",
            ]
        )

    props.append("_not_found")

    # Find all data blocks
    blocks = []
    line_pair = []
    for line_idx, line in enumerate(file_content):
        # Differentiate between the normal atom and the bcp case
        if prefix == "bcp_":
            if "Connected atoms:" in line:
                line_pair = [line_idx - 1]
        else:
            if line.startswith(" Note: Unless otherwise specified, all units are in a.u."):
                line_pair = [line_idx - 1]

        if "eta index:" in line and len(line_pair) == 1:
            line_pair.append(line_idx + 1)
        if len(line_pair) == 2:
            sub_dict: Dict[str, Any] = {p: None for p in props}
            blocks.append((line_pair, sub_dict))
            line_pair = []

    # Loop over the blocks and extract the properties
    for line_pair, data_dict in blocks:
        lines = file_content[line_pair[0] : line_pair[1]]
        for line_idx, line in enumerate(lines):
            splitted = line.split(":")

            # Differentiate between the normal atom and the bcp case
            if prefix == "bcp_":
                if "----------------" in line:
                    splitted = line.split()
                    data_dict["cp_index"] = int(splitted[2].split(",")[0])
                    data_dict["cp_type"] = splitted[4]

                if "Connected atoms:" in line:
                    start_idx_str, end_idx_str = splitted[-1].split("--")
                    start_idx = int(start_idx_str.split("(")[0])
                    end_idx = int(end_idx_str.split("(")[0])
                    data_dict["_atoms"] = (start_idx, end_idx)

                if "Position (Angstrom):" in line:
                    data_dict[f"{prefix}coordinates"] = ",".join(splitted[-1].split())

            if "Density of all electrons:" in line:
                data_dict[f"{prefix}electron_density"] = float(splitted[-1])

            if "Density of Alpha electrons:" in line:
                data_dict[f"{prefix}electron_density_alpha"] = float(splitted[-1])

            if "Density of Beta electrons:" in line:
                data_dict[f"{prefix}electron_density_beta"] = float(splitted[-1])

            if "Spin density of electrons:" in line:
                data_dict[f"{prefix}spin_density"] = float(splitted[-1])

            if "Lagrangian kinetic energy G(r):" in line:
                data_dict[f"{prefix}lagrangian_kinetic_energy"] = float(splitted[-1])

            if "Hamiltonian kinetic energy K(r):" in line:
                data_dict[f"{prefix}hamiltonian_kinetic_energy"] = float(splitted[-1])

            if "Potential energy density V(r):" in line:
                data_dict[f"{prefix}potential_energy_density"] = float(splitted[-1])

            if "Energy density E(r) or H(r):" in line:
                data_dict[f"{prefix}energy_density"] = float(splitted[-1])

            if "Laplacian of electron density:" in line:
                data_dict[f"{prefix}laplacian_of_electron_density"] = float(splitted[-1])

            if "Electron localization function (ELF):" in line:
                data_dict[f"{prefix}electron_localization_function"] = float(splitted[-1])

            if "Localized orbital locator (LOL):" in line:
                data_dict[f"{prefix}localized_orbital_locator"] = float(splitted[-1])

            if "Local information entropy:" in line:
                data_dict[f"{prefix}local_information_entropy"] = float(splitted[-1])

            if "Interaction region indicator (IRI):" in line:
                data_dict[f"{prefix}interaction_region_indicator"] = float(splitted[-1])

            if "Reduced density gradient (RDG):" in line:
                data_dict[f"{prefix}reduced_density_gradient"] = float(splitted[-1])

            if "Reduced density gradient with promolecular approximation:" in line:
                data_dict[f"{prefix}reduced_density_gradient_promolecular"] = float(splitted[-1])

            if "Sign(lambda2)*rho:" in line:
                data_dict[f"{prefix}sign_second_largest_eigenvalue_electron_density_hessian"] = (
                    float(splitted[-1])
                )

            if "Sign(lambda2)*rho with promolecular approximation:" in line:
                data_dict[
                    f"{prefix}sign_second_largest_eigenvalue_electron_density_hessian_promolecular"
                ] = float(splitted[-1])

            if "Average local ionization energy (ALIE):" in line:
                data_dict[f"{prefix}average_local_ionization_energy"] = float(splitted[-1])

            if "van der Waals potential (probe atom: C ):" in line:
                # Fix erroneous print out (missing E)
                num = splitted[-1].split("kcal")[0].strip()
                if "+" in num and "E" not in num:
                    num = num.replace("+", "E+")
                data_dict[f"{prefix}vdw_potential"] = float(num)

            if "Delta-g (under promolecular approximation):" in line:
                data_dict[f"{prefix}delta_g_promolecular"] = float(splitted[-1])

            if "Delta-g (under Hirshfeld partition):" in line:
                data_dict[f"{prefix}delta_g_hirshfeld"] = float(splitted[-1])

            if "ESP from nuclear charges:" in line:
                data_dict[f"{prefix}electrostatic_potential_nuclear_charges"] = float(splitted[-1])

            if "ESP from electrons:" in line:
                data_dict[f"{prefix}electrostatic_potential_electrons"] = float(splitted[-1])

            if "Total ESP:" in line:
                data_dict[f"{prefix}electrostatic_potential"] = float(splitted[-1].split("a.u.")[0])

            if "Components of gradient in x/y/z are:" in line:
                splitted = [str(float(x)) for x in lines[line_idx + 1].split()]
                data_dict[f"{prefix}gradient_components_x_y_z"] = ",".join(splitted)

            if "Norm of gradient is:" in line:
                data_dict[f"{prefix}gradient_norm"] = float(splitted[-1])

            if "Components of Laplacian in x/y/z are:" in line:
                splitted = [str(float(x)) for x in lines[line_idx + 1].split()]
                data_dict[f"{prefix}laplacian_components_x_y_z"] = ",".join(splitted)
                splitted = lines[line_idx + 2].split(":")
                data_dict[f"{prefix}laplacian_total"] = float(splitted[-1])

            if "Eigenvalues of Hessian:" in line:
                splitted = [str(float(x)) for x in splitted[-1].split()]
                data_dict[f"{prefix}hessian_eigenvalues"] = ",".join(splitted)

            if "Determinant of Hessian:" in line:
                data_dict[f"{prefix}hessian_determinant"] = float(splitted[-1])

            if "Ellipticity of electron density:" in line:
                data_dict[f"{prefix}electron_density_ellipticity"] = float(splitted[-1])

            if "eta index:" in line:
                data_dict[f"{prefix}eta_index"] = float(splitted[-1])

    # Only keep the extracted data
    data = [b[-1] for b in blocks]

    # Check for properties that were not found
    for d in data:
        d["_not_found"] = []
        for feature_name, value in d.items():
            if value is None:
                d["_not_found"].append(feature_name)

    return data
