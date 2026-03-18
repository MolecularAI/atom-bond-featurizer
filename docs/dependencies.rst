##############
 Dependencies
##############

*********
 ALFABET
*********

``ALFABET`` is a **graph neural network model** for predicting **bond dissociation energies** and
**free energies** of non-cyclic bonds in organic molecules. It was trained on a large dataset of
DFT-calculated data obtained at the M06-2X/def2-TZVP level of theory. The training dataset contained
neutral closed-shell molecules composed of C, H, N, and O. :footcite:`stjohn_alfabet_2020`

********
 DBSTEP
********

``DBSTEP`` is a software package for calculating **steric descriptors** from 3D molecular
structures. It can be used through BONAFIDE to calculate the buried volume and buried shell volume
descriptors. :footcite:`luchini_dbstep_2022`

*********
 DScribe
*********

``DScribe`` is a Python library that converts 3D atomic or molecular structures into **fixed-length
numerical fingerprints**, *e.g.*, as input for machine learning models. 
:footcite:`himanen_dscribe_2020` :footcite:`laakso_dscribe_2023` Within BONAFIDE, DScribe was used to implement Atom-centered
Symmetry Functions, Smooth Overlap of Atomic Positions, Local Many-body Tensor Representation, and
an atomic Coulomb Matrix-based descriptor. :footcite:`surajit_atomic_coulomb_matrix_2025`

**********
 kallisto
**********

``kallisto`` is a command-line tool for rapidly calculating various atomic features from 3D
molecular structures, that is, **coordination number**, **partial charge**, (relative)
**polarizability**, the **proximity shell** descriptor, and the **van der Waals radius**.
:footcite:`caldeweyher_kallisto_2021`

***********
 mendeleev
***********

``mendeleev`` is a Python package that provides access to an extensive database of the **properties
of the chemical elements** (*e.g.*, various atomic radii or electronegativity scales), which can be
accessed through BONAFIDE. :footcite:`mentel_mendeleev_2014`

*********
 MORFEUS
*********

``MORFEUS`` is a software package that provides a broad range of molecular as well as atom and bond
descriptors calculated from 3D molecular structures. BONAFIDE implements MORFEUS' **buried volume**,
**dispersion**, **local force constant**, and **solvent-accessible surface area** as well as several 
**angle-based** atom and bond descriptors, respectively. :footcite:`jacot_morfeus_2022`

**********
 Multiwfn
**********

``Multiwfn`` is a multifunctional stand-alone program for the **analysis of the electronic structure
of a molecule after a quantum chemical calculation**. :footcite:`lu_multiwfn_2012`
:footcite:`lu_multiwfn_2024` It provides access to a large number of atom- and bond-level features
that can be calculated through BONAFIDE. For an overview of the available features, see
:doc:`feature_list`.

******
 Psi4
******

``Psi4`` is an open-source quantum chemistry software package that is used in BONAFIDE to perform
**single-point energy calculations** - *e.g.*, with density functional theory (DFT), including
dispersion corrections and implicit solvation models. :footcite:`turney_psi4_2012`
:footcite:`parrish_psi4_2017` :footcite:`smith_psi4_2020`

********
 qmdesc
********

``qmdesc`` is a graph neural network model for predicting atom and bond property descriptors that
are typically obtained through quantum chemical calculations. This includes **Hirshfeld partial
charges**, **Fukui indices**, **NMR shielding constants**, as well as **bond orders** and
**lengths**. The large training dataset contained neutral closed-shell molecules composed of C, H,
O, N, P, S, F, Cl, Br, and I and was obtained at the B3LYP/def2-SVP level of theory.
:footcite:`guan_qmdesc_2021`

*******
 RDKit
*******

``RDKit`` is the general cheminformatics package used in BONAFIDE. It also implements a number of
atom and bond descriptors which can be calculated through BONAFIDE. For the full list, see the
:doc:`feature_list`. :footcite:`rdkit`

*****
 xtb
*****

``xtb`` is a semi-empirical quantum chemistry software package that is used in BONAFIDE to perform
**single-point energy calculations** as well as for the calculation of various **atom features**
such as reactivity indices from conceptual DFT (C-DFT). :footcite:`bannwarth_xtb_2021`

----

.. footbibliography::