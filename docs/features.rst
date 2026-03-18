#####################
 Feature calculation
#####################

After reading in a molecule (either as SMILES string, from a file, or as RDKit molecule object, see
:doc:`input`), **features for atoms and bonds** can be calculated. Please note the following
points:

-  For many 3D descriptors, data on the **electronic structure** of the molecule (and/or its
   one-electron reduced or oxidized form) is required, which must be calculated before using
   BONAFIDE or, alternatively, can also be calculated from scratch (see
   :doc:`electronic_structure`).

-  For some features, the **charge** and **multiplicity** of the molecule must be set with the
   respective set methods (see :doc:`electronic_structure`).

-  **Features for bonds** can only be calculated if the chemical bonds within the molecules are
   defined (see :doc:`bonds`).

-  See :doc:`notes` for additional important information.

The :meth:`list_atom_features() <bonafide.bonafide.AtomBondFeaturizer.list_atom_features>` and
:meth:`list_bond_features() <bonafide.bonafide.AtomBondFeaturizer.list_bond_features>` methods can
be used to list available features with the **feature indices** along additional information. The
feature index/indices (``INDEX``) must be specified to request the calculation of a single or a
group of features (see :doc:`feature_list`).

After reading in a molecule, the :meth:`show_molecule()
<bonafide.bonafide.AtomBondFeaturizer.show_molecule>` method can be used to find out the **indices
of the atoms or bonds** for which features should be calculated (see :doc:`input`).

After feature calculation, the results can be accessed through the :meth:`return_atom_features()
<bonafide.bonafide.AtomBondFeaturizer.return_atom_features>` and :meth:`return_bond_features()
<bonafide.bonafide.AtomBondFeaturizer.return_bond_features>` methods (see :doc:`output`).

.. important::

   Check the log file for **info messages** or any **warnings** or **errors** that might have
   occurred during the feature calculation. The log file is typically created in the current working
   directory.

***************
 Atom features
***************

Atom features are calculated with the :meth:`featurize_atoms()
<bonafide.bonafide.AtomBondFeaturizer.featurize_atoms>` method. It accepts **two required
parameters**:

-  ``atom_indices``: A single atom index or a list of atom indices for which descriptors should be
   calculated. It is also possible to set ``atom_indices="all"`` to calculate features for all atoms
   in the molecule.

-  ``feature_indices``: A single feature index or a list of feature indices to be calculated. It is
   also possible to set ``feature_indices="all"`` to calculate all 2D or 3D atom features (depending
   on the user input).

The example below shows how to calculate all 2D RDKit features for all atoms in a molecule starting
from a SMILES string.

.. code:: python

   >>> from bonafide import AtomBondFeaturizer
   >>> f = AtomBondFeaturizer()
   >>> # Get the list of the desired features indices
   >>> fdf = f.list_atom_features(origin="RDKit", dimensionality="2D")
   >>> fidx_list = fdf.index.to_list()
   >>> print(len(fidx_list))
   45
   >>> # Read in the molecule and calculate the features
   >>> f.read_input("O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl", "diclofenac")
   >>> f.featurize_atoms(atom_indices="all", feature_indices=fidx_list)

***************
 Bond features
***************

Bond features are calculated with the :meth:`featurize_bonds()
<bonafide.bonafide.AtomBondFeaturizer.featurize_bonds>` method. It accepts **two required
parameters**:

-  ``bond_indices``: A single bond index or a list of bond indices for which descriptors should be
   calculated. It is also possible to set ``bond_indices="all"`` to calculate features for all bonds
   in the molecule.

-  ``feature_indices``: A single feature index or a list of feature indices to be calculated. It is
   also possible to set ``feature_indices="all"`` to calculate all 2D or 3D bond features (depending
   on the user input).

The example below shows how to calculate all 3D Multiwfn features for all bonds in a molecule
starting from an XYZ file.

.. code:: python

   >>> from bonafide import AtomBondFeaturizer
   >>> f = AtomBondFeaturizer()
   >>> # Get the list of the desired features indices
   >>> fdf = f.list_bond_features(origin="Multiwfn", dimensionality="3D")
   >>> fidx_list = fdf.index.to_list()
   >>> print(len(fidx_list))
   57
   >>> # Read in the molecule and calculate the features
   >>> f.read_input("diclo.xyz", "diclofenac", input_format="file", read_energy=True, output_directory="diclo_output_files")
   >>> f.determine_bonds()
   >>> f.set_charge(0)
   >>> f.set_multiplicity(1)
   >>> f.featurize_bonds(bond_indices="all", feature_indices=fidx_list)

*******************
 Log file analysis
*******************

The BONAFIDE package includes a small utility class for **analyzing its log files**. It can be used
to check for any info messages, warnings, or errors that might have occurred during feature
calculation as well as for analyzing runtimes. The :class:`LogFileAnalyzer
<bonafide.log_file_analysis.LogFileAnalyzer>` can be used as follows.

.. code:: python

   >>> from bonafide import LogFileAnalyzer
   >>> a = LogFileAnalyzer(<log file path>)
   >>> error_logs = a.get_level_log_messages()
   >>> print(error_logs)
   ...
   >>> total_runtime = a.get_total_runtime()
   >>> print(total_runtime)
   ...

***************
 Feature cache
***************

After atom or bond features have been calculated, they are stored in internal feature caches. It is
possible to **clear** these caches with the :meth:`clear_atom_feature_cache()
<bonafide.bonafide.AtomBondFeaturizer.clear_atom_feature_cache>` and
:meth:`clear_bond_feature_cache() <bonafide.bonafide.AtomBondFeaturizer.clear_bond_feature_cache>`
methods, *e.g.*, to recalculate a given feature with a different configuration setting.

It is possible to provide an argument to the respective clear method to specify features from which
featurization program should be cleared. By default, the entire feature cache is cleared.

.. code:: python

   >>> from bonafide import AtomBondFeaturizer
   >>> f = AtomBondFeaturizer()
   >>> # Get the list of the desired features indices
   >>> fdf = f.list_atom_features(origin="RDKit", dimensionality="2D")
   >>> fidx_list = fdf.index.to_list()
   >>> print(len(fidx_list))
   45
   >>> # Read in the molecule and calculate the features
   >>> f.read_input("O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl", "diclofenac")
   >>> f.featurize_atoms(atom_indices="all", feature_indices=fidx_list)
   >>> results = f.return_atom_features()
   >>> print(results)
   ...
   >>> # Clear the entire atom feature cache
   >>> f.clear_atom_feature_cache()
