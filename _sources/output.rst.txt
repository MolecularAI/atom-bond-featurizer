################
 Feature output
################

After the feature calculation finished (see :doc:`features`), the results can be accessed through
the :meth:`return_atom_features() <bonafide.bonafide.AtomBondFeaturizer.return_atom_features>` and
:meth:`return_bond_features() <bonafide.bonafide.AtomBondFeaturizer.return_bond_features>` methods,
respectively. The methods allow to **customize the output** in several ways by accepting input by
the user (see the :doc:`API documentation <bonafide_user>` for details).

The example below shows how to calculate all 2D RDKit features for all atoms in a molecule starting
from a SMILES string and how to return the output.

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
   >>> # Get the output as a pandas DataFrame
   >>> f.return_atom_features()
   ...

In general, the output can be obtained as a **pandas DataFrame**, a **hierarchical dictionary**,
or as an **RDKit molecule object** with the features stored as atom and bond properties,
respectively.

.. note::

   It is possible that the feature of a given atom or bond is set to **_inaccessible**. This happens
   if that specific feature is not defined for the given atom or bond. One example are the features
   that are calculated based on the contribution of an atom to the molecular surface. If the atom is
   completely buried and does not contribute to the surface, the surface feature for that atom will
   be _inaccessible.
