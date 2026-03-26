#################
 Custom features
#################

It is possible to implement **custom atom or bond featurization methods** in BONAFIDE. This is
demonstrated and explained with an example here.

.. hint::

   A potential scenario during the implementation of a custom feature is the **incompatibility of a
   required Python package with the BONAFIDE environment**. In this case, it is possible to use the
   :func:`external_driver() <bonafide.utils.driver.external_driver>` function that allows to run
   Python scripts with the interpreter of an external environment. By doing that, the custom feature
   can make use of packages that are not installed in the BONAFIDE environment. At the same time,
   this function can be used to call any other external program with a custom input. See
   :doc:`external_environment` for more information.

********************************************
 Distance to fixed point in cartesian space
********************************************

As an example, the **distance of an atom to a predefined point in 3-dimensional space** will be
implemented as custom featurization method.

1) Feature factory classes
==========================

Within BONAFIDE, each feature has its own **factory class** that either directly or indirectly
inherits from the :class:`BaseFeaturizer <bonafide.utils.base_featurizer.BaseFeaturizer>` class. So
at first, the :class:`BaseFeaturizer <bonafide.utils.base_featurizer.BaseFeaturizer>` class must be
imported, and the custom class for calculating the feature must inherit from it. We also import
``numpy`` for calculating the distance (see below).

.. code:: python

   import numpy as np
   from bonafide import AtomBondFeaturizer
   from bonafide.utils.base_featurizer import BaseFeaturizer

   f = AtomBondFeaturizer()

2) Implementation of the custom feature factory
===============================================

The custom featurization class must fulfill **two requirements**.

-  It must implement the ``calculate()`` method to calculate the custom feature.

-  The attribute ``extraction_mode`` must be set either to "single" or "multi". This signals to the
   framework if the ``calculate()`` method yields the feature for *all* atoms or bonds when called
   once ("multi") or if it yields the feature *only for the current atom or bond* ("single").

In the chosen example, ``extraction_mode`` is set to "single" because the distance between the fixed
point and a given atom is calculated one at the time.

.. code:: python

   class Custom3DAtomFixedPointDistance(BaseFeaturizer):
       """Feature factory for the custom3D-atom-fixed-point-distance feature."""

       def __init__(self) -> None:
           self.extraction_mode = "single"
           super().__init__()

       def calculate(self) -> None:
           """Calculate the distance of an atom to a fixed point in 3D space."""
           # Get the position vector of the currently treated atom
           pos = self.mol.GetConformer().GetAtomPosition(self.atom_bond_idx)
           atom_coordinates = np.array([pos.x, pos.y, pos.z])

           # Calculate the distance
           self.fixed_point = np.array(self.fixed_point)
           value = np.linalg.norm(atom_coordinates - self.fixed_point)

           # Write the data to the results dictionary
           self.results[self.atom_bond_idx] = {self.feature_name: float(value)}

3) Saving the results
=====================

In order for the ``calculate()`` method to save the calculated data, it must write it to the
**results dictionary** ``self.results``. The key(s) of this dictionaries are the atom or bond
indices (atom indices in the example). The value(s) are dictionaries with their keys being the
feature name and the values being the calculated data. It is important to follow this structure
exactly.

In case ``calculate()`` directly computes the features for all atoms or bonds
(``extraction_mode="multi"``), all data should directly be written to the results dictionary through
a loop.

4) Attributes of the factory class and configuration settings
=============================================================

By inheriting from :class:`BaseFeaturizer <bonafide.utils.base_featurizer.BaseFeaturizer>`, the
custom class automatically exposes a list of **attributes** that can/must be used to calculate
features.

-  ``atom_bond_idx``: Index of the currently treated atom or bond.
-  ``charge``: Charge of the molecule (``None`` if not set).
-  ``conformer_idx``: Index of the currently treated conformer.
-  ``conformer_name``: Name of the currently treated conformer.
-  ``coordinates``: Cartesian coordinates of the currently treated conformer (``None`` in the 2D
   case).
-  ``electronic_struc_n``: Paths to the electronic structure files for the actual molecule (see
   :doc:`electronic_structure`).
-  ``electronic_struc_n_minus1``: Paths to the electronic structure files for the one-electron
   oxidized molecule (see :doc:`electronic_structure`).
-  ``electronic_struc_n_plus1``: Paths to the electronic structure files for the one-electron
   reduced molecule (see :doc:`electronic_structure`).
-  ``elements``: List of the chemical elements of the atoms in the molecule.
-  ``feature_cache``: Cache of previously computed features that could be used to calculate the new
   custom features.
-  ``feature_name``: Name of the feature.
-  ``mol``: RDKit molecule object of the currently treated molecule.
-  ``multiplicity``: Multiplicity of the molecule (``None`` if not set).

Additionally, it is possible to give the custom featurization class access to **specific
configuration settings**. They will also be exposed as attributes. In the chosen example, this is
the fixed point (arbitrarily chosen to be the point of origin) in 3D space to which the distance is
calculated.

.. code:: python

   fixed_point_feature_config = {"fixed_point": [0, 0, 0]}

5) Metadata
===========

Before the custom featurizer can be added to BONAFIDE, it is required to define the **metadata** of
the custom feature, such as whether it is an atom or bond feature, if it is a 2D or 3D feature, or
the name of the feature.

.. code:: python

   feature_info_dict = {
       "name": "custom3D-atom-fixed_point_distance",
       "origin": "custom",
       "feature_type": "atom",
       "dimensionality": "3D",
       "data_type": "float",
       "requires_electronic_structure_data": False,
       "requires_bond_data": False,
       "requires_charge": False,
       "requires_multiplicity": False,
       "config_path": fixed_point_feature_config,
       "factory": Custom3DAtomFixedPointDistance,
   }

6) Adding the custom featurizer to BONAFIDE
===========================================

Lastly, the custom featurizer can be **added to the framework** through the
:meth:`add_custom_featurizer() <bonafide.bonafide.AtomBondFeaturizer.add_custom_featurizer>` method.
It takes as its only argument the metadata dictionary (``feature_info_dict`` in the example). After
that, the custom feature can be calculated like any other feature.

.. code:: python

   f.add_custom_featurizer(feature_info_dict)

.. note::

   The **data type and format** passed to a custom feature factory through a configuration settings
   dictionary (``fixed_point_feature_config`` in the example) is not checked. The user must ensure
   that it is correct. For all already implemented features, the configuration settings are
   automatically checked.

7) Calculating the custom feature
=================================

The custom feature is now appended to the collection of all available features (see
:doc:`feature_list`) and can be calculated with its respective feature index (see :doc:`features`).

.. code:: python

   print(f.list_atom_features())
   ...

   # Calculate the custom feature
   custom_feature_idx = f.list_atom_features().index.to_list()[-1]
   f.read_input("diclo.xyz", "diclofenac", input_format="file")
   f.featurize_atoms(atom_indices="all", feature_indices=custom_feature_idx)
   f.return_atom_features()
   ...
