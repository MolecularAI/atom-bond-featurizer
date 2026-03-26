################
 Notes on usage
################

After installing the package (see :doc:`Installation <installation>`), BONAFIDE can be used by
instantiating the ``AtomBondFeaturizer`` class. The user has the opportunity to specify a custom
**log file** name by passing data to the ``log_file_name`` parameter. By default, the log file is
called ``bonafide.log`` and is created in the current working directory.

.. code:: python

   >>> from bonafide import AtomBondFeaturizer
   >>> f = AtomBondFeaturizer()

The log file will contain all messages logged during the usage of BONAFIDE.

*********************************
 Atom, bond, and feature indices
*********************************

All indices within BONAFIDE, including those for atoms, chemical bonds, and the respective features
are **zero-based**. This means that the first atom or bond in a molecule has index 0.

*****************
 Feature vectors
*****************

In case a given atom or bond feature is not represented by a single value but is rather a
**vector**, the datatype of this feature is ``str`` and the individual values are
**comma-separated** within the string.
