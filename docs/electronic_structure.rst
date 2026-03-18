######################################
 Energy and electronic structure data
######################################

*************
 Energy data
*************

Besides reading the individual conformer energies of an ensemble directly from the input file
through the :meth:`read_input() <bonafide.bonafide.AtomBondFeaturizer.read_input>` method (see
:doc:`input`), it is also possible to **attach conformer energies** after reading the input with the
:meth:`attach_energy() <bonafide.bonafide.AtomBondFeaturizer.attach_energy>` method. This is done by
passing a list of 2-tuples to the method, where the first element of each tuple is the energy value
and the second the corresponding energy unit. Supported units are "Eh", "kcal/mol", and "kJ/mol".

As for the :meth:`read_input() <bonafide.bonafide.AtomBondFeaturizer.read_input>` method, it is
possible to **prune the conformer ensemble** based on relative energies by setting the
``prune_by_energy`` parameter to a 2-tuple in which the first entry is the relative energy cutoff
value and the second entry is the respective energy unit.

The :meth:`attach_energy() <bonafide.bonafide.AtomBondFeaturizer.attach_energy>` method can not
only be used to attach conformer energies for the **actual molecule** ("n") but also for its
**one-electron reduced** ("n+1") and/or **one-electron oxidized** ("n-1") form. This is done by
passing the respective string identifier to the ``state`` parameter. The attached energies will then
be used accordingly to calculate respective features.

.. code:: python

   >>> from bonafide import AtomBondFeaturizer
   >>> f = AtomBondFeaturizer()
   >>> f.read_input("diclo.xyz", "diclofenac", input_format="file")
   >>> # Attach energies for the actual molecule
   >>> f.attach_energy([(-55.759066148201, "eh"), (-55.759066573014, "eh"), (-55.754661166000, "eh")], state="n")
   >>> # Attach energies for the one-electron reduced molecule
   >>> f.attach_energy([(-55.933808034517, "eh"), (-55.933808212075, "eh"), (-55.927604286309, "eh")], state="n+1")

***************************
 Electronic structure data
***************************

Many descriptors that can be calculated with BONAFIDE rely on knowledge of the **electronic
structure** of the molecule. There are two different options this information can be made available
for a single 3D structure or an ensemble of conformers.

-  **Attach precomputed electronic structure data**
-  **Calculate it from scratch**

Attaching electronic structure data
===================================

Typical **electronic structure data files** such as ``*.molden`` or ``*.fchk`` can be attached with
the :meth:`attach_electronic_structure()
<bonafide.bonafide.AtomBondFeaturizer.attach_electronic_structure>` method. This is done by
specifying the path to the file as a string (in case of a single conformer) or by passing a list of
file paths (in case of an ensemble of conformers).

.. code:: python

   >>> from bonafide import AtomBondFeaturizer
   >>> f = AtomBondFeaturizer()
   >>> f.read_input("diclo.xyz", "diclofenac", input_format="file")
   >>> f.attach_electronic_structure(["diclo_00.molden", "diclo_01.molden", "diclo_02.molden"])

For some descriptors from conceptual DFT, it is required to not only have data for the actual
molecule under consideration but also for its **one-electron reduced** and/or **one-electron
oxidized** form (see above the :meth:`attach_energy()
<bonafide.bonafide.AtomBondFeaturizer.attach_energy>` method). It is possible to address these three
individual states, the *actual molecule* ("n"), its *one-electron reduced* form ("n+1"), and its
*one-electron oxidized* form ("n-1"), separately by passing the respective string identifier to the
``state`` parameter, which defaults to "n". The attached files will then be used accordingly to
calculate respective features.

.. warning::

   The **atom order** of any electronic structure data file must match the order defined within the
   molecule vault before attaching the data. If this is not the case, erroneous features will be
   calculated.

Calculating electronic structure data from scratch
==================================================

The BONAFIDE framework implements an interface to the semi-empirical DFT package **xtb** as well as
to the quantum chemistry package **Psi4** to run **single-point energy calculations** with the
:meth:`calculate_electronic_structure()
<bonafide.bonafide.AtomBondFeaturizer.calculate_electronic_structure>` method for all conformers of
the ensemble. The desired quantum chemistry engine must be specified through the ``engine``
parameter (either "xtb" or "psi4").

   Before it is possible to run single-point energy calculations, the user must set the **molecular
   charge** and **multiplicity** of the molecule with the :meth:`set_charge()
   <bonafide.bonafide.AtomBondFeaturizer.set_charge>` and :meth:`set_multiplicity()
   <bonafide.bonafide.AtomBondFeaturizer.set_multiplicity>` methods, respectively.

.. code:: python

   >>> from bonafide import AtomBondFeaturizer
   >>> f = AtomBondFeaturizer()
   >>> f.read_input("diclo.xyz", "diclofenac", input_format="file")
   >>> f.set_charge(0)
   >>> f.set_multiplicity(1)
   >>> f.calculate_electronic_structure("xtb")

By default, the parameter ``redox`` of the :meth:`calculate_electronic_structure()
<bonafide.bonafide.AtomBondFeaturizer.calculate_electronic_structure>` method is set to "n", which
means that only the single-point energy calculation for the actual molecule under consideration is
performed (see above). Alternatively, this can be set to "n+1" or "n-1" to calculate the electronic
structure of the one-electron reduced and one-electron oxidized state, respectively, in addition to
calculating the data for the "n" state. By selecting "all", the electronic structure data for all
three states is calculated.

The exact **settings** for the electronic structure calculations (*e.g.*, **method**, **basis set**,
**solvation model**, **parallelization**) can be inspected and adjusted through the
:meth:`print_options() <bonafide.bonafide.AtomBondFeaturizer.print_options>` and :meth:`set_options()
<bonafide.bonafide.AtomBondFeaturizer.set_options>` methods, respectively (see
:doc:`config_settings`).

After the single-point energy calculation(s) are completed, the energy and electronic structure data
are **automatically attached to the conformer ensemble** and can be used for the calculation of
descriptors.

As for the :meth:`read_input() <bonafide.bonafide.AtomBondFeaturizer.read_input>` and
:meth:`attach_energy() <bonafide.bonafide.AtomBondFeaturizer.attach_energy>` methods (see above), it
is possible to **prune the conformer ensemble** based on relative energies by setting the
``prune_by_energy`` parameter to a 2-tuple in which the first entry is the relative energy cutoff
value and the second entry is the respective energy unit.

.. note::

   As a featurization tool for atoms and bonds in molecules, BONAFIDE *cannot* be used to **generate
   conformer ensembles** of molecules or to run **structure optimizations** or **frequency
   calculations**. If 3D descriptors are intended to be calculated, the user must address these
   tasks before using BONAFIDE. One option to achieve this is to make use of the **morfeus**
   cheminformatics package (see the `morfeus documentation
   <https://digital-chemistry-laboratory.github.io/morfeus/conformer.html>`_ for further details).
