##############################
 Definition of chemical bonds
##############################

After reading a molecule from an **XYZ file**, there is **no information on the connectivity of the
atoms** defined. However, this information is required for calculating bond and also several atom
features. In addition to defining the bonds separately and specifying them in an SD file, BONAFIDE
provides two options to introduce bond information to 3D molecules and conformer ensembles:

-  **Internally define chemical bonds**
-  **Attach a SMILES string**

*************************************
 Internally determine chemical bonds
*************************************

It is possible to **automatically define chemical bonds** with the :meth:`determine_bonds()
<bonafide.bonafide.AtomBondFeaturizer.determine_bonds>` method after reading a molecule from an XYZ
file. This method implements RDKit's ``rdkit.Chem.rdDetermineBonds.DetermineBonds`` and assigns
**atom connectivity** and **bond information**. This is done for all conformers in case multiple
have been provided. The charge of the molecule must be set before using this method (see the
:meth:`determine_bonds() <bonafide.bonafide.AtomBondFeaturizer.determine_bonds>` method).

.. code:: python

   >>> from bonafide import AtomBondFeaturizer
   >>> f = AtomBondFeaturizer()
   >>> f.read_input("diclo.xyz", "diclofenac", input_format="file")
   >>> f.set_charge(0)
   >>> f.determine_bonds()

Several optional arguments can be passed to modify the exact procedure of bond determination (see
the :meth:`API documentation <bonafide.bonafide.AtomBondFeaturizer.determine_bonds>`).

***************************
 Attaching a SMILES string
***************************

Alternatively, it is possible to **attach a SMILES string to a 3D conformer ensemble**, which
exactly defines the chemical bonding between the atoms (within the SMILE system). This is done
through the :meth:`attach_smiles() <bonafide.bonafide.AtomBondFeaturizer.attach_smiles>` method. By
using the default ``align=True``, the atom indices (atom order) of the initially read molecule are
preserved; if set to ``False``, the atoms are reordered according to the SMILES string. The
:meth:`attach_smiles() <bonafide.bonafide.AtomBondFeaturizer.attach_smiles>` method also requires
the charge of the molecule to be set beforehand. This is because the charge is required for
determining the atom connectivity of the molecule to which the SMILES string is attached.

.. code:: python

   >>> from bonafide import AtomBondFeaturizer
   >>> f = AtomBondFeaturizer()
   >>> f.read_input("diclo.xyz", "diclofenac", input_format="file")
   >>> f.set_charge(0)
   >>> f.attach_smiles("[H]OC(=O)C([H])([H])c1c([H])c([H])c([H])c([H])c1N([H])c1c(Cl)c([H])c([H])c([H])c1Cl")
