###################
 Notes on features
###################

This page lists some additional notes on some of the features implemented within BONAFIDE.

********************************
 Working with functional groups
********************************

BONAFIDE implements ``SMARTS-RX``, which is a hierarchical collection of **406 SMARTS patterns** for
the **identification of functional groups** within molecules. :footcite:`kogej_smarts-rx_2025` The
respective feature is called :class:`functional_group_match
<bonafide.features.functional_group.Bonafide2DAtomFunctionalGroupMatch>`, (feature index 27). In
case the desired functional group is not part of SMARTS-RX, it can be added through the
configuration setting under ``bonafide.functional_group.custom_groups`` (see
:doc:`config_settings`).

.. code:: python

   >>> from bonafide import AtomBondFeaturizer
   >>> f = AtomBondFeaturizer()
   >>> # Add the aromatic C-H group as custom functional group
   >>> f.set_options(("bonafide.functional_group.custom_groups", [("AromaticCH", "[c;H1]")]))

The :class:`functional_group_match
<bonafide.features.functional_group.Bonafide2DAtomFunctionalGroupMatch>` feature will identify one
representative atom per functional group match for which, in a second step, features can be
calculated. The :class:`functional_group_match
<bonafide.features.functional_group.Bonafide2DAtomFunctionalGroupMatch>` feature can potentially be
combined with the :mod:`is_symmetric_to <bonafide.features.symmetric_sites>` features (feature index
30 for atoms and 52 for bonds) to identify symmetry-equivalent positions or with the structural
:mod:`distance <bonafide.features.distance>` features to explore the vicinity around the identified
functional group or its structural relation to other parts of the molecule.

**********************************************
 Conceptual density functional theory (C-DFT)
**********************************************

The condensed C-DFT descriptors that can be calculated with BONAFIDE are obtained as shown below.
:footcite:`lu_multiwfn_2012` :footcite:`lu_multiwfn_2024` :math:`q` refers to the atomic partial
charge obtained with a given partitioning scheme, and :math:`N` stands for the number of electrons.

----

**Fukui coefficient for electrophilic attack** (scale for nucleophilicity)

.. math::

   f(-) = q_{N-1} - q_{N}

**Fukui coefficient for nucleophilic attack** (scale for electrophilicity)

.. math::

   f(+) = q_{N} - q_{N+1}

**Fukui coefficient for radical attack** (scale for radical reactivity)

.. math::

   f(0) = \frac{q_{N-1} - q_{N+1}}{2}

**Dual descriptor**

.. math::

   f^{dual} = f(+) - f(-)

Additionally, :math:`f(-)`, :math:`f(+)`, :math:`f(0)`, and :math:`f^{dual}` can be calculated
through an orbital weighting scheme as an alternative to the finite difference approach described
above.

----

**Relative electrophilicity**

.. math::

   \omega_{rel} = \frac{f(+)}{f(-)}

**Relative nucleophilicity**

.. math::

   N_{rel} = \frac{f(-)}{f(+)}

----

**Local electrophilicity**

.. math::

   \omega_{loc} = \omega \cdot f(+)

**Local nucleophilicity**

.. math::

   N_{loc} = N \cdot f(-)

:math:`\omega` is the global electrophilicity and :math:`N` the global nucleophilicity, which can be
obtained with the frontier molecular orbital (FMO) or ionization potential/electron affinity (redox)
approach.

*FMO approach*:

.. math::

   \Delta_{HL} &= E_{LUMO} - E_{HOMO} \\
   \mu^{FMO} &= \frac{E_{HOMO} + E_{LUMO}}{2} \\
   \eta^{FMO} &= \frac{\Delta_{HL}}{2} \\
   S^{FMO} &= \frac{1}{\eta^{FMO}} \\
   \omega^{FMO} &= \frac{(\mu^{FMO})^2}{2 \cdot \eta^{FMO}} \\
   N^{FMO} &= \frac{1}{\omega^{FMO}}

*Redox approach*:

.. math::

   IP &= E_{N-1} - E_{N} \\
   EA &= -(E_{N+1} - E_{N}) \\
   \mu^{redox} &= -\frac{IP + EA}{2} \\
   \eta^{redox} &= \frac{IP - EA}{2} \\
   S^{redox} &= \frac{1}{\eta^{redox}} \\
   \omega^{redox} &= \frac{(\mu^{redox})^2}{2 \cdot \eta^{redox}} \\
   N^{redox} &= -IP

In which :math:`E_{LUMO}` is the energy of the lowest unoccupied molecular orbital,
:math:`E_{HOMO}` the energy of the highest occupied molecular orbital, :math:`\Delta_{HL}` the
HOMO-LUMO gap, :math:`\mu` the chemical potential, :math:`\eta` the hardness, and :math:`S` the
softness. :math:`E_{N-1}` is the energy of the one-electron-oxidized species, :math:`E_{N+1}` the
energy of the one-electron-reduced species, and :math:`E_{N}` the energy of the actual molecule.
:math:`IP` stands for the first ionization potential and :math:`EA` for the first electron affinity.

----

Based on the above listed global descriptors, further local features can be calculated.

**Local hardness for electrophilic attack**

.. math::

   \eta_{loc}(-) = \eta \cdot f(-)

**Local hardness for nucleophilic attack**

.. math::

   \eta_{loc}(+) = \eta \cdot f(+)

**Local hardness for radical attack**

.. math::

   \eta_{loc}(0) = \eta \cdot f(0)

**Local softness for electrophilic attack**

.. math::

   S_{loc}(-) = S \cdot f(-)

**Local softness for nucleophilic attack**

.. math::

   S_{loc}(+) = S \cdot f(+)

**Local softness for radical attack**

.. math::

   S_{loc}(0) = S \cdot f(0)

**Local hyperhardness**

.. math::

   \eta^{dual} = \eta^2 \cdot f^{dual}

**Local hypersoftness**

.. math::

   S^{dual} = S^2 \cdot f^{dual}

**************************
 Autocorrelation features
**************************

BONAFIDE allows to calculate **atom-centered autocorrelation vectors** :math:`\mathbf{AC}_i` for an
atom with index :math:`i` within a molecule with :math:`N` atoms. It is also possible to scale the
values by the number of atoms at depth :math:`d`. A given maximum depth :math:`d_{max}` will result
in a feature vector of length :math:`d_{max}+1` for a given property :math:`p`. Every atom property
of numeric type (integer or float) can be used to calculate autocorrelation features, and multiple
properties can be used simultaneously through the iterable option (see :doc:`config_settings` for
details).

.. math::

   \mathbf{AC}_i &= [A_0, A_1, A_2, \ldots, A_{d,max}] \\
   A_d &= \sum_{j=0}^{N-1} \delta_{d_{ij},d} \cdot f(p_i, p_j) \\
   A_d^{scaled} &= \sum_{j=0}^{N-1} \delta_{d_{ij},d} \cdot \frac{f(p_i, p_j)}{\sum_{j=0}^{N-1} \delta_{d_{ij},d}}

:math:`\delta_{d_{ij},d}` is equal to 1 if :math:`d_{ij} = d` and 0 otherwise, with :math:`d_{ij}`
being the topological distance between atoms :math:`i` and :math:`j`. :math:`f(p_i, p_j)` is a
function that combines the property values of atoms :math:`i` and :math:`j`. It can be **addition**,
**subtraction**, **multiplication**, **averaging**, or the **absolute difference**.

----

.. footbibliography::
