###############
 Configuration
###############

BONAFIDE allows to **configure the behavior of its individual dependencies** as it would be possible
when using the packages and programs directly. After instantiating the ``AtomBondFeaturizer`` class,
the default parameters are read from a toml file in the installation directory and can be inspected
with the :meth:`print_options() <bonafide.bonafide.AtomBondFeaturizer.print_options>` method. The
output of this method can be filtered by passing the name or a list of names of the featurizer
engines as a parameter.

.. code:: python

   >>> from bonafide import AtomBondFeaturizer
   >>> f = AtomBondFeaturizer()
   >>> f.print_options("kallisto")
   Default configuration settings at:
   <Path to the _feature_config.toml file with the default settings will be displayed here.>

   kallisto
       cntype: cov
       size: [2, 3]
       vdwtype: rahm
       angstrom: False

*************************************
 Changing the configuration settings
*************************************

It is possible to **change the value of each parameter** through the :meth:`set_options()
<bonafide.bonafide.AtomBondFeaturizer.set_options>` method. This is done by passing a 2-tuple or a
list of 2-tuples to the method, where the first element of each 2-tuple is the point-separated path
to the parameter (including the name of the parameter) and the second element is the new value of
the parameter. It is automatically ensured that the new value is of the correct data type and
format.

.. code:: python

   >>> from bonafide import AtomBondFeaturizer
   >>> f = AtomBondFeaturizer()
   >>> # Print default kallisto settings
   >>> f.print_options("kallisto")
   Default configuration settings at:
   <Path to the _feature_config.toml file with the default settings will be displayed here.>

   kallisto
       cntype: cov
       size: [2, 3]
       vdwtype: rahm
       angstrom: False
   >>> # Change kallisto settings
   >>> f.set_options([("kallisto.cntype", "exp"), ("kallisto.angstrom", True)])
   >>> f.print_options("kallisto")
   Default configuration settings at:
   <Path to the _feature_config.toml file with the default settings will be displayed here.>

   kallisto
       cntype: exp
       size: [2, 3]
       vdwtype: rahm
       angstrom: True

**********************************
 Features with *iterable options*
**********************************

There are a few features implemented within BONAFIDE that come with a so-called **iterable option**.
By default, this option is populated with one input. If the user selects multiple inputs (from a
collection of available options), BONAFIDE will **calculate the given feature for each of the
selected iterable option inputs separately**. This allows to investigate how the different available
options influence the given feature. The set of available inputs to the individual
``iterable_option`` configuration settings is given in the default settings ``_feature_config.toml``
file. The path to this file is printed by the :meth:`print_options()
<bonafide.bonafide.AtomBondFeaturizer.print_options>` method (see above).
