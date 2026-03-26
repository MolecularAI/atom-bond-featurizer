##############
 Installation
##############

Please complete the following steps to install ``BONAFIDE`` and its :doc:`dependencies
<dependencies>`.

**************************************
 BONAFIDE and its Python dependencies
**************************************

#. Clone the **GitHub repository** and change to its root directory.

.. code:: shell

   $ git clone https://github.com/MolecularAI/atom-bond-featurizer.git
   $ cd atom-bond-featurizer

2. Create a new **conda environment** and activate it.

.. code:: shell

   $ conda env create -n bonafide_env -f bonafide_env.yml python=3.12
   $ conda activate bonafide_env

.. note::

   ``bonafide_env.yml`` provides the necessary dependencies for calculating features with BONAFIDE,
   whereas ``bonafide_env_dev.yml`` contains additional dependencies for development purposes (*e.g.*,
   testing, documentation building).

3. Install the ``kallisto`` package (in case its usage is intended). This is done separately due to
   version conflicts between kallisto dependencies and other BONAFIDE dependencies.

.. code:: shell

   $ pip install kallisto --no-deps

4. Install BONAFIDE and run the ``post_install_setup.py`` script to complete the installation.

.. code:: shell

   $ pwd
   .../atom-bond-featurizer
   $ pip install .
   $ python post_install_setup.py

*********
 ALFABET
*********

The environment for the machine learning model ``ALFABET`` is not compatible with BONAFIDE's
environment. It must therefore be installed in a **separate environment** if the model is intended
to be used through BONAFIDE.

.. code:: shell

   $ conda create -n alfabet_env -c conda-forge python=3.7 rdkit
   $ conda activate alfabet_env
   $ pip install tensorflow
   $ pip install alfabet

To allow BONAFIDE to discover the environment in which ALFABET is installed, go through the
following steps.

#. Create an instance of BONAFIDE's ``AtomBondFeaturizer`` class and inspect **ALFABET's
   configuration settings**.

.. code:: python

   >>> from bonafide import AtomBondFeaturizer
   >>> f = AtomBondFeaturizer()
   >>> f.print_options("alfabet")
   Default configuration settings at:
   .../src/bonafide/_feature_config.toml

   alfabet
      python_interpreter_path: ~/miniconda3/envs/alfabet_env/bin/python

2. Go to the ``_feature_config.toml`` file and change the value of ``python_interpreter_path`` to
   point to the Python interpreter of ``alfabet_env``. The path to the file is shown at the top of
   the output from the ``print_options()`` method. The path to the Python interpreter can be found
   out with the following commands.

.. code:: shell

   $ conda activate alfabet_env
   $ which python
   .../alfabet_env/bin/python

**********
 Multiwfn
**********

``Multiwfn`` is not a Python dependency and therefore must be **installed separately**. Go through
the following steps to install the program (if not already installed). Additional information on the
installation can be found in section 2.1 of the `Multiwfn manual
<http://sobereva.com/multiwfn/download.html>`_. BONAFIDE depends on the version 3.8 of Multiwfn.

#. Download the ``Multiwfn_3.8_bin_Linux_noGUI.zip`` file from this `webpage
   <http://sobereva.com/multiwfn/download.html>`_.

#. Unzip the folder and place it in an appropriate directory which can be accessed by BONAFIDE.

#. Ensure that the downloaded ``Multiwfn_noGUI`` file has **executable permission**. If not, this
   can be added by changing the current working directory to where Multiwfn was saved followed by
   executing the following command.

.. code:: shell

   $ chmod +x Multiwfn_noGUI

4. Add the following lines to your ``.bashrc`` file. Replace ``<path to parent directory>`` with the
   correct path.

.. code:: bash

   export Multiwfnpath=<path to parent directory>/Multiwfn_3.8_bin_Linux_noGUI
   export PATH=$PATH:<path to parent directory>/Multiwfn_3.8_bin_Linux_noGUI

***************************
 Finalization and checking
***************************

At any stage of the installation process, the ``check_bonafide_installation.py`` script can be
executed to inspect the **status of the installation**. This script checks if all dependencies are
installed correctly and whether BONAFIDE can access them. Ensure that BONAFIDE's conda environment
is activated before running the script.

.. code:: shell

   $ conda activate bonafide_env
   $ python check_bonafide_installation.py
   ...

If the output of this script ends with ``==> Hurray, all dependencies are installed correctly!``,
the installation process was completed **successfully**.
