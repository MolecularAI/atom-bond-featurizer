####################################
 External programs and environments
####################################

It is possible to implement custom features (see :doc:`custom`) that make use of **additional
external programs**, including Python packages that cannot be installed in the BONAFIDE environment.
This can be achieved with the :func:`external_driver() <bonafide.utils.driver.external_driver>`
function. This function allows to run arbitrary (Python) scripts as a subprocess, thereby providing
access to further programs and external Python packages that are not installed in the BONAFIDE
environment. In the case of Python scripts, the external program is the Python interpreter of an
external environment.

The :func:`external_driver() <bonafide.utils.driver.external_driver>` function can be imported as
follows.

.. code:: python

   from bonafide.utils.driver import external_driver

It takes four **required arguments**:

-  the path to the external program (str),
-  the input script to run (str),
-  its desired file extension (str), and
-  the name of the molecule (namespace) which is required for logging purposes (str).

For the case of Python scripts, it is optionally possible to pass a list of required dependencies
for the external environment. It is then ensured that these dependencies are installed.
Additionally, it is possible to pass additional keyword arguments to the run method of the
subprocess as demonstrated below.

Example usage with a Python script:

.. code:: python

   """Example usage of the external_driver() function to run a Python script
   in an external environment."""

   from bonafide.utils.driver import external_driver

   smiles = "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl"

   # Python script to run in the external environment
   script = [
       "import pandas as pd",
       "from external_package import something",
       f"res = something.predict({smiles})",
       "res.to_csv('external_out.csv')",
   ]
   script = "\n".join(script)

   # Run the script
   result = external_driver(
       program_path="~/miniconda3/envs/external_env/bin/python",
       program_input=script,
       input_file_extension=".py",
       namespace="example_molecule",
       dependencies=["pandas", "external_package"],
       capture_output=True,
       text=True,
       check=False,
   )

After using the external driver, it is possible to further process the ``external_out.csv`` file to
extract the desired features.
