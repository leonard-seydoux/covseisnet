Installation
============

.. note::

    The package is not yet available on the `PyPI repository <https://pypi.org/>`_. The following instructions are for installing the package directly from the GitHub repository. These installation instruction assume that you have `Python <https://www.python.org/downloads/>`_ and `pip <https://pip.pypa.io/en/stable/installing/>`_ installed on your system. We recommend using a `virtual environment <https://docs.python.org/3/library/venv.html>`_ to install the package.

Create an Anaconda virtual environment
---------------------------------------

This example shows how to create a new virtual environment using `Anaconda <https://www.anaconda.com/products/distribution>`_ and install the package in it. You can also choose another virtual environment manager, such as `virtualenv <https://virtualenv.pypa.io/en/latest/>`_. First, we create a new virtual environment using Anaconda. We suggest the environment name ``covseisnet``, but if your environment is specific to a project, you can name it accordingly. We also install Python 3.12 in the environment, which comes with Pip (Python's package installer). Run the following command in your terminal:

.. code-block:: text
    
    conda create --name covseisnet python=3.12

Then, activate the virtual environment and use the newly installed Python version:

.. code-block:: bash

    conda activate covseisnet

Installation with Pip from GitHub
---------------------------------

We can now install the package using Pip. Running the following command will download the package from the GitHub repository and install it in the virtual environment:

.. code-block:: bash

    pip install git+https://github.com/leonard-seydoux/covseisnet.git


Testing the installation
------------------------

After the installation completes, you can check if the package was installed correctly by running a simple test in a Python environment (such as a Python shell). Execute the following command:

.. code-block:: python

    >>> import covseisnet as csn

If you don't encounter any errors, the package has been installed successfully. If you need to run more complex tests, you can run a `PyTest <https://docs.pytest.org/en/stable/>`_ test suite. First, install the PyTest package using Pip:

.. code-block:: bash

    pip install pytest

Then, navigate to the package's ``tests`` directory and run the following command:

.. code-block:: bash

    pytest