Home
====

**Covseisnet** is a Python package for array signal processing, with a focus on data from seismic networks. The central mathematical object of the package is the network covariance matrix, used for signal detection, source separation, localisation, and plane-wave beamforming. 

The core signal detection algorithms are based on the analysis of the spectrum of the network covariance matrix. Eigendecomposition provides a basis for signal detection and blind source separation. In addition, the covariance matrix can be used as input for classical array processing tools such as beamforming and inter-station cross-correlations. 

The package builts on the `ObsPy <https://github.com/obspy/obspy/wiki/>`_ and `Numpy <https://numpy.org>`_. The code repository is hosted on GitHub at https://github.com/covseisnet/covseisnet and is distributed under the GNU General Public License v3.0.

.. image:: https://zenodo.org/badge/263984678.svg
   :target: https://zenodo.org/doi/10.5281/zenodo.10990031
   
.. image:: https://img.shields.io/pypi/v/covseisnet.svg
   :target: https://pypi.org/project/covseisnet/

.. image:: https://img.shields.io/conda/v/conda-forge/covseisnet
   :target: https://anaconda.org/conda-forge/covseisnet

.. image:: https://img.shields.io/conda/l/conda-forge/covseisnet
   :target: https://www.gnu.org/licenses/lgpl.html

.. image:: https://img.shields.io/pypi/pyversions/covseisnet
   :target: https://pypi.org/project/covseisnet/

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

.. image:: https://img.shields.io/conda/pn/conda-forge/covseisnet
   :target: https://anaconda.org/conda-forge/covseisnet

.. image:: https://codecov.io/gh/covseisnet/covseisnet/branch/develop/graph/badge.svg?token=N462A7PPRF
   :target: https://codecov.io/gh/covseisnet/covseisnet


Contents
--------

.. toctree::
   :maxdepth: 1

   installation
   auto_examples/index
   api
   license

How to cite this package
------------------------

Citation statement   
~~~~~~~~~~~~~~~~~~~
   
This package is released under the GNU General Public License v3.0 (see :ref:`license`). Please include the following statement in the ackowledgements section of your publication, and/or reference one the paper listed in the :ref:`references` section below.

   This work made use of the Covseisnet package, a Python package for array signal processing, developed by Léonard Seydoux, Jean Soubestre, Cyril Journeau, Francis Tong & Nikolai Shapiro.

Publication history
~~~~~~~~~~~~~~~~~~~

The method was first introduced in :cite:`seydoux_detecting_2016` with an
application to the monitoring of the Piton de la Fournaise eruptions. We then
applied it to the analysis of USArray data in :cite:`seydoux_spatial_2016`. We then proposed a pre-processing method for ambient noise cross-correlations in :cite:`seydoux_pre-processing_2017`. The method was then applied to the detection and classification of seismovolcanic tremors in :cite:`soubestre_network-based_2018` and to the depth migration of seismovolcanic tremor sources in :cite:`soubestre_depth_2019`. The method was also applied to the study of localized modes on a metasurface in :cite:`lott_localized_2020`. Finally, the method was applied to the detection, classification, and location of seismovolcanic signals at the Piton de la Fournaise volcano in :cite:`journeau_detection_2020`.

.. _references:

References
~~~~~~~~~~

.. bibliography:: references.bib
   :style: unsrt