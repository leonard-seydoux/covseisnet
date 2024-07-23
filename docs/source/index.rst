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

   
This package is released under the GNU General Public License v3.0 (see :ref:`license`). Please include the following statement in the ackowledgements section of your publication, and please include a reference to one the paper listed in the :ref:`references` section below.

.. admonition:: Citation statement

   This work made use of the Covseisnet package, a Python package for array signal processing, developed by Léonard Seydoux, Jean Soubestre, Cyril Journeau, Francis Tong & Nikolai Shapiro.



.. _references:

References
----------

The following publications have made use of this package, or of a development version of it. Please refer to the description provided therein in order to get in-depth descriptions of the tools defined in the package as well as an overview of possible applications.

1. Seydoux, L., Shapiro, N. M., de Rosny, J., Brenguier, F., & Landès, M. (2016). "Detecting seismic activity with a covariance matrix analysis of data recorded on seismic arrays." Geophysical Journal International, 204(3), 1430-1442. https://doi.org/10.1093/gji/ggv531


2. Seydoux, L., Shapiro, N. M., de Rosny, J., & Landès, M. (2016). "Spatial coherence of the seismic wavefield continuously recorded by the USArray." Geophysical Research Letters, 43, 9644-9652. https://doi.org/10.1002/2016GL070320


3. Seydoux, L., de Rosny, J., & Shapiro, N. M. (2017). "Pre-processing ambient noise cross-correlations with equalizing the covariance matrix eigenspectrum." Geophysical Journal International, 210(3), 1432-1449. https://doi.org/10.1093/gji/ggx250


4. Soubestre, J., Seydoux, L., Shapiro, N. M., de Rosny, J., & others. (2018). "Network-based detection and classification of seismovolcanic tremors: Example from the Klyuchevskoy Volcanic Group in Kamchatka." Journal of Geophysical Research: Solid Earth, 123(1), 564-582. https://doi.org/10.1002/2017JB014726


5. Soubestre, J., Seydoux, L., Shapiro, N. M., de Rosny, J., Droznin, D. V., Droznina, S. Ya., Senyukov, S. L., & Gordeev, E. I. (2019). "Depth migration of seismovolcanic tremor sources below the Klyuchevskoy Volcanic Group (Kamchatka) determined from a network-based analysis." Geophysical Research Letters, 46(14), 8018-8030. https://doi.org/10.1029/2019GL083465


6. Lott, M., Roux, P., Seydoux, L., Tallon, B., Pelat, A., Skipetrov, S., & Colombi, A. (2020). "Localized modes on a metasurface through multi-wave interactions." Physical Review Materials, 4(6), 065203. https://doi.org/10.1103/PhysRevMaterials.4.065203


7. Journeau, C., Shapiro, N. M., Seydoux, L., Soubestre, J., Ferrazzini, V., & Peltier, A. (2020). "Detection, classification, and location of seismovolcanic signals with multicomponent seismic data: Example from the Piton de la Fournaise volcano (La Réunion, France)." Journal of Geophysical Research: Solid Earth, 125, e2019JB019333. https://doi.org/10.1029/2019JB019333

