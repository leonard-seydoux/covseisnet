# Covseisnet

<img src="docs/source/_static/logo_covseisnet_normal.svg" alt="Covseisnet Logo Normal" width="400">

[![DOI](https://zenodo.org/badge/263984678.svg)](https://zenodo.org/doi/10.5281/zenodo.10990031) [![PyPI Version](https://img.shields.io/pypi/v/covseisnet.svg)](https://pypi.org/project/covseisnet/) [![Conda Version](https://img.shields.io/conda/v/conda-forge/covseisnet)](https://anaconda.org/conda-forge/covseisnet) [![License](https://img.shields.io/conda/l/conda-forge/covseisnet)](https://www.gnu.org/licenses/lgpl.html) [![Python Versions](https://img.shields.io/pypi/pyversions/covseisnet)](https://pypi.org/project/covseisnet/) [![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/covseisnet)](https://anaconda.org/conda-forge/covseisnet) [![Code Coverage](https://codecov.io/gh/covseisnet/covseisnet/branch/develop/graph/badge.svg?token=N462A7PPRF)](https://codecov.io/gh/covseisnet/covseisnet)

**Covseisnet** is a Python package for array signal processing, with a focus on data from seismic networks. The core mathematical object of the package is the network covariance matrix, used for signal detection, source separation, localisation, and plane-wave beamforming. The signal detection and processing methods are based on the analysis of the covariance matrix spectrum. The covariance matrix can be used as input for classical array processing tools such as beamforming and inter-station cross-correlations.

To provide user-friendly and efficient tools, the package builds on [ObsPy](https://github.com/obspy/obspy/wiki/), a Python toolbox for seismology, as well as [Scipy](https://www.scipy.org) and [Numpy](https://numpy.org), two Python libraries for scientific computing and linear algebra.

This library is hosted on GitHub at
[github.com/covseisnet/covseisnet](https://github.com/covseisnet/covseisnet) and is distributed under the GNU
General Public License v3.0. Contributions are welcome, and can be made via
pull requests on the GitHub repository.



