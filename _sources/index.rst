Home
====

**Covseisnet** is a Python package for array signal processing, with a focus
on data from seismic networks. The core mathematical object of the package is
the network covariance matrix, used for signal detection, source separation,
localisation, and plane-wave beamforming. More precisely, the signal detection
and processing methods are based on the analysis of the covariance matrix
spectrum. The covariance matrix can be used as input for classical array
processing tools such as beamforming and inter-station cross-correlations. 

In order to provide tools that are user-friendly and efficient, the package
builts on `ObsPy <https://github.com/obspy/obspy/wiki/>`_, a Python toolbox
for seismology, on `Scipy <https://www.scipy.org>`_, and `Numpy
<https://numpy.org>`_, two Python libraries for scientific computing and
linear algebra. 

This library is hosted on GitHub at
https://github.com/leonard-seydoux/covseisnet and is distributed under the GNU
General Public License v3.0. Contributions are welcome, and can be made via
pull requests on the GitHub repository.

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
   
This package is released under the GNU General Public License v3.0 (see the
:ref:`license` section for more information). Please include the following
statement in the ackowledgements section of your publication, and/or reference
one the paper listed in the :ref:`references` section below.

   This work made use of Covseisnet, a Python package for array signal
   processing, developed by Léonard Seydoux, Jean Soubestre, Cyril Journeau,
   Francis Tong & Nikolai Shapiro.

Publications summary
~~~~~~~~~~~~~~~~~~~~

The method was first introduced in :cite:t:`seydoux_detecting_2016` with an
application to the monitoring of the Piton de la Fournaise eruptions. We then
applied it to the analysis of USArrays data :cite:p:`seydoux_spatial_2016`. We
also proposed a pre-processing method for ambient noise cross-correlations
:cite:p:`seydoux_pre-processing_2017`. 

The method was then applied to the detection and classification of
seismovolcanic tremors :cite:p:`soubestre_network-based_2018` and to the depth
migration of seismovolcanic tremor sources :cite:p:`soubestre_depth_2019`. The
method was also applied to the study of localized modes on a metasurface
:cite:p:`lott_localized_2020`, and to the detection, classification, and
location of seismovolcanic signals at the Piton de la Fournaise volcano in
:cite:p:`journeau_detection_2020`.

In the context of unsuperised learning, we also used the covariance matrix
representation to reveal patterns in the continuous seismic wavefield during
the 2009 L'Aquila earthquake :cite:p:`shi_unsupervised_2021`. We also
investigated the dynamics of the Kamchatka volcanic systen in
:cite:t:`journeau_seismic_2022`.

.. _references:

References
~~~~~~~~~~

.. bibliography:: references.bib
   :style: unsrt

About us
--------

This package was mainly developed by `Léonard Seydoux
<https://sites.google.com/view/leonard-seydoux/accueil>`_ during his PhD at
the Institut de Physique du Globe de Paris, under the supervision of Nikolai
Shapiro. Several contributions for the core program were made by Jean
Soubestre and Cyril Journeau. Francis Tong contributed to the distribution of
the packge via PyPI and Conda. 

Several other versions of this package are available (not distributed yet,
planned for the future). A first version was developed in Matlab by Léonard
Seydoux. For computational efficiency, a second version was developed in C++
by Matthieu Landès. For now, it works only with SAC files. 

If you have any questions, please contact us via the GitHub repository at
https://github.com/covseisnet/covseisnet. You can also consider opening an
issue on the repository, or creating pull requests.