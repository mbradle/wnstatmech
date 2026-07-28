Overview
========

.. image:: https://zenodo.org/badge/1019497078.svg
  :target: https://doi.org/10.5281/zenodo.17561282

wnstatmech is a python package for handling the statistical
mechanics of fermions and bosons.

|pypi| |doc_stat| |license| |test| |lint-test| |black|

Installation
------------

Install from `PyPI <https://pypi.org/project/wnstatmech>`_ with pip by
typing in your favorite terminal::

    $ pip install wnstatmech

Performance controls
--------------------

Fermion and Boson objects accept optional ``integration_epsabs`` and
``integration_epsrel`` arguments to control numerical integration accuracy.
Their scalar exact-result caches retain 1,024 entries by default; pass
``cache_size=0`` to disable them or call ``clear_cache()`` to clear a
particle's cached values and chemical-potential warm start.

For an application-specific direct inversion, install a chemical-potential
function that accepts scalar temperature and number-density values and returns
``alpha`` or ``None`` to use the numerical fallback::

    electron.update_chemical_potential_function(my_direct_alpha)

For better throughput over many thermodynamic states, pass vectorized
``temperature`` and ``alpha`` arrays to ``compute_quantity`` rather than
calling it repeatedly in a Python loop::

    import wnstatmech as ws
    import numpy as np

    electron = ws.fermion.create_electron(integration_epsrel=1.0e-6)
    temperatures = np.geomspace(1.0e7, 1.0e10, 24)
    alphas = np.linspace(-8.0, 8.0, 24)
    pressures = electron.compute_quantity("pressure", temperatures, alphas)

Scalar call signatures remain backwards compatible with 1.0.x usage.

Performance benchmarks
----------------------

Optional benchmarks measure representative scalar, batched, and
fixed-density derivative workloads.  They are separate from the correctness
tests and are not run by the normal build.  Install the benchmark extra and
run them with::

    $ pip install -e '.[benchmark]'
    $ pytest benchmarks --benchmark-only

Benchmark results are most useful when compared before and after a change on
the same machine.  The benchmark cases intentionally create a fresh particle
for scalar calls so that they measure a cold numerical solve rather than a
cache hit.

Authors
-------

- Bradley S. Meyer <mbradle@g.clemson.edu>
- Lucas S. Walls <lwalls@g.clemson.edu>

Contribute
----------

- Issue Tracker: `<https://github.com/mbradle/wnstatmech/issues/>`_
- Source Code: `<https://github.com/mbradle/wnstatmech/>`_
- `Contributing guide <https://github.com/mbradle/wnstatmech/blob/main/CONTRIBUTING.md>`_

License
-------

The project is licensed under the GNU Public License v3 (or later).

Documentation
-------------

The project documentation is available at `<https://wnstatmech.readthedocs.io>`_.

Usage
-----

The best way to get started using wnstatmech is to follow the
`tutorial <https://github.com/mbradle/wnstatmech/tree/main/tutorial>`_.

.. |pypi| image:: https://badge.fury.io/py/wnstatmech.svg 
    :target: https://badge.fury.io/py/wnstatmech
.. |license| image:: https://img.shields.io/github/license/mbradle/wnstatmech
    :alt: GitHub
.. |doc_stat| image:: https://readthedocs.org/projects/wnstatmech/badge/?version=latest 
    :target: https://wnstatmech.readthedocs.io/en/latest/?badge=latest 
    :alt: Documentation Status
.. |test| image:: https://github.com/mbradle/wnstatmech/actions/workflows/test.yml/badge.svg?branch=main&event=push
        :target: https://github.com/mbradle/wnstatmech/actions/workflows/test.yml
.. |lint| image:: https://img.shields.io/badge/linting-pylint-yellowgreen
    :target: https://github.com/pylint-dev/pylint
.. |lint-test| image:: https://github.com/mbradle/wnstatmech/actions/workflows/lint.yml/badge.svg?branch=main&event=push
        :target: https://github.com/mbradle/wnstatmech/actions/workflows/lint.yml
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
