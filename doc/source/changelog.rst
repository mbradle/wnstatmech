Changelog
=========

All notable changes to this project will be documented in this file.  This
project adheres to `Semantic Versioning <http://semver.org/spec/v2.0.0.html>`_.

Version 1.2.0
-------------

New:

  * The test suite now resides in a top-level ``tests`` directory.
  * Optional ``pytest-benchmark`` benchmarks cover representative scalar,
    batched, and derivative thermodynamics workloads.
  * A direct chemical-potential callback can provide alpha or request the
    existing numerical fallback for each state.
  * Scalar exact-result caches now use a bounded least-recently-used policy.
    Cache size is configurable and cached state can be cleared explicitly.

Improve:

  * Vectorized degenerate fermion calculations now use small alpha-sorted
    batches instead of a single global integration grid.
  * Python 3.10 and newer are now explicitly supported, with CI coverage
    configured for Python 3.10 through 3.14.
  * The tutorial now demonstrates a direct chemical-potential approximation
    with numerical fallback and corrects its classical-neutron examples.

Version 1.1.1
-------------

Fix:

  * The ubuntu version has been updated for proper building of documentation.

Version 1.1.0
-------------

New:

  * Scalar chemical-potential solves now use a faster brentq-based path with
    caching and warm-starting from the previous scalar solve.
  * Quantity calculations support vectorized inputs and batched integration.
  * Integration tolerances are configurable.
  * Temperature derivatives at fixed number density use a faster implicit
    derivative path with a finite-difference fallback.

Fix:

  * The minimum scipy version has been updated for the elementwise root-finding
    API.
  * User-supplied quantity functions that return 0 are now honored.
  * Expected overflow warnings in vectorized boson integrands are suppressed.

Version 1.0.4
-------------

Fix:

  * The minimum python version has been corrected.

Version 1.0.3
-------------

Fix:

  * The minimum scipy version needed for the package has been specified (to allow
    for scipy differentiate).
  * The minimum python version has been updated.

Version 1.0.2
-------------

Fix:

  * The required python version has been updated.

Version 1.0.1
-------------

New:

  * The zenodo doi has been added.
  * A .readthedocs.yaml file has been added for proper documentation building.


Version 1.0.0
-------------

New:

  * Initial release.
