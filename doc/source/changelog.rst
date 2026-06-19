Changelog
=========

All notable changes to this project will be documented in this file.  This
project adheres to `Semantic Versioning <http://semver.org/spec/v2.0.0.html>`_.

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

