Installation
============

Requirements
------------

- Python **3.10.x** exactly (``>=3.10,<3.11``). MARLA checks this at CLI
  startup and refuses to run on any other version -- see
  :mod:`marla.utils.python_version`.
- `NASimEmu <https://github.com/jaromiru/NASimEmu>`_, installed separately
  -- see "Installing NASimEmu" below. ``marla init`` will look for a
  ``NASimEmu/scenarios/`` directory near your current directory to fill
  in a starter config automatically; see :doc:`quickstart`.
- A CUDA-capable GPU if you plan to set ``device: gpu`` -- otherwise
  ``device: cpu`` or ``device: auto`` (the default) both work on CPU alone,
  just slower for the recurrent PPO forward/backward passes.

Base install
------------

From PyPI:

.. code-block:: bash

   pip install marla-agents

The PyPI *distribution* is named ``marla-agents`` (PyPI's anti-typosquat
check blocks new names within edit-distance 1 of an existing project, and
plain ``marla`` collides with two already-registered ones) -- but this
only affects the ``pip install`` line above. Everything else keeps the
short name: ``import marla`` and the ``marla`` CLI command are unaffected.

This gets you the ``marla`` CLI (``marla version``, ``marla validate``,
``marla init``) and everything needed to *read* an experiment or a
finished run's metrics. It does **not** get you NASimEmu -- see above --
so ``marla run`` still needs the separate NASimEmu install before it can
do anything.

From a repository clone, for development (editable install, plus the test
suite):

.. code-block:: bash

   pip install -e ".[dev]"

This installs MARLA itself (editable) plus the ``dev`` extra (pytest and
pytest-asyncio), which is enough to run the **baseline** variant (no
Gatekeeper, no Plan Maker) and the test suite.

Installing NASimEmu
--------------------

MARLA does not declare NASimEmu as a regular ``pip`` dependency (it isn't
published on PyPI under a name/version this project can pin), so it's
installed as a second, separate editable install. A repository clone
vendors two directories for exactly this purpose:

.. code-block:: bash

   pip install -e ./gym-0.21.0   # NASimEmu pins gym==0.21.0; PyPI's own
                                  # 0.21.0 has known build issues on modern
                                  # Python/setuptools, so this vendored,
                                  # buildable copy is used instead.
   pip install -e ./NASimEmu

Do this before ``pip install -e ".[dev]"`` or after -- order doesn't
matter, only that both end up installed in the same environment.
``marla version`` reports ``nasimemu <version>`` once this worked; it
reports ``nasimemu not installed`` otherwise (and ``marla run``/``marla
scenario check`` fail immediately and specifically, rather than with a
bare ``ImportError``, if it's missing).

If you're not working from a MARLA repository clone (e.g. installed
``marla-agents`` from PyPI on its own), clone `NASimEmu
<https://github.com/jaromiru/NASimEmu>`_ (and, if its own ``gym==0.21.0``
pin fails to build, a working ``gym==0.21.0`` from any source) yourself
and install both the same way.

Optional extras
----------------

``local-lm`` -- real local Plan Maker backend
    Needed to actually run the **assisted** variant with
    ``model.backend: local`` (a real Hugging Face ``transformers`` model
    loaded and run on this machine):

    .. code-block:: bash

       pip install -e ".[local-lm]"

    This pulls in ``transformers`` and ``accelerate``. Without it,
    constructing a local backend raises an ``ImportError`` with this exact
    install command (see :mod:`marla.models.local_backend`).

``docs`` -- build this documentation
    .. code-block:: bash

       pip install -e ".[docs]"
       sphinx-build -b html docs docs/_build/html

    See the repository's ``README.md`` for more on the project layout;
    this extra only affects building the docs themselves.

Verifying the install
----------------------

.. code-block:: bash

   marla version
   marla --help

``marla version`` prints MARLA's own version plus the resolved versions of
its key dependencies (torch, torch_geometric, spade, pydantic, typer,
nasimemu) and the current git commit, if any -- the same information
recorded in every run's ``metadata.json`` (see :doc:`metrics`).
