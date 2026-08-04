Installation
============

Requirements
------------

- Python **3.10.x** exactly (``>=3.10,<3.11``). MARLA checks this at CLI
  startup and refuses to run on any other version -- see
  :mod:`marla.utils.python_version`.
- `NASimEmu <https://github.com/jaromiru/NASimEmu>`_, cloned alongside (or
  vendored inside) the MARLA repository. ``marla init`` will look for it
  automatically; see :doc:`quickstart`.
- A CUDA-capable GPU if you plan to set ``device: gpu`` -- otherwise
  ``device: cpu`` or ``device: auto`` (the default) both work on CPU alone,
  just slower for the recurrent PPO forward/backward passes.

Base install
------------

From the repository root:

.. code-block:: bash

   pip install -e ".[dev]"

This installs MARLA itself (editable) plus the ``dev`` extra (pytest and
pytest-asyncio), which is enough to run the **baseline** variant (no
Gatekeeper, no Plan Maker) and the test suite.

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
