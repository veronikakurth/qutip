QuTiP documentation
===================

This directory contains the source files for the QuTiP documentation.
It lives inside the main [qutip/qutip](https://github.com/qutip/qutip) repository; there is no longer a separate documentation repository.

For pre-built documentation, see https://qutip.readthedocs.io/en/stable/ (also linked from https://www.qutip.org/documentation.html).

Building
--------

The main Python requirements for the documentation are `sphinx`, `sphinx_rtd_theme`, `numpydoc`, `sphinxcontrib-bibtex`, `sphinx-copybutton`, `matplotlib` and `ipython`.
The full set of extensions in use is listed in `conf.py`.
You will also need a sensible copy of `make`, and if you want to build the LaTeX documentation then also a `pdflatex` distribution.

The simplest way to get a functional build environment is to use the `doc/requirements.txt` file, which completely defines a known-good `pip` environment.
It is the same file the Read the Docs build uses, and it is currently pinned against Python 3.13, though other versions supported by QuTiP (3.11+) may well work.
If you typically use conda, the way to do this is

```bash
$ conda create -n qutip-doc-build python=3.13
$ conda activate qutip-doc-build
$ pip install -r /path/to/qutip/doc/requirements.txt
```

We recommend installing the documentation dependencies with `pip` rather than `conda`, because several of the packages can be slower to update their `conda` recipes.

### Installing QuTiP into the build environment

The documentation build runs many components of the main QuTiP library to generate figures, to test the output and to generate all of the API documentation, so you need a version of QuTiP in the same environment.
The build exercises almost all of the optional features, so install the `full` extra as well; if you get failure messages in red, it is usually a missing optional dependency.

If you simply want to build the documentation without editing the main library, install a release version:

```bash
$ pip install 'qutip[full]'
```

If you are also modifying the main library, install your working copy instead.
QuTiP builds with `meson-python`, so an editable install needs the build tools present in the environment and `--no-build-isolation`.
Note that `doc/requirements.txt` pins `cython`, `numpy` and `scipy` but **not** `meson-python` or `ninja`, so those have to be added separately:

```bash
$ pip install meson-python ninja
$ pip install --no-build-isolation --editable '.[full]'
```

The `--no-build-isolation` flag is required, not merely an optimisation; see the "Contributing to QuTiP" page of this documentation for the details and for the available build options.

### Running the build

After you have done this, you can effect the build with `make`.
The targets you might want are `html`, `latexpdf` and `clean`, which build the HTML pages, build the PDFs, and delete all built files respectively.
For example, to build the HTML files only, use

```bash
$ make html
```

Two further targets are useful before opening a pull request:

- `make doctest` runs the code examples embedded in the documentation.
- `make linkcheck` verifies that external links still resolve.

Re-run `make html` any time you make changes; it should only rebuild files that have changed.

### Read the Docs

The hosted build is configured in `.readthedocs.yaml` at the repository root.
It builds on Python 3.13 and installs `doc/requirements.txt` followed by `pip install .[full]`, so that file is the source of truth for the documentation dependencies.
Note that Read the Docs sets `fail_on_warning: true`: a build that emits Sphinx warnings will fail there even if `make html` completed locally, so check the output of your local build for warnings as well.

Writing user guides
-------------------

The user guide provides an overview of QuTiP's functionality.
It is composed of individual reStructuredText (`.rst`) files which each get rendered as a webpage, and each page typically tackles one area of functionality.
To learn more about how to write `.rst` files, it is useful to follow the [Sphinx guide](https://www.sphinx-doc.org/en/master/usage/index.html).

Code examples in the guide are written with two Sphinx directives:
[`doctest`](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html), which allows the examples to be tested by `make doctest`, and matplotlib's
[`plot`](https://matplotlib.org/stable/api/sphinxext_plot_directive_api.html) directive, which renders figures.
For guidelines on how to use each of them, including which one to reach for when an example needs both testing and a figure, see the *Working with the QuTiP Documentation* page (`doc/development/docs.rst`) in this documentation.
Additional extensions can be configured in the `conf.py` file.

Changes to the documentation, like all other changes, need a `towncrier` entry in `doc/changes`; see the contributing guide for details.
