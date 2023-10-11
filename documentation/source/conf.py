# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import pathlib
import sys

# add root directory to path
target_dir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, target_dir)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Coreax"
copyright = '2023, ""'
author = '""'
release = "v0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# sphinx extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinxcontrib.bibtex",
]

# file sources
source_suffix = [".rst", ".md"]
myst_enable_extensions = ["html_image"]

# bibtex references path
bibtex_bibfiles = ["references.bib"]

templates_path = ["_templates"]
exclude_patterns = []

# Display type annotations only in compiled description.
autodoc_typehints = "description"

# set intersphinx mapping to auto-link to external documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "classic"
html_static_path = ["_static"]
