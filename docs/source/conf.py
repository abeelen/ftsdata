#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sphinx configuration for the standalone ftsdata documentation."""

import os
import sys

# Make the ftsdata package importable from the source tree.
sys.path.insert(0, os.path.abspath("../.."))

# -- General configuration ------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    "sphinx_copybutton",
    "myst_parser",
    "sphinxcontrib.mermaid",
]

# MyST-Parser extensions — dollarmath enables $...$ and $$...$$ in .md files
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

plot_html_show_source_link = False
plot_html_show_formats = False

mermaid_init_js = """
mermaid.initialize({
    startOnLoad: true,
    theme: 'base',
    themeVariables: {
        primaryColor: '#2980b9',
        primaryTextColor: '#fff',
        primaryBorderColor: '#1a5276',
        lineColor: '#555',
        secondaryColor: '#85c1e9',
        tertiaryColor: '#d6eaf8',
        edgeLabelBackground: '#eaf4fb',
        clusterBkg: '#d6eaf8',
        titleColor: '#1a5276',
        fontFamily: 'Lato, sans-serif',
        fontSize: '14px'
    }
});
"""

autosummary_generate = True
numpydoc_show_class_members = False

templates_path = ["_templates"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

project = "ftsdata"
copyright = "2019, Alexandre Beelen"
author = "Alexandre Beelen"

try:
    from importlib.metadata import version as _version

    version = _version("ftsdata")
except Exception:
    version = "0.1.0"
release = version

language = "en"

exclude_patterns = []

pygments_style = "sphinx"

todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

html_theme = "sphinx_rtd_theme"

html_theme_path = []

html_static_path = ["_static"]

html_sidebars = {
    "**": [
        "relations.html",
        "searchbox.html",
    ]
}


# -- Options for HTMLHelp output ------------------------------------------

htmlhelp_basename = "ftsdata"


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {}

latex_documents = [
    (master_doc, "ftsdata.tex", "ftsdata Documentation", "Contributors", "manual"),
]


# -- Options for manual page output ---------------------------------------

man_pages = [(master_doc, "ftsdata", "ftsdata Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "ftsdata",
        "ftsdata Documentation",
        author,
        "ftsdata",
        "Fourier-transform spectrometer data containers and transforms.",
        "Miscellaneous",
    ),
]


# -- Example configuration for intersphinx --------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}
