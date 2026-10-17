"""Configuration file for the Sphinx documentation builder.

This file does only contain a selection of the most common options. For a full
list see the documentation: http://www.sphinx-doc.org/en/stable/config.
"""

import importlib.metadata
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass, field
import sphinxcontrib.bibtex.plugin
from sphinxcontrib.bibtex.style.referencing import BracketStyle
from sphinxcontrib.bibtex.style.referencing.author_year import (
    AuthorYearReferenceStyle,
)


# Define the bracket style for the references
def bracket_style() -> BracketStyle:
    return BracketStyle(left="(", right=")")


# Define the reference style
@dataclass
class BracketReferenceStyle(AuthorYearReferenceStyle):
    bracket_parenthetical: BracketStyle = field(default_factory=bracket_style)
    bracket_textual: BracketStyle = field(default_factory=bracket_style)
    bracket_author: BracketStyle = field(default_factory=bracket_style)
    bracket_label: BracketStyle = field(default_factory=bracket_style)
    bracket_year: BracketStyle = field(default_factory=bracket_style)


# Register the plugin
sphinxcontrib.bibtex.plugin.register_plugin(
    "sphinxcontrib.bibtex.style.referencing",
    "author_year_round",
    BracketReferenceStyle,
)


# Project information
project = "covseisnet"
copyright = "2022, The Covseisnet Team"
author = "Léonard Seydoux, Éric Beaucé, Francis Tong, Jean Soubestre, Cyril Journeau & Nikolai Shapiro"

# The full version, including alpha/beta/rc tags
release = importlib.metadata.version("covseisnet")

# The short X.Y version
version = ".".join(release.split(".")[:2])

# Sphinx extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinxcontrib.bibtex",
    "sphinx_new_tab_link",
    "matplotlib.sphinxext.plot_directive",
]

# Smart quotes
smartquotes = False
language = "en"

# Plot directive
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("svg", 150)]
# Run plot code from the repository root so that docstrings can use paths
# such as "docs/source/data/undervolc.mseed", as the doctests do.
plot_working_directory = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_preprocess_types = True

# Autosummary
autosummary_generate = True
autosummary_generate_overwrite = True

# Autodoc
autodoc_member_order = "bysource"
autoclass_content = "both"

# Templates
templates_path = ["_templates"]

# Sources
source_suffix = [".rst"]
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year_round"

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["examples/matched_field_processing_fuji.ipynb"]
add_module_names = True

# Theme
html_theme = "sphinx_book_theme"
html_css_path = ["_static"]
html_static_path = ["_static"]
html_css_files = ["custom.css", "nbsphinx-gallery.css"]
html_logo = "_static/logo_covseisnet_normal.svg"
html_title = "covseisnet"
html_favicon = "_static/logo_covseisnet_small.svg"
htmlhelp_basename = "covseisnetdoc"
html_theme_options = {
    "logo": {"alt_text": "Covseisnet logo"},
    "repository_url": "https://github.com/leonard-seydoux/covseisnet",
    "repository_branch": "main",
    "path_to_docs": "docs/source",
    "use_repository_button": True,
    "use_download_button": False,
    "use_issues_button": True,
    "home_page_in_toc": True,
    "show_toc_level": 1,
}

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "obspy": ("https://docs.obspy.org", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

nbsphinx_execute = "auto"
nbsphinx_prompt_width = "0"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg'}",
    "--InlineBackend.rc={'figure.dpi': 150}",
]

# Linkcheck configuration
linkcheck_ignore = [
    r"https://doi\.org/.*",  # DOI services can be intermittently slow
    r"https://arxiv\.org/.*",  # arXiv throttling can trigger false negatives
    r"https://academic\.oup\.com/.*",  # often blocked by 403
    r"https://(onlinelibrary\.)?wiley\.com/.*",  # often blocked by 403
    r"http://doi\.wiley\.com/.*",  # legacy Wiley DOI resolver 403s
    r"https://(link\.)?aps\.org/.*",  # APS links frequently 403 without cookies
    r"https://journals\.aps\.org/.*",  # APS journal abstracts 403
    r"https://www\.science\.org/.*",  # Science magazine paywalled
    r"https://www\.sciencedirect\.com/.*",  # Elsevier/SD 403 without auth
]
linkcheck_retries = 2
linkcheck_timeout = 30
linkcheck_workers = 5
linkcheck_anchors = True

# MathJax configuration: define common symbol macros for consistent notation
mathjax3_config = {
    "tex": {
        "macros": {
            # Coherence-related symbols
            "width": r"\\sigma",
            "entropy": r"\\mathrm{h}",
            "diversity": r"\\mathrm{D}",
            "eigv": r"\\lambda",
        }
    }
}

# Provide handy RST substitutions to keep symbols consistent in prose
rst_prolog = r"""
.. |sigma| replace:: :math:`\\sigma`
.. |h| replace:: :math:`h`
.. |D| replace:: :math:`D`
.. |lambda| replace:: :math:`\\lambda`
"""
