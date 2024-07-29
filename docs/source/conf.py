"""Configuration file for the Sphinx documentation builder.

This file does only contain a selection of the most common options. For a full
list see the documentation: http://www.sphinx-doc.org/en/stable/config.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass, field
import sphinxcontrib.bibtex.plugin
from sphinxcontrib.bibtex.style.referencing import BracketStyle
from sphinxcontrib.bibtex.style.referencing.author_year import (
    AuthorYearReferenceStyle,
)


# Define the bracket style for the references
def bracket_style() -> BracketStyle:
    return BracketStyle(
        left="(",
        right=")",
    )


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
author = "Léonard Seydoux, Jean Soubestre, Cyril Journeau, Francis Tong & Nikolai Shapiro"

# The short X.Y version
version = "1.0"

# The full version, including alpha/beta/rc tags
release = "1.0.0"

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
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
    "matplotlib.sphinxext.plot_directive",
]


# Plot directive
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("svg", 150)]

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


# Gallery
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
    "image_srcset": ["4x"],
    "download_all_examples": True,
    "remove_config_comments": True,
    "within_subsection_order": "FileNameSortKey",
    "capture_repr": (),
}

# Templates
templates_path = ["_templates"]

# Sources
source_suffix = [".rst", ".md"]
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year_round"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation for
# a list of supported languages. This is also used if you do content
# translation via gettext catalogs. Usually you set "language" from the
# command line for these cases.
language = "english"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []
add_module_names = True


# Theme
# pygments_style = "paraiso-light"
# pygments_style_dark = "monokai"
html_theme = "sphinx_book_theme"
html_css_path = ["_static"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo_covseisnet_normal.svg"
html_title = "covseisnet"
html_favicon = "_static/logo_covseisnet_small.svg"
htmlhelp_basename = "covseisnetdoc"
html_theme_options = {
    "logo": {"alt_text": "Covseisnet logo"},
    "repository_url": "https://github.com/covseisnet/covseisnet",
    "use_repository_button": True,
    "use_download_button": False,
    "use_issues_button": True,
    "pygments_light_style": "paraiso-light",
    "pygments_dark_style": "paraiso-dark",
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
