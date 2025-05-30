# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mvcluster'
copyright = '2025, Hamady GACKOU'
author = 'Hamady GACKOU'
release = '1.10'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

suppress_warnings = ['ref.ref']

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # pour que Sphinx tro

# Extensions
extensions = [
    'sphinx.ext.autodoc',    # generate documentation from docstrings
    'sphinx.ext.napoleon',   # support Google / NumPy docstrings
    'sphinx.ext.viewcode',   # link to source code
]
extensions += ['sphinx.ext.intersphinx']
intersphinx_mapping = {
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

autodoc_default_options = {
    'exclude-members': '__get__',  # ou tout le sous-module sklearn.utils._metadata_requests
}



templates_path = ['_templates']
exclude_patterns = []

language = 'en'  # Set the language to English

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
