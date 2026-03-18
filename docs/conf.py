# Project information #############################################################################
project = "BONAFIDE"
copyright = "2026, Molecular AI, AstraZeneca Gothenburg"
author = "Lukas M. Sigmund"
release = "0.1.0"

# General configuration ###########################################################################
extensions = [
    "sphinxcontrib.jquery",
    "sphinx_datatables",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinxcontrib.bibtex",
]
pygments_style = "friendly"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# API doc settings ################################################################################
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
numpydoc_show_class_members = False
autodoc_typehints = "none"


# Options for HTML output #########################################################################
html_theme = "furo"
html_static_path = ["_static"]

html_theme_options = {
    "light_logo": "doc_logo_light.png",
    "dark_logo": "doc_logo_dark.png",
    "sidebar_hide_name": True,
}

html_favicon = "_static/favicon.png"
html_title = "BONAFIDE"

html_last_updated_fmt = "%b %d, %Y"

# Datatables ######################################################################################

datatables_version = "2.3.0"
datatables_class = "sphinx-datatable"

datatables_extensions = [
    "sphinxcontrib.jquery",
    "sphinx_datatables",
]

html_css_files = [
    "https://cdn.datatables.net/buttons/3.0.2/css/buttons.dataTables.min.css",
    "custom_datatable.css",
]

html_js_files = [
    "https://cdn.datatables.net/buttons/3.0.2/js/dataTables.buttons.min.js",
    "https://cdn.datatables.net/buttons/3.0.2/js/buttons.colVis.min.js",
]

datatables_options = {
    "pageLength": 10,
    "language": {"lengthLabels": {"-1": "Show all"}},
    "lengthMenu": [10, 25, 50, -1],
    "dom": "Blfrtip",
    "buttons": [{"extend": "colvis", "text": "Show or hide columns"}],
}


# Bibtex ##########################################################################################
bibtex_bibfiles = ["references.bib"]
bibtex_footbibliography_header = ".. rubric:: References"
