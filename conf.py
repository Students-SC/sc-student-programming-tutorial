# Configuration file for the Sphinx documentation builder.

from pathlib import Path


project = "Hands-On Introduction to HPC and AI"
author = "Students@SC"
copyright = "2026, Students@SC"

root_doc = "index"
source_suffix = {
    ".md": "markdown",
}

extensions = [
    "myst_parser",
    "sphinx_copybutton",
]

myst_heading_anchors = 3
copybutton_prompt_text = r"^(\$ |>>> |\.\.\. )"
copybutton_prompt_is_regexp = True

exclude_patterns = [
    "_build",
    ".git",
    ".github",
    ".venv",
    "venv",
    "__pycache__",
    "setup/vllm-server/README.md",
]

html_theme = "furo"
html_title = project
html_copy_source = False
html_show_sourcelink = False
html_extra_path = [".nojekyll"] if Path(".nojekyll").exists() else []
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["force-light.js"]
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
    ]
}

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#368fd8",
        "color-brand-content": "#368fd8",
        "color-admonition-background": "#f8fafc",
    },
}
