"""Repo-relative paths, resolved from this package's location.

Using the package location (not the process cwd) means charts and reports land
in the project's ``reports/`` dir regardless of who spawned the server.
"""

import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PKG_DIR)
REPORTS_DIR = os.path.join(REPO_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)
