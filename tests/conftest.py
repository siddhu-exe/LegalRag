"""
Test configuration for the LegalRAG local unit-test suite.

Local tests must never run the production pipeline. The runtime environment is forced to
``local_stub`` here (before any application module is imported) so the suite never
downloads artifacts, loads heavy models, or calls external providers.
"""

import os

os.environ["ENVIRONMENT"] = "local_stub"
