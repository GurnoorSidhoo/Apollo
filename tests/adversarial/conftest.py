import os
from pathlib import Path

import pytest

from . import helpers


def pytest_addoption(parser):
    parser.addoption(
        "--adversarial-report-dir",
        action="store",
        default="",
        help="Directory to write adversarial JSON/text reports to.",
    )


@pytest.fixture(scope="session")
def adversarial_manifest():
    return helpers.load_adversarial_manifest()


@pytest.fixture(scope="session")
def adversarial_cases(adversarial_manifest):
    return list(adversarial_manifest.get("cases", []))


@pytest.fixture(scope="session")
def adversarial_taxonomy():
    return helpers.load_failure_taxonomy()


@pytest.fixture(scope="session")
def adversarial_context_presets():
    return helpers.load_context_presets()


@pytest.fixture(scope="session")
def adversarial_baseline():
    return helpers.load_baseline()


@pytest.fixture(scope="session")
def adversarial_report_dir(pytestconfig):
    configured = pytestconfig.getoption("--adversarial-report-dir") or os.environ.get("APOLLO_ADVERSARIAL_REPORT_DIR")
    return Path(configured) if configured else helpers.DEFAULT_REPORT_DIR
