from pathlib import Path
import tomllib

import mcp_llm_router
import mcp_llm_router.judge


REPO_ROOT = Path("/home/runner/work/mcp-llm-router/mcp-llm-router")


def test_package_versions_match_release_metadata():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    version = pyproject["project"]["version"]

    assert mcp_llm_router.__version__ == version
    assert mcp_llm_router.judge.__version__ == version


def test_python_support_metadata_is_explicit():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]

    assert project["requires-python"] == ">=3.12"
    classifiers = set(project["classifiers"])
    assert "Programming Language :: Python :: 3.12" in classifiers
    assert "Programming Language :: Python :: 3.13" in classifiers


def test_readme_config_snippets_reference_embedded_router_consistently():
    readme = (REPO_ROOT / "README.md").read_text()

    assert "This project is tested on **Python 3.12 and 3.13**." in readme
    assert "#### Canonical minimal config" in readme
    assert "#### Provider override example" in readme
    assert '"args": ["-m", "mcp_llm_router.server"]' in readme
    assert '"ROUTER_BRAIN_PROVIDER": "deepseek"' in readme
    assert '"EMBEDDINGS_PROVIDER": "ollama"' in readme
