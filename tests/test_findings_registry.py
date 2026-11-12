"""Tests tethering docs/findings/findings_registry.json to the results files."""

import json

import pytest

from src.analysis.config import FINDINGS_REGISTRY_PATH, PROJECT_ROOT, RESULTS_DIR

REGISTRY = json.loads(FINDINGS_REGISTRY_PATH.read_text())


def resolve(ref: str):
    """Look up '<file>:<dot.path>' inside a results JSON file."""
    fname, path = ref.split(":", 1)
    node = json.loads((RESULTS_DIR / fname).read_text())
    for key in path.split("."):
        node = node[key]
    return node


def _values():
    """Every (claim id, metric ref, registered value) in the registry."""
    for finding in REGISTRY["findings"]:
        for claim in finding["claims"] + finding.get("methods_notes", []):
            for v in claim.get("values", []):
                yield pytest.param(v["metric"], v["value"], id=f"{claim['id']}:{v['metric']}")


@pytest.mark.parametrize("ref,expected", list(_values()))
def test_registered_value_matches_results(ref, expected):
    """A registered value equals the value currently in the results file."""
    actual = resolve(ref)
    if isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=1e-9)
    else:
        assert actual == expected


def test_claim_sources_and_code_exist():
    """Every claim's sources are declared and its code file exists."""
    sources = REGISTRY["sources"]
    for src in sources.values():
        assert (PROJECT_ROOT / src["path"]).exists(), src["path"]
    for finding in REGISTRY["findings"]:
        assert (PROJECT_ROOT / finding["document"]).exists()
        for claim in finding["claims"] + finding.get("methods_notes", []):
            assert set(claim["sources"]) <= set(sources), claim["id"]
            code_file = claim["code"].split(":")[0].split(" ")[0]
            assert (PROJECT_ROOT / code_file).exists(), claim["id"]


def test_claim_ids_unique():
    """Claim ids are unique across findings."""
    ids = [c["id"] for f in REGISTRY["findings"] for c in f["claims"] + f.get("methods_notes", [])]
    assert len(ids) == len(set(ids))
