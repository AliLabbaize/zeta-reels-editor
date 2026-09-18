"""The vendored render path is the reason the output is correct. Freeze it.

If either file drifts, hard rules 1-5 are no longer guaranteed by code that was
proven upstream. A deliberate re-sync updates docs/VENDOR.md and this test in
the same commit; anything else is an accident and this test is how we find out.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDORED = {
    "render.py": "5f908e643b4c96d186e6eed8d5e384ca200be86f",
    "timeline_view.py": "dea86d6e20722d0656e54a8d986951f3b2917dc7",
}


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def test_vendored_files_are_unmodified():
    for name, expected in VENDORED.items():
        path = REPO_ROOT / "helpers" / name
        assert path.exists(), f"{name} is missing from helpers/"
        assert _sha1(path) == expected, (
            f"helpers/{name} was modified. Vendored upstream code is read-only; "
            f"put the change in a wrapper instead. If this was a deliberate "
            f"re-sync, update docs/VENDOR.md and this test together.")


def test_vendor_doc_matches_the_pinned_hashes():
    doc = (REPO_ROOT / "docs" / "VENDOR.md").read_text(encoding="utf-8")
    for name, expected in VENDORED.items():
        assert expected in doc, f"docs/VENDOR.md does not record the {name} sha1"
    assert re.search(r"[0-9a-f]{40}", doc), "VENDOR.md records no upstream commit"
