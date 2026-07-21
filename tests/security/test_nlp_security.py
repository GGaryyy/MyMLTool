"""Security tests for the NLP pipeline.

Covers: safe YAML loading, no hard-coded secrets, PII masking in generated
reports, safe model deserialization, and that no China-origin / prohibited
packages are imported by the pipeline source.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.security

SRC_NLP = Path(__file__).resolve().parents[2] / "src" / "nlp"
PROHIBITED_IMPORTS = ("jieba", "pkuseg", "cleanlab")
SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|passwd|password|token)\s*=\s*['\"][^'\"]{8,}['\"]"),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"-----BEGIN (RSA |EC )?PRIVATE KEY-----"),
]


def _python_files():
    return list(SRC_NLP.rglob("*.py"))


def test_config_uses_safe_yaml_load():
    config_src = (SRC_NLP / "config.py").read_text(encoding="utf-8")
    assert "yaml.safe_load" in config_src
    assert "yaml.load(" not in config_src  # unsafe full loader must not be used


def test_no_prohibited_or_unsafe_imports():
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        for banned in PROHIBITED_IMPORTS:
            # allow the words in comments/messages, but not as an import statement
            assert not re.search(rf"^\s*import\s+{banned}\b", text, re.MULTILINE), \
                f"{path.name} imports prohibited package {banned}"
            assert not re.search(rf"^\s*from\s+{banned}\b", text, re.MULTILINE), \
                f"{path.name} imports prohibited package {banned}"


def test_no_hardcoded_secrets():
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        for pattern in SECRET_PATTERNS:
            assert not pattern.search(text), f"possible secret in {path.name}"


def test_torch_checkpoint_load_is_weights_only():
    torch_base = (SRC_NLP / "models" / "_torch_base.py").read_text(encoding="utf-8")
    assert "weights_only=True" in torch_base
    assert "weights_only=False" not in torch_base


def test_pii_masked_in_eda_report(tmp_path):
    from src.nlp.config import config_from_dict
    from src.nlp.eda import run_eda
    from src.nlp.report import write_eda_report

    texts = [
        "主旨：當事人身分證字號 A123456789，聯絡電話 0912-345-678，請查照。",
        "主旨：本案函轉相關單位辦理，說明如后。",
        "公告：資通安全維護計畫修訂，自即日生效。",
    ] * 6
    labels = ["人事", "採購", "資訊安全"] * 6
    config = config_from_dict({"data": {"task_type": "multiclass"}, "segment": {"engine": "char"}})
    report = run_eda(texts, labels, config)
    written = write_eda_report(report, str(tmp_path))
    md = Path(written["markdown"]).read_text(encoding="utf-8")
    js = Path(written["json"]).read_text(encoding="utf-8")
    # the raw national id must never appear verbatim in generated artifacts
    assert "A123456789" not in md
    assert "A123456789" not in js
    # but the report must still flag that PII was detected
    assert report.pii.docs_with_pii > 0


def test_pii_scan_masks_snippets():
    from src.nlp.analysis.pii import scan_pii

    report = scan_pii(["聯絡人 A123456789 電話 0912-345-678"])
    for finding in report.findings:
        assert "A123456789" not in finding.masked
        assert "0912-345-678" not in finding.masked
