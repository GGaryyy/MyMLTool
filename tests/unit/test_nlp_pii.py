"""Unit tests for src.nlp.analysis.pii — PII scan, masking, normalization."""

import pytest

from src.nlp.analysis.pii import (
    ADDRESS_KIND,
    MAX_FINDINGS,
    OCR_SUSPECT_CHARS,
    PII_PATTERNS,
    NormalizationReport,
    PiiReport,
    check_normalization,
    mask_snippet,
    scan_pii,
)

pytestmark = pytest.mark.unit

ALL_KINDS = {"national_id", "mobile", "landline", "email", ADDRESS_KIND}


# --------------------------------------------------------------------------- #
# pattern hits on crafted 繁中 strings
# --------------------------------------------------------------------------- #
def test_pattern_keys_are_the_documented_four():
    assert set(PII_PATTERNS) == {"national_id", "mobile", "landline", "email"}


def test_national_id_detected_and_masked():
    report = scan_pii(["承辦人身分證字號A123456789，請查照。"])
    assert report.counts["national_id"] == 1
    assert report.docs_with_pii == 1
    finding = [f for f in report.findings if f.kind == "national_id"][0]
    assert finding.row == 0
    assert finding.masked == "A********9"
    assert "123456789" not in finding.masked


def test_national_id_requires_gender_digit_1_or_2():
    report = scan_pii(["案件編號A323456789不是身分證。"])
    assert report.counts["national_id"] == 0


def test_mobile_detected_with_and_without_dashes():
    report = scan_pii(["行動電話0912-345-678。", "手機0987654321。"])
    assert report.counts["mobile"] == 2
    assert report.counts["landline"] == 0


def test_landline_detected():
    report = scan_pii(["聯絡電話02-2345-6789。"])
    assert report.counts["landline"] == 1
    assert report.counts["mobile"] == 0


def test_landline_does_not_double_count_mobile_in_same_text():
    report = scan_pii(["電話02-2345-6789，行動電話0912-345-678。"])
    assert report.counts["landline"] == 1
    assert report.counts["mobile"] == 1
    assert len(report.findings) == 2


def test_email_detected():
    report = scan_pii(["承辦人信箱test.user@example.gov.tw，請逕洽。"])
    assert report.counts["email"] == 1


def test_address_detected():
    report = scan_pii(["會議地點：臺中市西屯區文心路100號3樓會議室。"])
    assert report.counts[ADDRESS_KIND] == 1


def test_clean_text_has_no_hits():
    report = scan_pii(["旨揭案件業經本機關核定，請依說明段配合辦理。"])
    assert report.docs_with_pii == 0
    assert report.findings == []
    assert set(report.counts) == ALL_KINDS
    assert all(count == 0 for count in report.counts.values())


# --------------------------------------------------------------------------- #
# report aggregation
# --------------------------------------------------------------------------- #
def test_docs_with_pii_counts_documents_not_matches():
    report = scan_pii([
        "身分證A123456789，電話02-2345-6789。",  # two hits, one doc
        "信箱someone@example.com。",
        "無任何個資的一般文件內容。",
    ])
    assert report.total_docs == 3
    assert report.docs_with_pii == 2
    assert report.counts["national_id"] == 1
    assert report.counts["landline"] == 1
    assert report.counts["email"] == 1


def test_findings_capped_but_counts_are_not():
    texts = [f"信箱user{i}@example.com" for i in range(MAX_FINDINGS + 50)]
    report = scan_pii(texts)
    assert len(report.findings) == MAX_FINDINGS
    assert report.counts["email"] == MAX_FINDINGS + 50
    assert report.docs_with_pii == MAX_FINDINGS + 50


def test_to_dict_is_nested_plain_data():
    data = scan_pii(["身分證A123456789。"]).to_dict()
    assert data["total_docs"] == 1
    assert isinstance(data["findings"][0], dict)
    assert data["findings"][0]["kind"] == "national_id"


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #
def test_scan_pii_rejects_non_str_elements():
    with pytest.raises(TypeError, match="texts\\[1\\]"):
        scan_pii(["正常文字", 123])


def test_scan_pii_rejects_bare_string():
    with pytest.raises(TypeError, match="single str"):
        scan_pii("A123456789")


def test_scan_pii_rejects_non_iterable():
    with pytest.raises(TypeError):
        scan_pii(42)


def test_check_normalization_rejects_non_str_elements():
    with pytest.raises(TypeError):
        check_normalization([None])


# --------------------------------------------------------------------------- #
# empty input -> zeroed reports
# --------------------------------------------------------------------------- #
def test_scan_pii_empty_input_zeroed():
    report = scan_pii([])
    assert isinstance(report, PiiReport)
    assert report.total_docs == 0
    assert report.docs_with_pii == 0
    assert report.findings == []
    assert all(count == 0 for count in report.counts.values())


def test_check_normalization_empty_input_zeroed():
    report = check_normalization([])
    assert isinstance(report, NormalizationReport)
    assert report.docs_with_fullwidth == 0
    assert report.fullwidth_char_count == 0
    assert report.docs_with_ocr_suspects == 0
    assert report.mixed_width_digit_docs == 0
    assert set(report.ocr_suspect_counts) == set(OCR_SUSPECT_CHARS)
    assert all(count == 0 for count in report.ocr_suspect_counts.values())


# --------------------------------------------------------------------------- #
# mask_snippet
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw, expected", [
    ("A123456789", "A********9"),
    ("0912-345-678", "0**********8"),
    ("abc", "a*c"),
    ("ab", "**"),
    ("a", "*"),
    ("", ""),
])
def test_mask_snippet(raw, expected):
    assert mask_snippet(raw) == expected


def test_mask_snippet_rejects_non_str():
    with pytest.raises(TypeError):
        mask_snippet(12345)


# --------------------------------------------------------------------------- #
# normalization checks
# --------------------------------------------------------------------------- #
def test_fullwidth_detected_and_counted():
    report = check_normalization(["案件編號１２３ＡＢ號。", "一般半形文字123。"])
    assert report.docs_with_fullwidth == 1
    assert report.fullwidth_char_count == 5


def test_ocr_suspects_counted_per_char():
    report = check_normalization(["文件編號〇〇一ｌ號○。"])
    assert report.docs_with_ocr_suspects == 1
    assert report.ocr_suspect_counts == {"〇": 2, "ｌ": 1, "Ｏ": 0, "○": 1}


def test_mixed_width_digit_docs():
    report = check_normalization([
        "半形123與全形１２３並存。",  # mixed
        "只有半形123。",
        "只有全形１２３。",
    ])
    assert report.mixed_width_digit_docs == 1
    assert report.docs_with_fullwidth == 2


def test_normalization_to_dict():
    data = check_normalization(["編號１２３"]).to_dict()
    assert data["docs_with_fullwidth"] == 1
    assert data["fullwidth_char_count"] == 3
