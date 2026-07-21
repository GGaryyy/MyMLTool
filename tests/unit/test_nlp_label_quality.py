"""Unit tests for src.nlp.analysis.label_quality — planted noise recall, guards."""

import json

import pytest

from src.nlp.analysis.label_quality import MAX_SUSPECTS, find_label_issues
from src.nlp.config import TASK_MULTICLASS, TASK_MULTILABEL
from src.nlp.synth import TOPICS, generate_synthetic_gov_docs, texts_and_labels

pytestmark = pytest.mark.unit

FLIP_ROWS = (3, 17, 30, 44, 58, 71, 85, 99)


def _flip_labels(labels, rows):
    """Deterministically reassign each chosen row to a different topic."""
    flipped = list(labels)
    for row in rows:
        others = [t for t in TOPICS if t != flipped[row]]
        flipped[row] = others[row % len(others)]
    return flipped


@pytest.fixture(scope="module")
def mc_corpus():
    df = generate_synthetic_gov_docs(mode="balanced", n_docs=120, seed=5,
                                     inject_defects=False)
    return texts_and_labels(df)


@pytest.fixture(scope="module")
def clean_report(mc_corpus):
    texts, labels = mc_corpus
    return find_label_issues(texts, labels, TASK_MULTICLASS, seed=0)


@pytest.fixture(scope="module")
def flipped_report(mc_corpus):
    texts, labels = mc_corpus
    return find_label_issues(texts, _flip_labels(labels, FLIP_ROWS),
                             TASK_MULTICLASS, seed=0)


# --------------------------------------------------------------------------- #
# planted-noise recall on multiclass
# --------------------------------------------------------------------------- #
def test_flipped_rows_are_recalled(flipped_report):
    suspect_rows = {s.row for s in flipped_report.suspects}
    assert len(suspect_rows & set(FLIP_ROWS)) >= 4


def test_flipped_suspect_ratio_is_sane(flipped_report):
    assert 0.0 < flipped_report.suspect_ratio < 0.3
    assert flipped_report.n_docs == 120


def test_clean_ratio_below_flipped_ratio(clean_report, flipped_report):
    assert clean_report.suspect_ratio < flipped_report.suspect_ratio


def test_suspects_sorted_ascending_and_capped(flipped_report):
    confidences = [s.self_confidence for s in flipped_report.suspects]
    assert confidences == sorted(confidences)
    assert len(flipped_report.suspects) <= MAX_SUSPECTS
    assert flipped_report.n_suspects >= len(flipped_report.suspects)
    for suspect in flipped_report.suspects:
        assert suspect.suggested != suspect.given
        assert suspect.suggested in TOPICS


def test_per_class_thresholds_keys_match_classes(flipped_report):
    assert set(flipped_report.per_class_thresholds) == set(TOPICS)
    for threshold in flipped_report.per_class_thresholds.values():
        assert 0.0 <= threshold <= 1.0


def test_report_to_dict_is_json_serializable(flipped_report):
    json.dumps(flipped_report.to_dict())


# --------------------------------------------------------------------------- #
# guard rails
# --------------------------------------------------------------------------- #
def test_tiny_corpus_returns_zero_suspects_with_note():
    texts = ["人事案文件甲", "人事案文件乙", "人事案文件丙",
             "預算案文件甲", "預算案文件乙", "預算案文件丙"]
    labels = ["人事", "人事", "人事", "預算", "預算", "預算"]
    report = find_label_issues(texts, labels, TASK_MULTICLASS, seed=0)
    assert report.n_suspects == 0
    assert report.suspects == []
    assert report.suspect_ratio == 0.0
    assert report.notes


def test_single_class_returns_zero_suspects_with_note():
    texts = [f"公文內容第{i}號，人員陞遷案。" for i in range(20)]
    labels = ["人事"] * 20
    report = find_label_issues(texts, labels, TASK_MULTICLASS, seed=0)
    assert report.n_suspects == 0
    assert any("2 distinct classes" in note for note in report.notes)


def test_invalid_task_type_raises():
    with pytest.raises(ValueError, match="task_type"):
        find_label_issues(["文件"], ["人事"], "bogus")


# --------------------------------------------------------------------------- #
# multilabel smoke
# --------------------------------------------------------------------------- #
def test_multilabel_smoke():
    df = generate_synthetic_gov_docs(mode="multilabel", n_docs=60, seed=3,
                                     inject_defects=False)
    texts, labels = texts_and_labels(df)
    report = find_label_issues(texts, labels, TASK_MULTILABEL, seed=0)

    assert report.n_docs == 60
    json.dumps(report.to_dict())
    for suspect in report.suspects:
        assert suspect.suggested[0] in "+-"
        assert suspect.suggested[1:] in TOPICS
    confidences = [s.self_confidence for s in report.suspects]
    assert confidences == sorted(confidences)
