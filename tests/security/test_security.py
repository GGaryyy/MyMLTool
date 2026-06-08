"""Security tests for the MyMLTool data-prep and feature-shift modules.

Covers:
1. SAST / secret-leak scan of src/*.py source text (no hardcoded secrets,
   no eval/exec/os.system/insecure pickle.load).
2. Robustness against malformed and empty CSV input (controlled, specific
   exceptions rather than bare crashes).
3. CSV content is never executed as code (formula/script-like cells stay
   plain values).
4. FeatureShiftDetector.detect rejects mismatched/missing columns instead
   of silently passing.

All tests are offline and deterministic.
"""

import re
from pathlib import Path

import pandas as pd
import pytest

from src.data_prep import load_dataset, prepare_data, split_features_target
from src.feature_shift import FeatureShiftDetector

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
SRC_FILES = sorted(p for p in SRC_DIR.glob("*.py") if p.name != "__init__.py")

# Dangerous code-execution / deserialization patterns that must not appear.
DANGEROUS_PATTERNS = {
    "eval(": re.compile(r"\beval\s*\("),
    "exec(": re.compile(r"\bexec\s*\("),
    "os.system(": re.compile(r"\bos\.system\s*\("),
    "subprocess shell=True": re.compile(r"shell\s*=\s*True"),
    "pickle.load": re.compile(r"\bpickle\.loads?\s*\("),
    "yaml.load (unsafe)": re.compile(r"\byaml\.load\s*\((?!.*Loader)"),
    "__import__(": re.compile(r"\b__import__\s*\("),
}

# Hardcoded-secret heuristics: an assignment of a secret-named var to a
# non-trivial string literal.
SECRET_PATTERNS = {
    "api_key": re.compile(
        r"(?i)\b(api[_-]?key|apikey|secret|password|passwd|pwd|token|"
        r"access[_-]?key|private[_-]?key|aws[_-]?secret)\b\s*[:=]\s*"
        r"['\"][^'\"]{6,}['\"]"
    ),
    "aws_access_key_id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key_block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "bearer_token": re.compile(r"(?i)bearer\s+[A-Za-z0-9\-_\.=]{20,}"),
}


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# 1. SAST: dangerous patterns absent from source
# --------------------------------------------------------------------------- #
@pytest.mark.security
def test_src_files_discovered():
    """Sanity check so the scan tests are not silently scanning nothing."""
    names = {p.name for p in SRC_FILES}
    assert {"data_prep.py", "feature_shift.py"} <= names


@pytest.mark.security
@pytest.mark.parametrize("src_file", SRC_FILES, ids=lambda p: p.name)
def test_no_dangerous_code_execution_patterns(src_file):
    """No eval/exec/os.system/insecure pickle/yaml/__import__ in source."""
    source = _read_source(src_file)
    found = [name for name, pat in DANGEROUS_PATTERNS.items() if pat.search(source)]
    assert not found, f"{src_file.name} contains dangerous pattern(s): {found}"


@pytest.mark.security
@pytest.mark.parametrize("src_file", SRC_FILES, ids=lambda p: p.name)
def test_no_hardcoded_secrets(src_file):
    """No hardcoded API keys, passwords, tokens or private keys in source."""
    source = _read_source(src_file)
    leaks = [name for name, pat in SECRET_PATTERNS.items() if pat.search(source)]
    assert not leaks, f"{src_file.name} appears to contain secret(s): {leaks}"


# --------------------------------------------------------------------------- #
# 2. Robustness: malformed / empty CSV raise controlled, specific exceptions
# --------------------------------------------------------------------------- #
@pytest.mark.security
def test_empty_csv_raises_value_error(tmp_path):
    """A completely empty file maps to a controlled ValueError, not a crash."""
    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")
    with pytest.raises(ValueError):
        load_dataset(str(empty))


@pytest.mark.security
def test_header_only_csv_raises_value_error(tmp_path):
    """A header with no data rows is rejected with ValueError."""
    header_only = tmp_path / "header_only.csv"
    header_only.write_text("age,salary,region,purchased\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(str(header_only))


@pytest.mark.security
def test_random_bytes_csv_raises_specific_error(tmp_path):
    """Random binary garbage must raise a specific (pandas/Value) error.

    The contract is that it is NOT swallowed and NOT a bare Exception leak:
    we accept the recognised pandas parser/encoding errors or ValueError.
    """
    garbage = tmp_path / "garbage.csv"
    garbage.write_bytes(bytes(range(256)) * 64)
    expected = (
        ValueError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
    )
    with pytest.raises(expected):
        load_dataset(str(garbage))


@pytest.mark.security
def test_missing_file_raises_file_not_found(tmp_path):
    """A non-existent path raises FileNotFoundError, not a generic error."""
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_dataset(str(missing))


@pytest.mark.security
def test_single_column_csv_rejected(tmp_path):
    """A frame with no feature/target split raises a controlled ValueError."""
    one_col = tmp_path / "one_col.csv"
    one_col.write_text("only\n1\n2\n3\n", encoding="utf-8")
    df = load_dataset(str(one_col))
    with pytest.raises(ValueError):
        split_features_target(df)


# --------------------------------------------------------------------------- #
# 3. CSV content is never executed as code (no formula/injection evaluation)
# --------------------------------------------------------------------------- #
@pytest.mark.security
def test_csv_formula_cell_read_as_plain_value(tmp_path):
    """A formula/script-like cell must be loaded as a literal string value."""
    payload = '=SUM(1+1)'
    csv_path = tmp_path / "formula.csv"
    csv_path.write_text(
        "age,note,purchased\n"
        f'30,"{payload}",1\n'
        '40,"__import__(\'os\').system(\'echo pwned\')",0\n',
        encoding="utf-8",
    )

    df = load_dataset(str(csv_path))
    notes = list(df["note"])
    assert payload in notes
    assert any("__import__" in str(v) for v in notes)
    # No evaluation happened: the formula stayed a 6-char string, not 2.
    assert df.loc[df["note"] == payload, "note"].iloc[0] == payload


@pytest.mark.security
def test_prepare_data_does_not_execute_csv_content(tmp_path):
    """prepare_data treats script-like cells as data, returning them verbatim."""
    injection = "__import__('os').system('echo pwned')"
    csv_path = tmp_path / "inject.csv"
    csv_path.write_text(
        "age,note,purchased\n"
        f'30,"{injection}",1\n'
        '50,"benign",0\n',
        encoding="utf-8",
    )

    prepared = prepare_data(str(csv_path), test_size=0.5, random_state=0)
    # 'note' is a feature column; the malicious string survives as a plain value.
    note_idx = prepared.feature_names.index("note")
    all_rows = list(prepared.X_train[:, note_idx]) + list(prepared.X_test[:, note_idx])
    assert injection in [str(v) for v in all_rows]


# --------------------------------------------------------------------------- #
# 4. FeatureShiftDetector rejects mismatched / missing columns
# --------------------------------------------------------------------------- #
@pytest.mark.security
def test_detect_missing_column_raises_value_error(feature_reference_df, feature_nodrift_df):
    """Dropping a reference column must raise ValueError (no silent pass)."""
    detector = FeatureShiftDetector(feature_reference_df).fit()
    incoming = feature_nodrift_df.drop(columns=["salary"])
    with pytest.raises(ValueError):
        detector.detect(incoming)


@pytest.mark.security
def test_detect_renamed_column_raises_value_error(feature_reference_df, feature_nodrift_df):
    """A renamed (effectively missing) column is rejected with ValueError."""
    detector = FeatureShiftDetector(feature_reference_df).fit()
    incoming = feature_nodrift_df.rename(columns={"region": "zone"})
    with pytest.raises(ValueError):
        detector.detect(incoming)


@pytest.mark.security
def test_detect_empty_incoming_raises_value_error(feature_reference_df):
    """An empty incoming batch is rejected with ValueError, not silently passed."""
    detector = FeatureShiftDetector(feature_reference_df).fit()
    empty = pd.DataFrame(columns=feature_reference_df.columns)
    with pytest.raises(ValueError):
        detector.detect(empty)


@pytest.mark.security
def test_detect_non_dataframe_raises_type_error(feature_reference_df):
    """Non-DataFrame incoming input is rejected with TypeError."""
    detector = FeatureShiftDetector(feature_reference_df).fit()
    with pytest.raises(TypeError):
        detector.detect([1, 2, 3])


@pytest.mark.security
def test_unknown_categorical_feature_rejected(feature_reference_df):
    """Constructing with an unknown categorical column raises ValueError."""
    with pytest.raises(ValueError):
        FeatureShiftDetector(feature_reference_df, categorical_features=["nonexistent"])
