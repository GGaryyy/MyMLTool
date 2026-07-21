"""Unit tests for src.nlp.config — dict/YAML parsing, validation, defaults."""

import pytest

from src.nlp.config import (
    DEFAULT_LABEL_SEPARATOR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    DEFAULT_THRESHOLD,
    DEFAULT_VAL_SIZE,
    TASK_MULTICLASS,
    TASK_MULTILABEL,
    DataConfig,
    DeviceConfig,
    ModelConfig,
    RunConfig,
    SegmentConfig,
    config_from_dict,
    load_config,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# config_from_dict — happy paths
# --------------------------------------------------------------------------- #
def test_config_from_dict_happy_path_with_nested_models():
    raw = {
        "data": {
            "csv_path": "data/gov_docs.csv",
            "text_col": "text",
            "label_col": "label",
            "task_type": "multilabel",
            "label_separator": "|",
            "test_size": 0.2,
            "val_size": 0.1,
        },
        "segment": {"engine": "char"},
        "device": {"device": "cpu", "precision": "fp32", "compile": False},
        "models": [
            {"name": "tfidf_logreg"},
            {
                "name": "bert_finetune",
                "pretrained_path": "models/bert-base-chinese",
                "max_length": 256,
                "batch_size": 8,
                "epochs": 2,
                "learning_rate": 3e-5,
                "class_weight": "balanced",
                "threshold": 0.4,
                "params": {"C": 1.0},
            },
        ],
        "seed": 42,
        "output_dir": "output/nlp_run",
    }
    cfg = config_from_dict(raw)
    assert isinstance(cfg, RunConfig)
    assert cfg.data.csv_path == "data/gov_docs.csv"
    assert cfg.data.task_type == TASK_MULTILABEL
    assert cfg.segment.engine == "char"
    assert cfg.device.device == "cpu"
    assert len(cfg.models) == 2
    assert cfg.models[0].name == "tfidf_logreg"
    assert cfg.models[1].pretrained_path == "models/bert-base-chinese"
    assert cfg.models[1].batch_size == 8
    assert cfg.models[1].class_weight == "balanced"
    assert cfg.models[1].threshold == 0.4
    assert cfg.models[1].params == {"C": 1.0}
    assert cfg.seed == 42
    assert cfg.output_dir == "output/nlp_run"


def test_config_from_dict_empty_dict_gives_valid_defaults():
    cfg = config_from_dict({})
    assert cfg.data.task_type == TASK_MULTICLASS
    assert cfg.segment.engine == "spacy"
    assert cfg.models == []
    assert cfg.seed == DEFAULT_SEED
    assert cfg.output_dir == DEFAULT_OUTPUT_DIR


def test_config_from_dict_root_not_mapping_raises():
    with pytest.raises(ValueError, match="root must be a mapping"):
        config_from_dict(["not", "a", "dict"])


# --------------------------------------------------------------------------- #
# load_config — YAML files
# --------------------------------------------------------------------------- #
def test_load_config_from_yaml_file(tmp_path):
    content = (
        "data:\n"
        "  csv_path: data/gov_docs.csv\n"
        "  task_type: multiclass\n"
        "  test_size: 0.25\n"
        "segment:\n"
        "  engine: char\n"
        "device:\n"
        "  device: cpu\n"
        "models:\n"
        "  - name: tfidf_logreg\n"
        "  - name: bert_finetune\n"
        "    epochs: 2\n"
        "    batch_size: 8\n"
        "seed: 7\n"
        "output_dir: output/nlp_test\n"
    )
    path = tmp_path / "run.yaml"
    path.write_text(content, encoding="utf-8")

    cfg = load_config(str(path))
    assert cfg.data.csv_path == "data/gov_docs.csv"
    assert cfg.data.test_size == 0.25
    assert cfg.segment.engine == "char"
    assert cfg.device.device == "cpu"
    assert [m.name for m in cfg.models] == ["tfidf_logreg", "bert_finetune"]
    assert cfg.models[1].epochs == 2
    assert cfg.models[1].batch_size == 8
    assert cfg.seed == 7
    assert cfg.output_dir == "output/nlp_test"


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "does_not_exist.yaml"))


def test_load_config_empty_file_raises(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_config(str(path))


def test_load_config_invalid_yaml_syntax_raises(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("data: [unclosed, flow\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config(str(path))


# --------------------------------------------------------------------------- #
# Unknown-key rejection
# --------------------------------------------------------------------------- #
def test_unknown_top_level_key_raises():
    with pytest.raises(ValueError, match="Unknown top-level"):
        config_from_dict({"datas": {"csv_path": "x.csv"}})


def test_unknown_section_key_raises():
    with pytest.raises(ValueError, match="config section 'data'"):
        config_from_dict({"data": {"csv": "x.csv"}})


def test_unknown_model_key_raises():
    with pytest.raises(ValueError, match=r"models\[0\]"):
        config_from_dict({"models": [{"name": "m", "lr": 0.1}]})


def test_models_not_a_list_raises():
    with pytest.raises(ValueError, match="must be a list"):
        config_from_dict({"models": {"name": "m"}})


def test_section_not_mapping_raises():
    with pytest.raises(ValueError, match="must be a mapping"):
        config_from_dict({"data": 5})


# --------------------------------------------------------------------------- #
# Validation — data section
# --------------------------------------------------------------------------- #
def test_bad_task_type_raises():
    with pytest.raises(ValueError, match="task_type must be one of"):
        config_from_dict({"data": {"task_type": "binary"}})


def test_test_plus_val_size_too_large_raises():
    with pytest.raises(ValueError, match="leave room"):
        config_from_dict({"data": {"test_size": 0.5, "val_size": 0.6}})


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("test_size", 0.0, "test_size"),
        ("test_size", 1.0, "test_size"),
        ("val_size", -0.1, "val_size"),
        ("label_separator", "", "label_separator"),
    ],
)
def test_invalid_data_field_raises(field, value, match):
    with pytest.raises(ValueError, match=match):
        config_from_dict({"data": {field: value}})


# --------------------------------------------------------------------------- #
# Validation — segment / device / top-level
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("engine", ["jieba", "pkuseg", "bogus"])
def test_invalid_segment_engine_raises(engine):
    with pytest.raises(ValueError, match="segment.engine"):
        config_from_dict({"segment": {"engine": engine}})


def test_invalid_device_raises():
    with pytest.raises(ValueError, match="device must be one of"):
        config_from_dict({"device": {"device": "tpu"}})


def test_invalid_precision_raises():
    with pytest.raises(ValueError, match="precision must be one of"):
        config_from_dict({"device": {"precision": "int8"}})


def test_non_integer_seed_raises():
    with pytest.raises(ValueError, match="seed must be an integer"):
        config_from_dict({"seed": "zero"})


def test_empty_output_dir_raises():
    with pytest.raises(ValueError, match="output_dir"):
        config_from_dict({"output_dir": ""})


# --------------------------------------------------------------------------- #
# Validation — model entries
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "field, value, match",
    [
        ("threshold", 0.0, "threshold"),
        ("threshold", 1.0, "threshold"),
        ("epochs", 0, "epochs"),
        ("batch_size", 0, "batch_size"),
        ("learning_rate", 0.0, "learning_rate"),
        ("learning_rate", -1e-5, "learning_rate"),
        ("class_weight", "auto", "class_weight"),
        ("max_length", 4, "max_length"),
    ],
)
def test_invalid_model_field_raises(field, value, match):
    with pytest.raises(ValueError, match=match):
        config_from_dict({"models": [{"name": "m", field: value}]})


def test_model_empty_name_raises():
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        config_from_dict({"models": [{"name": ""}]})


def test_model_params_not_mapping_raises():
    with pytest.raises(ValueError, match="params must be a mapping"):
        config_from_dict({"models": [{"name": "m", "params": [1, 2]}]})


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
def test_run_config_defaults():
    cfg = RunConfig()
    assert cfg.data == DataConfig()
    assert cfg.segment == SegmentConfig()
    assert cfg.device == DeviceConfig()
    assert cfg.models == []
    assert cfg.seed == DEFAULT_SEED
    assert cfg.output_dir == DEFAULT_OUTPUT_DIR

    assert cfg.data.csv_path == ""
    assert cfg.data.text_col == 0
    assert cfg.data.label_col == -1
    assert cfg.data.task_type == TASK_MULTICLASS
    assert cfg.data.label_separator == DEFAULT_LABEL_SEPARATOR
    assert cfg.data.test_size == DEFAULT_TEST_SIZE
    assert cfg.data.val_size == DEFAULT_VAL_SIZE
    assert cfg.segment.engine == "spacy"
    assert cfg.device.device == "auto"
    assert cfg.device.precision == "auto"
    assert cfg.device.compile is False


def test_model_config_defaults():
    model = ModelConfig()
    assert model.name == "tfidf_logreg"
    assert model.pretrained_path is None
    assert model.max_length == 512
    assert model.batch_size == 16
    assert model.epochs == 3
    assert model.learning_rate == pytest.approx(2e-5)
    assert model.class_weight == "none"
    assert model.threshold == DEFAULT_THRESHOLD
    assert model.params == {}


def test_run_config_to_dict_is_plain_dict():
    d = RunConfig().to_dict()
    assert isinstance(d, dict)
    assert set(d) == {"data", "segment", "device", "models", "seed", "output_dir"}
    assert isinstance(d["data"], dict)
