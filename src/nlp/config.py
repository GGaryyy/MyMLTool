"""Configuration system for the 公文 NLP pipeline.

YAML files are parsed with ``yaml.safe_load`` and validated into typed
dataclasses. Unknown keys are rejected so config typos fail fast. Heavy
libraries (torch / transformers / spacy) are never imported here.
"""

from dataclasses import asdict, dataclass, field, fields
from typing import Optional, Union

TASK_MULTICLASS = "multiclass"
TASK_MULTILABEL = "multilabel"
VALID_TASK_TYPES = (TASK_MULTICLASS, TASK_MULTILABEL)
VALID_SEGMENT_ENGINES = ("spacy", "char", "bert", "ckip")
VALID_DEVICES = ("auto", "cuda", "cpu")
VALID_PRECISIONS = ("auto", "bf16", "fp16", "fp32")
VALID_CLASS_WEIGHTS = ("none", "balanced")
DEFAULT_LABEL_SEPARATOR = "|"
DEFAULT_THRESHOLD = 0.5
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.1
DEFAULT_SEED = 0
DEFAULT_OUTPUT_DIR = "output/nlp"


@dataclass
class DataConfig:
    """Where the labelled CSV lives and how to interpret its columns."""

    csv_path: str = ""
    text_col: Union[int, str] = 0
    label_col: Union[int, str] = -1
    task_type: str = TASK_MULTICLASS
    label_separator: str = DEFAULT_LABEL_SEPARATOR
    test_size: float = DEFAULT_TEST_SIZE
    val_size: float = DEFAULT_VAL_SIZE
    metadata_cols: list = field(default_factory=list)  # structured columns for feature-selection analysis


@dataclass
class SegmentConfig:
    """Which Chinese segmenter backs token-level statistics."""

    engine: str = "spacy"


@dataclass
class DeviceConfig:
    """Device / precision policy for the PyTorch model families."""

    device: str = "auto"
    precision: str = "auto"
    compile: bool = False


@dataclass
class ModelConfig:
    """One benchmark entry: which model to run and its hyper-parameters."""

    name: str = "tfidf_logreg"
    pretrained_path: Optional[str] = None
    max_length: int = 512
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    class_weight: str = "none"
    threshold: float = DEFAULT_THRESHOLD
    params: dict = field(default_factory=dict)


@dataclass
class RunConfig:
    """Top-level bundle consumed by the EDA and benchmark drivers."""

    data: DataConfig = field(default_factory=DataConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    models: list = field(default_factory=list)
    seed: int = DEFAULT_SEED
    output_dir: str = DEFAULT_OUTPUT_DIR

    def to_dict(self) -> dict:
        return asdict(self)


def _build_section(cls, mapping: Optional[dict], section: str):
    """Instantiate a config dataclass from a dict, rejecting unknown keys."""
    if mapping is None:
        return cls()
    if not isinstance(mapping, dict):
        raise ValueError(f"Config section '{section}' must be a mapping, got {type(mapping).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(mapping) - known)
    if unknown:
        raise ValueError(f"Unknown key(s) in config section '{section}': {', '.join(unknown)}")
    return cls(**mapping)


def config_from_dict(raw: dict) -> RunConfig:
    """Build a validated :class:`RunConfig` from a plain dict."""
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")
    known = {"data", "segment", "device", "models", "seed", "output_dir"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown top-level config key(s): {', '.join(unknown)}")

    models_raw = raw.get("models") or []
    if not isinstance(models_raw, list):
        raise ValueError("Config key 'models' must be a list")
    models = [_build_section(ModelConfig, m, f"models[{i}]") for i, m in enumerate(models_raw)]

    cfg = RunConfig(
        data=_build_section(DataConfig, raw.get("data"), "data"),
        segment=_build_section(SegmentConfig, raw.get("segment"), "segment"),
        device=_build_section(DeviceConfig, raw.get("device"), "device"),
        models=models,
        seed=raw.get("seed", DEFAULT_SEED),
        output_dir=raw.get("output_dir", DEFAULT_OUTPUT_DIR),
    )
    validate_config(cfg)
    return cfg


def load_config(path: str) -> RunConfig:
    """Load and validate a YAML config file (parsed with ``yaml.safe_load``)."""
    import yaml  # local import: keeps this module importable without pyyaml

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError:
        raise
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file {path}: {exc}") from exc

    if raw is None:
        raise ValueError(f"Config file is empty: {path}")
    return config_from_dict(raw)


def validate_config(cfg: RunConfig) -> None:
    """Raise ``ValueError`` on any out-of-range or inconsistent setting."""
    data, seg, dev = cfg.data, cfg.segment, cfg.device

    if data.task_type not in VALID_TASK_TYPES:
        raise ValueError(f"task_type must be one of {VALID_TASK_TYPES}, got '{data.task_type}'")
    if not data.label_separator:
        raise ValueError("label_separator must be a non-empty string")
    if not 0.0 < data.test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {data.test_size}")
    if not 0.0 <= data.val_size < 1.0:
        raise ValueError(f"val_size must be in [0, 1), got {data.val_size}")
    if data.test_size + data.val_size >= 1.0:
        raise ValueError("test_size + val_size must leave room for a training split")
    if not isinstance(data.metadata_cols, list):
        raise ValueError(f"metadata_cols must be a list, got {type(data.metadata_cols).__name__}")

    if seg.engine not in VALID_SEGMENT_ENGINES:
        raise ValueError(f"segment.engine must be one of {VALID_SEGMENT_ENGINES}, got '{seg.engine}'")

    if dev.device not in VALID_DEVICES:
        raise ValueError(f"device must be one of {VALID_DEVICES}, got '{dev.device}'")
    if dev.precision not in VALID_PRECISIONS:
        raise ValueError(f"precision must be one of {VALID_PRECISIONS}, got '{dev.precision}'")

    if not isinstance(cfg.seed, int):
        raise ValueError(f"seed must be an integer, got {type(cfg.seed).__name__}")
    if not cfg.output_dir:
        raise ValueError("output_dir must be a non-empty string")

    for i, model in enumerate(cfg.models):
        _validate_model(model, i)


def _validate_model(model: ModelConfig, index: int) -> None:
    prefix = f"models[{index}] ({model.name})"
    if not model.name or not isinstance(model.name, str):
        raise ValueError(f"models[{index}]: name must be a non-empty string")
    if model.max_length < 8:
        raise ValueError(f"{prefix}: max_length must be >= 8, got {model.max_length}")
    if model.batch_size < 1:
        raise ValueError(f"{prefix}: batch_size must be >= 1, got {model.batch_size}")
    if model.epochs < 1:
        raise ValueError(f"{prefix}: epochs must be >= 1, got {model.epochs}")
    if model.learning_rate <= 0:
        raise ValueError(f"{prefix}: learning_rate must be > 0, got {model.learning_rate}")
    if model.class_weight not in VALID_CLASS_WEIGHTS:
        raise ValueError(f"{prefix}: class_weight must be one of {VALID_CLASS_WEIGHTS}, got '{model.class_weight}'")
    if not 0.0 < model.threshold < 1.0:
        raise ValueError(f"{prefix}: threshold must be in (0, 1), got {model.threshold}")
    if not isinstance(model.params, dict):
        raise ValueError(f"{prefix}: params must be a mapping")
