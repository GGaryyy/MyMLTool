"""GPU / device detection and diagnostics for the Chinese-text NLP pipeline.

Resolves a :class:`src.nlp.config.DeviceConfig` policy (device ``auto`` /
``cuda`` / ``cpu`` plus precision) into a concrete runtime plan BEFORE any
model is built, and cross-checks the GPU compute capability against the
kernel architectures compiled into the installed torch wheel
(``torch.cuda.get_arch_list()``).

Motivation: a torch wheel built without ``sm_120`` / ``sm_121`` kernels
imports cleanly and still reports ``torch.cuda.is_available() == True`` on
RTX 50-series cards (Blackwell, compute capability 12.0) and GB10
Grace-Blackwell machines (12.1, aarch64) — then dies mid-training with the
cryptic "no kernel image is available for execution on the device".
Blackwell requires torch>=2.7 CUDA 12.8 wheels, so the mismatch is surfaced
up front instead.

The wheel check follows CUDA compatibility rules rather than an exact
``sm_<major><minor>`` match: an ``sm_XY`` cubin runs on any device with the
SAME major and a minor >= the cubin's (e.g. torch 2.11 cu128 ships sm_86 but
not sm_89 — an RTX 4070 (8.9) still runs it), and a ``compute_XY`` PTX entry
can be JIT-compiled on any numerically-newer device. detect_device records
warnings only (PTX-only matches warn about first-run JIT cost);
:func:`assert_wheel_compatible` is the hard gate the training harness calls
before CUDA training starts.

torch is imported lazily through :func:`_import_torch`, so this module stays
importable on machines without torch installed.
"""

import importlib
import platform
import random
import re
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Optional

import numpy as np

from src.nlp.config import DeviceConfig, VALID_DEVICES, VALID_PRECISIONS

ARCH_ADA = "ada"
ARCH_BLACKWELL = "blackwell"
ARCH_BLACKWELL_GB10 = "blackwell-gb10"
ARCH_UNKNOWN = "unknown"
AARCH64_MACHINES = ("aarch64", "arm64")
MIN_TORCH_HINT = "torch>=2.7 with CUDA 12.8 (see requirements-nlp.txt / docs/nlp/INSTALL.md)"


@dataclass
class DeviceInfo:
    """Snapshot of the resolved device / precision / wheel-compatibility state."""

    requested_device: str                # device string from DeviceConfig
    resolved_device: str                 # "cuda" | "cpu"
    torch_available: bool
    torch_version: Optional[str]
    cuda_available: bool
    cuda_version: Optional[str]          # torch.version.cuda
    device_name: Optional[str]           # torch.cuda.get_device_name(0)
    compute_capability: Optional[tuple]  # (major, minor)
    architecture: Optional[str]          # ada / blackwell / blackwell-gb10 / unknown; None on cpu
    platform_machine: str                # platform.machine()
    bf16_supported: bool
    precision: str                       # resolved: bf16 / fp16 / fp32
    arch_list: list                      # torch.cuda.get_arch_list() or []
    wheel_supports_device: Optional[bool]  # None when resolved to cpu / no cuda
    warnings: list                       # human-readable degradation notes


def _import_torch() -> Optional[ModuleType]:
    """Return the torch module, or ``None`` when torch is unavailable.

    Reads ``sys.modules`` first so tests can inject a fake module or force
    absence by setting the entry to ``None``; otherwise imports via
    :func:`importlib.import_module` and treats ``ImportError`` as missing.
    """
    if "torch" in sys.modules:
        return sys.modules["torch"]
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def _resolve_device(requested: str, cuda_available: bool, warnings: list) -> str:
    """Map the requested device policy onto what the machine can provide."""
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if cuda_available:
            return "cuda"
        warnings.append("device 'cuda' requested but CUDA is not available; falling back to cpu")
        return "cpu"
    return "cuda" if cuda_available else "cpu"  # "auto"


_ARCH_ENTRY_RE = re.compile(r"^(sm|compute)_(\d+)")


def _wheel_supports(major: int, minor: int, arch_list: list) -> tuple:
    """Check wheel arch entries against a device compute capability.

    Returns ``(supported, mode)`` with mode one of ``"binary"`` (an sm_XY
    cubin with the same major and minor <= device minor), ``"ptx"`` (only a
    compute_XY PTX entry numerically <= the device, JIT at first run) or
    ``"none"``. Suffixed entries like ``sm_120a`` parse by their digits.
    """
    has_binary = False
    has_ptx = False
    for entry in arch_list:
        match = _ARCH_ENTRY_RE.match(entry)
        if match is None:
            continue
        num = int(match.group(2))
        entry_major, entry_minor = num // 10, num % 10
        if match.group(1) == "sm":
            if entry_major == major and entry_minor <= minor:
                has_binary = True
        elif (entry_major, entry_minor) <= (major, minor):
            has_ptx = True
    if has_binary:
        return True, "binary"
    if has_ptx:
        return True, "ptx"
    return False, "none"


def _map_architecture(major: int, minor: int, machine: str) -> str:
    """Classify a compute capability into a known target architecture."""
    if (major, minor) == (8, 9):
        return ARCH_ADA
    if major == 10:
        return ARCH_BLACKWELL
    if major == 12:
        if machine.lower() in AARCH64_MACHINES:
            return ARCH_BLACKWELL_GB10
        return ARCH_BLACKWELL
    return ARCH_UNKNOWN


def _resolve_precision(requested: str, resolved_device: str, bf16_supported: bool,
                       warnings: list) -> str:
    """Turn the requested precision policy into a concrete dtype choice."""
    if requested == "auto":
        if resolved_device == "cuda":
            return "bf16" if bf16_supported else "fp16"
        return "fp32"
    if resolved_device == "cpu":
        if requested in ("bf16", "fp16"):
            warnings.append(
                f"precision '{requested}' requested but resolved device is cpu; using fp32"
            )
            return "fp32"
        return requested
    if requested == "bf16" and not bf16_supported:
        warnings.append(
            "precision 'bf16' requested but the GPU does not support bfloat16; using fp16"
        )
        return "fp16"
    return requested


def detect_device(device_config: Optional[DeviceConfig] = None) -> DeviceInfo:
    """Resolve a :class:`DeviceConfig` into a fully populated :class:`DeviceInfo`.

    Never raises when torch or a GPU is missing — every degradation is
    recorded in ``DeviceInfo.warnings`` instead. A wheel/device kernel
    mismatch (e.g. sm_120 GPU on a pre-Blackwell wheel) is likewise only a
    warning here; call :func:`assert_wheel_compatible` for a hard failure.
    Raises ``ValueError`` for invalid ``device`` / ``precision`` strings.
    """
    cfg = device_config if device_config is not None else DeviceConfig()
    if cfg.device not in VALID_DEVICES:
        raise ValueError(f"device must be one of {VALID_DEVICES}, got '{cfg.device}'")
    if cfg.precision not in VALID_PRECISIONS:
        raise ValueError(f"precision must be one of {VALID_PRECISIONS}, got '{cfg.precision}'")

    machine = platform.machine()
    warnings: list = []
    torch = _import_torch()

    if torch is None:
        if cfg.device == "cuda":
            warnings.append(f"device 'cuda' requested but torch not installed; {MIN_TORCH_HINT}")
        else:
            warnings.append(f"torch not installed; {MIN_TORCH_HINT}")
        precision = _resolve_precision(cfg.precision, "cpu", False, warnings)
        return DeviceInfo(
            requested_device=cfg.device,
            resolved_device="cpu",
            torch_available=False,
            torch_version=None,
            cuda_available=False,
            cuda_version=None,
            device_name=None,
            compute_capability=None,
            architecture=None,
            platform_machine=machine,
            bf16_supported=False,
            precision=precision,
            arch_list=[],
            wheel_supports_device=None,
            warnings=warnings,
        )

    torch_version = getattr(torch, "__version__", None)
    cuda_available = bool(torch.cuda.is_available())
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    resolved = _resolve_device(cfg.device, cuda_available, warnings)

    device_name = None
    compute_capability = None
    architecture = None
    bf16_supported = False
    arch_list: list = []
    wheel_supports_device = None

    if resolved == "cuda":
        device_name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        compute_capability = (major, minor)
        architecture = _map_architecture(major, minor, machine)
        if architecture == ARCH_UNKNOWN:
            warnings.append(
                f"compute capability {major}.{minor} ({device_name}) is not a recognised "
                f"target architecture; treating as unknown"
            )
        arch_list = list(torch.cuda.get_arch_list())
        wheel_supports_device, support_mode = _wheel_supports(major, minor, arch_list)
        if not wheel_supports_device:
            warnings.append(
                f"installed torch wheel has no sm_{major}{minor}-compatible kernels "
                f"for {device_name}; reinstall {MIN_TORCH_HINT}"
            )
        elif support_mode == "ptx":
            warnings.append(
                f"wheel has no sm_{major}{minor}-compatible binary kernels for "
                f"{device_name}; relying on PTX JIT (first run will be slower)"
            )
        try:
            bf16_supported = bool(torch.cuda.is_bf16_supported())
        except (RuntimeError, AssertionError):
            bf16_supported = False

    precision = _resolve_precision(cfg.precision, resolved, bf16_supported, warnings)

    return DeviceInfo(
        requested_device=cfg.device,
        resolved_device=resolved,
        torch_available=True,
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        device_name=device_name,
        compute_capability=compute_capability,
        architecture=architecture,
        platform_machine=machine,
        bf16_supported=bf16_supported,
        precision=precision,
        arch_list=arch_list,
        wheel_supports_device=wheel_supports_device,
        warnings=warnings,
    )


def assert_wheel_compatible(info: DeviceInfo) -> None:
    """Hard gate: raise ``RuntimeError`` when the wheel cannot drive the GPU.

    :func:`detect_device` only warns on a kernel mismatch because PTX forward
    compatibility can occasionally save the run; the training harness calls
    this before CUDA work starts, preferring an actionable failure over a
    cryptic mid-run "no kernel image is available" error.
    """
    if info.wheel_supports_device is not False:
        return
    major, minor = info.compute_capability
    raise RuntimeError(
        f"Installed torch wheel has no sm_{major}{minor}-compatible kernels for {info.device_name} "
        f"(compute capability {major}.{minor}). Compiled arch list: "
        f"{', '.join(info.arch_list) or 'empty'}. Training would fail with "
        f"'no kernel image is available for execution on the device'. "
        f"Reinstall {MIN_TORCH_HINT}."
    )


def seed_everything(seed: int) -> None:
    """Seed ``random``, ``numpy`` and (when installed) torch + CUDA.

    numpy only accepts 32-bit seeds, so the seed is reduced mod 2**32 there.
    Raises ``TypeError`` when ``seed`` is not an int.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch = _import_torch()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def format_diagnostics(info: Optional[DeviceInfo] = None) -> str:
    """Render a multi-line, human-readable device report.

    Runs :func:`detect_device` with defaults when ``info`` is not supplied.
    The final line is a PASS/FAIL-style wheel/device compatibility verdict.
    """
    if info is None:
        info = detect_device()

    capability = "-"
    if info.compute_capability is not None:
        capability = f"{info.compute_capability[0]}.{info.compute_capability[1]}"

    rows = [
        ("requested device", info.requested_device),
        ("resolved device", info.resolved_device),
        ("torch available", info.torch_available),
        ("torch version", info.torch_version or "-"),
        ("cuda available", info.cuda_available),
        ("cuda version", info.cuda_version or "-"),
        ("device name", info.device_name or "-"),
        ("compute capability", capability),
        ("architecture", info.architecture or "-"),
        ("platform machine", info.platform_machine),
        ("bf16 supported", info.bf16_supported),
        ("precision", info.precision),
        ("wheel arch list", ", ".join(info.arch_list) or "-"),
    ]
    width = max(len(label) for label, _ in rows)
    lines = ["NLP device diagnostics"]
    lines.extend(f"  {label:<{width}} : {value}" for label, value in rows)
    if info.warnings:
        lines.append("  warnings:")
        lines.extend(f"    {i}. {text}" for i, text in enumerate(info.warnings, 1))
    if info.wheel_supports_device is None:
        verdict = "n/a (cpu)"
    elif info.wheel_supports_device:
        verdict = "OK"
    else:
        verdict = "MISMATCH"
    lines.append(f"  wheel/device compatibility: {verdict}")
    return "\n".join(lines)


def print_diagnostics() -> None:
    """Print :func:`format_diagnostics` for quick CLI / notebook checks."""
    print(format_diagnostics())
