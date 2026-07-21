"""Unit tests for src.nlp.device — torch is always faked via sys.modules.

Every test monkeypatches ``sys.modules["torch"]`` (a fake namespace for the
installed case, ``None`` for the not-installed case) so the outcomes never
depend on whether a real torch wheel exists in the venv.
"""

import platform
import random
import sys
import types

import numpy as np
import pytest

from src.nlp.config import DeviceConfig
from src.nlp import device as device_mod
from src.nlp.device import (
    ARCH_ADA,
    ARCH_BLACKWELL,
    ARCH_BLACKWELL_GB10,
    ARCH_UNKNOWN,
    MIN_TORCH_HINT,
    assert_wheel_compatible,
    detect_device,
    format_diagnostics,
    print_diagnostics,
    seed_everything,
)

pytestmark = pytest.mark.unit

RTX_4070 = "NVIDIA GeForce RTX 4070"
RTX_5070_TI = "NVIDIA GeForce RTX 5070 Ti"
ADA_ARCH_LIST = ("sm_80", "sm_86", "sm_89", "sm_90")


def make_fake_torch(cuda_available=True, cc=(8, 9), name=RTX_4070,
                    arch_list=ADA_ARCH_LIST, bf16=True, cuda_version="12.8",
                    version="2.7.1", bf16_raises=False):
    """Build a fake torch module; records manual_seed calls in fake._calls."""
    calls = {"manual_seed": [], "cuda_manual_seed_all": []}

    def is_bf16_supported():
        if bf16_raises:
            raise RuntimeError("bf16 probe unsupported on this build")
        return bf16

    fake = types.SimpleNamespace(
        __version__=version,
        version=types.SimpleNamespace(cuda=cuda_version),
        manual_seed=lambda s: calls["manual_seed"].append(s),
        cuda=types.SimpleNamespace(
            is_available=lambda: cuda_available,
            get_device_name=lambda idx=0: name,
            get_device_capability=lambda idx=0: cc,
            get_arch_list=lambda: list(arch_list),
            is_bf16_supported=is_bf16_supported,
            manual_seed_all=lambda s: calls["cuda_manual_seed_all"].append(s),
        ),
    )
    fake._calls = calls
    return fake


# --------------------------------------------------------------------------- #
# _import_torch
# --------------------------------------------------------------------------- #
def test_import_torch_returns_module_from_sys_modules(monkeypatch):
    fake = make_fake_torch()
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert device_mod._import_torch() is fake


def test_import_torch_treats_none_entry_as_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert device_mod._import_torch() is None


# --------------------------------------------------------------------------- #
# detect_device — torch not installed
# --------------------------------------------------------------------------- #
def test_no_torch_defaults_to_cpu_fp32(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    info = detect_device()
    assert info.torch_available is False
    assert info.resolved_device == "cpu"
    assert info.precision == "fp32"
    assert info.cuda_available is False
    assert info.architecture is None
    assert info.arch_list == []
    assert info.wheel_supports_device is None
    assert any("torch not installed" in w for w in info.warnings)


def test_no_torch_explicit_cuda_still_cpu_with_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    info = detect_device(DeviceConfig(device="cuda"))
    assert info.requested_device == "cuda"
    assert info.resolved_device == "cpu"
    assert any("torch not installed" in w and MIN_TORCH_HINT in w for w in info.warnings)


# --------------------------------------------------------------------------- #
# detect_device — device resolution with torch present
# --------------------------------------------------------------------------- #
def test_cpu_only_torch_auto_resolves_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", make_fake_torch(cuda_available=False))
    info = detect_device(DeviceConfig(device="auto"))
    assert info.torch_available is True
    assert info.cuda_available is False
    assert info.resolved_device == "cpu"
    assert info.precision == "fp32"
    assert info.warnings == []


def test_cpu_only_torch_cuda_request_warns(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", make_fake_torch(cuda_available=False))
    info = detect_device(DeviceConfig(device="cuda"))
    assert info.resolved_device == "cpu"
    assert any("cuda" in w.lower() and "not available" in w.lower() for w in info.warnings)


def test_explicit_cpu_request_skips_cuda_probing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", make_fake_torch(cuda_available=True))
    info = detect_device(DeviceConfig(device="cpu"))
    assert info.resolved_device == "cpu"
    assert info.cuda_available is True
    assert info.device_name is None
    assert info.compute_capability is None
    assert info.architecture is None
    assert info.arch_list == []
    assert info.wheel_supports_device is None
    assert info.precision == "fp32"


# --------------------------------------------------------------------------- #
# detect_device — architecture mapping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cc,machine,expected",
    [
        ((8, 9), "x86_64", ARCH_ADA),
        ((10, 0), "x86_64", ARCH_BLACKWELL),
        ((12, 0), "AMD64", ARCH_BLACKWELL),
        ((12, 0), "aarch64", ARCH_BLACKWELL_GB10),
        ((12, 1), "aarch64", ARCH_BLACKWELL_GB10),
        ((12, 1), "arm64", ARCH_BLACKWELL_GB10),
        ((7, 5), "x86_64", ARCH_UNKNOWN),
    ],
)
def test_architecture_mapping(monkeypatch, cc, machine, expected):
    fake = make_fake_torch(cc=cc, arch_list=(f"sm_{cc[0]}{cc[1]}",))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    info = detect_device()
    assert info.architecture == expected
    assert info.platform_machine == machine


def test_unknown_arch_appends_warning(monkeypatch):
    fake = make_fake_torch(cc=(7, 5), name="Tesla T4", arch_list=("sm_75",))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    info = detect_device()
    assert info.architecture == ARCH_UNKNOWN
    assert any("unknown" in w.lower() for w in info.warnings)


# --------------------------------------------------------------------------- #
# detect_device — Ada / Blackwell / GB10 end-to-end
# --------------------------------------------------------------------------- #
def test_ada_4070_happy_path(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", make_fake_torch())
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    info = detect_device(DeviceConfig(device="auto"))
    assert info.resolved_device == "cuda"
    assert info.device_name == RTX_4070
    assert info.compute_capability == (8, 9)
    assert info.architecture == ARCH_ADA
    assert info.torch_version == "2.7.1"
    assert info.cuda_version == "12.8"
    assert info.bf16_supported is True
    assert info.precision == "bf16"
    assert info.wheel_supports_device is True
    assert info.warnings == []
    assert_wheel_compatible(info)  # must not raise


def test_blackwell_5070ti_good_wheel(monkeypatch):
    fake = make_fake_torch(cc=(12, 0), name=RTX_5070_TI,
                           arch_list=("sm_80", "sm_90", "sm_100", "sm_120"))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    info = detect_device()
    assert info.architecture == ARCH_BLACKWELL
    assert info.wheel_supports_device is True
    assert info.warnings == []
    assert_wheel_compatible(info)  # must not raise


def test_blackwell_old_wheel_flags_mismatch(monkeypatch):
    fake = make_fake_torch(cc=(12, 0), name=RTX_5070_TI, arch_list=ADA_ARCH_LIST)
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    info = detect_device()
    assert info.wheel_supports_device is False
    assert any("sm_120" in w and RTX_5070_TI in w for w in info.warnings)
    assert any(MIN_TORCH_HINT in w for w in info.warnings)


def test_assert_wheel_compatible_raises_on_mismatch(monkeypatch):
    fake = make_fake_torch(cc=(12, 0), name=RTX_5070_TI, arch_list=ADA_ARCH_LIST)
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    info = detect_device()
    with pytest.raises(RuntimeError) as excinfo:
        assert_wheel_compatible(info)
    message = str(excinfo.value)
    assert "sm_120" in message
    assert RTX_5070_TI in message
    assert "einstall" in message  # Reinstall / reinstall hint
    assert MIN_TORCH_HINT in message


def test_assert_wheel_compatible_noop_for_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    info = detect_device()
    assert info.wheel_supports_device is None
    assert assert_wheel_compatible(info) is None  # must not raise


def test_ada_supported_via_same_major_lower_minor(monkeypatch):
    """Real-world case: torch 2.11 cu128 ships sm_86 but NOT sm_89 — an RTX
    4070 (8.9) runs sm_86 cubins, so the wheel must count as compatible."""
    fake = make_fake_torch(cc=(8, 9),
                           arch_list=("sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    info = detect_device()
    assert info.wheel_supports_device is True
    assert info.warnings == []
    assert_wheel_compatible(info)  # must not raise


def test_gb10_supported_via_sm120_binary(monkeypatch):
    """GB10 is cc 12.1; sm_120 cubins (same major, lower minor) must count."""
    fake = make_fake_torch(cc=(12, 1), name="NVIDIA GB10",
                           arch_list=("sm_90", "sm_100", "sm_120"))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    info = detect_device()
    assert info.wheel_supports_device is True
    assert info.warnings == []


def test_ptx_only_match_supported_with_jit_warning(monkeypatch):
    fake = make_fake_torch(cc=(12, 0), name=RTX_5070_TI, arch_list=("compute_120",))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    info = detect_device()
    assert info.wheel_supports_device is True
    assert any("PTX JIT" in w for w in info.warnings)
    assert_wheel_compatible(info)  # must not raise


def test_suffixed_arch_entry_parses(monkeypatch):
    fake = make_fake_torch(cc=(12, 0), name=RTX_5070_TI, arch_list=("sm_120a",))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    info = detect_device()
    assert info.wheel_supports_device is True


def test_higher_minor_binary_does_not_count(monkeypatch):
    """A sm_89 cubin cannot run on an 8.6 device — minor must be <= device."""
    fake = make_fake_torch(cc=(8, 6), name="NVIDIA GeForce RTX 3090",
                           arch_list=("sm_89",))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    info = detect_device()
    assert info.wheel_supports_device is False


def test_gb10_grace_blackwell(monkeypatch):
    fake = make_fake_torch(cc=(12, 1), name="NVIDIA GB10",
                           arch_list=("sm_90", "sm_100", "sm_120", "sm_121"))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    info = detect_device()
    assert info.architecture == ARCH_BLACKWELL_GB10
    assert info.platform_machine == "aarch64"
    assert info.compute_capability == (12, 1)
    assert info.wheel_supports_device is True
    assert info.warnings == []


# --------------------------------------------------------------------------- #
# detect_device — precision resolution
# --------------------------------------------------------------------------- #
def test_precision_auto_without_bf16_uses_fp16(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", make_fake_torch(bf16=False))
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    info = detect_device(DeviceConfig(precision="auto"))
    assert info.bf16_supported is False
    assert info.precision == "fp16"


def test_precision_explicit_bf16_downgrades_without_support(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", make_fake_torch(bf16=False))
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    info = detect_device(DeviceConfig(precision="bf16"))
    assert info.precision == "fp16"
    assert any("bf16" in w for w in info.warnings)


@pytest.mark.parametrize("requested", ["bf16", "fp16"])
def test_precision_explicit_half_on_cpu_falls_back_fp32(monkeypatch, requested):
    monkeypatch.setitem(sys.modules, "torch", make_fake_torch(cuda_available=False))
    info = detect_device(DeviceConfig(device="cpu", precision=requested))
    assert info.precision == "fp32"
    assert any(requested in w for w in info.warnings)


@pytest.mark.parametrize("requested", ["fp32", "fp16"])
def test_precision_explicit_valid_stays_on_cuda(monkeypatch, requested):
    monkeypatch.setitem(sys.modules, "torch", make_fake_torch())
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    info = detect_device(DeviceConfig(precision=requested))
    assert info.precision == requested
    assert info.warnings == []


def test_bf16_probe_error_treated_as_unsupported(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", make_fake_torch(bf16_raises=True))
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    info = detect_device()
    assert info.bf16_supported is False
    assert info.precision == "fp16"


def test_invalid_device_string_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError):
        detect_device(DeviceConfig(device="tpu"))


def test_invalid_precision_string_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError):
        detect_device(DeviceConfig(precision="int8"))


# --------------------------------------------------------------------------- #
# seed_everything
# --------------------------------------------------------------------------- #
def test_seed_everything_records_torch_and_cuda_seeds(monkeypatch):
    fake = make_fake_torch(cuda_available=True)
    monkeypatch.setitem(sys.modules, "torch", fake)
    seed_everything(42)
    assert fake._calls["manual_seed"] == [42]
    assert fake._calls["cuda_manual_seed_all"] == [42]


def test_seed_everything_skips_cuda_when_unavailable(monkeypatch):
    fake = make_fake_torch(cuda_available=False)
    monkeypatch.setitem(sys.modules, "torch", fake)
    seed_everything(7)
    assert fake._calls["manual_seed"] == [7]
    assert fake._calls["cuda_manual_seed_all"] == []


def test_seed_everything_without_torch_is_reproducible(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    seed_everything(123)
    expected_py = random.random()
    expected_np = float(np.random.rand())
    seed_everything(123)
    assert random.random() == expected_py
    assert float(np.random.rand()) == expected_np


def test_seed_everything_large_seed_wraps_for_numpy(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    seed_everything(2 ** 40)  # would overflow np.random.seed without mod 2**32


@pytest.mark.parametrize("bad_seed", ["42", 1.5, None])
def test_seed_everything_rejects_non_int(monkeypatch, bad_seed):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(TypeError):
        seed_everything(bad_seed)


# --------------------------------------------------------------------------- #
# format_diagnostics / print_diagnostics
# --------------------------------------------------------------------------- #
def test_format_diagnostics_reports_mismatch(monkeypatch):
    fake = make_fake_torch(cc=(12, 0), name=RTX_5070_TI, arch_list=ADA_ARCH_LIST)
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    report = format_diagnostics(detect_device())
    assert RTX_5070_TI in report
    assert ARCH_BLACKWELL in report
    assert "warnings" in report.lower()
    assert "1." in report  # numbered warning entries
    assert "wheel/device compatibility: MISMATCH" in report


def test_format_diagnostics_reports_ok(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", make_fake_torch())
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    report = format_diagnostics(detect_device())
    assert RTX_4070 in report
    assert ARCH_ADA in report
    assert "wheel/device compatibility: OK" in report


def test_format_diagnostics_without_info_runs_detection(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    report = format_diagnostics()
    assert "torch available" in report
    assert "wheel/device compatibility: n/a (cpu)" in report


def test_print_diagnostics_prints_report(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", None)
    print_diagnostics()
    out = capsys.readouterr().out
    assert "NLP device diagnostics" in out
    assert "wheel/device compatibility" in out
