#!/usr/bin/env python3
"""
Environment Checker for the ML4ME Textbook

Run this after bootstrap_env.py to confirm the ml4me-student environment can
actually run the course material:

    conda run -n ml4me-student python check_env.py

Prints one PASS/WARN/FAIL line per check and exits non-zero if anything
required is broken.
"""

import importlib
import importlib.util
import os
import platform
import sys
import traceback
from pathlib import Path

# Headless-safe plotting: several course modules configure matplotlib on import.
os.environ.setdefault("MPLBACKEND", "Agg")

SEED = 42
REPO_ROOT = Path(__file__).resolve().parent
GEN_MODELS_DIR = REPO_ROOT / "part2" / "gen_models"

# (import name, pip name) for packages declared in pyproject.toml.
REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("sklearn", "scikit-learn"),
    ("PIL", "Pillow"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("geomloss", "geomloss"),
    ("engibench", "engibench"),
    ("ipywidgets", "ipywidgets"),
    ("notebook", "notebook"),
]

# Imported by notebooks but not declared in pyproject.toml, so missing ones are
# reported as warnings rather than failures.
OPTIONAL_PACKAGES = [
    ("plotly", "plotly", "part2/gen_models/latent_generative_models.ipynb"),
    ("torchdiffeq", "torchdiffeq", "part2/gen_models/continuous_flows.ipynb"),
    ("gymnasium", "gymnasium[classic-control]", "part2/introduction_to_reinforcement_learning.ipynb"),
    ("tqdm", "tqdm", "part2/introduction_to_reinforcement_learning.ipynb"),
    ("pyro", "pyro-ppl", "part1/introduction_to_probabilistic_programming.ipynb"),
    ("wandb", "wandb", "part2/gen_models/EngiBench_latent_gen_models.ipynb"),
    ("geomstats", "geomstats", "part1/distribution_distance.ipynb"),
    ("requests", "requests", "part1/linear_decompositions.ipynb"),
]

failures = []
warnings = []


def report(status, message):
    """Print a single result line and record failures and warnings."""
    print(f"[{status:4}] {message}")
    if status == "FAIL":
        failures.append(message)
    elif status == "WARN":
        warnings.append(message)


def section(title):
    """Print a section heading."""
    print(f"\n{title}\n{'-' * len(title)}")


def package_version(module):
    """Best-effort version string for an imported module."""
    return getattr(module, "__version__", "unknown")


def check_interpreter():
    """Report which Python and which environment is actually being tested."""
    section("Interpreter")
    print(f"       Python     {sys.version.split()[0]} ({platform.machine()})")
    print(f"       Executable {sys.executable}")
    print(f"       Prefix     {sys.prefix}")

    env_name = Path(sys.prefix).name
    if env_name == "ml4me-student":
        report("PASS", "Running inside the ml4me-student environment")
    else:
        report("WARN", f"Environment is '{env_name}', expected 'ml4me-student'")

    if sys.version_info < (3, 11):
        report("FAIL", f"Python {sys.version_info.major}.{sys.version_info.minor} is below the required 3.11")


def check_packages():
    """Import every package the course material relies on."""
    section("Required packages")
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(import_name)
        except Exception as exc:
            report("FAIL", f"{pip_name}: {type(exc).__name__}: {exc}")
        else:
            report("PASS", f"{pip_name} {package_version(module)}")

    section("Optional packages (used by notebooks, not declared in pyproject.toml)")
    for import_name, pip_name, used_by in OPTIONAL_PACKAGES:
        if importlib.util.find_spec(import_name) is None:
            report("WARN", f"{pip_name} missing - needed by {used_by}")
        else:
            module = importlib.import_module(import_name)
            report("PASS", f"{pip_name} {package_version(module)}")


def check_torch():
    """Verify PyTorch computes on CPU and, where present, on the GPU."""
    section("PyTorch")
    try:
        import torch
    except Exception as exc:
        report("FAIL", f"torch import failed: {type(exc).__name__}: {exc}")
        return

    torch.manual_seed(SEED)
    print(f"       torch {torch.__version__}  CUDA build: {torch.version.cuda}")

    try:
        result = torch.randn(64, 64) @ torch.randn(64, 64)
        assert result.shape == (64, 64)
        report("PASS", "CPU matrix multiply")
    except Exception as exc:
        report("FAIL", f"CPU matrix multiply failed: {exc}")

    if not torch.cuda.is_available():
        # The bootstrap script installs CUDA wheels whenever nvidia-smi is
        # found, so a CPU-only result on such a machine means a broken install.
        report("WARN", "torch.cuda.is_available() is False - GPU acceleration unavailable")
        return

    report("PASS", f"CUDA available: {torch.cuda.get_device_name(0)}")
    try:
        gpu_result = (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).cpu()
        assert gpu_result.shape == (64, 64)
        report("PASS", "GPU matrix multiply")
    except Exception as exc:
        report("FAIL", f"GPU matrix multiply failed: {exc}")


def load_course_module(name):
    """Import a module from part2/gen_models, which is not an installed package."""
    path = GEN_MODELS_DIR / f"{name}.py"
    if not path.exists():
        return None
    if str(GEN_MODELS_DIR) not in sys.path:
        sys.path.insert(0, str(GEN_MODELS_DIR))
    return importlib.import_module(name)


def check_course_code():
    """Run the repository's own modules, not just third-party imports."""
    section("Course code")
    if not GEN_MODELS_DIR.exists():
        report("WARN", f"{GEN_MODELS_DIR} not found - run this script from the repository root")
        return

    try:
        import torch

        models = load_course_module("engibench_gen_models")
        generator = models.CGANGenerator(latent_dim=8, n_conds=2, design_shape=(64, 64)).eval()
        with torch.no_grad():
            design = generator(torch.randn(2, 8, 1, 1), torch.randn(2, 2, 1, 1))
        assert design.shape == (2, 1, 64, 64), f"unexpected shape {tuple(design.shape)}"
        report("PASS", f"engibench_gen_models.CGANGenerator forward pass -> {tuple(design.shape)}")
    except Exception as exc:
        report("FAIL", f"engibench_gen_models failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    try:
        utilities = load_course_module("gen_models_utilities")
        points, assignments = utilities.create_ring_gaussians(n_samples=256, n_modes=8)
        assert points.shape == (256, 2) and assignments.shape == (256,)
        report("PASS", f"gen_models_utilities.create_ring_gaussians -> {points.shape}, device={utilities.device}")
    except Exception as exc:
        report("FAIL", f"gen_models_utilities failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()


def check_jupyter_kernel():
    """A working environment VS Code cannot see is still broken for students."""
    section("Jupyter kernel")
    try:
        from jupyter_client.kernelspec import KernelSpecManager
    except Exception as exc:
        report("FAIL", f"jupyter_client unavailable: {type(exc).__name__}: {exc}")
        return

    specs = KernelSpecManager().find_kernel_specs()
    if not specs:
        report("WARN", "No Jupyter kernels registered")
        return

    print(f"       Kernels: {', '.join(sorted(specs))}")
    report("PASS", "Jupyter kernelspecs discoverable")


def main():
    """Run every check and summarise the result."""
    print("ML4ME Environment Check")
    print("=" * 50)

    check_interpreter()
    check_packages()
    check_torch()
    check_course_code()
    check_jupyter_kernel()

    section("Summary")
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for item in failures:
            print(f"  - {item}")
    if warnings:
        print(f"{len(warnings)} warning(s):")
        for item in warnings:
            print(f"  - {item}")
    if not failures and not warnings:
        print("All checks passed.")
    elif not failures:
        print("All required checks passed.")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
