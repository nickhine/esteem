"""Smoke tests: ESTEEM imports, task classes construct, and OpenQP inputs are written.

These need only numpy and ase. None of them run an external code.
"""

import importlib
import pkgutil

import pytest

import esteem

# Modules that need an optional heavy dependency, skipped when it is not installed
OPTIONAL = {"esteem.wrappers.mace": "mace"}

MODULES = sorted(m.name for m in pkgutil.walk_packages(esteem.__path__, "esteem."))


@pytest.mark.parametrize("module", MODULES)
def test_import(module):
    if module in OPTIONAL:
        pytest.importorskip(OPTIONAL[module])
    importlib.import_module(module)


@pytest.mark.parametrize("module, cls", [
    ("esteem.tasks.solutes", "SolutesTask"),
    ("esteem.tasks.solvate", "SolvateTask"),
    ("esteem.tasks.clusters", "ClustersTask"),
    ("esteem.tasks.spectra", "SpectraTask"),
    ("esteem.tasks.qmd_trajectories", "QMDTrajTask"),
    ("esteem.tasks.ml_training", "MLTrainingTask"),
    ("esteem.tasks.ml_testing", "MLTestingTask"),
    ("esteem.tasks.ml_trajectories", "MLTrajTask"),
])
def test_task_construct(module, cls):
    # Each task's __init__ fills in its defaults from its argument parser
    task = getattr(importlib.import_module(module), cls)()
    assert task.make_parser() is not None


def test_openqp_mrsf_gradient_input(tmp_path, monkeypatch):
    from ase.build import molecule
    from esteem.wrappers.openqp import OpenQPWrapper

    monkeypatch.chdir(tmp_path)
    w = OpenQPWrapper()
    w._write_input(molecule("H2O"), "frame", "def2-SV(P)", "BHHLYP", 0, 0, "grad",
                   mrsf_nstate=3, grad_state=[1, 2])
    lines = (tmp_path / "frame.inp").read_text().splitlines()
    for line in ["runtype=grad", "method=tdhf", "type=mrsf", "nstate=3", "save_mol=true",
                 "grad=1,2", "export=true", "title=frame", "td_prop=true"]:
        assert line in lines, line
