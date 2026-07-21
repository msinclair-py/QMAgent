"""The CPU-only DFT path selects pyscf, the GPU path selects gpu4pyscf.

``load_dft`` is the single place the ``gpu`` flag decides which engine backs the
mean-field. Neither real engine is installed in the test environment, so we
inject lightweight fakes for ``pyscf`` and ``gpu4pyscf`` and assert the flag
routes ``dft.RKS`` to the right module. This locks in the demo's ``use_gpu=False``
plumbing without needing PySCF/CUDA.
"""

import sys
import types

import pytest

from qmagent.agents.distributed import load_dft
from qmagent.utils.pydantic_models import QMConfig


def _fake_engine(name: str, tag: str) -> types.ModuleType:
    """A stand-in module exposing a ``dft.RKS`` that records which engine ran."""
    mod = types.ModuleType(name)
    dft = types.ModuleType(f'{name}.dft')

    class RKS:
        def __init__(self, mol):
            self.mol = mol
            self.engine = tag
            self.grids = types.SimpleNamespace(level=None)

    dft.RKS = RKS
    mod.dft = dft
    return mod


def _fake_pyscf_gto() -> types.ModuleType:
    """A minimal ``pyscf`` package whose ``gto.M`` returns a sentinel Mole."""
    pyscf = types.ModuleType('pyscf')
    gto = types.ModuleType('pyscf.gto')
    gto.M = lambda **kwargs: types.SimpleNamespace(**kwargs)
    pyscf.gto = gto
    return pyscf


@pytest.fixture
def qm_config():
    return QMConfig(functional='b3lyp', basis='sto-3g', dispersion='d3bj',
                    charge=0, multiplicity=1, grid_level=3)


@pytest.fixture
def fake_engines(monkeypatch):
    """Install fake pyscf (with gto + dft) and gpu4pyscf (dft) modules."""
    pyscf = _fake_pyscf_gto()
    pyscf_dft = _fake_engine('pyscf', 'cpu').dft
    pyscf.dft = pyscf_dft
    gpu4pyscf = _fake_engine('gpu4pyscf', 'gpu')

    monkeypatch.setitem(sys.modules, 'pyscf', pyscf)
    monkeypatch.setitem(sys.modules, 'pyscf.gto', pyscf.gto)
    monkeypatch.setitem(sys.modules, 'pyscf.dft', pyscf_dft)
    monkeypatch.setitem(sys.modules, 'gpu4pyscf', gpu4pyscf)
    monkeypatch.setitem(sys.modules, 'gpu4pyscf.dft', gpu4pyscf.dft)


def test_cpu_path_uses_pyscf(fake_engines, qm_config):
    mf = load_dft(geom_str='H 0 0 0', qm_config=qm_config, verbose=0,
                  max_memory=1000, symmetry=False, gpu=False)
    assert mf.engine == 'cpu'
    # config was applied to the CPU mean-field
    assert mf.xc == 'b3lyp'
    assert mf.disp == 'd3bj'
    assert mf.grids.level == 3


def test_gpu_path_uses_gpu4pyscf(fake_engines, qm_config):
    mf = load_dft(geom_str='H 0 0 0', qm_config=qm_config, verbose=0,
                  max_memory=1000, symmetry=False, gpu=True)
    assert mf.engine == 'gpu'
