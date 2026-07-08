import numpy as np
from pathlib import Path
from pydantic import (BaseModel,
                      ConfigDict,
                      Field,
                      model_validator)
from typing import Any, Self
from .file_ops import NDArray, XYZContents

class AMBERConfig(BaseModel):
    """Settings for ambertools usage"""
    sdf_file: Path
    mol2_file: Path
    frcmod_file: Path
    lib_files: Path
    resp_charges: Path
    prmtop: Path
    amberhome: Path
    resname: str
    charge: int

class AMBERResultSet(BaseModel):
    """Files produced by AMBER/GAFF2 force field integration."""
    mol2_file: Path
    frcmod_file: Path
    lib_file: Path
    prmtop: Path
    inpcrd: Path
    metadata: dict[str, Any] = Field(default_factory=dict)

class ESPCalculation(BaseModel):
    """Electrostatic potential charge calculation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    esp_total: NDArray
    energy: float
    solvated: bool

class ESPResult(BaseModel):
    """Set of ESP calculations"""
    calculations: list[ESPCalculation]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __iter__(self):
        return iter(self.calculations)

    def __len__(self) -> int:
        return len(self.calculations)

    def __getitem__(self, 
                    idx: int) -> ESPCalculation:
        return self.calculations[idx]

    def save(self, 
             path: Path, 
             *,
             pretty: bool=True) -> None:
        """Save the scan to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        indent = 2 if pretty else None
        path.write_text(self.model_dump_json(indent=indent))

    @classmethod
    def load(cls,
             path: Path) -> Self:
        """Load a scan set from JSON file."""
        return cls.model_validate_json(path.read_text())

class GeomOptResult(BaseModel):
    """Geometry optimization result."""
    xyz_file: Path
    energy: float
    metadata: dict[str, str] = Field(default_factory=dict)

class OptimizationResult(BaseModel):
    """Raw output of a single geometry optimization stage."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    e_final: float
    coords: NDArray

class QMConfig(BaseModel):
    """Basic QM config for running most calculations."""
    functional: str
    basis: str
    dispersion: str
    charge: int
    multiplicity: int
    grid_level: int

class RESPCharges(BaseModel):
    """RESP charge assignments"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    elements: list[str] = Field(default_factory=list)
    charges: NDArray
    metadata: dict[str, Any] = Field(default_factory=dict)

class ScanPoint(BaseModel):
    """A single point in a torsion scan."""
    xyz_file: Path
    energy: float
    angle: float

class TorsionScanResult(BaseModel):
    """A complete scan over one torsion (one rotatable bond)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    torsion: tuple[int, int, int, int]
    points: list[ScanPoint]

    @model_validator(mode='after')
    def _sort_by_angle(self) -> Self:
        self.points.sort(key=lambda p: p.angle)
        return self

    # Derived quantities are plain properties (not computed_field): they are
    # recomputed from ``points`` on access and deliberately kept out of the
    # serialized JSON, so a loaded scan reproduces them from the stored points.
    @property
    def angles(self) -> np.ndarray:
        return np.array([p.angle for p in self.points])

    @property
    def raw_energies(self) -> np.ndarray:
        return np.array([p.energy for p in self.points])

    @property
    def relative_energies(self) -> np.ndarray:
        """Energies relative to the minimum, in kcal/mol"""
        e = self.raw_energies.copy()
        e -= np.min(e)
        return e * 627.5095

class TorsionScanSet(BaseModel):
    """A collection of torsion scans, e.g. all rotatable bonds of a molecule
    or a whole benchmark dataset."""
    scans: list[TorsionScanResult]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __iter__(self):
        return iter(self.scans)

    def __len__(self) -> int:
        return len(self.scans)

    def __getitem__(self, 
                    idx: int) -> TorsionScanResult:
        return self.scans[idx]

    def save(self, 
             path: Path, 
             *,
             pretty: bool=True) -> None:
        """Save the scan to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        indent = 2 if pretty else None
        path.write_text(self.model_dump_json(indent=indent))

    @classmethod
    def load(cls,
             path: Path) -> Self:
        """Load a scan set from JSON file."""
        return cls.model_validate_json(path.read_text())

class TorsionFitResult(BaseModel):
    """Fitted AMBER dihedral parameters for one torsion."""
    torsion: tuple[int, int, int, int]
    atom_types: tuple[str, str, str, str]
    frcmod_file: Path | None = None

class TorsionFitSet(BaseModel):
    """All per-torsion fits merged into a single refined frcmod."""
    fits: list[TorsionFitResult]
    refined_frcmod: Path
    metadata: dict[str, str] = Field(default_factory=dict)

    def __iter__(self):
        return iter(self.fits)

    def __len__(self) -> int:
        return len(self.fits)

    def save(self,
             path: Path,
             *,
             pretty: bool=True) -> None:
        """Save the fit set to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        indent = 2 if pretty else None
        path.write_text(self.model_dump_json(indent=indent))

class QMExperiment(BaseModel):
    """Main object storing in silico experimental results."""
    smiles: str
    mol2_file: Path
    molecule: XYZContents
    geometry_optimizations: list[GeomOptResult]
    electrostatic_potential: ESPResult
    resp_charges: RESPCharges
    torsion_scan: TorsionScanSet

