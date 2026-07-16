from parsl import python_app
from pathlib import Path
import re
import subprocess

def parse_paramfit_k(log_file: Path) -> float | None:
    """Recover the fitted QM/MM energy offset K from a paramfit K_ONLY log.

    paramfit reports the fitted K in the log of a ``PARAMETERS_TO_FIT=K_ONLY``
    run; the main dihedral fit should then hold that K fixed (the documented
    K_ONLY -> record K -> LOAD-fit workflow). The exact wording varies across
    AmberTools versions, so several patterns are tried and ``None`` is returned
    when none match, letting the caller fall back to not setting K rather than
    guessing a wrong value.

    Arguments:
        log_file (Path): The paramfit K_ONLY log to parse.

    Returns:
        (float | None): The fitted K (kcal/mol), or None if it could not be found.
    """
    try:
        text = log_file.read_text()
    except OSError:
        return None

    patterns = (
        r'value of K[^\-\d]*(-?\d+\.\d+)',   # "...value of K to be:  -12.34 KCal/mol"
        r'\bK\b[^\-\d=]*(-?\d+\.\d+)\s*KCal', # "K =  -12.34 KCal/mol"
        r'\bK\s*=\s*(-?\d+\.\d+)',            # "K= -12.34"
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None

def run_cmd(cmd, description='') -> bool:
    """Run a shell command and check for errors."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        return False
    return True

@python_app
def run_antechamber(sdf_file: Path,
                    resp_charges: Path,
                    mol2_output: Path,
                    resname: str,
                    amberhome: Path,
                    charge: int) -> bool:
    """Use antechamber to assign GAFF2 atom types.

    We can provide our pre-computed RESP2 charges via the -cf flag,
    so antechamber only does atom typing and not charge setting.

    Arguments:
        sdf_file (Path): sdf file path
        resp_charges (Path): RESP charges file path
        mol2_output (Path): Path for where mol2 file will be written
        resname (str): The name of our designed residue (e.g. LIG)
        amberhome (Path): Path to where the ambertools binaries are
        charge (int): Net charge of our design

    Returns:
        (bool): Whether antechamber was successful or not
    """
    from .amber_apps import run_cmd

    antechamber = amberhome / 'bin' / 'antechamber'
    cmd = [
        str(antechamber),
        '-i', str(sdf_file),
        '-fi', 'sdf',
        '-o', str(mol2_output),
        '-fo', 'mol2',
        '-at', 'gaff2',
        '-rn', resname,
        '-c', 'rc',
        '-nc', str(charge),
        '-pf', 'y',
        '-dr', 'no',
        '-cf', str(resp_charges)
    ]

    success = run_cmd(cmd)

    return success

@python_app
def run_parmchk2(mol2: Path,
                 frcmod_output: Path,
                 amberhome: Path) -> bool:
    """Use parmchk2 to identify missing GAFF2 parameters and generate frcmod.
    Verifies that parameters are at the minimum reasonable.

    Arguments:
        mol2 (Path): Mol2 file path
        frcmod_output (Path): Path to where frcmod_output will be written
        amberhome (Path): Path to where the ambertools binaries are

    Returns:
        (bool): Whether parmchk2 was successful or not
    """
    from .amber_apps import run_cmd
    
    parmchk2 = amberhome / 'bin' / 'parmchk2'
    cmd = [
        str(parmchk2),
        '-i', str(mol2),
        '-f', 'mol2',
        '-o', str(frcmod_output),
        '-s', 'gaff2',
        '-a', 'Y',
    ]

    success = run_cmd(cmd, 'parmchk2')

    return success

@python_app
def get_lib_files(mol2: Path,
                  frcmod_file: Path,
                  prmtop: Path,
                  resname: str,
                  amberhome: Path) -> bool:
    """Sanity check, can tleap actually build this molecule.

    Arguments:
        mol2 (Path): Mol2 file path
        frcmod_file (Path): frcmod file path
        prmtop (Path): Path to prmtop to be written. Inpcrd and lib files are
            inferred from this
        resname (str): The name of the residue being parameterized
        amberhome (Path): Path to where the ambertools binaries are

    Returns:
        (bool): Whether tleap was successful or not
    """
    from .amber_apps import run_cmd

    tleap_script = prmtop.parent / 'tleap.in'
    lib_file = prmtop.parent / f'{resname}.lib'

    tleap_content = [
        'source leaprc.gaff2',
        f'loadamberparams {frcmod_file}',
        f'{resname} = loadmol2 {mol2}',
        f'check {resname}',
        f'saveoff {resname} {lib_file}',
        f'saveamberparm {resname} {prmtop} {prmtop.with_suffix(".inpcrd")}',
        'quit'
    ]

    tleap_script.write_text('\n'.join(tleap_content))
    
    tleap = amberhome / 'bin' / 'tleap'
    cmd = [
        str(tleap),
        '-f', str(tleap_script)
    ]

    success = run_cmd(cmd, 'tleap check')

    return success

def run_paramfit(job_ctrl: Path,
                 prmtop: Path,
                 mdcrd: Path,
                 qm_energies: Path,
                 log_file: Path,
                 amberhome: Path) -> bool:
    """Run a single paramfit pass in AmberTools.

    paramfit performs a global fit of dihedral parameters to reproduce
    QM energy surfaces, handling coupled torsions and proper MM baseline
    subtraction automatically. It is invoked twice per torsion: once with
    K_ONLY to fit the QM/MM energy offset, then with LOAD to fit V_n / phase
    for each periodicity. Which pass runs is defined by the job control file.

    Arguments:
        job_ctrl (Path): paramfit job control file (defines RUNTYPE etc.)
        prmtop (Path): Topology file for the parameterized residue
        mdcrd (Path): AMBER trajectory of the scan geometries
        qm_energies (Path): QM reference energies, one per frame (Hartree)
        log_file (Path): Path for the paramfit stdout/stderr log
        amberhome (Path): Path to where the ambertools binaries are

    Returns:
        (bool): Whether paramfit was successful or not
    """
    paramfit = amberhome / 'bin' / 'paramfit'
    cmd = [
        str(paramfit),
        '-i', str(job_ctrl),
        '-p', str(prmtop),
        '-c', str(mdcrd),
        '-q', str(qm_energies),
    ]

    with open(log_file, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    return result.returncode == 0
