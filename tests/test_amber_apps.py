"""Tests for the dependency-light helpers in qmagent.agents.amber_apps.

The antechamber/parmchk2/tleap/paramfit wrappers shell out to AmberTools and are
out of scope here; this pins the pure-Python K parser used to forward paramfit's
fitted energy offset from the K_ONLY pre-pass into the main dihedral fit.
"""

import pytest

from qmagent.agents.amber_apps import parse_paramfit_k


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Info: Calculated the value of K to be:  -1234.5678 KCal/mol\n", -1234.5678),
        ("Final results:\n   K =  42.1000 KCal/mol\n", 42.1000),
        ("K= -0.5\n", -0.5),
    ],
)
def test_parse_paramfit_k_recovers_value(tmp_path, text, expected):
    log = tmp_path / "fit_K.log"
    log.write_text(text)
    assert parse_paramfit_k(log) == pytest.approx(expected)


def test_parse_paramfit_k_returns_none_when_absent(tmp_path):
    log = tmp_path / "fit_K.log"
    log.write_text("paramfit ran but printed nothing about the offset.\n")
    assert parse_paramfit_k(log) is None


def test_parse_paramfit_k_missing_file_is_none(tmp_path):
    assert parse_paramfit_k(tmp_path / "does_not_exist.log") is None
