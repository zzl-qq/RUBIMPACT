"""Tests for shared kernels: geometry decomposition and ROM DOF mapping."""
import pytest
import numpy as np
from rubimpact.kernels.shared import geometry_decompose, rom_dof_map


def test_geometry_decompose():
    """F_n and F_t decompose correctly into Y/Z components."""
    F_n, F_t = 1000.0, 300.0
    yc, zc, r_val = 0.8, 0.6, 1.0

    Fy_n, Fz_n, Fy_t, Fz_t = geometry_decompose(F_n, F_t, yc, zc, r_val)

    assert Fy_n == pytest.approx(800.0, rel=1e-12)   # 1000 * 0.8 / 1.0
    assert Fz_n == pytest.approx(600.0, rel=1e-12)   # 1000 * 0.6 / 1.0
    assert Fy_t == pytest.approx(-180.0, rel=1e-12)  # -300 * 0.6 / 1.0
    assert Fz_t == pytest.approx(240.0, rel=1e-12)   # 300 * 0.8 / 1.0


def test_geometry_decompose_zero_r():
    """Zero radius should not divide by zero."""
    Fy_n, Fz_n, Fy_t, Fz_t = geometry_decompose(100.0, 50.0, 0.6, 0.8, 0.0)
    assert np.isfinite(Fy_n)
    assert np.isfinite(Fz_n)


def test_rom_dof_map_basic():
    """Forces accumulate to correct DOF indices."""
    n_r = 5
    F_total = np.zeros(n_r)
    F_normal = np.zeros(n_r)
    F_friction = np.zeros(n_r)

    rom_dof_map(F_total, F_normal, F_friction,
                1600.0, 1200.0, -360.0, 480.0,  # Fy_n, Fz_n, Fy_t, Fz_t
                0, 1, True, True)               # ky, kz, has_normal, has_friction

    assert F_total[0] == pytest.approx(1240.0)    # 1600 - 360
    assert F_total[1] == pytest.approx(1680.0)    # 1200 + 480
    assert F_normal[0] == pytest.approx(1600.0)
    assert F_normal[1] == pytest.approx(1200.0)
    assert F_friction[0] == pytest.approx(-360.0)
    assert F_friction[1] == pytest.approx(480.0)


def test_rom_dof_map_no_buffers():
    """Without F_normal/F_friction buffers, only F_total accumulates."""
    n_r = 5
    F_total = np.zeros(n_r)
    F_normal = np.zeros(0)
    F_friction = np.zeros(0)

    rom_dof_map(F_total, F_normal, F_friction,
                800.0, 600.0, -180.0, 240.0,
                0, 1, False, False)

    # Only F_total should have values
    assert F_total[0] == pytest.approx(620.0)
    assert F_total[1] == pytest.approx(840.0)
