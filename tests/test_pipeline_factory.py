"""Tests for PipelineFactory — protocol-driven @njit pipeline generation."""
import numpy as np
import pytest
from numba import njit
from rubimpact.core.registry import components, KernelSpec
from rubimpact.core.module_base import Module, PipelineStage, PipelineProtocol
from rubimpact.core.pipeline_factory import PipelineFactory


# ── Test kernels ──

@njit(fastmath=True)
def _test_normal_force_kernel(delta, h_loc, ep_loc, alpha_loc, params, out):
    """Simplified: sigma = E * delta / h, F_n = A * sigma"""
    E, A_cell = params[0], params[1]
    if h_loc > 1e-12:
        sigma = E * delta / h_loc
    else:
        sigma = 0.0
    out[0] = A_cell * sigma  # F_n
    out[1] = sigma            # sigma
    out[2] = 0.0              # dgamma


@njit(fastmath=True)
def _test_friction_kernel(F_n, v_rel, params, out):
    """F_t = mu * F_n"""
    out[0] = params[0] * F_n


@njit(fastmath=True)
def _test_geometry_decompose(Fy_n_out, Fz_n_out, Fy_t_out, Fz_t_out,
                             F_n, F_t, yc, zc, r_val):
    """Shared geometry decomposition."""
    inv_r = 1.0 / max(r_val, 1e-12)
    Fy_n_out[0] = F_n * yc * inv_r
    Fz_n_out[0] = F_n * zc * inv_r
    Fy_t_out[0] = -F_t * zc * inv_r
    Fz_t_out[0] = F_t * yc * inv_r


@njit(fastmath=True)
def _test_rom_map_kernel(F_total, F_normal, F_friction,
                          Fy_n, Fz_n, Fy_t, Fz_t,
                          ky, kz, has_normal, has_friction):
    """Shared ROM DOF mapping."""
    n = F_total.shape[0]
    if 0 <= ky < n:
        if has_normal:
            F_normal[ky] += Fy_n
        if has_friction:
            F_friction[ky] += Fy_t
        F_total[ky] += Fy_n + Fy_t
    if 0 <= kz < n:
        if has_normal:
            F_normal[kz] += Fz_n
        if has_friction:
            F_friction[kz] += Fz_t
        F_total[kz] += Fz_n + Fz_t


# ── Register test kernels ──

components.register("constitutive", "TEST_PCL",
    KernelSpec(fn=_test_normal_force_kernel, signature="pcl_return_mapping",
               stage="compute_normal_force"))
components.register("friction", "TEST_COULOMB",
    KernelSpec(fn=_test_friction_kernel, signature="coulomb_friction",
               stage="compute_friction_force"))
components.register("shared", "TEST_GEOMETRY",
    KernelSpec(fn=_test_geometry_decompose, signature="geometry_decompose",
               stage="geometry_decompose"))
components.register("shared", "TEST_ROM_MAP",
    KernelSpec(fn=_test_rom_map_kernel, signature="rom_dof_map",
               stage="rom_dof_map"))


# ── Test module ──

class TestContactModule(Module):
    __test__ = False

    def __init__(self, db, ctx, E=200e3, A=1.0):
        super().__init__(db, ctx)
        self.E = E
        self.A = A

    def get_pipeline_protocol(self):
        return PipelineProtocol(
            stages=[
                PipelineStage(
                    name="compute_normal_force",
                    kernel_ref="constitutive/TEST_PCL",
                    depends_on=[],
                ),
            ],
            params={"E": self.E, "A_cell": self.A},
        )


class TestFrictionModule(Module):
    __test__ = False

    def __init__(self, db, ctx, mu=0.25):
        super().__init__(db, ctx)
        self.mu = mu

    def get_pipeline_protocol(self):
        return PipelineProtocol(
            stages=[
                PipelineStage(
                    name="compute_friction_force",
                    kernel_ref="friction/TEST_COULOMB",
                    depends_on=["compute_normal_force"],
                ),
            ],
            params={"mu": self.mu},
        )


def test_pipeline_factory_builds_pcl_coulomb():
    """PipelineFactory.build() 生成 PCL + COULOMB 管道并能正确计算。"""
    contact = TestContactModule(None, {}, E=200e3, A=1.0)
    friction = TestFrictionModule(None, {}, mu=0.3)

    modules = {"contact": contact, "friction": friction}

    pipeline = PipelineFactory.build(
        modules=modules,
        protocol="ForceAssembler",
        shared_kernels={
            "geometry_decompose": "shared/TEST_GEOMETRY",
            "rom_dof_map": "shared/TEST_ROM_MAP",
        },
    )

    # Test inputs: 2 nodes, node0 penetrates, node1 does not
    pen = np.array([0.01, -0.001], dtype=np.float64)
    coords = np.array([
        [0.0, 0.1, 0.8, 0.6, 1.0],   # node0: yc=0.8, zc=0.6, r=1.0
        [0.5, 0.2, 0.6, 0.8, 1.0],   # node1: yc=0.6, zc=0.8, r=1.0
    ], dtype=np.float64)
    interp = np.array([
        [1.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0],   # h_loc=1.0
        [1.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float64)
    dof_idx = np.array([[0, 0, 1], [0, 2, 3]], dtype=np.int64)

    n_r = 5
    F_total = np.zeros(n_r, dtype=np.float64)
    F_normal = np.zeros(n_r, dtype=np.float64)
    F_friction = np.zeros(n_r, dtype=np.float64)

    pipeline(pen, coords, interp, 100.0, dof_idx,
             F_total, F_normal, F_friction,
             np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))

    # node0: delta=0.01, sigma=200000*0.01/1.0=2000, F_n=2000
    # F_n geometry: Fy_n=2000*0.8/1.0=1600, Fz_n=2000*0.6/1.0=1200
    # F_t=0.3*2000=600
    # Fy_t=-600*0.6/1.0=-360, Fz_t=600*0.8/1.0=480
    # DOF ky=0: F_total+=1600-360=1240, F_normal+=1600, F_friction+=-360
    # DOF kz=1: F_total+=1200+480=1680, F_normal+=1200, F_friction+=480
    assert F_total[0] == pytest.approx(1240.0, rel=1e-10)
    assert F_total[1] == pytest.approx(1680.0, rel=1e-10)
    assert F_normal[0] == pytest.approx(1600.0, rel=1e-10)
    assert F_friction[1] == pytest.approx(480.0, rel=1e-10)
    # node1: no penetration → no contribution
    assert F_total[2] == 0.0
    assert F_total[3] == 0.0


def test_pipeline_factory_empty_modules():
    """空模块列表应返回一个 no-op 管道。"""
    pipeline = PipelineFactory.build(
        modules={}, protocol="ForceAssembler",
        extra_params={"k_penalty": 0.0})
    pen = np.ones(3, dtype=np.float64)
    coords = np.ones((3, 5), dtype=np.float64)
    interp = np.ones((3, 9), dtype=np.float64)
    dof_idx = np.ones((3, 3), dtype=np.int64)
    F_total = np.zeros(5, dtype=np.float64)

    pipeline(pen, coords, interp, 100.0, dof_idx, F_total,
             np.zeros(0), np.zeros(0),
             np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))
    # No modules → no force contributions
    assert np.all(F_total == 0.0)
