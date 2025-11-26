import os
import sys
import numpy as np
import pytest

_SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

from hamiltonian.geometry.graphene_generator import GrapheneGenerator, generate_graphene_xyz
from hamiltonian import Hamiltonian


def _nearest_distance(coords: np.ndarray) -> float:
    dmin = float('inf')
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = np.linalg.norm(coords[i] - coords[j])
            if 0 < d < dmin:
                dmin = d
    return dmin


def test_graphene_generator_counts_and_spacing():
    nx, ny = 3, 4
    gen = GrapheneGenerator.generate(nx=nx, ny=ny, orientation='zigzag', periodic_dirs=None)
    coords = np.asarray(gen.c_positions)
    expected = nx * (4 * ny + 2)
    assert len(coords) == expected
    assert _nearest_distance(coords) == pytest.approx(1.42, abs=1e-2)


def test_graphene_generator_rectangular_axes():
    gen = GrapheneGenerator.generate(nx=4, ny=3, orientation='zigzag', periodic_dirs=None)
    coords = np.asarray(gen.c_positions)
    assert coords[:, 0].min() == pytest.approx(0.0, abs=1e-6)
    assert coords[:, 1].min() == pytest.approx(0.0, abs=1e-6)
    widths = coords.max(axis=0) - coords.min(axis=0)
    assert widths[0] > 0.0 and widths[1] > 0.0


def test_graphene_generator_orientation_change():
    gen_arm = GrapheneGenerator.generate(nx=2, ny=2, orientation='armchair', periodic_dirs=None)
    coords_arm = np.asarray(gen_arm.c_positions)
    width_arm_y = np.ptp(coords_arm[:, 1])
    width_arm_x = np.ptp(coords_arm[:, 0])
    gen_zig = GrapheneGenerator.generate(nx=2, ny=2, orientation='zigzag', periodic_dirs=None)
    coords_zig = np.asarray(gen_zig.c_positions)
    width_zig_y = np.ptp(coords_zig[:, 1])
    width_zig_x = np.ptp(coords_zig[:, 0])
    assert abs(width_arm_y - width_zig_y) > 1e-3 or abs(width_arm_x - width_zig_x) > 1e-3


def test_graphene_hamiltonian_initialises_from_string():
    H = Hamiltonian(structure='graphene-armchair', nx=2, ny=2, periodic_dirs=None).initialize()
    generated = GrapheneGenerator.generate(nx=2, ny=2, orientation='armchair', periodic_dirs=None)
    expected_nodes = len(generated.c_positions)
    assert H.num_of_nodes == expected_nodes
    assert pytest.approx(H.nn_distance, rel=1e-3) == 1.4342
    xyz = generate_graphene_xyz(nx=2, ny=2, orientation='armchair', periodic_dirs=None)
    H2 = Hamiltonian(structure='graphene', nx=2, ny=2, graphene_orientation='armchair', periodic_dirs=None, xyz=xyz).initialize()
    assert H2.num_of_nodes == H.num_of_nodes


def test_graphene_passivation_counts_and_extent():
    nx, ny = 3, 4
    base = GrapheneGenerator.generate(nx=nx, ny=ny, orientation='zigzag', periodic_dirs=None, passivate_edges=False)
    assert len(base.h_positions) == 0

    y_only = GrapheneGenerator.generate(nx=nx, ny=ny, orientation='zigzag', periodic_dirs=None, passivate_edges=True, passivate_x=True)
    assert len(y_only.h_positions) > 0
    carbon = np.asarray(y_only.c_positions)
    hyd_y = np.asarray(y_only.h_positions)
    assert hyd_y[:, 1].min() < carbon[:, 1].min()
    assert hyd_y[:, 1].max() > carbon[:, 1].max()

    xy_pass = GrapheneGenerator.generate(nx=nx, ny=ny, orientation='zigzag', periodic_dirs=None, passivate_edges=True, passivate_x=False)
    assert len(xy_pass.h_positions) > len(y_only.h_positions)
    hyd_xy = np.asarray(xy_pass.h_positions)
    assert hyd_xy[:, 0].min() < carbon[:, 0].min()
    assert hyd_xy[:, 0].max() > carbon[:, 0].max()

    periodic_x = GrapheneGenerator.generate(nx=nx, ny=ny, orientation='zigzag', periodic_dirs='x', passivate_edges=True, passivate_x=False)
    carbon_periodic = np.asarray(periodic_x.c_positions)
    hyd_periodic = np.asarray(periodic_x.h_positions)
    assert 0 < len(hyd_periodic) <= len(hyd_xy)
    assert hyd_periodic[:, 0].min() >= carbon_periodic[:, 0].min() - 1e-3
    assert hyd_periodic[:, 0].max() <= carbon_periodic[:, 0].max() + 1e-3
