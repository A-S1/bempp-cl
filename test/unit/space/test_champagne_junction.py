"""Tests for the Champagne wire-to-surface junction basis geometry."""

import numpy as np

from bempp.api.grid import Grid, LineGrid
from bempp.api.space import find_champagne_junctions


def _plate_and_wire():
    vertices = np.array(
        [
            [0.0, 1.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    elements = np.array(
        [[0, 0, 0, 0], [1, 2, 3, 4], [2, 3, 4, 1]],
        dtype=np.uint32,
    )
    surface = Grid(vertices, elements)
    wire = LineGrid(
        np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.75]]),
        np.array([[0], [1]], dtype=np.uint32),
        wire_radius=0.01,
    )
    return surface, wire


def test_champagne_junction_has_unit_surface_flux_and_charge_balance():
    """Equations (2)--(3) must conserve charge across the junction."""
    surface, wire = _plate_and_wire()
    junctions = find_champagne_junctions(surface, wire)

    assert len(junctions) == 1
    junction = junctions[0]
    np.testing.assert_allclose(
        [data.flux_weight for data in junction.triangles],
        0.25,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(junction.integrated_surface_divergence, 1.0)
    np.testing.assert_allclose(junction.integrated_wire_divergence, -1.0)
    np.testing.assert_allclose(junction.charge_balance, 0.0, atol=1.0e-14)


def test_champagne_surface_value_has_the_published_constant_divergence():
    """Finite differences of equation (1) must reproduce equation (3)."""
    surface, wire = _plate_and_wire()
    junction = find_champagne_junctions(surface, wire)[0]
    element = 0
    point = np.array([0.3, 0.2, 0.0])
    step = 1.0e-6

    plus_x = junction.surface_value(element, point + [step, 0.0, 0.0])
    minus_x = junction.surface_value(element, point - [step, 0.0, 0.0])
    plus_y = junction.surface_value(element, point + [0.0, step, 0.0])
    minus_y = junction.surface_value(element, point - [0.0, step, 0.0])
    numerical_divergence = (
        (plus_x[0] - minus_x[0]) + (plus_y[1] - minus_y[1])
    ) / (2.0 * step)

    np.testing.assert_allclose(
        numerical_divergence,
        junction.surface_divergence(element),
        rtol=2.0e-9,
        atol=2.0e-9,
    )


def test_champagne_wire_rooftop_points_outward_and_tapers_to_zero():
    """The wire part must be one at the junction and zero at the next node."""
    surface, wire = _plate_and_wire()
    junction = find_champagne_junctions(surface, wire)[0]
    values = junction.wire_value([0.0, 0.5 * junction.wire_length, junction.wire_length])

    np.testing.assert_allclose(values[:, 0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(values[:, 1], [0.0, 0.0, 0.5])
    np.testing.assert_allclose(values[:, 2], 0.0, atol=1.0e-15)
    np.testing.assert_allclose(junction.wire_divergence, -1.0 / 0.75)
