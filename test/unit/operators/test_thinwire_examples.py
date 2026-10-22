"""Validation tests for the thin-wire tutorial geometries and loads."""

import numpy as np

import bempp.api
from examples.maxwell.thin_wire_geometries import (
    latitude_wire_sphere_grid,
    parallel_wire_grid,
    plane_wave_load,
)
from examples.maxwell.thin_wire_transmission_line import solve_transmission_line


def test_parallel_wire_differential_solution_has_no_common_mode():
    """A symmetric differential source must produce antisymmetric currents."""
    coordinate, first, second, common_mode_ratio = solve_transmission_line(
        number_of_elements=8,
    )

    np.testing.assert_allclose(first, -second, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(first[[0, -1]], 0.0, atol=1.0e-14)
    assert common_mode_ratio < 1.0e-12
    assert len(coordinate) == 9


def test_spherical_wire_grid_contains_closed_loops_on_one_sphere():
    """Every cage vertex must lie on the sphere and carry one PWL dof."""
    grid = latitude_wire_sphere_grid(
        number_of_latitudes=3,
        elements_per_loop=8,
        sphere_radius=1.5,
    )
    space = bempp.api.function_space(grid, "PWL", 0)

    np.testing.assert_allclose(np.linalg.norm(grid.vertices, axis=0), 1.5)
    assert grid.number_of_vertices == 24
    assert grid.number_of_elements == 24
    assert space.global_dof_count == 24
    assert set(grid.domain_indices) == {0, 1, 2}


def test_plane_wave_load_is_nonzero_and_respects_transversality():
    """The line projection should excite the cage and reject invalid waves."""
    grid = parallel_wire_grid(number_of_elements=4)
    space = bempp.api.function_space(grid, "PWL", 0)
    loading = plane_wave_load(
        grid,
        space,
        wavenumber=1.3,
        direction=[1.0, 0.0, 0.0],
        polarization=[0.0, 0.0, 1.0],
    )
    assert np.linalg.norm(loading) > 0.0

    try:
        plane_wave_load(
            grid,
            space,
            wavenumber=1.3,
            direction=[1.0, 0.0, 0.0],
            polarization=[1.0, 0.0, 0.0],
        )
    except ValueError as error:
        assert "transverse" in str(error)
    else:
        raise AssertionError("non-transverse polarization was accepted")
