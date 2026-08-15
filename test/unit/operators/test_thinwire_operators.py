"""Regression tests for the Pocklington thin-wire formulation."""

import numpy as np

import bempp.api
from bempp.api.grid import LineGrid
from bempp.core.numba_kernels import thinwire_helmholtz_potential
from examples.maxwell.thin_wire_dipole import (
    delta_gap_load,
    solve_dipole,
    straight_wire_grid,
)


def _line_grid(direction, number_of_vertices=7, length=2.0, radius=0.01):
    """Create a uniformly meshed straight wire in the given direction."""
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    coordinate = np.linspace(-0.5 * length, 0.5 * length, number_of_vertices)
    vertices = direction[:, None] * coordinate
    elements = np.vstack(
        [np.arange(number_of_vertices - 1), np.arange(1, number_of_vertices)]
    ).astype(np.uint32)
    return LineGrid(vertices, elements, wire_radius=radius)


def _parameters(regular=10, singular=20):
    parameters = bempp.api.DefaultParameters()
    parameters.quadrature.regular = regular
    parameters.quadrature.singular = singular
    return parameters


def test_pwl_space_excludes_open_wire_endpoints_by_default():
    """The PEC endpoint condition must be represented in the PWL dof map."""
    grid = _line_grid([0.0, 0.0, 1.0], number_of_vertices=7)

    interior_space = bempp.api.function_space(grid, "PWL", 0)
    full_space = bempp.api.function_space(
        grid, "PWL", 0, include_boundary_dofs=True
    )

    assert interior_space.global_dof_count == 5
    assert full_space.global_dof_count == 7
    assert interior_space.local_multipliers[0, 0] == 0.0
    assert interior_space.local_multipliers[-1, 1] == 0.0
    assert np.all(interior_space.local_multipliers[1:-1] == 1.0)


def test_wire_radius_update_reaches_numba_grid_data():
    """Changing the public radius must affect double and single assembly data."""
    grid = _line_grid([0.0, 0.0, 1.0])
    grid.wire_radius = 0.025

    np.testing.assert_allclose(grid.wire_radius, 0.025)
    np.testing.assert_allclose(grid.data("double").wire_radius, 0.025)
    np.testing.assert_allclose(grid.data("single").wire_radius, 0.025)


def test_delta_gap_load_is_independent_of_mesh_parity():
    """The feed must stay centred for meshes with even or odd element counts."""
    expected_values = {
        20: np.array([1.0]),
        21: np.array([0.5, 0.5]),
    }

    for number_of_elements, expected in expected_values.items():
        grid = straight_wire_grid(number_of_elements)
        space = bempp.api.function_space(grid, "PWL", 0)
        loading = delta_gap_load(grid, space)
        nonzero_values = np.sort(loading[loading != 0.0].real)

        np.testing.assert_allclose(nonzero_values, expected)
        np.testing.assert_allclose(np.sum(loading), 1.0)


def test_thinwire_current_converges_under_uniform_refinement():
    """Successive PWL solutions must approach one mesh-independent curve."""
    comparison_points = np.linspace(-1.75, 1.75, 141)
    interpolated_currents = []

    for number_of_elements in (8, 16, 32):
        coordinate, current = solve_dipole(
            number_of_elements,
            regular_order=8,
            singular_order=16,
        )
        np.testing.assert_allclose(current[[0, -1]], 0.0, atol=1e-14)
        np.testing.assert_allclose(current, current[::-1], rtol=2e-11, atol=2e-11)
        feed_index = np.argmin(np.abs(coordinate))
        assert np.abs(current[feed_index]) > np.abs(current[1])
        assert np.abs(current[feed_index]) > np.abs(current[-2])
        interpolated_currents.append(
            np.interp(comparison_points, coordinate, current.real)
            + 1j * np.interp(comparison_points, coordinate, current.imag)
        )

    coarse_change = np.linalg.norm(
        interpolated_currents[0] - interpolated_currents[1]
    ) / np.linalg.norm(interpolated_currents[1])
    fine_change = np.linalg.norm(
        interpolated_currents[1] - interpolated_currents[2]
    ) / np.linalg.norm(interpolated_currents[2])

    assert fine_change < 0.5 * coarse_change
    assert fine_change < 0.04


def test_thinwire_boundary_operator_is_symmetric_and_rotation_invariant():
    """The Galerkin matrix must preserve reciprocity and rigid rotations."""
    parameters = _parameters()
    wavenumber = 2.0 * np.pi / 8.0
    matrices = []

    for direction in ([0.0, 0.0, 1.0], [1.0, 2.0, 3.0]):
        grid = _line_grid(direction)
        space = bempp.api.function_space(grid, "PWL", 0)
        operator = bempp.api.operators.boundary.maxwell.electric_field(
            space, space, space, wavenumber, parameters=parameters
        )
        matrix = operator.weak_form().to_dense()
        np.testing.assert_allclose(matrix, matrix.T, rtol=2e-12, atol=2e-12)
        matrices.append(matrix)

    np.testing.assert_allclose(matrices[0], matrices[1], rtol=2e-12, atol=2e-12)


def test_thinwire_boundary_operator_matches_direct_galerkin_quadrature():
    """The assembled matrix must implement the weak form in Eq. (3.13)."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.3, 0.6],
            [-1.0, -0.3, 0.2, 0.7],
        ]
    )
    elements = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.uint32)
    radius = 0.02
    wavenumber = 1.7
    grid = LineGrid(vertices, elements, wire_radius=radius)
    space = bempp.api.function_space(grid, "PWL", 0)
    parameters = _parameters(regular=20, singular=30)

    operator = bempp.api.operators.boundary.maxwell.electric_field(
        space, space, space, wavenumber, parameters=parameters
    )
    actual = operator.weak_form().to_dense()

    points, weights = np.polynomial.legendre.leggauss(120)
    points = 0.5 * (points + 1.0)
    weights *= 0.5
    expected = np.zeros_like(actual)

    for test_element in range(grid.number_of_elements):
        test_start = vertices[:, elements[0, test_element]]
        test_end = vertices[:, elements[1, test_element]]
        test_length = np.linalg.norm(test_end - test_start)
        test_tangent = (test_end - test_start) / test_length
        test_global_points = test_start[:, None] + (
            test_end - test_start
        )[:, None] * points
        test_basis = np.vstack([1.0 - points, points])
        test_divergence = np.array([-1.0 / test_length, 1.0 / test_length])

        for trial_element in range(grid.number_of_elements):
            trial_start = vertices[:, elements[0, trial_element]]
            trial_end = vertices[:, elements[1, trial_element]]
            trial_length = np.linalg.norm(trial_end - trial_start)
            trial_tangent = (trial_end - trial_start) / trial_length
            trial_global_points = trial_start[:, None] + (
                trial_end - trial_start
            )[:, None] * points
            trial_basis = np.vstack([1.0 - points, points])
            trial_divergence = np.array(
                [-1.0 / trial_length, 1.0 / trial_length]
            )

            distance = np.sqrt(
                np.sum(
                    (
                        test_global_points[:, :, None]
                        - trial_global_points[:, None, :]
                    )
                    ** 2,
                    axis=0,
                )
                + radius**2
            )
            green_function = np.exp(1j * wavenumber * distance) / (
                4.0 * np.pi * distance
            )
            quadrature_weights = (
                weights[:, None]
                * weights[None, :]
                * test_length
                * trial_length
            )

            for test_local_index in range(2):
                if space.local_multipliers[test_element, test_local_index] == 0.0:
                    continue
                test_dof = space.local2global[test_element, test_local_index]
                for trial_local_index in range(2):
                    if (
                        space.local_multipliers[
                            trial_element, trial_local_index
                        ]
                        == 0.0
                    ):
                        continue
                    trial_dof = space.local2global[
                        trial_element, trial_local_index
                    ]
                    weak_factor = (
                        np.dot(test_tangent, trial_tangent)
                        * test_basis[test_local_index, :, None]
                        * trial_basis[trial_local_index, None, :]
                        - test_divergence[test_local_index]
                        * trial_divergence[trial_local_index]
                        / wavenumber**2
                    )
                    expected[test_dof, trial_dof] += np.sum(
                        quadrature_weights * weak_factor * green_function
                    )

    np.testing.assert_allclose(actual, expected, rtol=5e-7, atol=1e-8)


def test_thinwire_potential_kernel_matches_finite_difference():
    """The closed-form axial field kernel must equal (d2/dz2 + k2)G."""
    wavenumber = 2.3
    rho = 0.4
    axial_distance = 0.7
    trial_point = np.zeros((3, 1))
    trial_tangent = np.array([[0.0], [0.0], [1.0]])
    observation_point = np.array([rho, 0.0, axial_distance])

    actual = thinwire_helmholtz_potential(
        observation_point,
        trial_point,
        None,
        trial_tangent,
        np.array([wavenumber, 0.0]),
        np.array([0.01]),
    )[0]

    def green_function(z_coordinate):
        distance = np.sqrt(rho * rho + z_coordinate * z_coordinate)
        return np.exp(1j * wavenumber * distance) / (4.0 * np.pi * distance)

    step = 1.0e-4
    expected = (
        green_function(axial_distance + step)
        - 2.0 * green_function(axial_distance)
        + green_function(axial_distance - step)
    ) / step**2 + wavenumber**2 * green_function(axial_distance)

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=3e-9)


def test_thinwire_potential_operator_has_correct_line_jacobian():
    """Potential assembly must include exactly one line Jacobian factor."""
    grid = _line_grid([0.0, 0.0, 1.0], number_of_vertices=5)
    space = bempp.api.function_space(
        grid, "PWL", 0, include_boundary_dofs=True
    )
    parameters = _parameters(regular=20)
    coefficients = np.ones(space.global_dof_count, dtype=np.complex128)
    current = bempp.api.GridFunction(
        space, coefficients=coefficients, parameters=parameters
    )
    observation_point = np.array([[0.4], [0.0], [0.7]])
    wavenumber = 2.3

    potential = bempp.api.operators.potential.maxwell.electric_field(
        space, observation_point, wavenumber, parameters=parameters
    )
    actual = potential * current

    points, weights = np.polynomial.legendre.leggauss(80)
    points = 0.5 * (points + 1.0)
    weights *= 0.5
    z_vertices = grid.vertices[2]
    expected = 0.0j
    for start, end in zip(z_vertices[:-1], z_vertices[1:]):
        z_coordinate = start + (end - start) * points
        rho_squared = 0.4**2
        distance = np.sqrt(rho_squared + (0.7 - z_coordinate) ** 2)
        kernel = (
            np.exp(1j * wavenumber * distance)
            / (4.0 * np.pi * distance**5)
            * (
                (1.0 - 1j * wavenumber * distance)
                * (2.0 * distance**2 - 3.0 * rho_squared)
                + wavenumber**2 * rho_squared * distance**2
            )
        )
        expected += (end - start) * np.dot(weights, kernel)

    np.testing.assert_allclose(actual[:2, 0], 0.0, atol=1e-14)
    np.testing.assert_allclose(actual[2, 0], expected, rtol=2e-12, atol=2e-12)
