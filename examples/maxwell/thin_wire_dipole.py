"""Solve a normalized delta-gap problem on a straight thin-wire dipole."""

import numpy as np

import bempp.api
from bempp.api.grid import LineGrid


def straight_wire_grid(number_of_elements, length=4.0, wire_radius=0.01):
    """Create a uniformly meshed straight wire along the z axis."""
    coordinate = np.linspace(
        -0.5 * length, 0.5 * length, number_of_elements + 1
    )
    vertices = np.vstack(
        [np.zeros_like(coordinate), np.zeros_like(coordinate), coordinate]
    )
    elements = np.vstack(
        [
            np.arange(number_of_elements),
            np.arange(1, number_of_elements + 1),
        ]
    ).astype(np.uint32)
    return LineGrid(vertices, elements, wire_radius=wire_radius)


def delta_gap_load(grid, space, feed_coordinate=0.0, axis=2):
    """Return the Galerkin load for a unit point source at the feed.

    For a delta-gap source, the right-hand side is ``b_i = phi_i(z_feed)``.
    This definition is independent of the element length. If the feed is
    inside an element, the load is shared between its two nodal basis
    functions. Consequently, both even and odd mesh counts preserve a source
    at the same physical coordinate.
    """
    vertex_coordinates = grid.vertices[axis]
    coordinate_scale = max(np.ptp(vertex_coordinates), 1.0)
    tolerance = 100.0 * np.finfo(float).eps * coordinate_scale
    loading = np.zeros(space.global_dof_count, dtype=np.complex128)
    feed_found = False

    for element in range(grid.number_of_elements):
        start_vertex, end_vertex = grid.elements[:, element]
        start = vertex_coordinates[start_vertex]
        end = vertex_coordinates[end_vertex]
        element_length = end - start
        if abs(element_length) <= tolerance:
            continue

        local_coordinate = (feed_coordinate - start) / element_length
        if -tolerance <= local_coordinate <= 1.0 + tolerance:
            local_coordinate = np.clip(local_coordinate, 0.0, 1.0)
            local_values = (1.0 - local_coordinate, local_coordinate)
            for local_index, value in enumerate(local_values):
                if space.local_multipliers[element, local_index] != 0.0:
                    dof = space.local2global[element, local_index]
                    # At a mesh vertex two elements find the same feed. Taking
                    # the maximum avoids counting the delta functional twice.
                    loading[dof] = max(loading[dof].real, value)
            feed_found = True

    if not feed_found or not np.any(loading):
        raise ValueError("The feed must lie in the interior of the wire.")

    return loading


def coefficients_at_vertices(grid, space, coefficients):
    """Map the PWL coefficients to mesh vertices, including zero endpoints."""
    current = np.zeros(grid.number_of_vertices, dtype=np.complex128)
    for element in range(grid.number_of_elements):
        for local_index in range(2):
            if space.local_multipliers[element, local_index] != 0.0:
                vertex = grid.elements[local_index, element]
                dof = space.local2global[element, local_index]
                current[vertex] = coefficients[dof]
    return current


def solve_dipole(
    number_of_elements,
    length=4.0,
    wavelength=8.0,
    wire_radius=0.01,
    feed_coordinate=0.0,
    regular_order=12,
    singular_order=24,
):
    """Solve the normalized delta-gap problem and return nodal current."""
    grid = straight_wire_grid(number_of_elements, length, wire_radius)

    # Endpoint degrees of freedom are excluded, enforcing I=0 at both ends.
    space = bempp.api.function_space(grid, "PWL", 0)
    expected_dof_count = number_of_elements - 1
    if space.global_dof_count != expected_dof_count:
        raise RuntimeError(
            "The imported Bempp package does not contain the thin-wire "
            "endpoint fix. Expected "
            f"{expected_dof_count} PWL dofs, found {space.global_dof_count}. "
            f"Bempp was imported from {bempp.__file__!r}. Install this "
            "checkout with `python -m pip install -e .` and rerun."
        )
    parameters = bempp.api.DefaultParameters()
    parameters.quadrature.regular = regular_order
    parameters.quadrature.singular = singular_order

    wavenumber = 2.0 * np.pi / wavelength
    operator = bempp.api.operators.boundary.maxwell.electric_field(
        space, space, space, wavenumber, parameters=parameters
    )
    matrix = operator.weak_form().to_dense()
    loading = delta_gap_load(grid, space, feed_coordinate)
    coefficients = np.linalg.solve(matrix, loading)

    return grid.vertices[2], coefficients_at_vertices(
        grid, space, coefficients
    )


def main():
    """Run and plot one dipole solve."""
    import matplotlib.pyplot as plt

    # An odd count deliberately places the feed inside an element. The source
    # projection above keeps it centred and symmetric nonetheless.
    coordinate, current = solve_dipole(number_of_elements=51)

    order = np.argsort(coordinate)
    plt.plot(coordinate[order], np.abs(current[order]))
    plt.xlabel("Wire coordinate")
    plt.ylabel("Current magnitude (arbitrary units)")
    plt.title("Thin-wire dipole current")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
