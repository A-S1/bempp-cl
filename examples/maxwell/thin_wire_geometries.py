"""Reusable meshes and loads for thin-wire tutorial examples."""

import numpy as np

from bempp.api.grid import LineGrid


def parallel_wire_grid(
    number_of_elements=24,
    length=2.0,
    separation=0.2,
    wire_radius=0.01,
):
    """Create two parallel, equally oriented open wires.

    Domain indices 0 and 1 identify the two conductors. Keeping the element
    orientation equal makes a differential excitation appear as equal and
    opposite coefficient vectors.
    """
    if number_of_elements < 2:
        raise ValueError("number_of_elements must be at least two")
    if length <= 0.0 or separation <= 0.0 or wire_radius <= 0.0:
        raise ValueError("length, separation, and wire_radius must be positive")

    coordinate = np.linspace(
        -0.5 * length, 0.5 * length, number_of_elements + 1
    )
    first = np.vstack(
        [
            np.full_like(coordinate, -0.5 * separation),
            np.zeros_like(coordinate),
            coordinate,
        ]
    )
    second = first.copy()
    second[0] = 0.5 * separation
    vertices = np.hstack([first, second])

    first_elements = np.vstack(
        [np.arange(number_of_elements), np.arange(1, number_of_elements + 1)]
    )
    offset = number_of_elements + 1
    second_elements = first_elements + offset
    elements = np.hstack([first_elements, second_elements]).astype(np.uint32)
    domains = np.repeat([0, 1], number_of_elements).astype(np.uint32)
    return LineGrid(
        vertices,
        elements,
        domain_indices=domains,
        wire_radius=wire_radius,
    )


def latitude_wire_sphere_grid(
    number_of_latitudes=7,
    elements_per_loop=32,
    sphere_radius=1.0,
    wire_radius=0.01,
    polar_fraction=0.8,
):
    """Create a spherical cage from disjoint closed latitude loops.

    The loops are separate conductors. This avoids introducing artificial
    current-continuity constraints at wire crossings before a network-junction
    space is available.
    """
    if number_of_latitudes < 1:
        raise ValueError("number_of_latitudes must be positive")
    if elements_per_loop < 3:
        raise ValueError("elements_per_loop must be at least three")
    if sphere_radius <= 0.0 or wire_radius <= 0.0:
        raise ValueError("sphere_radius and wire_radius must be positive")
    if not 0.0 < polar_fraction < 1.0:
        raise ValueError("polar_fraction must lie between zero and one")

    heights = np.linspace(
        -polar_fraction * sphere_radius,
        polar_fraction * sphere_radius,
        number_of_latitudes,
    )
    angles = 2.0 * np.pi * np.arange(elements_per_loop) / elements_per_loop
    vertices = []
    elements = []
    domains = []
    vertex_offset = 0

    for domain, height in enumerate(heights):
        loop_radius = np.sqrt(sphere_radius**2 - height**2)
        loop = np.vstack(
            [
                loop_radius * np.cos(angles),
                loop_radius * np.sin(angles),
                np.full(elements_per_loop, height),
            ]
        )
        vertices.append(loop)
        start = vertex_offset + np.arange(elements_per_loop)
        end = vertex_offset + np.roll(np.arange(elements_per_loop), -1)
        elements.append(np.vstack([start, end]))
        domains.append(np.full(elements_per_loop, domain, dtype=np.uint32))
        vertex_offset += elements_per_loop

    return LineGrid(
        np.hstack(vertices),
        np.hstack(elements).astype(np.uint32),
        domain_indices=np.concatenate(domains),
        wire_radius=wire_radius,
    )


def plane_wave_load(
    grid,
    space,
    wavenumber,
    direction,
    polarization,
    quadrature_order=8,
):
    """Project a plane-wave electric field onto a PWL line space."""
    direction = np.asarray(direction, dtype=np.float64)
    polarization = np.asarray(polarization, dtype=np.complex128)
    direction /= np.linalg.norm(direction)
    if abs(np.vdot(direction, polarization)) > 1.0e-12:
        raise ValueError("polarization must be transverse to direction")

    points, weights = np.polynomial.legendre.leggauss(quadrature_order)
    points = 0.5 * (points + 1.0)
    weights *= 0.5
    loading = np.zeros(space.global_dof_count, dtype=np.complex128)

    for element in range(grid.number_of_elements):
        start = grid.vertices[:, grid.elements[0, element]]
        end = grid.vertices[:, grid.elements[1, element]]
        segment = end - start
        length = np.linalg.norm(segment)
        tangent = segment / length
        global_points = start[:, None] + segment[:, None] * points
        phase = np.exp(1j * wavenumber * (direction @ global_points))
        tangential_field = np.dot(tangent, polarization) * phase
        local_shapes = (1.0 - points, points)

        for local_index, shape in enumerate(local_shapes):
            multiplier = space.local_multipliers[element, local_index]
            if multiplier == 0.0:
                continue
            dof = space.local2global[element, local_index]
            loading[dof] += (
                multiplier
                * length
                * np.dot(weights, shape * tangential_field)
            )

    return loading


def domain_vertex_data(grid, values, domain_index):
    """Return coordinates and nodal values belonging to one line domain."""
    element_indices = np.flatnonzero(grid.domain_indices == domain_index)
    vertices = np.unique(grid.elements[:, element_indices])
    return grid.vertices[:, vertices], np.asarray(values)[vertices]
