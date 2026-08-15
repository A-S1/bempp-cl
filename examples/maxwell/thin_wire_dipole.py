"""Solve a normalized delta-gap problem on a straight thin-wire dipole."""

import matplotlib.pyplot as plt
import numpy as np

import bempp.api
from bempp.api.grid import LineGrid


length = 4.0
wavelength = 8.0
wavenumber = 2.0 * np.pi / wavelength
wire_radius = 0.01
number_of_elements = 50

z = np.linspace(-0.5 * length, 0.5 * length, number_of_elements + 1)
vertices = np.vstack([np.zeros_like(z), np.zeros_like(z), z])
elements = np.vstack(
    [np.arange(number_of_elements), np.arange(1, number_of_elements + 1)]
).astype(np.uint32)
grid = LineGrid(vertices, elements, wire_radius=wire_radius)

# Endpoint degrees of freedom are excluded by default, enforcing I=0 at both
# ends of the open PEC wire.
space = bempp.api.function_space(grid, "PWL", 0)
parameters = bempp.api.DefaultParameters()
parameters.quadrature.regular = 12
parameters.quadrature.singular = 24

operator = bempp.api.operators.boundary.maxwell.electric_field(
    space, space, space, wavenumber, parameters=parameters
)
matrix = operator.weak_form().to_dense()

loading = np.zeros(space.global_dof_count, dtype=np.complex128)
loading[space.global_dof_count // 2] = 1.0
coefficients = np.linalg.solve(matrix, loading)

# Map the interior PWL coefficients back to their geometric vertices for a
# direct plot. The mesh vertex numbering need not follow the wire coordinate.
vertex_current = np.zeros(grid.number_of_vertices, dtype=np.complex128)
for element in range(grid.number_of_elements):
    for local_index in range(2):
        if space.local_multipliers[element, local_index] != 0.0:
            vertex = grid.elements[local_index, element]
            dof = space.local2global[element, local_index]
            vertex_current[vertex] = coefficients[dof]

order = np.argsort(grid.vertices[2])
plt.plot(grid.vertices[2, order], np.abs(vertex_current[order]))
plt.xlabel("Wire coordinate")
plt.ylabel("Normalized current magnitude")
plt.title("Thin-wire dipole current")
plt.grid(True)
plt.tight_layout()
plt.show()
