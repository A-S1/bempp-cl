import bempp.api
import numpy as np
import matplotlib.pyplot as plt

wavelength = 1
k = 2 * np.pi / wavelength

grid = bempp.api.import_grid("examples/maxwell/line_mesh.msh")
# grid.plot()

if hasattr(grid, 'line_mask'):
    line_elements = grid.elements[:, grid.line_mask]
else:
    # Otherwise, assume grid4 is the wire grid and use that.
    line_elements = None

if hasattr(grid, 'junction_mask'):
    junc_elements = grid.elements[:, grid.junction_mask]
else:
    junc_elements = None

space = bempp.api.function_space(grid, "PWL", 0)
elec = bempp.api.operators.boundary.maxwell.electric_field(space, space, space, k)


print("line elements =", line_elements)
print("junction elements =", junc_elements)

print("all elements =", grid.elements)
print("grid type =", grid.type)

print("efie operator =", elec)