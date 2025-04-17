import bempp.api
import numpy as np
import matplotlib.pyplot as plt

wavelength = 1
k = 2 * np.pi / wavelength

grid_surface = bempp.api.import_grid("examples/maxwell/plane2.msh")

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

spce_surface_domain = bempp.api.function_space(grid_surface, "RWG", 0)
spce_surface_range = bempp.api.function_space(grid_surface, "SNC", 0)	

space = bempp.api.function_space(grid, "PWL", 0)
elec = bempp.api.operators.boundary.maxwell.electric_field(space, space, space, k)
elec_surface = bempp.api.operators.boundary.maxwell.electric_field(spce_surface_domain, spce_surface_domain, spce_surface_range, k)

# mat_surface = elec_surface.weak_form().to_dense()
# print("surface matrix =", mat_surface)


mat = elec.weak_form().to_dense()


print("line elements =", line_elements)
print("junction elements =", junc_elements)

print("all elements =", grid.elements)
print("grid type =", grid.type)

print("efie operator =", elec)