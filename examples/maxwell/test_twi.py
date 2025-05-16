import bempp.api
import numpy as np
import matplotlib.pyplot as plt

wavelength = 1
k = 2 * np.pi / wavelength

# grid_surface = bempp.api.import_grid("examples/maxwell/plane2.msh")

grid = bempp.api.import_grid("examples/maxwell/line_mesh.msh")

grid2 = bempp.api.import_grid("examples/maxwell/line_mesh_rotated.msh")
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

# spce_surface_domain = bempp.api.function_space(grid_surface, "RWG", 0)
# spce_surface_range = bempp.api.function_space(grid_surface, "SNC", 0)	
from types import SimpleNamespace

class _DenseAssembly(object):
    """Dense assembly options."""

    def __init__(self):
        """Iniitalize dense assembly parameters."""
        self.workgroup_size_multiple = 2

parameters = SimpleNamespace()
parameters.quadrature = SimpleNamespace()
parameters.quadrature.regular = 30
parameters.quadrature.singular = 30

parameters.assembly = SimpleNamespace()
parameters.assembly.dense = _DenseAssembly()
parameters.assembly.always_promote_to_double = False
parameters.assembly.discretization_type = "galerkin"

parameters.verbosity = SimpleNamespace()
parameters.verbosity.extended_verbosity = False

parameters.output = SimpleNamespace()
parameters.output.gmsh_use_binary = True

parameters.fmm = SimpleNamespace()
parameters.fmm.expansion_order = 5
parameters.fmm.depth = 4
parameters.fmm.ncrit = 400
parameters.fmm.near_field_representation = "evaluate"
parameters.fmm.debug = False
parameters.fmm.dense_evaluation = False




space = bempp.api.function_space(grid, "PWL", 0)
space2 = bempp.api.function_space(grid2, "PWL", 0)

elec = bempp.api.operators.boundary.maxwell.electric_field(space, space, space, k, parameters=parameters)
elec2 = bempp.api.operators.boundary.maxwell.electric_field(space2, space2, space2, k, parameters=parameters)
mat = elec.weak_form().to_dense()
mat2 = elec2.weak_form().to_dense()

print("difference between matrices =", mat - mat2)
print("difference between matrices norm =", np.linalg.norm(mat - mat2))
print("difference between matrices norm (2) =", np.linalg.norm(mat - mat2, ord=2))

mat_norm_ref = np.linalg.norm(mat)




testing_quad_order = np.arange(3, 30, 1)
norm_list = []

error_list = []

for order in testing_quad_order:
    parameters.quadrature.singular = order
    parameters.quadrature.regular = order

    elec = bempp.api.operators.boundary.maxwell.electric_field(space, space, space, k, parameters=parameters)
    mat = elec.weak_form().to_dense()
    mat_norm = np.linalg.norm(mat)
    norm_list.append(mat_norm)

    error = np.abs(mat_norm - mat_norm_ref)/mat_norm_ref
    error_list.append(error)


plt.plot(testing_quad_order, norm_list)
plt.xlabel("Quadrature Order")
plt.ylabel("Matrix Norm")
plt.title("Matrix Norm vs Quadrature Order")

plt.figure()
plt.plot(testing_quad_order, error_list)
plt.xlabel("Quadrature Order")
plt.ylabel("Error")
plt.title("Error vs Quadrature Order")
plt.yscale("log")
plt.grid()
plt.tight_layout()


plt.show()



# elec_surface = bempp.api.operators.boundary.maxwell.electric_field(spce_surface_domain, spce_surface_domain, spce_surface_range, k)

# mat_surface = elec_surface.weak_form().to_dense()
# print("surface matrix =", mat_surface)


mat = elec.weak_form().to_dense()
print("matrix =", mat)

mat_norm = np.linalg.norm(mat)
print("matrix norm =", mat_norm)


# print("line elements =", line_elements)
# print("junction elements =", junc_elements)

# print("all elements =", grid.elements)
# print("grid type =", grid.type)

# print("efie operator =", elec)