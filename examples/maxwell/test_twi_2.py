import bempp.api
import numpy as np
import matplotlib.pyplot as plt

wavelength = 30
k = 2 * np.pi / wavelength

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

#### Straight wire, define the grid ####
grid = bempp.api.import_grid("examples/maxwell/line_mesh_long.msh")

#### Define space of Piecewise Linear Functions, with 0 at the boundaries  ####
space = bempp.api.function_space(grid, "PWL", 0)

#### Assemble the Matrix Z as a dens EFIE operator and the RHS with a central impulse trace function ####
@bempp.api.complex_callable
def trace_function(x, n, domain_index, result):
    """Trace function for the wire."""

    value = 1 if np.isclose(x[0], 0) else 0
    result[2] = value  # Set the third component for the wire trace

coeffs = [ 0, 0, 1, 0, 0]

elec = bempp.api.operators.boundary.maxwell.electric_field(space, space, space, k, parameters=parameters)
rhs = bempp.api.GridFunction(space, coefficients=coeffs, parameters=parameters)


#### Solve the system using LU decomposition ####
from bempp.api.linalg import lu

mat = elec.weak_form().to_dense()
# print("Matrix shape:", mat)
print("Is symmetric:", np.allclose(mat, mat.T))
# print("error:", np.abs(mat - mat.T))

lambda_data = np.linalg.solve(mat, coeffs)

# Print the solution
print("Lambda data (solution):", lambda_data)

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(np.abs(lambda_data), label='Solution Lambda Data')
plt.title('Solution of the Maxwell Electric Field Boundary Operator')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()

