import bempp.api
import numpy as np
import matplotlib.pyplot as plt

wavelength = 50
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
grid7 = bempp.api.import_grid("examples/maxwell/line_mesh_long_7.msh")
grid11 = bempp.api.import_grid("examples/maxwell/line_mesh_long_11.msh")
grid15 = bempp.api.import_grid("examples/maxwell/line_mesh_long_15.msh")
grid21 = bempp.api.import_grid("examples/maxwell/line_mesh_long_21.msh")
grid31 = bempp.api.import_grid("examples/maxwell/line_mesh_long_31.msh")
grid51 = bempp.api.import_grid("examples/maxwell/line_mesh_long_51.msh")

grid_list = [grid7]#, grid11, grid15, grid21, grid31, grid51]
max_list = []

plt.figure(figsize=(10, 6))
for i, grid in enumerate(grid_list):
#### Define space of Piecewise Linear Functions, with 0 at the boundaries  ####
    space = bempp.api.function_space(grid, "PWL", 0, include_boundary_dofs=True)	


    coeffs = np.zeros(space.global_dof_count-2, dtype=np.complex128)
    coeffs[(space.global_dof_count-2)//2] =  1/np.max(grid.diameters)  # Set the central impulse



    elec = bempp.api.operators.boundary.maxwell.electric_field(space, space, space, k, parameters=parameters)
    rhs = bempp.api.GridFunction(space, coefficients=coeffs, parameters=parameters)


    #### Solve the system using LU decomposition ####
    from bempp.api.linalg import lu

    mat = elec.weak_form().to_dense()
    mat = mat[2:, 2:]  # Extract the relevant part of the matrix

    lambda_data = np.linalg.solve(mat, coeffs)

    plot_data_1 = np.zeros(space.global_dof_count, dtype=np.complex128)
    plot_data_1[1:-1] = lambda_data.copy()
    plot_data_1[0] = 0
    plot_data_1[-1] = 0  # Set the last value to zero for plotting

    max_list.append(np.max(np.abs(lambda_data)))

    element_size = np.max(grid.diameters)

    N = len(plot_data_1)
    h = 2.0/(N-1)
    x_nodes = np.linspace(-1, 1, N)
    x_plot = np.linspace(-1, 1, 500)

    # Compute weighted sum of hat functions
    y = np.zeros_like(x_plot, dtype=complex)
    for n in range(N):
        f_n = np.maximum((1 - np.abs((x_plot - x_nodes[n])) / h), 0)
        y += plot_data_1[n] * f_n

    # Absolute real and imaginary parts
    y_real_abs =  np.abs(y.real)
    y_imag_abs =  np.abs(y.imag)

    # max_value = np.max(y_real_abs)
    # y_real_abs /= max_value

    # Plot
    plt.plot(x_plot, y_real_abs, label='|Re(I(z))|', linewidth=2)
    plt.plot(x_plot, y_imag_abs, label='|Im(I(z))|', linestyle='--', linewidth=2)

plt.xlim(-1, 1)
plt.ylim(bottom=0)
plt.xlabel('z')
plt.ylabel('current')
plt.title('Current Distribution for Small Dipole Antenna')
plt.legend()
plt.grid(True)
plt.tight_layout()

    # Refine the grid for better resolution
# plt.show()

nx = 300
nz = 300
extent = 3
x, y, z = np.mgrid[-extent : extent : nx * 1j, 0:0:1j, -extent : extent : nz * 1j]
points = np.vstack((x.ravel(), y.ravel(), z.ravel()))
# -

# We now initialise the electric field potential operator.

slp_pot = bempp.api.operators.potential.maxwell.electric_field(space, points, k)


print("Max values for each grid size:", max_list)
