import bempp.api
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

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
parameters.quadrature.regular = 15
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

grid_list = [grid7, grid11, grid15, grid21, grid31, grid51]
max_list = []

wavelength_list = [8] # Different wave numbers for each grid

plt.figure(figsize=(10, 6))
for i, grid in enumerate(grid_list):
#### Define space of Piecewise Linear Functions, with 0 at the boundaries  #### 
    for j in range(len(wavelength_list)):

        
        wvl = wavelength_list[j]  # Use the corresponding wave number for the grid
        k = 2 * np.pi / wvl  # Convert wavelength to wave number
        print(f"Processing grid {i+1} with wave number {k:.2f}")
        space = bempp.api.function_space(grid, "PWL", 0, include_boundary_dofs=True)	


        coeffs = np.zeros(space.global_dof_count-2, dtype=np.complex128)
        coeffs[(space.global_dof_count-2)//2] =  1/np.max(grid.diameters)  # Set the central impulse


        coeffs2 = np.zeros(space.global_dof_count, dtype=np.complex128)
        coeffs2[1:-1] = coeffs.copy()  # Copy coefficients to the interior dofs
        coeffs2[0] = 0  # Set the first value to zero
        coeffs2[-1] = 0  # Set the last value

        elec = bempp.api.operators.boundary.maxwell.electric_field(space, space, space, k, parameters=parameters)
        rhs = bempp.api.GridFunction(space, coefficients=coeffs2, parameters=parameters)


        #### Solve the system using LU decomposition ####
        from bempp.api.linalg import lu
        N = space.global_dof_count
        h = 2.0/(N-1)

        mat = elec.weak_form().to_dense()
        mat = mat[2:, 2:]  # Extract the relevant part of the matrix

        lambda_data = np.linalg.solve(mat, coeffs)


        plot_data_1 = np.zeros(space.global_dof_count, dtype=np.complex128)
        plot_data_1[1:-1] = lambda_data.copy()
        plot_data_1[0] = 0
        plot_data_1[-1] = 0  # Set the last value to zero for plotting

        max_list.append(np.max(np.abs(lambda_data)))

        element_size = np.max(grid.diameters)

        
        x_nodes = np.linspace(-1, 1, N)
        x_plot = np.linspace(-1, 1, 500)

        # Compute weighted sum of hat functions
        y = np.zeros_like(x_plot, dtype=complex)
        for n in range(N):
            f_n = np.maximum((1 - np.abs((x_plot - x_nodes[n])) / h), 0) * h 
            y += plot_data_1[n] * f_n

        # Absolute real and imaginary parts
        y_real_abs =  np.abs(y.real)
        y_imag_abs =  np.abs(y.imag)

        y_mag = np.abs(y)
        fractional_wavelength = Fraction(4,wvl)

        # Plot
        plt.plot(x_plot, y_mag, label= f"L= {fractional_wavelength} λ, grid {i+1}", linewidth=2)

        reactance = 1/np.max(grid.diameters) / np.max(y_imag_abs) 
        resistance = 1/np.max(grid.diameters) / np.max(y_real_abs)

        print(f"Reactance for grid {i+1} with {fractional_wavelength}: {reactance:.4f}")
        print(f"Resistance for grid {i+1} with {fractional_wavelength}: {resistance:.4f}")
       

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


z = np.linspace(-2, 2, 500)
I =  y

# Sample theta for the cut
theta = np.linspace(0, np.pi, 500)

# Compute the far-field pattern F(θ) via numerical integration
F = np.array([np.trapz(I * np.exp(1j * k * z * np.cos(t)), z)*np.sin(t) for t in theta])

F_norm = np.abs(F) / np.max(np.abs(F))
# F_norm *= F_norm

P_db_down = -20*np.log10(F_norm)        #  dB
P_db_down = np.minimum(P_db_down, 40)

θ_full = np.concatenate((theta, theta+np.pi))
P_full = np.concatenate((P_db_down, P_db_down))

# --- 3) plot exactly like Balanis Fig 4.6 λ/2 curve ---
fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
ax.plot(θ_full, np.concatenate((F_norm, F_norm)), 'k-', linewidth=2)    # solid black for λ/2
ax.set_theta_zero_location('N')               # 0° at top
ax.set_theta_direction(-1)                    # increase clockwise
ax.set_rmax(0)                                # outer = 0 dB
ax.set_rmin(40)                               # inner = 40 dB down
ax.set_rticks([10,20,30])                     # 10,20,30 dB circles
# ax.set_xticklabels(['10','20','30'])
ax.set_title('λ/2 Dipole Elevation Plane (power, dB down)', va='bottom')
plt.show()
# nx = 300
# nz = 300
# extent = 3
# x, y, z = np.mgrid[-extent : extent : nx * 1j, 0:0:1j, -extent : extent : nz * 1j]
# points = np.vstack((x.ravel(), y.ravel(), z.ravel()))
# # -

points = [5, 0, 0]
# # We now initialise the electric field potential operator.
current = lu(elec, rhs)
slp_pot = bempp.api.operators.potential.maxwell.electric_field(space, points, k)
scattered_field = -slp_pot * current
scattered_field_squared = scattered_field**2
print("Scattered field shape:", scattered_field)
print("max scattered field value:", np.max(np.abs(scattered_field)))
# scattered_field = scattered_field.evaluate(points)'
# Reshape to match the grid dimensions
#plot the values at each point
plt.figure(figsize=(10, 6))
plt.imshow(scattered_field_squared, origin='lower', cmap='viridis')
plt.colorbar(label='|Scattered Field|')
plt.title('Scattered Field Magnitude')
plt.xlabel('x')
plt.ylabel('z')
plt.grid(True)
plt.tight_layout()
plt.show()
