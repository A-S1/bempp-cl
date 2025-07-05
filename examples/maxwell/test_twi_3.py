import bempp.api
import numpy as np
import matplotlib.pyplot as plt

wavelength = 50
k = 2 * np.pi / wavelength

grid = bempp.api.import_grid("examples/maxwell/transmission_line_mesh.msh")

space = bempp.api.function_space(grid, "PWL", 0)
elec = bempp.api.operators.boundary.maxwell.electric_field(space, space, space, k)

mat_elec = elec.weak_form().to_dense()

identity = bempp.api.operators.boundary.sparse.identity(space, space, space)
mat_id = identity.weak_form().to_dense()

coeffs_impedance = np.zeros(space.global_dof_count, dtype=np.complex128)
coeffs_impedance[0] = 1
coeffs_impedance[3] = 1

coeffs_impedance[1] = -1
coeffs_impedance[2] = -1

impedance_mat = np.zeros((space.global_dof_count, space.global_dof_count), dtype=np.complex128)
for i in range(space.global_dof_count):
    if coeffs_impedance[i] != 0:
        impedance_mat[i, :] = mat_id[i,:] * coeffs_impedance[i]
    
mat_system = mat_elec - impedance_mat

potential_loading = np.ones(space.global_dof_count, dtype=np.complex128)

lambda_data = np.linalg.solve(mat_system, coeffs_impedance)

print("Impedance matrix size:", impedance_mat.shape)
print("Impeedance Matrix:\n", impedance_mat)

print("Matrix size:", mat_elec.shape, "\n Identity size:", mat_id.shape)
