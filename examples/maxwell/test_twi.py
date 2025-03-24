import bempp.api
import numpy as np
import matplotlib.pyplot as plt

corners1 = np.array([[-0.5, -1, 0], [-0.5, 1, 0], [-2, 1, 2], [-2, -1, 2]])
corners2 = np.array([[0.5, -1, 0], [0.5, 1, 0], [2, 1, 2], [2, -1, 2]])
corners3 = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]])

wire_1 = np.array([[-0.5, -1, 0], [-0.5, 1, 1]])

grid1 = bempp.api.shapes.screen(corners1)
grid2 = bempp.api.shapes.screen(corners2)
grid3 = bempp.api.shapes.screen(corners3)

grid4 = bempp.api.shapes.wire(wire_1, 0.1, 0.1)
grid5 = bempp.api.grid.union([grid1, grid2, grid3])
grid = bempp.api.grid.union([grid5, grid4])

# grid5.plot()
# grid4.plot()
# grid.plot()	

if hasattr(grid, 'line_mask'):
    line_elements = grid.elements[:, grid.line_mask]
else:
    # Otherwise, assume grid4 is the wire grid and use that.
    line_elements = grid4.elements

# Create the PWL function space on the wire grid.
# This function space should be similar to what we defined as pwl0_function_space.
# For example:
wire_space = bempp.api.function_space(grid4, "PWL", 0)

# For testing, we evaluate the wire space's basis functions on the first (and perhaps only) wire element.
# Assume we can access the underlying grid data via wire_space.grid_data.
# We also assume that our PWL evaluator returns a (3, 2, npoints) array.
# We choose local evaluation points in the parametric coordinate [0,1]:
local_points = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])

# Evaluate the basis functions on the first element (element index 0).
# The function _numba_pwl0_evaluate should be linked as the evaluator for the space.
eval_result = wire_space.numba_evaluator(
    0,  # element index
    None,  # shapeset_evaluate placeholder (unused for PWL)
    local_points,
    wire_space.grid_data,
    wire_space.local_multipliers,
    wire_space.normal_multipliers
)

# Print the evaluated basis functions.
print("Evaluated basis functions on wire element 0 (vector components for each basis function):")
print(eval_result)

# Plot the scalar coefficients along the wire for visual verification.
# For a straight segment from (-0.5,-1,0) to (-0.5,1,1) the tangent vector is computed internally.
# We extract the x-component (or norm) of the basis functions for each dof.
# Since our evaluator returns a 3x2xnpoints array, where the second index corresponds to the two basis functions,
# we plot the magnitude (norm) of each basis function along the wire.

npoints = local_points.shape[1]
s_vals = local_points[0, :]

basis0 = eval_result[:, 0, :]  # shape (3, npoints)
basis1 = eval_result[:, 1, :]  # shape (3, npoints)
mag_basis0 = np.linalg.norm(basis0, axis=0)
mag_basis1 = np.linalg.norm(basis1, axis=0)

plt.figure()
plt.plot(s_vals, mag_basis0, label='Basis Function at Wire Node 0')
plt.plot(s_vals, mag_basis1, label='Basis Function at Wire Node 1')
plt.xlabel('Local coordinate s (0 to 1)')
plt.ylabel('Magnitude of vector basis function')
plt.legend()
plt.title('PWL Basis Functions on a Wire Element')
plt.grid(True)
plt.show()

print("PWL function space test completed successfully.")