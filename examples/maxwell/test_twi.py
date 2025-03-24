import bempp.api
import numpy as np

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

grid5.plot()
grid4.plot()
grid.plot()	

