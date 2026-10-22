# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Champagne wire-to-surface junction basis
#
# Champagne, Johnson, and Wilton construct one composite basis for a wire
# attached to a triangulated conductor. Its surface part has the required
# $1/r$ current variation at the attachment point, while its wire part is a
# rooftop on the first segment. The integrated divergences are +1 and -1,
# respectively, so the composite basis conserves charge.

import matplotlib.pyplot as plt
import numpy as np

from bempp.api.grid import Grid, LineGrid
from bempp.api.space import find_champagne_junctions


def square_plate_and_wire():
    """Create a four-triangle plate fan and one attached wire segment."""
    vertices = np.array(
        [
            [0.0, 1.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    elements = np.array(
        [[0, 0, 0, 0], [1, 2, 3, 4], [2, 3, 4, 1]],
        dtype=np.uint32,
    )
    surface = Grid(vertices, elements)
    wire = LineGrid(
        np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.75]]),
        np.array([[0], [1]], dtype=np.uint32),
        wire_radius=0.01,
    )
    return surface, wire


def main():
    """Visualize the surface part and report discrete charge balance."""
    surface, wire = square_plate_and_wire()
    junction = find_champagne_junctions(surface, wire)[0]
    print("Surface flux weights:", [data.flux_weight for data in junction.triangles])
    print(f"Integrated surface divergence: {junction.integrated_surface_divergence:.6f}")
    print(f"Integrated wire divergence: {junction.integrated_wire_divergence:.6f}")
    print(f"Composite charge balance: {junction.charge_balance:.3e}")

    centroids = surface.centroids.T
    values = np.column_stack(
        [
            junction.surface_value(element, centroids[:, element])
            for element in junction.surface_elements
        ]
    )

    try:
        get_ipython().run_line_magic("matplotlib", "inline")
        ipython = True
    except NameError:
        ipython = False

    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_trisurf(
        surface.vertices[0],
        surface.vertices[1],
        surface.vertices[2],
        triangles=surface.elements.T,
        alpha=0.25,
        color="tab:blue",
        edgecolor="k",
    )
    axis.quiver(
        centroids[0],
        centroids[1],
        centroids[2],
        values[0],
        values[1],
        values[2],
        length=0.15,
        normalize=True,
        color="tab:red",
    )
    axis.plot(wire.vertices[0], wire.vertices[1], wire.vertices[2], "k-", lw=4)
    axis.set_box_aspect((1, 1, 0.6))
    axis.set_title("Champagne composite junction basis")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    figure.tight_layout()

    if not ipython:
        plt.savefig("example-champagne_junction_basis.png", dpi=160)


if __name__ == "__main__":
    main()
