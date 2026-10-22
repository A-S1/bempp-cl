# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Plane-wave scattering from a spherical cage of wires
#
# The cage consists of electrically disjoint latitude loops on a common
# sphere. Keeping the loops disjoint makes this a pure wire-wire interaction
# example; no artificial current-continuity condition is imposed at crossings.
# A plane wave travels in the positive z direction and is x-polarized.

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import bempp.api

try:
    from examples.maxwell.thin_wire_dipole import coefficients_at_vertices
    from examples.maxwell.thin_wire_geometries import (
        latitude_wire_sphere_grid,
        plane_wave_load,
    )
except ModuleNotFoundError:  # Support direct execution from this directory.
    from thin_wire_dipole import coefficients_at_vertices
    from thin_wire_geometries import latitude_wire_sphere_grid, plane_wave_load


def solve_wire_sphere(
    number_of_latitudes=7,
    elements_per_loop=24,
    sphere_radius=1.0,
    wavelength=4.0,
    wire_radius=0.01,
):
    """Solve the normalized plane-wave problem on a spherical wire cage."""
    grid = latitude_wire_sphere_grid(
        number_of_latitudes,
        elements_per_loop,
        sphere_radius,
        wire_radius,
    )
    space = bempp.api.function_space(grid, "PWL", 0)
    wavenumber = 2.0 * np.pi / wavelength
    parameters = bempp.api.DefaultParameters()
    parameters.quadrature.regular = 10
    parameters.quadrature.singular = 20
    operator = bempp.api.operators.boundary.maxwell.electric_field(
        space,
        space,
        space,
        wavenumber,
        parameters=parameters,
    )
    matrix = operator.weak_form().to_dense()
    loading = plane_wave_load(
        grid,
        space,
        wavenumber,
        direction=[0.0, 0.0, 1.0],
        polarization=[1.0, 0.0, 0.0],
    )
    coefficients = np.linalg.solve(matrix, loading)
    return grid, coefficients_at_vertices(grid, space, coefficients)


def main():
    """Run the spherical-cage example and plot the current magnitude."""
    grid, current = solve_wire_sphere()

    try:
        get_ipython().run_line_magic("matplotlib", "inline")
        ipython = True
    except NameError:
        ipython = False

    segments = np.stack(
        [
            grid.vertices[:, grid.elements[0]].T,
            grid.vertices[:, grid.elements[1]].T,
        ],
        axis=1,
    )
    segment_current = 0.5 * (
        np.abs(current[grid.elements[0]])
        + np.abs(current[grid.elements[1]])
    )

    fig = plt.figure(figsize=(8, 7))
    axis = fig.add_subplot(111, projection="3d")
    collection = Line3DCollection(segments, cmap="viridis", linewidths=3.0)
    collection.set_array(segment_current)
    axis.add_collection3d(collection)
    extent = 1.05 * np.max(np.abs(grid.vertices))
    axis.set_xlim(-extent, extent)
    axis.set_ylim(-extent, extent)
    axis.set_zlim(-extent, extent)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_title("Current magnitude on a spherical wire cage")
    fig.colorbar(collection, ax=axis, shrink=0.7, label="Current magnitude")
    fig.tight_layout()

    if not ipython:
        plt.savefig("example-thin_wire_sphere.png", dpi=160)


if __name__ == "__main__":
    main()
