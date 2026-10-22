# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Differential excitation of a two-wire transmission line
#
# This example uses the thin-wire EFIE to excite two identical parallel
# conductors with equal and opposite delta-gap loads. Symmetry requires the
# two current distributions to have equal magnitude and opposite phase. The
# common-mode residual is therefore a compact validation metric for mutual
# wire coupling.

import matplotlib.pyplot as plt
import numpy as np

import bempp.api

try:
    from examples.maxwell.thin_wire_dipole import (
        coefficients_at_vertices,
        delta_gap_load,
    )
    from examples.maxwell.thin_wire_geometries import (
        domain_vertex_data,
        parallel_wire_grid,
    )
except ModuleNotFoundError:  # Support direct execution from this directory.
    from thin_wire_dipole import coefficients_at_vertices, delta_gap_load
    from thin_wire_geometries import domain_vertex_data, parallel_wire_grid


def solve_transmission_line(
    number_of_elements=24,
    length=2.0,
    separation=0.2,
    wavelength=4.0,
    wire_radius=0.01,
):
    """Solve a normalized differential two-wire problem."""
    grid = parallel_wire_grid(
        number_of_elements,
        length,
        separation,
        wire_radius,
    )
    space = bempp.api.function_space(grid, "PWL", 0)
    wavenumber = 2.0 * np.pi / wavelength
    parameters = bempp.api.DefaultParameters()
    parameters.quadrature.regular = 12
    parameters.quadrature.singular = 24

    operator = bempp.api.operators.boundary.maxwell.electric_field(
        space,
        space,
        space,
        wavenumber,
        parameters=parameters,
    )
    matrix = operator.weak_form().to_dense()
    loading = delta_gap_load(grid, space, domain_index=0)
    loading += delta_gap_load(
        grid,
        space,
        domain_index=1,
        amplitude=-1.0,
    )
    coefficients = np.linalg.solve(matrix, loading)
    current = coefficients_at_vertices(grid, space, coefficients)

    first_coordinates, first_current = domain_vertex_data(grid, current, 0)
    second_coordinates, second_current = domain_vertex_data(grid, current, 1)
    common_mode_ratio = np.linalg.norm(first_current + second_current) / np.linalg.norm(
        first_current - second_current
    )
    return (
        first_coordinates[2],
        first_current,
        second_current,
        common_mode_ratio,
    )


def main():
    """Run the transmission-line example and plot both conductor currents."""
    coordinate, first_current, second_current, common_mode_ratio = (
        solve_transmission_line()
    )
    print(f"Common-mode residual: {common_mode_ratio:.3e}")

    try:
        get_ipython().run_line_magic("matplotlib", "inline")
        ipython = True
    except NameError:
        ipython = False

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    axes[0].plot(coordinate, np.abs(first_current), "o-", label="wire 1")
    axes[0].plot(coordinate, np.abs(second_current), ".--", label="wire 2")
    axes[0].set_ylabel("Current magnitude")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(coordinate, np.unwrap(np.angle(first_current)), "o-")
    axes[1].plot(coordinate, np.unwrap(np.angle(second_current)), ".--")
    axes[1].set_xlabel("Axial coordinate")
    axes[1].set_ylabel("Current phase [rad]")
    axes[1].grid(True)
    fig.suptitle("Differential two-wire transmission line")
    fig.tight_layout()

    if not ipython:
        plt.savefig("example-thin_wire_transmission_line.png", dpi=160)


if __name__ == "__main__":
    main()
