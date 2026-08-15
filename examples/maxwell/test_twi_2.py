"""Demonstrate convergence of the thin-wire dipole current."""

import matplotlib.pyplot as plt
import numpy as np

try:
    from examples.maxwell.thin_wire_dipole import solve_dipole
except ModuleNotFoundError:  # Support ``python examples/maxwell/test_twi_2.py``.
    from thin_wire_dipole import solve_dipole


def main():
    """Solve successive meshes and report changes on a common coordinate grid."""
    element_counts = (8, 16, 32, 64)
    comparison_points = np.linspace(-1.9, 1.9, 501)
    previous_current = None

    for number_of_elements in element_counts:
        coordinate, current = solve_dipole(number_of_elements)
        interpolated_current = np.interp(
            comparison_points, coordinate, current.real
        ) + 1j * np.interp(comparison_points, coordinate, current.imag)

        if previous_current is not None:
            relative_change = np.linalg.norm(
                interpolated_current - previous_current
            ) / np.linalg.norm(interpolated_current)
            print(
                f"{number_of_elements:3d} elements: "
                f"relative change = {relative_change:.3e}"
            )

        plt.plot(
            coordinate,
            np.abs(current),
            marker=".",
            label=f"{number_of_elements} elements",
        )
        previous_current = interpolated_current

    plt.xlabel("Wire coordinate")
    plt.ylabel("Normalized current magnitude")
    plt.title("Thin-wire dipole mesh convergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
