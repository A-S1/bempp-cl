"""Maxwell potential operators."""
import numpy as _np


def electric_field(
    space,
    points,
    wavenumber,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Maxwell electric field potential operator."""
    from bempp.api.operators import OperatorDescriptor
    from bempp.api.assembly.potential_operator import PotentialOperator
    from bempp.api.assembly.assembler import PotentialAssembler
    import bempp.api


    if precision is None:
        precision = bempp.api.DEFAULT_PRECISION

    if "triangle" in space.grid.type.lower():
        if space.identifier != "rwg0":
            raise ValueError("Space must be an RWG type function space for triangular elements.")

        operator_descriptor = OperatorDescriptor(
            "maxwell_electric_field_potential",  # Identifier
            [_np.real(wavenumber), _np.imag(wavenumber)],  # Options
            "helmholtz_single_layer",  # Kernel type
            "maxwell_electric_field",  # Assembly type
            precision,  # Precision
            True,  # Is complex
            None,  # Singular part
            3,  # Kernel dimension
        )

        return PotentialOperator(
            PotentialAssembler(
                space, points, operator_descriptor, device_interface, assembler, parameters
            )
        )
    
    elif "line" in space.grid.type.lower():
        if space.identifier != "pwl0":
            raise ValueError("Space must be an PWL type function space for line elements.")

        operator_descriptor = OperatorDescriptor(
            "maxwell_electric_field_potential",  # Identifier
            [_np.real(wavenumber), _np.imag(wavenumber)],  # Options
            "thinwire_helmholtz_potential",  # Kernel type
            "maxwell_electric_field_thinwire",  # Assembly type
            precision,  # Precision
            True,  # Is complex
            None,  # Singular part
            3,  # Kernel dimension
        )

        return PotentialOperator(
            PotentialAssembler(
                space, points, operator_descriptor, device_interface, assembler, parameters
            )
        )


def magnetic_field(
    space,
    points,
    wavenumber,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Maxwell magnetic field potential operator."""
    from bempp.api.operators import OperatorDescriptor
    from bempp.api.assembly.potential_operator import PotentialOperator
    from bempp.api.assembly.assembler import PotentialAssembler
    import bempp.api

    if space.identifier != "rwg0":
        raise ValueError("Space must be an RWG type function space.")

    if precision is None:
        precision = bempp.api.DEFAULT_PRECISION
    
    operator_descriptor = OperatorDescriptor(
        "maxwell_magnetic_field_potential",  # Identifier
        [_np.real(wavenumber), _np.imag(wavenumber)],  # Options
        "helmholtz_single_layer",  # Kernel type
        "maxwell_magnetic_field",  # Assembly type
        precision,  # Precision
        True,  # Is complex
        None,  # Singular part
        3,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )
