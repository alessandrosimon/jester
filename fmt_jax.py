from dataclasses import dataclass

import jax
import jax.numpy as jnp

from dsl import Functional

Array = jax.Array

@dataclass(frozen=True)
class FMTWeightDefinition:
    component_shape: tuple[int, ...]

FMT_WEIGHT_DEFINITIONS = {
    "scalar-0": FMTWeightDefinition(()),
    "scalar-1": FMTWeightDefinition(()),
    "scalar-2": FMTWeightDefinition(()),
    "scalar-3": FMTWeightDefinition(()),
    "vector-1": FMTWeightDefinition((3,)),
    "vector-2": FMTWeightDefinition((3,)),
}

@dataclass(frozen=True)
class PreparedWeightedDensity:
    name: str
    component_shape: tuple[int, ...]

    kernels: Array

def prepare_fmt_weighted_densities(
        functional: Functional,
        grid: FourierGrid
) -> tuple[PreparedWeightedDensity, ...]:
    species_weight_bank = tuple(
        build_fmt_weight_bank(
            grid=grid,
            radius=species.radius,
        )
        for species in functional.species)

    prepared = []

    for decl in functional.weighted_densities:
        definition = FMT_WEIGHT_DEFINITIONS[
            decl.weight
        ]

        kernels = jnp.stack([
            bank[decl.weight]
            for bank in species_weight_bank
        ])

        prepared.append(
            PreparedWeightedDensity(
                name=decl.name,
                component_shape=definition.component_shape,
                kernels=kernels
            )
        )

    return tuple(prepared)
        

def build_fmt_weight_bank(
    grid: FourierGrid,
    radius: float,
) -> dict[str, Array]:
    k_norm = grid.k_norm
    k_norm_safe = grid.k_norm_safe
    k_vec = grid.k_vec

    R = radius
    kR = k_norm * R
    is_k_zero = k_norm < grid.epsilon

    w3 = (
        4.0
        * jnp.pi
        * (jnp.sin(kR) - kR * jnp.cos(kR))
        / k_norm_safe**3
    )

    w2 = (
        4.0
        * jnp.pi
        * R
        * jnp.sin(kR)
        / k_norm_safe
    )

    w3 = jnp.where(
        is_k_zero,
        (4.0 / 3.0) * jnp.pi * R**3,
        w3,
    )

    w2 = jnp.where(
        is_k_zero,
        4.0 * jnp.pi * R**2,
        w2,
    )

    wv2 = -1.0j * k_vec * w3[None, ...]

    return {
        "scalar-0": w2 / (4.0 * jnp.pi * R**2),
        "scalar-1": w2 / (4.0 * jnp.pi * R),
        "scalar-2": w2,
        "scalar-3": w3,
        "vector-1": wv2 / (4.0 * jnp.pi * R),
        "vector-2": wv2,
    }
