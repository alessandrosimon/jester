from dataclasses import dataclass
from typing import ValuesView
from math import prod
from collections.abc import Callable

import jax
import jax.numpy as jnp
from numpy import number

from dsl import Functional
from eval_jax import Value, eval_sexp

Array = jax.Array

@dataclass(frozen=True)
class FourierGrid:
    shape: tuple[int, int, int]
    lengths: tuple[float, float, float]

    spacing: tuple[float, float, float]
    volume_element: float

    kx: Array
    ky: Array
    kz: Array

    k_vec: Array
    k_norm: Array
    k_norm_safe: Array
    is_k_zero: Array

    epsilon: float

    @classmethod
    def create(
            cls,
            shape: tuple[int, int, int],
            lengths: tuple[float, float, float],
            *,
            epsilon: float = 1e-12,
    ) -> "FourierGrid":
        if len(shape) != 3:
            ValueError()

        if len(lengths) != 3:
            ValueError()

        # add more input checks later

        spacing = tuple(
            length / n 
            for n, length in zip(shape, lengths, strict=True)
        )

        k_axes = tuple(
            2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dx)
            for n, dx in zip(shape, spacing, strict=True)
        )

        kx, ky, kz = jnp.meshgrid(
            *k_axes,
            indexing="ij",
        )

        k_vec = jnp.stack(
            (kx, ky, kz),
            axis=0
        )

        k_norm = jnp.sqrt(
            kx**2 + ky**2 + kz**2
        )

        is_k_zero = k_norm < epsilon

        k_norm_safe = jnp.where(
            is_k_zero,
            1.0,
            k_norm
        )
                
        return cls(
            shape=shape,
            lengths=lengths,
            spacing=spacing,
            volume_element=prod(spacing),
            kx=kx, ky=ky, kz=kz,
            k_vec=k_vec,
            k_norm=k_norm,
            k_norm_safe=k_norm_safe,
            is_k_zero=is_k_zero,
            epsilon=epsilon
        )
 

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

@dataclass(frozen=True)
class CompiledFMTFunctional:
    weighted_densities: Callable[
        [Array],
        dict[str, Array]]

    excess_energy_density: Callable[
        [Array],
        Array]

    excess_energy: Callable[
        [Array],
        Array]

    functional_derivative: Callable[
        [Array],
        Array]

    direct_correlation: Callable[
        [Array],
        Array,
    ]

def compile_fmt(
        functional: Functional,
        grid: FourierGrid
) -> CompiledFMTFunctional:
    prepared = prepare_fmt_weighted_densities(
        functional,
        grid)

    expression = functional.free_energy_density
    number_of_species = len(functional.species)
    dV = grid.volume_element

    def weighted_densities(
            rho: Array
    ) -> dict[str, Array]:
        return evaluate_fmt_weighted_densities(
            rho=rho,
            prepared=prepared,
            number_of_species=number_of_species)

    def excess_energy_density(
            rho: Array
    ) -> Array:
        environment = weighted_densities(rho)

        return eval_sexp(
            expression,
            environment
        )

    def excess_energy(
            rho: Array
    ) -> Array:
        return dV * jnp.sum(excess_energy_density(rho))

    discrete_grad = jax.grad(excess_energy)

    def functional_derivative(rho: Array) -> Array:
        return discrete_grad(rho) / dV

    def direct_correlation(rho: Array) -> Array:
        return -functional_derivative(rho)

    return CompiledFMTFunctional(
        weighted_densities=jax.jit(weighted_densities),
        excess_energy_density=jax.jit(excess_energy_density),
        excess_energy=jax.jit(excess_energy),
        functional_derivative=jax.jit(functional_derivative),
        direct_correlation=jax.jit(direct_correlation)
    )

def evaluate_fmt_weighted_densities(
        rho: Array,
        prepared: tuple[PreparedWeightedDensity, ...],
        number_of_species: int,
) -> dict[str, Array]:
    if rho.ndim == 3:
        if number_of_species != 1:
            raise ValueError()
        rho = rho[None, ...]

    if rho.shape[0] != number_of_species:
        raise ValueError()

    rho_ft = jnp.fft.fftn(
        rho,
        axes=(-3, -2, -1)
    )

    result: dict[str, Array] = {}

    for weighted_density in prepared:
        component_ndim = len(
            weighted_density.component_shape
        )

        expanded_rho_ft = rho_ft.reshape(
            (rho_ft.shape[0],)
            + (1,) * component_ndim
            + rho_ft.shape[1:]
        )

        weighted_ft = jnp.sum(
            expanded_rho_ft
            * weighted_density.kernels,
            axis=0
        )

        result[weighted_density.name] = jnp.fft.ifftn(
            weighted_ft,
            axes=(-3, -2, -1),
        ).real

    return result
    
        

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
